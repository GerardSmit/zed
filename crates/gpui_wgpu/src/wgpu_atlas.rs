use anyhow::{Context as _, Result};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

pub struct WgpuAtlas(Mutex<WgpuAtlasState>);

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    pending_uploads: Vec<PendingUpload>,
    upload_ring: Option<crate::upload_ring::UploadRing>,
    merged_uploads: bool,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
}

impl WgpuAtlas {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Mutex::new(WgpuAtlasState {
            device,
            queue,
            max_texture_size,
            color_texture_format,
            storage: WgpuAtlasStorage::default(),
            tiles_by_key: Default::default(),
            pending_uploads: Vec::new(),
            upload_ring: None,
            merged_uploads: false,
        }))
    }

    pub fn from_context(context: &WgpuContext) -> Self {
        Self::new(
            context.device.clone(),
            context.queue.clone(),
            context.color_texture_format(),
        )
    }

    pub(crate) fn before_external_frame(&self) -> Result<bool> {
        self.0.lock().flush_ring_uploads()
    }

    /// Copy each frame's new tiles into an atlas texture with as few copy regions as fit the
    /// staging buffer, instead of one region per tile; see [`merge_upload_regions`]. Only the
    /// external-target path (`before_external_frame`) merges.
    pub fn set_merged_uploads(&self, merged: bool) {
        self.0.lock().merged_uploads = merged;
    }

    pub fn before_frame(&self) {
        let mut lock = self.0.lock();
        lock.flush_uploads();
    }

    pub(crate) fn texture_bind_group(
        &self,
        id: AtlasTextureId,
        create: impl FnOnce(&wgpu::TextureView) -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        let lock = self.0.lock();
        let texture = &lock.storage[id];
        texture
            .bind_group
            .get_or_init(|| create(&texture.view))
            .clone()
    }

    pub fn get_texture_info(&self, id: AtlasTextureId) -> WgpuTextureInfo {
        let lock = self.0.lock();
        let texture = &lock.storage[id];
        WgpuTextureInfo {
            view: texture.view.clone(),
        }
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    pub fn clear(&self) {
        let mut lock = self.0.lock();
        lock.storage = WgpuAtlasStorage::default();
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
        lock.upload_ring = None;
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        let mut lock = self.0.lock();
        lock.device = context.device.clone();
        lock.queue = context.queue.clone();
        lock.color_texture_format = context.color_texture_format();
        lock.storage = WgpuAtlasStorage::default();
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
        lock.upload_ring = None;
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            profiling::scope!("new tile");
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock
                .allocate(size, key.texture_kind())
                .context("failed to allocate")?;
            lock.upload_texture(tile.texture_id, tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();

        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;

        let Some(texture_slot) = lock.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                lock.pending_uploads
                    .retain(|upload| upload.id != texture.id);
                lock.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl WgpuAtlasState {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | ATLAS_TEST_USAGE,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
            bind_group: Default::default(),
            shadow: (self.merged_uploads && texture_bytes(size, format) <= SHADOW_MAX_BYTES).then(|| {
                let bpp = format.block_copy_size(None).unwrap_or(4) as usize;
                vec![0; size.width.0 as usize * size.height.0 as usize * bpp]
            }),
            initialized: false,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());
        if let Some(texture) = self.storage.get_mut(id) {
            texture.write_shadow(bounds, &data);
        }

        self.pending_uploads
            .push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn flush_ring_uploads(&mut self) -> Result<bool> {
        if self.pending_uploads.is_empty() { return Ok(true); }
        let mut separate = 0u64;
        for upload in &self.pending_uploads {
            let Some(texture) = self.storage.get(upload.id) else { continue; };
            let width = u64::try_from(upload.bounds.size.width.0)?;
            let height = u64::try_from(upload.bounds.size.height.0)?;
            let x = u32::try_from(upload.bounds.origin.x.0)?;
            let y = u32::try_from(upload.bounds.origin.y.0)?;
            anyhow::ensure!(width > 0 && height > 0, "Empty atlas upload");
            anyhow::ensure!(u64::from(x).checked_add(width).is_some_and(|right| right <= u64::from(texture.texture.width()))
                && u64::from(y).checked_add(height).is_some_and(|bottom| bottom <= u64::from(texture.texture.height())),
                "Atlas upload exceeds texture bounds");
            let row = width.checked_mul(texture.bytes_per_pixel() as u64).context("Atlas row overflow")?;
            anyhow::ensure!(row.checked_mul(height) == Some(upload.data.len() as u64), "Invalid atlas upload size");
            u32::try_from(staging_pitch(row)).context("Atlas pitch exceeds copy layout")?;
            separate = separate.checked_add(staging_pitch(row) * height).context("Atlas staging overflow")?;
        }
        if separate == 0 { self.pending_uploads.clear(); return Ok(true); }
        anyhow::ensure!(separate <= self.device.limits().max_buffer_size && separate <= 64 * 1024 * 1024,
            "Atlas upload exceeds staging budget");
        // Merging never asks for more staging than one region per tile would, beyond the
        // smallest ring, so the ring grows exactly as it did without merging.
        let ring_capacity = self.upload_ring.as_ref().map_or(0, |ring| ring.capacity());
        let budget = separate.max(ring_capacity).max(MIN_STAGING_BYTES);
        let plan = self.plan_upload_regions(separate, budget);
        let required = plan.iter().map(|(_, region)| region.staging_bytes).sum::<u64>();
        if self.upload_ring.as_ref().is_none_or(|ring| ring.capacity() < required) {
            let capacity = required.max(MIN_STAGING_BYTES).checked_next_power_of_two().context("Atlas staging capacity overflow")?;
            anyhow::ensure!(capacity <= self.device.limits().max_buffer_size, "Atlas staging exceeds device limit");
            self.upload_ring = Some(crate::upload_ring::UploadRing::new(&self.device, capacity, false));
        }
        let ring = self.upload_ring.as_ref().context("Atlas upload ring missing")?;
        anyhow::ensure!(!ring.failed(), "Atlas staging mapping failed");
        let Some(index) = ring.acquire() else { return Ok(false); };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpui_atlas_uploads"),
        });
        let mut offset = 0;
        let mut staging = Vec::new();
        let mut init_regions = 0;
        let mut init_bytes = 0;
        for (position, (id, region)) in plan.iter().enumerate() {
            let texture = &self.storage[*id];
            let rect = region.rect;
            texture.compose(rect, &self.pending_uploads, &mut staging);
            ring.write(index, offset, &staging)?;
            let first = plan[..position].iter().all(|(earlier, _)| earlier != id);
            if first && !texture.initialized && !rect.covers(texture.texture.width(), texture.texture.height()) {
                // wgpu zero-fills the whole texture before the first copy that leaves texels
                // undefined, with one region per 512 KiB of its zero buffer.
                let bytes = texture_bytes(Size {
                    width: DevicePixels(texture.texture.width() as i32),
                    height: DevicePixels(texture.texture.height() as i32),
                }, texture.format);
                init_regions += bytes.div_ceil(ZERO_BUFFER_BYTES) as u32;
                init_bytes += bytes;
            }
            encoder.copy_buffer_to_texture(wgpu::TexelCopyBufferInfo {
                buffer: ring.buffer(index),
                layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(region.pitch as u32), rows_per_image: None },
            }, wgpu::TexelCopyTextureInfo {
                texture: &texture.texture, mip_level: 0,
                origin: wgpu::Origin3d { x: rect.x0, y: rect.y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            }, wgpu::Extent3d { width: rect.width(), height: rect.height(), depth_or_array_layers: 1 });
            offset += region.staging_bytes;
        }
        for (id, _) in &plan {
            if let Some(texture) = self.storage.get_mut(*id) { texture.initialized = true; }
        }
        ring.unmap(index);
        crate::pass_trace::note(|| format!("atlas_uploads n={} regions={} bytes={required} separate_bytes={separate}",
            self.pending_uploads.len(), plan.len()));
        crate::pass_trace::transfer(crate::pass_trace::Transfer::Atlas, plan.len() as u32, required);
        crate::pass_trace::transfer(crate::pass_trace::Transfer::AtlasInit, init_regions, init_bytes);
        self.queue.submit([encoder.finish()]);
        ring.submitted(index);
        self.pending_uploads.clear();
        Ok(true)
    }

    /// The copy regions for the pending uploads, per texture in the order the textures first
    /// appear among them. Without merging, one region per upload.
    fn plan_upload_regions(&self, separate: u64, budget: u64) -> Vec<(AtlasTextureId, UploadRegion)> {
        let mut plan = Vec::new();
        let mut textures: Vec<AtlasTextureId> = Vec::new();
        for upload in &self.pending_uploads {
            if self.storage.get(upload.id).is_some() && !textures.contains(&upload.id) {
                textures.push(upload.id);
            }
        }
        let mut total = separate;
        for id in textures {
            let texture = &self.storage[id];
            let bpp = texture.bytes_per_pixel() as u64;
            let rects: Vec<Rect> = self.pending_uploads.iter()
                .filter(|upload| upload.id == id)
                .map(|upload| Rect::of(upload.bounds))
                .collect();
            let regions = if self.merged_uploads {
                let obstacles = if texture.shadow.is_some() {
                    Vec::new()
                } else {
                    let span = rects.iter().copied().reduce(Rect::union).expect("texture has uploads");
                    // Tiles uploaded in earlier frames hold texels the staging cannot reproduce.
                    self.tiles_by_key.values()
                        .filter(|tile| tile.texture_id == id)
                        .map(|tile| Rect::of(tile.bounds))
                        .filter(|rect| rect.intersects(span) && !rects.contains(rect))
                        .collect()
                };
                merge_upload_regions(&rects, &obstacles, bpp, &mut total, budget)
            } else {
                rects
            };
            plan.extend(regions.into_iter().map(|rect| {
                let pitch = staging_pitch(u64::from(rect.width()) * bpp);
                (id, UploadRegion { rect, pitch, staging_bytes: pitch * u64::from(rect.height()) })
            }));
        }
        plan
    }
}

/// Tests read atlas textures back.
const ATLAS_TEST_USAGE: wgpu::TextureUsages = if cfg!(test) { wgpu::TextureUsages::COPY_SRC } else { wgpu::TextureUsages::empty() };
const MIN_STAGING_BYTES: u64 = 512 * 1024;
/// wgpu-core's zero buffer, which clears textures in copies of at most this size.
const ZERO_BUFFER_BYTES: u64 = 512 * 1024;
/// Textures up to this size keep a CPU copy, so that merged uploads can include texels of
/// tiles uploaded in earlier frames: a 1024x1024 monochrome atlas, not a colour one.
const SHADOW_MAX_BYTES: u64 = 1024 * 1024;

fn staging_pitch(row: u64) -> u64 {
    row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64)
}

fn texture_bytes(size: Size<DevicePixels>, format: wgpu::TextureFormat) -> u64 {
    let bpp = format.block_copy_size(None).unwrap_or(4) as u64;
    staging_pitch(size.width.0.max(0) as u64 * bpp) * size.height.0.max(0) as u64
}

struct UploadRegion {
    rect: Rect,
    pitch: u64,
    staging_bytes: u64,
}

/// Half-open texel rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl Rect {
    fn of(bounds: Bounds<DevicePixels>) -> Self {
        let x0 = bounds.origin.x.0.max(0) as u32;
        let y0 = bounds.origin.y.0.max(0) as u32;
        Self { x0, y0, x1: x0 + bounds.size.width.0.max(0) as u32, y1: y0 + bounds.size.height.0.max(0) as u32 }
    }
    fn width(self) -> u32 { self.x1 - self.x0 }
    fn height(self) -> u32 { self.y1 - self.y0 }
    fn union(self, other: Self) -> Self {
        Self { x0: self.x0.min(other.x0), y0: self.y0.min(other.y0), x1: self.x1.max(other.x1), y1: self.y1.max(other.y1) }
    }
    fn intersects(self, other: Self) -> bool {
        self.x0 < other.x1 && other.x0 < self.x1 && self.y0 < other.y1 && other.y0 < self.y1
    }
    fn contains(self, other: Self) -> bool {
        self.x0 <= other.x0 && self.y0 <= other.y0 && other.x1 <= self.x1 && other.y1 <= self.y1
    }
    fn covers(self, width: u32, height: u32) -> bool {
        self.x0 == 0 && self.y0 == 0 && self.x1 >= width && self.y1 >= height
    }
    fn staging_bytes(self, bpp: u64) -> u64 {
        staging_pitch(u64::from(self.width()) * bpp) * u64::from(self.height())
    }
}

/// Merges upload rectangles into fewer, larger copy regions. PowerVR's Vulkan driver runs
/// one transfer job per copy region, each with a fixed cost of a few hundred microseconds,
/// so a region that also copies unused texels is far cheaper than another region. A merged
/// region never covers an obstacle (texels it cannot reproduce), and merging stops once the
/// staging bytes of all regions, `total`, would exceed `budget`.
fn merge_upload_regions(rects: &[Rect], obstacles: &[Rect], bpp: u64, total: &mut u64, budget: u64) -> Vec<Rect> {
    let mut sorted = rects.to_vec();
    sorted.sort_by_key(|rect| (rect.y0, rect.x0));
    sorted.dedup();
    let mut regions: Vec<Rect> = Vec::new();
    let try_merge = |a: Rect, b: Rect, total: u64| -> Option<(Rect, u64)> {
        let merged = a.union(b);
        let next = (total + merged.staging_bytes(bpp)).checked_sub(a.staging_bytes(bpp) + b.staging_bytes(bpp))?;
        (next <= budget && !obstacles.iter().any(|obstacle| obstacle.intersects(merged))).then_some((merged, next))
    };
    for rect in sorted {
        if regions.iter().any(|region| region.contains(rect)) {
            *total -= rect.staging_bytes(bpp);
            continue;
        }
        let merged = regions.iter().enumerate().rev()
            .find_map(|(index, region)| try_merge(*region, rect, *total).map(|merge| (index, merge)));
        match merged {
            Some((index, (region, next))) => { regions[index] = region; *total = next; }
            None => regions.push(rect),
        }
    }
    // A merge can make two regions mergeable that were not before.
    'merge: loop {
        for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                if let Some((region, next)) = try_merge(regions[i], regions[j], *total) {
                    regions[i] = region;
                    regions.swap_remove(j);
                    *total = next;
                    continue 'merge;
                }
            }
        }
        break;
    }
    regions
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get(id.index as usize)
            .and_then(|t| t.as_ref())
    }

    fn get_mut(&mut self, id: AtlasTextureId) -> Option<&mut WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get_mut(id.index as usize)
            .and_then(|t| t.as_mut())
    }
}

impl ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize]
            .as_ref()
            .expect("texture must exist")
    }
}

struct WgpuAtlasTexture {
    bind_group: std::sync::OnceLock<wgpu::BindGroup>,
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
    /// The texels of every tile uploaded so far, rows packed without padding.
    shadow: Option<Vec<u8>>,
    /// Whether a copy has reached the texture, after which wgpu no longer zero-fills it.
    initialized: bool,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn write_shadow(&mut self, bounds: Bounds<DevicePixels>, data: &[u8]) {
        let bpp = self.bytes_per_pixel() as usize;
        let stride = self.texture.width() as usize * bpp;
        let Some(shadow) = self.shadow.as_mut() else { return; };
        let rect = Rect::of(bounds);
        let row = rect.width() as usize * bpp;
        if row == 0 || data.len() != row * rect.height() as usize { return; }
        for (y, source) in (rect.y0 as usize..).zip(data.chunks_exact(row)) {
            let start = y * stride + rect.x0 as usize * bpp;
            if let Some(target) = shadow.get_mut(start..start + row) {
                target.copy_from_slice(source);
            }
        }
    }

    /// Fills `staging` with the rows of `rect` at the staging pitch: from the shadow when the
    /// texture has one, else zeros overlaid with the pending uploads in order.
    fn compose(&self, rect: Rect, uploads: &[PendingUpload], staging: &mut Vec<u8>) {
        let bpp = self.bytes_per_pixel() as usize;
        let row = rect.width() as usize * bpp;
        let pitch = staging_pitch(row as u64) as usize;
        staging.clear();
        staging.resize(pitch * rect.height() as usize, 0);
        if let Some(shadow) = &self.shadow {
            let stride = self.texture.width() as usize * bpp;
            for (y, target) in (rect.y0 as usize..).zip(staging.chunks_exact_mut(pitch)) {
                let start = y * stride + rect.x0 as usize * bpp;
                target[..row].copy_from_slice(&shadow[start..start + row]);
            }
            return;
        }
        for upload in uploads.iter().filter(|upload| upload.id == self.id) {
            let source = Rect::of(upload.bounds);
            if !source.intersects(rect) { continue; }
            let clip = Rect {
                x0: source.x0.max(rect.x0), y0: source.y0.max(rect.y0),
                x1: source.x1.min(rect.x1), y1: source.y1.min(rect.y1),
            };
            let source_row = source.width() as usize * bpp;
            let length = clip.width() as usize * bpp;
            for y in clip.y0..clip.y1 {
                let from = (y - source.y0) as usize * source_row + (clip.x0 - source.x0) as usize * bpp;
                let to = (y - rect.y0) as usize * pitch + (clip.x0 - rect.x0) as usize * bpp;
                staging[to..to + length].copy_from_slice(&upload.data[from..from + length]);
            }
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        }
        _ => bytes.to_vec(),
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("wgpu_atlas_test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas
            .get_or_insert_with(&key, &mut build)?
            .expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn cached_binding_is_recreated_when_an_atlas_slot_is_reused() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device.clone(), queue, wgpu::TextureFormat::Bgra8Unorm);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let builds = std::cell::Cell::new(0);
        let binding = |view: &wgpu::TextureView| {
            builds.set(builds.get() + 1);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                }],
            })
        };
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let insert = || {
            atlas
                .get_or_insert_with(&key, &mut || {
                    Ok(Some((
                        gpui::size(DevicePixels(1), DevicePixels(1)),
                        Cow::Owned(vec![0, 0, 0, 255]),
                    )))
                })
                .map(|tile| tile.unwrap().texture_id)
        };
        let first = insert()?;
        let old_binding = atlas.texture_bind_group(first, binding);
        atlas.texture_bind_group(first, binding);
        assert_eq!(builds.get(), 1);
        atlas.remove(&key);
        let second = insert()?;
        assert_eq!(first, second);
        atlas.texture_bind_group(second, binding);
        assert_eq!(builds.get(), 2);
        // Keeping the old GPU binding alive must not associate it with the replacement texture.
        drop(old_binding);
        atlas.clear();
        let third = insert()?;
        assert_eq!(second, third);
        atlas.texture_bind_group(third, binding);
        assert_eq!(builds.get(), 3);
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: &AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(&keeper_key, small);
        let tile_a = insert(&big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(&big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, bpp: u32) -> Vec<u8> {
        let row = texture.width() * bpp;
        let pitch = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(pitch * texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(pitch), rows_per_image: None },
            },
            texture.size(),
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let bytes = readback.slice(..).get_mapped_range();
        bytes.chunks_exact(pitch as usize).flat_map(|line| &line[..row as usize]).copied().collect()
    }

    /// Inserts and removes glyph- and image-sized tiles over several frames, flushing each frame
    /// the way external frames do, and returns every texture's texels with the live tiles.
    fn render_atlas_frames(merged: bool) -> anyhow::Result<(Vec<(AtlasTextureId, Vec<u8>)>, Vec<(AtlasTile, Vec<u8>)>)> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device.clone(), queue.clone(), wgpu::TextureFormat::Bgra8Unorm);
        atlas.set_merged_uploads(merged);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move |range: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(range)) as u32
        };
        let mut live: Vec<(AtlasKey, AtlasTile, Vec<u8>)> = Vec::new();
        let mut next_id = 0usize;
        for _frame in 0..8 {
            for _ in 0..40 {
                if !live.is_empty() && random(5) == 0 {
                    let (key, _, _) = live.swap_remove(random(live.len() as u32) as usize);
                    atlas.remove(&key);
                    continue;
                }
                next_id += 1;
                let image = random(4) == 0;
                let size = if image {
                    gpui::size(DevicePixels(8 + random(90) as i32), DevicePixels(8 + random(90) as i32))
                } else {
                    gpui::size(DevicePixels(3 + random(30) as i32), DevicePixels(6 + random(4) as i32 * 8))
                };
                let key = if image {
                    AtlasKey::Image(RenderImageParams { image_id: ImageId(next_id), frame_index: 0 })
                } else {
                    AtlasKey::Svg(gpui::RenderSvgParams { path: format!("{next_id}").into(), size })
                };
                let bpp = if image { 4 } else { 1 };
                let bytes: Vec<u8> = (0..size.width.0 * size.height.0 * bpp).map(|_| random(255) as u8 + 1).collect();
                let tile = atlas
                    .get_or_insert_with(&key, &mut || Ok(Some((size, Cow::Owned(bytes.clone())))))?
                    .expect("tile");
                live.push((key, tile, bytes));
            }
            assert!(atlas.before_external_frame()?, "staging slot available");
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        }
        let lock = atlas.0.lock();
        let mut textures = Vec::new();
        for kind in [AtlasTextureKind::Monochrome, AtlasTextureKind::Polychrome] {
            for texture in lock.storage[kind].textures.iter().flatten() {
                assert_eq!(texture.shadow.is_some(), merged && kind == AtlasTextureKind::Monochrome);
                let bpp = texture.bytes_per_pixel() as u32;
                textures.push((texture.id, read_texture(&device, &queue, &texture.texture, bpp)));
            }
        }
        Ok((textures, live.into_iter().map(|(_, tile, bytes)| (tile, bytes)).collect()))
    }

    #[test]
    fn merged_uploads_write_the_same_texels_as_one_region_per_tile() -> anyhow::Result<()> {
        let (regions, regions_tiles) = render_atlas_frames(false)?;
        let (merged, merged_tiles) = render_atlas_frames(true)?;
        assert_eq!(regions.len(), merged.len());
        for ((tile, bytes), (merged_tile, _)) in regions_tiles.iter().zip(&merged_tiles) {
            assert_eq!(tile, merged_tile, "allocation does not depend on the upload mode");
            for (texels, label) in [(&regions, "regions"), (&merged, "merged")] {
                let (_, texels) = texels.iter().find(|(id, _)| *id == tile.texture_id).unwrap();
                let bpp = if tile.texture_id.kind == AtlasTextureKind::Monochrome { 1 } else { 4 };
                let stride = 1024 * bpp;
                let row = tile.bounds.size.width.0 as usize * bpp;
                for (y, expected) in bytes.chunks_exact(row).enumerate() {
                    let start = (tile.bounds.origin.y.0 as usize + y) * stride + tile.bounds.origin.x.0 as usize * bpp;
                    let mut expected = expected.to_vec();
                    if bpp == 4 {
                        expected = swizzle_upload_data(&expected, wgpu::TextureFormat::Bgra8Unorm);
                    }
                    assert_eq!(&texels[start..start + row], &expected[..], "{label} tile {tile:?} row {y}");
                }
            }
        }
        // The CPU copy of a monochrome texture reproduces even texels of removed tiles.
        for ((id, texels), (merged_id, merged_texels)) in regions.iter().zip(&merged) {
            assert_eq!(id, merged_id);
            if id.kind == AtlasTextureKind::Monochrome {
                assert!(texels == merged_texels, "monochrome texture {id:?} differs");
            }
        }
        Ok(())
    }

    fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    #[test]
    fn merged_regions_cover_uploads_and_avoid_obstacles() {
        // Two shelves of new glyphs; an older glyph sits between them on the left.
        let uploads = [rect(40, 0, 50, 16), rect(50, 0, 58, 16), rect(58, 0, 70, 12),
            rect(30, 32, 40, 48), rect(40, 32, 52, 48)];
        let obstacles = [rect(0, 16, 30, 32), rect(0, 0, 40, 16)];
        let separate: u64 = uploads.iter().map(|rect| rect.staging_bytes(1)).sum();
        let mut total = separate;
        let regions = merge_upload_regions(&uploads, &obstacles, 1, &mut total, MIN_STAGING_BYTES);
        assert!(regions.len() < uploads.len(), "{regions:?}");
        for upload in uploads {
            assert!(regions.iter().any(|region| region.contains(upload)), "{upload:?} in {regions:?}");
        }
        for region in &regions {
            assert!(!obstacles.iter().any(|obstacle| obstacle.intersects(*region)), "{region:?}");
        }
        assert_eq!(total, regions.iter().map(|region| region.staging_bytes(1)).sum::<u64>());

        // Without obstacles everything becomes one region, within the budget.
        let mut total = separate;
        let regions = merge_upload_regions(&uploads, &[], 1, &mut total, MIN_STAGING_BYTES);
        assert_eq!(regions, vec![rect(30, 0, 70, 48)]);

        // A budget no larger than the separate regions' staging allows no growth.
        let far = [rect(0, 0, 4, 4), rect(1000, 1000, 1004, 1004)];
        let separate: u64 = far.iter().map(|rect| rect.staging_bytes(4)).sum();
        let mut total = separate;
        assert_eq!(merge_upload_regions(&far, &[], 4, &mut total, separate), far.to_vec());
        assert_eq!(total, separate);
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm),
            input
        );
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }
}
