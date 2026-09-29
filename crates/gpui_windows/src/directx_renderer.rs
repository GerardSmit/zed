use std::{
    slice,
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
use smallvec::SmallVec;
use windows::{
    Win32::{
        Foundation::{HWND, RECT, S_OK},
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
        },
    },
    core::{HSTRING, Interface},
};

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::*;
use gpui::*;

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";
const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
const PATH_CACHE_BYTES: u64 = 16 * 1024 * 1024;
const LAYER_CACHE_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    atlas: Arc<DirectXAtlas>,
    devices: Option<DirectXRendererDevices>,
    resources: Option<DirectXResources>,
    globals: DirectXGlobalElements,
    pipelines: DirectXRenderPipelines,
    direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,

    width: u32,
    height: u32,

    last_background_appearance: Option<WindowBackgroundAppearance>,
    had_remote_surfaces: bool,

    /// Cached upload texture for remote-window surfaces (M1). Recreated when the frame size changes.
    surface_texture: Option<RemoteSurfaceTexture>,

    /// Offscreen render-target textures for cached view layers, keyed by `LayerId`. A layered view
    /// renders its subtree into its texture only when its content changes; on resize the texture is
    /// re-composited (stretched) without re-rendering. See `render_layers` / `draw_surfaces`.
    layers: std::collections::HashMap<u64, LayerTexture>,
    live_layers: std::collections::HashSet<u64>,
    staging_scene: Scene,
    path_vertices: Vec<PathRasterizationSprite>,
    path_sprites: Vec<PathSprite>,
    #[cfg(test)]
    scene_uploads: usize,
    #[cfg(test)]
    uploaded_primitive_bytes: usize,

    /// Whether we want to skip drwaing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    skip_draws: bool,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    partial_clear_context: Option<ID3D11DeviceContext1>,
    dxgi_device: Option<IDXGIDevice>,
    annotation: Option<ID3DUserDefinedAnnotation>,
}

struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,

    path_intermediate: Option<PathIntermediate>,
    path_cache: Vec<PathIntermediate>,
    path_intermediate_origin: Point<ScaledPixels>,
    #[cfg(test)]
    path_allocations: usize,

    // Cached viewport
    viewport: D3D11_VIEWPORT,
}

struct PathIntermediate {
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
    msaa_texture: ID3D11Texture2D,
    msaa_view: Option<ID3D11RenderTargetView>,
    size: (u32, u32),
}

impl PathIntermediate {
    fn bytes(&self) -> u64 {
        self.size.0 as u64 * self.size.1 as u64 * 4 * (PATH_MULTISAMPLE_COUNT + 1) as u64
    }
}

struct DirectXRenderPipelines {
    shadow_pipeline: PipelineState<Shadow>,
    quad_pipeline: PipelineState<Quad>,
    path_rasterization_pipeline: PipelineState<PathRasterizationSprite>,
    path_sprite_pipeline: PipelineState<PathSprite>,
    underline_pipeline: PipelineState<Underline>,
    mono_sprites: PipelineState<MonochromeSprite>,
    subpixel_sprites: PipelineState<SubpixelSprite>,
    poly_sprites: PipelineState<PolychromeSprite>,
    surface_pipeline: PipelineState<SurfaceSprite>,
}

/// One instance for the surface pipeline; mirrors the HLSL `SurfaceSprite` (two `Bounds`, a
/// `ContentFade` and a `float2` = 14 floats). `tex_size` is the layer texture's device size for a cached-view-layer
/// composite (1:1, crisp, alpha-preserving), or `[0, 0]` for a stretched opaque image surface.
#[derive(Clone, Copy)]
#[repr(C)]
struct SurfaceSprite {
    bounds: Bounds<ScaledPixels>,
    content_mask: Bounds<ScaledPixels>,
    content_fade: ContentFade<ScaledPixels>,
    tex_size: [f32; 2],
}

/// A cached dynamic BGRA texture (+ SRV) for uploading remote-window frames each draw.
struct RemoteSurfaceTexture {
    width: u32,
    height: u32,
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
}

/// An offscreen RGBA render target (RTV to draw into, SRV to composite from) backing one cached
/// view layer. Sized to the layered view's device bounds; recreated when that size changes.
struct LayerTexture {
    width: u32,
    height: u32,
    #[allow(dead_code)]
    texture: ID3D11Texture2D,
    rtv: Option<ID3D11RenderTargetView>,
    srv: Option<ID3D11ShaderResourceView>,
    /// Consecutive frames this layer was not composited. Once it exceeds `LAYER_EVICT_FRAMES` the
    /// texture is dropped (the layered view was closed/hidden), freeing its VRAM.
    unseen: u32,
}

/// Drop a layer texture after this many frames without being composited.
const LAYER_EVICT_FRAMES: u32 = 240;

struct DirectXGlobalElements {
    global_params_buffer: Option<ID3D11Buffer>,
    batch_params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
}

struct Annotation<'a>(&'a ID3DUserDefinedAnnotation);

impl<'a> Annotation<'a> {
    fn new(annotation: &'a ID3DUserDefinedAnnotation, label: HSTRING) -> Self {
        unsafe { annotation.BeginEvent(&label) };
        Self(annotation)
    }
}

impl Drop for Annotation<'_> {
    fn drop(&mut self) {
        unsafe { self.0.EndEvent() };
    }
}

struct DirectComposition {
    comp_device: IDCompositionDevice,
    comp_target: IDCompositionTarget,
    comp_visual: IDCompositionVisual,
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };
        let annotation = device_context.cast().ok();
        let mut options = D3D11_FEATURE_DATA_D3D11_OPTIONS::default();
        let supports_partial_clear = unsafe {
            device.CheckFeatureSupport(
                D3D11_FEATURE_D3D11_OPTIONS,
                &mut options as *mut _ as _,
                std::mem::size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.ClearView.as_bool();
        let partial_clear_context = supports_partial_clear
            .then(|| device_context.cast().ok())
            .flatten();

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            partial_clear_context,
            dxgi_device,
            annotation,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            direct_composition,
            font_info: Self::get_font_info(),
            width: 1,
            height: 1,
            last_background_appearance: None,
            had_remote_surfaces: false,
            surface_texture: None,
            layers: std::collections::HashMap::new(),
            live_layers: std::collections::HashSet::new(),
            staging_scene: Scene::default(),
            path_vertices: Vec::new(),
            path_sprites: Vec::new(),
            #[cfg(test)]
            scene_uploads: 0,
            #[cfg(test)]
            uploaded_primitive_bytes: 0,
            skip_draws: false,
        })
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(&self, clear_color: &[f32; 4], damage: SceneDamage) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let devices = self.devices.as_ref().expect("devices missing");
        let device_context = &devices.device_context;
        self.update_viewport(&resources.viewport)?;
        let render_target_view = resources
            .render_target_view
            .as_ref()
            .context("missing render target view")?;
        unsafe {
            if damage != SceneDamage::Full {
                let rects = damage_rects(damage, self.width, self.height);
                devices
                    .partial_clear_context
                    .as_ref()
                    .context("partial render target clears are unavailable")?
                    .ClearView(render_target_view, clear_color, Some(&rects));
            } else {
                device_context.ClearRenderTargetView(render_target_view, clear_color);
            }
            device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
            device_context
                .VSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            device_context
                .VSSetConstantBuffers(1, Some(slice::from_ref(&self.globals.batch_params_buffer)));
            device_context
                .PSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
        }
        Ok(())
    }

    fn update_viewport(&self, viewport: &D3D11_VIEWPORT) -> Result<()> {
        let device_context = &self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context;
        update_buffer(
            device_context,
            self.globals.global_params_buffer.as_ref().unwrap(),
            &[GlobalParams {
                gamma_ratios: self.font_info.gamma_ratios,
                viewport_size: [viewport.Width, viewport.Height],
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
                is_bgr: self.font_info.is_bgr as u32,
                _pad: [0; 3],
            }],
        )?;
        unsafe { device_context.RSSetViewports(Some(slice::from_ref(viewport))) };
        Ok(())
    }

    #[inline]
    fn present(&mut self, damage: SceneDamage) -> Result<bool> {
        let mut rects = if damage == SceneDamage::Full {
            SmallVec::<[RECT; 8]>::new()
        } else {
            damage_rects(damage, self.width, self.height)
        };
        let parameters = DXGI_PRESENT_PARAMETERS {
            DirtyRectsCount: rects.len() as u32,
            pDirtyRects: if rects.is_empty() {
                std::ptr::null_mut()
            } else {
                rects.as_mut_ptr()
            },
            ..Default::default()
        };
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present1(0, DXGI_PRESENT(0), &parameters)
        };
        result.ok().context("Presenting swap chain failed")?;
        Ok(result == S_OK)
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        self.last_background_appearance = None;
        let disable_direct_composition = self.direct_composition.is_none();

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            // Layer textures belong to the lost device; drop them so they're recreated fresh.
            self.layers.clear();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.globals = globals;
        self.pipelines = pipelines;
        self.direct_composition = direct_composition;
        self.skip_draws = true;
        Ok(())
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if self.skip_draws {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            return Ok(());
        }

        let remote_surfaces = crate::remote_surface::take_surfaces();
        let damage = if self.last_background_appearance != Some(background_appearance)
            || self.had_remote_surfaces
            || !remote_surfaces.is_empty()
        {
            SceneDamage::Full
        } else {
            scene.damage
        };
        let previous_background = self.last_background_appearance.take();
        // A failed clear, layer render or present invalidates the next incremental frame.
        // Offscreen layer changes must be uploaded even if the window's clip hides their damage.
        self.render_layers(scene, previous_background.is_none())?;
        self.evict_stale_layers(scene);
        if damage
            .pixel_rects(size(
                DevicePixels(self.width as i32),
                DevicePixels(self.height as i32),
            ))
            .len()
            == 0
        {
            self.last_background_appearance = previous_background;
            return Ok(());
        }
        let damage = self.effective_damage(damage);
        self.render(scene, background_appearance, damage)?;
        // Remote-window frames are pushed out-of-band (not as scene primitives, to avoid editing the
        // read-only base gpui). Composit them on top after the scene.
        self.draw_remote_surfaces(&remote_surfaces)?;
        if self.present(damage)? {
            self.last_background_appearance = Some(background_appearance);
        }
        self.had_remote_surfaces = !remote_surfaces.is_empty();
        Ok(())
    }

    fn effective_damage(&self, damage: SceneDamage) -> SceneDamage {
        if self
            .devices
            .as_ref()
            .is_some_and(|devices| devices.partial_clear_context.is_some())
        {
            damage
        } else {
            SceneDamage::Full
        }
    }

    /// Clear the render target for `background_appearance` and encode every
    /// primitive batch of `scene` into it, without presenting. Shared by
    /// [`draw`](Self::draw) (which then presents) and
    /// [`render_to_image`](Self::render_to_image) (which reads the target back
    /// instead), so the two cannot drift.
    fn render(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
        damage: SceneDamage,
    ) -> Result<()> {
        self.pre_draw(
            &match background_appearance {
                WindowBackgroundAppearance::Opaque => [1.0f32; 4],
                _ => [0.0f32; 4],
            },
            damage,
        )?;
        let render_target_view = self
            .resources
            .as_ref()
            .context("resources missing")?
            .render_target_view
            .clone();
        self.render_scene(scene, &render_target_view, damage)
    }

    /// Drop layer textures whose view stopped compositing (closed/hidden tab or panel), so their
    /// VRAM isn't held forever. A layer is "live" this frame if a `PaintSurface(Layer)` referenced
    /// it; the composite surface is replayed every frame (even on cache reuse), so this is reliable.
    fn evict_stale_layers(&mut self, scene: &Scene) {
        if self.layers.is_empty() {
            return;
        }
        fn collect_live(scene: &Scene, live: &mut std::collections::HashSet<u64>) {
            for surface in &scene.surfaces {
                if let PaintSurfaceSource::Layer(id) = surface.source {
                    live.insert(id.0);
                }
            }
            for layer in &scene.layers {
                if let Some(scene) = layer.scene.as_deref() {
                    collect_live(scene, live);
                }
            }
        }
        self.live_layers.clear();
        collect_live(scene, &mut self.live_layers);
        self.layers.retain(|id, texture| {
            if self.live_layers.contains(id) {
                texture.unseen = 0;
                true
            } else {
                texture.unseen += 1;
                texture.unseen < LAYER_EVICT_FRAMES
            }
        });
        let mut bytes: u64 = self
            .layers
            .values()
            .map(|t| t.width as u64 * t.height as u64 * 4)
            .sum();
        while bytes > LAYER_CACHE_BYTES {
            let Some(id) = self
                .layers
                .iter()
                .filter(|(_, t)| t.unseen > 0)
                .max_by_key(|(_, t)| (t.unseen, t.width as u64 * t.height as u64))
                .map(|(id, _)| *id)
            else {
                break;
            };
            let texture = self.layers.remove(&id).unwrap();
            bytes -= texture.width as u64 * texture.height as u64 * 4;
        }
    }

    /// Draw a scene's primitive batches into the currently-bound render target, which is
    /// `render_target_view`. Shared by the main pass and the per-layer offscreen passes: the path
    /// pipeline rasterizes into the shared MSAA intermediate and then rebinds `render_target_view`
    /// to copy the result back, so the pass it interrupts carries on where it left off.
    fn render_scene(
        &mut self,
        scene: &Scene,
        render_target_view: &Option<ID3D11RenderTargetView>,
        damage: SceneDamage,
    ) -> Result<()> {
        let mut staging = std::mem::take(&mut self.staging_scene);
        let scene = if damage == SceneDamage::Full {
            scene
        } else {
            scene.copy_primitives_for_damage(damage, &mut staging);
            &staging
        };
        let rendered = (|| {
            self.upload_scene_buffers(scene)?;
            let viewport = self
                .resources
                .as_ref()
                .context("resources missing")?
                .viewport;
            for bounds in damage.pixel_rects(size(
                DevicePixels(viewport.Width as i32),
                DevicePixels(viewport.Height as i32),
            )) {
                let scissor = device_rect(bounds);
                unsafe {
                    self.devices
                        .as_ref()
                        .context("devices missing")?
                        .device_context
                        .RSSetScissorRects(Some(slice::from_ref(&scissor)));
                }
                self.draw_scene_batches(
                    scene,
                    render_target_view,
                    SceneDamage::Partial(bounds.map(|p| ScaledPixels(p.0 as f32))),
                )?;
            }
            Ok(())
        })();
        self.staging_scene = staging;
        rendered
    }

    fn draw_scene_batches(
        &mut self,
        scene: &Scene,
        render_target_view: &Option<ID3D11RenderTargetView>,
        damage: SceneDamage,
    ) -> Result<()> {
        let annotation = self
            .devices
            .as_ref()
            .and_then(|devices| devices.annotation.clone())
            .filter(|annotation| unsafe { annotation.GetStatus().as_bool() });
        for batch in scene.batches_for_damage(damage) {
            let _annotation = annotation
                .as_ref()
                .map(|annotation| Annotation::new(annotation, HSTRING::from(batch.label())));
            match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(range.start, range.len()),
                PrimitiveBatch::Quads(range) => self.draw_quads(range.start, range.len()),
                PrimitiveBatch::Paths(range) => {
                    let Some(bounds) = scene.path_bounds_for_damage(damage, range.clone()) else {
                        continue;
                    };
                    let paths = &scene.paths[range];
                    self.draw_paths_to_intermediate(paths, render_target_view, damage, bounds)?;
                    self.draw_paths_from_intermediate(paths, bounds)
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(range.start, range.len()),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_monochrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    self.draw_subpixel_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_polychrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(&scene.surfaces[range]),
            }
            .with_context(|| {
                format!(
                    "scene too large:\
                    {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                )
            })?;
        }
        Ok(())
    }

    /// Render each dirty cached layer's sub-scene into its offscreen texture. Reused (non-dirty)
    /// layers keep their existing texture and are only re-composited by the main pass. Restores the
    /// main viewport size in the global params at the end; `pre_draw` rebinds the back buffer.
    fn render_layers(&mut self, scene: &Scene, force_full: bool) -> Result<()> {
        if scene.layers.is_empty() {
            return Ok(());
        }
        for layer in &scene.layers {
            let width = layer.size.width.0.max(1) as u32;
            let height = layer.size.height.0.max(1) as u32;
            if !layer.needs_render && !force_full && self.layers.contains_key(&layer.id.0) {
                continue;
            }
            let Some(sub_scene) = layer.scene.as_deref() else {
                continue;
            };
            self.render_layers(sub_scene, force_full)?;
            let recreated = self.ensure_layer_texture(layer.id.0, width, height)?;
            if !recreated && !force_full && sub_scene.damage == SceneDamage::None {
                continue;
            }
            let damage = if recreated || force_full {
                SceneDamage::Full
            } else {
                self.effective_damage(sub_scene.damage)
            };
            if damage
                .pixel_rects(size(
                    DevicePixels(width as i32),
                    DevicePixels(height as i32),
                ))
                .len()
                == 0
            {
                continue;
            }
            let Some(devices) = self.devices.clone() else {
                continue;
            };
            let rtv = self
                .layers
                .get(&layer.id.0)
                .and_then(|t| t.rtv.clone())
                .context("layer render target missing")?;

            let layer_viewport = D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            // The batch draw helpers read `resources.viewport` for their RSSetViewports; override it
            // so they rasterize into this layer (its size), not the main back buffer. Restored after.
            let saved_viewport = self
                .resources
                .as_ref()
                .context("resources missing")?
                .viewport;
            if let Some(resources) = self.resources.as_mut() {
                resources.viewport = layer_viewport;
            }
            let rendered = (|| {
                self.update_viewport(&layer_viewport)?;
                unsafe {
                    if damage != SceneDamage::Full {
                        let rects = damage_rects(damage, width, height);
                        devices
                            .partial_clear_context
                            .as_ref()
                            .context("partial clears unavailable")?
                            .ClearView(&rtv, &[0.0; 4], Some(&rects));
                    } else {
                        devices
                            .device_context
                            .ClearRenderTargetView(&rtv, &[0.0; 4]);
                    }
                    devices
                        .device_context
                        .OMSetRenderTargets(Some(slice::from_ref(&Some(rtv.clone()))), None);
                    devices.device_context.VSSetConstantBuffers(
                        0,
                        Some(slice::from_ref(&self.globals.global_params_buffer)),
                    );
                    devices.device_context.VSSetConstantBuffers(
                        1,
                        Some(slice::from_ref(&self.globals.batch_params_buffer)),
                    );
                    devices.device_context.PSSetConstantBuffers(
                        0,
                        Some(slice::from_ref(&self.globals.global_params_buffer)),
                    );
                }
                self.render_scene(sub_scene, &Some(rtv), damage)
            })();
            if let Some(resources) = self.resources.as_mut() {
                resources.viewport = saved_viewport;
            }
            if rendered.is_err() {
                self.layers.remove(&layer.id.0);
            }
            rendered?;
        }
        Ok(())
    }

    /// Ensure an offscreen render-target texture exists for `key` at `width`x`height`, recreating it
    /// when the size changed (the layered view resized).
    fn ensure_layer_texture(&mut self, key: u64, width: u32, height: u32) -> Result<bool> {
        if let Some(t) = self.layers.get(&key)
            && t.width == width
            && t.height == height
        {
            return Ok(false);
        }
        let device = &self.devices.as_ref().context("devices missing")?.device;
        let (texture, srv) = create_path_intermediate_texture(device, width, height)?;
        let mut rtv = None;
        unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut rtv))? };
        self.layers.insert(
            key,
            LayerTexture {
                width,
                height,
                texture,
                rtv,
                srv,
                unseen: 0,
            },
        );
        Ok(true)
    }

    /// Render `scene` to an offscreen CPU image **without presenting** so
    /// the window need never be shown or visible (the macOS headless path
    /// goes through MetalRenderer; this is the Windows analogue). Draws into
    /// the existing render target, copies it into a `D3D11_USAGE_STAGING`
    /// texture, maps it, and converts BGRA to RGBA.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        // A pending device-lost recovery (`skip_draws`) leaves the atlas holding
        // tile references from the previous device; drawing before the forced
        // re-render rebuilds them panics in `DirectXAtlasState::texture`.
        anyhow::ensure!(
            !self.skip_draws,
            "render_to_image unavailable while recovering from a lost device"
        );
        self.last_background_appearance = None;
        self.render_layers(scene, true)?;
        self.render(scene, background_appearance, SceneDamage::Full)?;
        self.readback_target()
    }

    #[cfg(any(test, feature = "test-support"))]
    fn readback_target(&self) -> Result<image::RgbaImage> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let device = &devices.device;
        let context = &devices.device_context;
        let resources = self.resources.as_ref().context("resources missing")?;
        let render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;

        // A CPU-readable copy of the render target.
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { render_target.GetDesc(&mut desc) };
        let width = desc.Width;
        let height = desc.Height;
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            MipLevels: 1,
            ArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..desc
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging))? };
        let staging = staging.context("creating staging texture")?;
        unsafe { context.CopyResource(&staging, render_target) };

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };
        let row_bytes = (width as usize) * 4;
        let mut pixels = vec![0u8; row_bytes * height as usize];
        // SAFETY: `Map` succeeded, so `pData` points at `RowPitch * height`
        // readable bytes for as long as the mapping is held, and `RowPitch >=
        // row_bytes` (it only ever adds trailing padding). `pixels` is sized
        // `row_bytes * height`, so every copy stays in bounds on both sides,
        // and the regions cannot overlap (`pixels` is a fresh allocation).
        unsafe {
            let src = mapped.pData as *const u8;
            for row in 0..height as usize {
                let s = src.add(row * mapped.RowPitch as usize);
                let d = pixels.as_mut_ptr().add(row * row_bytes);
                std::ptr::copy_nonoverlapping(s, d, row_bytes);
            }
            context.Unmap(&staging, 0);
        }
        // The render target is BGRA; image::RgbaImage expects RGBA.
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        image::RgbaImage::from_raw(width, height, pixels)
            .context("Failed to build RgbaImage from staging readback")
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        self.last_background_appearance = None;

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    RENDER_TARGET_FORMAT,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }

        resources.recreate_resources(devices, width, height)?;

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }

        Ok(())
    }

    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        #[cfg(test)]
        {
            self.scene_uploads += 1;
            self.uploaded_primitive_bytes += std::mem::size_of_val(scene.shadows.as_slice())
                + std::mem::size_of_val(scene.quads.as_slice())
                + std::mem::size_of_val(scene.underlines.as_slice())
                + std::mem::size_of_val(scene.monochrome_sprites.as_slice())
                + std::mem::size_of_val(scene.subpixel_sprites.as_slice())
                + std::mem::size_of_val(scene.polychrome_sprites.as_slice());
        }
        let devices = self.devices.as_ref().context("devices missing")?;

        if !scene.shadows.is_empty() {
            self.pipelines.shadow_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.shadows,
            )?;
        }

        if !scene.quads.is_empty() {
            self.pipelines.quad_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.quads,
            )?;
        }

        if !scene.underlines.is_empty() {
            self.pipelines.underline_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.underlines,
            )?;
        }

        if !scene.monochrome_sprites.is_empty() {
            self.pipelines.mono_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.monochrome_sprites,
            )?;
        }

        if !scene.subpixel_sprites.is_empty() {
            self.pipelines.subpixel_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.subpixel_sprites,
            )?;
        }

        if !scene.polychrome_sprites.is_empty() {
            self.pipelines.poly_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.polychrome_sprites,
            )?;
        }

        Ok(())
    }

    fn draw_shadows(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.shadow_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_quads(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.quad_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_paths_to_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        render_target_view: &Option<ID3D11RenderTargetView>,
        damage: SceneDamage,
        path_bounds: Bounds<ScaledPixels>,
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?.clone();
        let resources = self.resources.as_mut().context("resources missing")?;
        let viewport_size = size(
            DevicePixels(resources.viewport.Width as i32),
            DevicePixels(resources.viewport.Height as i32),
        );
        let Some(region) = SceneDamage::Partial(path_bounds).pixel_bounds(viewport_size) else {
            return Ok(());
        };
        let scissor = device_rect(
            damage
                .pixel_bounds(viewport_size)
                .context("path target damage missing")?,
        );
        let width = (region.size.width.0 as u32).div_ceil(64) * 64;
        let height = (region.size.height.0 as u32).div_ceil(64) * 64;
        resources.resize_path_intermediate(&devices, width, height)?;
        resources.path_intermediate_origin = region.origin.map(|p| ScaledPixels(p.0 as f32));
        let resources = self.resources.as_ref().context("resources missing")?;
        let target = resources
            .path_intermediate
            .as_ref()
            .context("path intermediate missing")?;
        let viewport = resources.viewport;
        let tile_viewport = D3D11_VIEWPORT {
            Width: width as f32,
            Height: height as f32,
            ..viewport
        };
        let tile_scissor = RECT {
            left: 0,
            top: 0,
            right: region.size.width.0,
            bottom: region.size.height.0,
        };
        self.update_viewport(&tile_viewport)?;
        // Clear intermediate MSAA texture
        unsafe {
            devices
                .device_context
                .ClearRenderTargetView(target.msaa_view.as_ref().unwrap(), &[0.0; 4]);
            // Set intermediate MSAA texture as render target
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&target.msaa_view)), None);
            devices
                .device_context
                .RSSetScissorRects(Some(slice::from_ref(&tile_scissor)));
        }

        // Collect all vertices and sprites for a single draw call
        self.path_vertices.clear();
        let region = region.map(|p| ScaledPixels(p.0 as f32));

        for path in paths {
            if !path.clipped_bounds().intersects(&region) {
                continue;
            }
            self.path_vertices
                .extend(path.vertices.iter().map(|v| PathRasterizationSprite {
                    xy_position: v.xy_position,
                    st_position: v.st_position,
                    color: path.color,
                    bounds: path.clipped_bounds(),
                    fade: path.content_mask.fade,
                    target_origin: resources.path_intermediate_origin,
                }));
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &self.path_vertices,
        )?;

        self.pipelines.path_rasterization_pipeline.draw(
            &devices.device_context,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            self.path_vertices.len() as u32,
            1,
        )?;

        // Resolve MSAA to non-MSAA intermediate texture
        unsafe {
            devices.device_context.ResolveSubresource(
                &target.texture,
                0,
                &target.msaa_texture,
                0,
                RENDER_TARGET_FORMAT,
            );
            // Resume the target pass in its coordinate system, with its original damage scissor.
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(render_target_view)), None);
            devices
                .device_context
                .RSSetScissorRects(Some(slice::from_ref(&scissor)));
        }
        self.update_viewport(&viewport)
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        bounds: Bounds<ScaledPixels>,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let target = resources
            .path_intermediate
            .as_ref()
            .context("path intermediate missing")?;
        let (width, height) = target.size;
        let tex_size = [width as f32, height as f32];
        let texture_origin = resources.path_intermediate_origin;
        self.path_sprites.clear();
        self.path_sprites.extend(
            paths
                .iter()
                .filter(|path| path.clipped_bounds().intersects(&bounds))
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds().intersect(&bounds),
                    tex_size,
                    texture_origin,
                }),
        );
        if self.path_sprites.is_empty() {
            return Ok(());
        }
        if paths.last().unwrap().order != first_path.order {
            let bounds = self
                .path_sprites
                .iter()
                .map(|sprite| sprite.bounds)
                .reduce(|a, b| a.union(&b))
                .unwrap();
            self.path_sprites.truncate(1);
            self.path_sprites[0].bounds = bounds;
        }

        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &self.path_sprites,
        )?;

        // Draw the sprites with the path texture
        self.pipelines.path_sprite_pipeline.draw_with_texture(
            &devices.device_context,
            slice::from_ref(&target.srv),
            slice::from_ref(&self.globals.sampler),
            self.path_sprites.len() as u32,
        )
    }

    fn draw_underlines(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.underline_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.mono_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.subpixel_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.poly_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    /// Composite cached view layers: for each `PaintSurface` whose source is a layer, draw its
    /// offscreen texture stretched to the surface bounds via the surface pipeline (the same path
    /// remote-window frames use). Runs in the main pass, so the back buffer + main viewport are
    /// bound and the layer texture lands at the view's real on-screen position.
    fn draw_surfaces(&mut self, surfaces: &[PaintSurface]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        let Some(devices) = self.devices.clone() else {
            return Ok(());
        };
        for surface in surfaces {
            let layer_id = match &surface.source {
                PaintSurfaceSource::Layer(layer_id) => *layer_id,
                // DirectX does not read its target back; `Style` falls back to the static backdrop.
                _ => continue,
            };
            let Some((srv, tex_w, tex_h)) = self
                .layers
                .get(&layer_id.0)
                .map(|t| (t.srv.clone(), t.width, t.height))
            else {
                continue;
            };
            if srv.is_none() {
                // Texture not rendered yet (e.g. a composite-only first frame) — skip rather than
                // draw garbage.
                continue;
            }
            let instance = SurfaceSprite {
                bounds: surface.bounds,
                content_mask: surface.content_mask.bounds,
                content_fade: surface.content_mask.fade,
                // A stretched composite samples the whole texture across its bounds.
                tex_size: if surface.stretch {
                    [surface.bounds.size.width.0, surface.bounds.size.height.0]
                } else {
                    [tex_w as f32, tex_h as f32]
                },
            };
            self.pipelines.surface_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &[instance],
            )?;
            self.pipelines.surface_pipeline.draw_with_texture(
                &devices.device_context,
                slice::from_ref(&srv),
                slice::from_ref(&self.globals.sampler),
                1,
            )?;
        }
        Ok(())
    }

    /// Composite the remote-window frames pushed this frame (via `crate::remote_surface::push_surface`)
    /// as textured quads. M1: one BGRA frame uploaded to a cached dynamic texture per draw.
    fn draw_remote_surfaces(
        &mut self,
        surfaces: &[crate::remote_surface::RemoteSurface],
    ) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }

        for surface in surfaces {
            if surface.width == 0 || surface.height == 0 {
                continue;
            }
            self.ensure_surface_texture(surface.width, surface.height)?;
            self.upload_surface(surface)?;

            let Some(devices) = self.devices.clone() else {
                continue;
            };
            let srv = self.surface_texture.as_ref().and_then(|t| t.srv.clone());

            let instance = SurfaceSprite {
                bounds: surface.bounds,
                content_mask: surface.bounds,
                content_fade: ContentFade::default(),
                tex_size: [0.0, 0.0],
            };
            self.pipelines.surface_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &[instance],
            )?;
            self.pipelines.surface_pipeline.draw_with_texture(
                &devices.device_context,
                slice::from_ref(&srv),
                slice::from_ref(&self.globals.sampler),
                1,
            )?;
        }
        Ok(())
    }

    fn ensure_surface_texture(&mut self, width: u32, height: u32) -> Result<()> {
        if let Some(t) = &self.surface_texture
            && t.width == width
            && t.height == height
        {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
        };
        let mut texture = None;
        unsafe {
            devices
                .device
                .CreateTexture2D(&desc, None, Some(&mut texture))?
        };
        let texture = texture.context("CreateTexture2D returned null")?;
        let mut srv = None;
        unsafe {
            devices
                .device
                .CreateShaderResourceView(&texture, None, Some(&mut srv))?
        };
        self.surface_texture = Some(RemoteSurfaceTexture {
            width,
            height,
            texture,
            srv,
        });
        Ok(())
    }

    fn upload_surface(&mut self, surface: &crate::remote_surface::RemoteSurface) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let Some(tex) = &self.surface_texture else {
            return Ok(());
        };
        unsafe {
            let mut mapped = std::mem::zeroed::<D3D11_MAPPED_SUBRESOURCE>();
            devices.device_context.Map(
                &tex.texture,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            )?;
            let src_stride = surface.stride as usize;
            let row_bytes = (surface.width as usize) * 4;
            let dst = mapped.pData as *mut u8;
            let dst_stride = mapped.RowPitch as usize;
            for row in 0..surface.height as usize {
                std::ptr::copy_nonoverlapping(
                    surface.bgra.as_ptr().add(row * src_stride),
                    dst.add(row * dst_stride),
                    row_bytes,
                );
            }
            devices.device_context.Unmap(&tex.texture, 0);
        }
        Ok(())
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
            let render_params: IDWriteRenderingParams1 =
                factory.CreateRenderingParams().unwrap().cast().unwrap();
            FontInfo {
                gamma_ratios: gpui::get_gamma_correction_ratios(render_params.GetGamma()),
                grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
                is_bgr: render_params.GetPixelGeometry() == DWRITE_PIXEL_GEOMETRY_BGR,
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
}

impl DirectXResources {
    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (render_target, render_target_view, viewport) =
            create_resources(devices, &swap_chain, width, height)?;
        set_rasterizer_state(&devices.device, &devices.device_context)?;

        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
            path_intermediate: None,
            path_cache: Vec::new(),
            path_intermediate_origin: Point::default(),
            #[cfg(test)]
            path_allocations: 0,
            viewport,
        })
    }

    fn resize_path_intermediate(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if self
            .path_intermediate
            .as_ref()
            .is_some_and(|target| target.size == (width, height))
        {
            return Ok(());
        }
        let target = if let Some(index) = self
            .path_cache
            .iter()
            .position(|target| target.size == (width, height))
        {
            self.path_cache.remove(index)
        } else {
            let (texture, srv) = create_path_intermediate_texture(&devices.device, width, height)?;
            let (msaa_texture, msaa_view) =
                create_path_intermediate_msaa_texture_and_view(&devices.device, width, height)?;
            #[cfg(test)]
            {
                self.path_allocations += 1;
            }
            PathIntermediate {
                texture,
                srv,
                msaa_texture,
                msaa_view,
                size: (width, height),
            }
        };
        if let Some(previous) = self.path_intermediate.replace(target) {
            if previous.bytes() <= PATH_CACHE_BYTES {
                self.path_cache.push(previous);
            }
        }
        while self.path_cache.len() > 3
            || self
                .path_cache
                .iter()
                .map(PathIntermediate::bytes)
                .sum::<u64>()
                > PATH_CACHE_BYTES
        {
            self.path_cache.remove(0);
        }
        self.path_intermediate_origin = Point::default();
        Ok(())
    }

    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (render_target, render_target_view, viewport) =
            create_resources(devices, &self.swap_chain, width, height)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        self.path_intermediate = None;
        self.path_cache.clear();
        self.viewport = viewport;
        Ok(())
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_blend_state_for_path_rasterization(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_blend_state_for_path_sprite(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?;
        let surface_pipeline = PipelineState::new(
            device,
            "surface_pipeline",
            ShaderModule::Surface,
            4,
            create_blend_state(device)?,
        )?;

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surface_pipeline,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, true) }?;
        let comp_visual = unsafe { comp_device.CreateVisual() }?;

        Ok(Self {
            comp_device,
            comp_target,
            comp_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.comp_visual.SetContent(swap_chain)?;
            self.comp_target.SetRoot(&self.comp_visual)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let global_params_buffer = create_constant_buffer::<GlobalParams>(device)?;
        let batch_params_buffer = create_constant_buffer::<BatchParams>(device)?;

        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        Ok(Self {
            global_params_buffer,
            batch_params_buffer,
            sampler,
        })
    }
}

#[derive(Debug, Default)]
#[repr(C)]
struct GlobalParams {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
struct BatchParams {
    start_index: u32,
    _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<BatchParams>() == 16);

struct PipelineState<T> {
    label: &'static str,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    buffer: ID3D11Buffer,
    buffer_size: usize,
    view: Option<ID3D11ShaderResourceView>,
    blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            vertex,
            fragment,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        if self.buffer_size < data.len() {
            let element_size = std::mem::size_of::<T>();
            let required_size = std::mem::size_of_val(data);
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "{} buffer needs {required_size} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}",
                self.label
            );
            let new_buffer_size = data
                .len()
                .next_power_of_two()
                .min(MAX_INSTANCE_BUFFER_SIZE / element_size);
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer(device_context, &self.buffer, data)
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            topology,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        sampler: &[Option<ID3D11SamplerState>],
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));

            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range(
        &self,
        device_context: &ID3D11DeviceContext,
        batch_params_buffer: &ID3D11Buffer,
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        batch_params_buffer: &ID3D11Buffer,
        sampler: &[Option<ID3D11SamplerState>],
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathRasterizationSprite {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
    fade: ContentFade<ScaledPixels>,
    target_origin: Point<ScaledPixels>,
}

/// One instance for the path sprite pass; mirrors the HLSL `PathSprite`.
#[derive(Clone, Copy)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
    tex_size: [f32; 2],
    texture_origin: Point<ScaledPixels>,
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device)?;
    let viewport = D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    Ok((render_target, render_target_view, viewport))
}

#[inline]
fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0) }?;
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };

    Ok((texture, Some(shader_resource_view.unwrap())))
}

#[inline]
fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

#[inline]
fn set_rasterizer_state(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Result<()> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: true.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let rasterizer_state = unsafe {
        let mut state = None;
        device.CreateRasterizerState(&desc, Some(&mut state))?;
        state.unwrap()
    };
    unsafe { device_context.RSSetState(&rasterizer_state) };
    Ok(())
}

// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_blend_desc
#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_rasterization(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_sprite(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_vertex_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_fragment_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_constant_buffer<T>(device: &ID3D11Device) -> Result<Option<ID3D11Buffer>> {
    const { assert!(std::mem::size_of::<T>() != 0 && std::mem::size_of::<T>().is_multiple_of(16)) };
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: std::mem::size_of::<T>() as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer)
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: (element_size * buffer_size) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
        StructureByteStride: element_size as u32,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, None, Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn device_rect(bounds: Bounds<DevicePixels>) -> RECT {
    RECT {
        left: bounds.left().0,
        top: bounds.top().0,
        right: bounds.right().0,
        bottom: bounds.bottom().0,
    }
}

fn damage_rects(damage: SceneDamage, width: u32, height: u32) -> SmallVec<[RECT; 8]> {
    damage
        .pixel_rects(size(
            DevicePixels(width as i32),
            DevicePixels(height as i32),
        ))
        .map(device_rect)
        .collect()
}

#[inline]
fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

#[inline]
fn update_batch_start(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    first_instance: u32,
) -> Result<()> {
    update_buffer(
        device_context,
        buffer,
        &[BatchParams {
            start_index: first_instance,
            _padding: [0; 3],
        }],
    )
}

#[inline]
fn set_pipeline_state(
    device_context: &ID3D11DeviceContext,
    buffer_view: &[Option<ID3D11ShaderResourceView>],
    topology: D3D_PRIMITIVE_TOPOLOGY,
    vertex_shader: &ID3D11VertexShader,
    fragment_shader: &ID3D11PixelShader,
    blend_state: &ID3D11BlendState,
) {
    unsafe {
        device_context.VSSetShaderResources(1, Some(buffer_view));
        device_context.PSSetShaderResources(1, Some(buffer_view));
        device_context.IASetPrimitiveTopology(topology);
        device_context.VSSetShader(vertex_shader, None);
        device_context.PSSetShader(fragment_shader, None);
        device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
    }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

const BUFFER_COUNT: usize = 3;

#[cfg(test)]
mod damage_tests {
    use super::*;
    use windows::{
        Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_POPUP,
        },
        core::w,
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn fixture(left: Option<f32>, layered: bool) -> Scene {
        let mut scene = Scene::default();
        let clip = rect(0., 0., 128., 128.);
        for (bounds, color) in [(clip, 0x24476670), (rect(82., 82., 10., 10.), 0x24db91ff)] {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask {
                    bounds: clip,
                    ..Default::default()
                },
                background: rgba(color).into(),
                ..Default::default()
            });
        }
        if let Some(left) = left {
            scene.insert_primitive(Quad {
                bounds: rect(left, 8., 16., 24.),
                content_mask: ContentMask {
                    bounds: clip,
                    ..Default::default()
                },
                background: rgba(0xc96bce80).into(),
                corner_radii: Corners::all(ScaledPixels(4.)),
                ..Default::default()
            });
        }
        let mut path = Path::new(point(px(8.), px(28.)));
        path.line_to(point(px(24.), px(40.)));
        path.line_to(point(px(40.), px(28.)));
        path.color = linear_gradient(
            25.,
            linear_color_stop(rgba(0xe0bb3080), 0.),
            linear_color_stop(rgba(0xe020ae80), 1.),
        );
        path.content_mask = ContentMask {
            bounds: clip.map(|p| px(p.0)),
            fade: ContentFade {
                top: px(28.),
                top_len: px(8.),
                ..Default::default()
            },
        };
        scene.insert_primitive(path.scale(1.));
        scene.finish();
        if !layered {
            return scene;
        }
        let mut root = Scene::default();
        let bounds = rect(16., 16., 96., 96.);
        root.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            source: PaintSurfaceSource::Layer(LayerId(1)),
            stretch: false,
        });
        root.layers.push(SceneLayer {
            id: LayerId(1),
            size: size(DevicePixels(96), DevicePixels(96)),
            needs_render: true,
            scene: Some(Box::new(scene)),
        });
        root.finish();
        root
    }

    fn assert_same_pixels(partial: &image::RgbaImage, full: &image::RgbaImage, context: &str) {
        assert_eq!(partial.dimensions(), full.dimensions());
        if let Some((x, y, pixel)) = partial
            .enumerate_pixels()
            .find(|(x, y, pixel)| full.get_pixel(*x, *y) != *pixel)
        {
            panic!(
                "{context}: pixel {x},{y} partial={pixel:?}, full={:?}",
                full.get_pixel(x, y)
            );
        }
    }

    fn two_layer_fixture(left: f32) -> Scene {
        let mut scene = Scene::default();
        for (id, origin, left) in [(1, 4., left), (2, 68., 40. - left)] {
            let bounds = rect(origin, origin, 56., 56.);
            scene.insert_primitive(PaintSurface {
                order: 0,
                bounds,
                content_mask: ContentMask {
                    bounds,
                    ..Default::default()
                },
                source: PaintSurfaceSource::Layer(LayerId(id)),
                stretch: false,
            });
            scene.layers.push(SceneLayer {
                id: LayerId(id),
                size: size(DevicePixels(56), DevicePixels(56)),
                needs_render: true,
                scene: Some(Box::new(fixture(Some(left), false))),
            });
        }
        scene.finish();
        scene
    }

    fn sparse_fixture(offset: f32) -> Scene {
        let mut scene = Scene::default();
        let clip = rect(0., 0., 128., 128.);
        for (bounds, color) in std::iter::once((clip, 0x24476670))
            .chain((0..500).map(|i| {
                (
                    rect(
                        4. + (i % 25) as f32 * 4.,
                        64. + (i / 25) as f32 * 2.,
                        2.,
                        1.,
                    ),
                    0x24db91ff,
                )
            }))
            .chain([
                (rect(8. + offset, 8., 12., 12.), 0xc96bce80),
                (rect(88. - offset, 8., 12., 12.), 0xe0bb3080),
            ])
        {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask {
                    bounds: clip,
                    ..Default::default()
                },
                background: rgba(color).into(),
                ..Default::default()
            });
        }
        scene.finish();
        scene
    }

    fn check_sparse_uploads(renderer: &mut DirectXRenderer) -> Result<()> {
        let mut frames = Vec::new();
        for offset in [4., 8., 0., 4.] {
            let scene = sparse_fixture(offset);
            let full = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
            frames.push((scene, full));
        }
        let mut previous = sparse_fixture(0.);
        let mut composed =
            renderer.render_to_image(&previous, WindowBackgroundAppearance::Transparent)?;
        assert!(renderer.present(SceneDamage::Full)?);
        for (mut scene, full) in frames {
            scene.update_damage(&previous, false);
            assert_eq!(
                scene
                    .damage
                    .pixel_rects(size(DevicePixels(128), DevicePixels(128)))
                    .len(),
                2
            );
            renderer.scene_uploads = 0;
            renderer.uploaded_primitive_bytes = 0;
            let mut expected_buffer = renderer.readback_target()?;
            renderer.render(
                &scene,
                WindowBackgroundAppearance::Transparent,
                scene.damage,
            )?;
            assert_eq!(
                renderer.scene_uploads, 1,
                "damage regions must share one upload"
            );
            assert_eq!(
                renderer.uploaded_primitive_bytes,
                3 * std::mem::size_of::<Quad>()
            );
            let partial = renderer.readback_target()?;
            // DXGI repairs pixels outside dirty rectangles during Present1; the rotated buffer
            // may be stale there. Verify every produced pixel and that drawing stayed in damage.
            for bounds in scene
                .damage
                .pixel_rects(size(DevicePixels(128), DevicePixels(128)))
            {
                for y in bounds.top().0..bounds.bottom().0 {
                    for x in bounds.left().0..bounds.right().0 {
                        let pixel = *partial.get_pixel(x as u32, y as u32);
                        expected_buffer.put_pixel(x as u32, y as u32, pixel);
                        composed.put_pixel(x as u32, y as u32, pixel);
                    }
                }
            }
            assert_same_pixels(
                &partial,
                &expected_buffer,
                "draw escaped Present1 dirty rectangles",
            );
            assert_same_pixels(&composed, &full, "sparse multi-region quad upload");
            assert!(renderer.present(scene.damage)?);
            previous = scene;
        }
        Ok(())
    }

    fn check_scratch_reuse(renderer: &mut DirectXRenderer) -> Result<()> {
        renderer.resize(size(DevicePixels(3840), DevicePixels(2160)))?;
        let allocations = renderer.resources.as_ref().unwrap().path_allocations;
        renderer.render(
            &Scene::default(),
            WindowBackgroundAppearance::Opaque,
            SceneDamage::Full,
        )?;
        assert!(
            renderer
                .resources
                .as_ref()
                .unwrap()
                .path_intermediate
                .is_none()
        );
        let path_scene = |width: f32| {
            let mut scene = Scene::default();
            let mut path = Path::new(point(px(8.), px(8.)));
            path.line_to(point(px(8. + width), px(8.)));
            path.line_to(point(px(8.), px(32.)));
            path.color = rgba(0xe0bb3080).into();
            path.content_mask.bounds = rect(0., 0., 3840., 2160.).map(|p| px(p.0));
            scene.insert_primitive(path.scale(1.));
            scene.finish();
            scene
        };
        for width in [24., 80.] {
            renderer.render(
                &path_scene(width),
                WindowBackgroundAppearance::Opaque,
                SceneDamage::Full,
            )?;
            assert_eq!(
                renderer
                    .resources
                    .as_ref()
                    .unwrap()
                    .path_intermediate
                    .as_ref()
                    .unwrap()
                    .size,
                (if width == 24. { 64 } else { 128 }, 64)
            );
        }
        let capacities = (
            renderer.path_vertices.capacity(),
            renderer.path_sprites.capacity(),
        );
        for frame in 0..12 {
            renderer.render(
                &path_scene(if frame % 2 == 0 { 24. } else { 80. }),
                WindowBackgroundAppearance::Opaque,
                SceneDamage::Full,
            )?;
            assert_eq!(
                renderer.resources.as_ref().unwrap().path_allocations,
                allocations + 2
            );
            assert_eq!(
                (
                    renderer.path_vertices.capacity(),
                    renderer.path_sprites.capacity()
                ),
                capacities
            );
        }
        let resources = renderer.resources.as_mut().unwrap();
        let devices = renderer.devices.as_ref().unwrap();
        for (width, height) in [(192, 64), (256, 64), (320, 64), (1024, 1024), (64, 64)] {
            resources.resize_path_intermediate(devices, width, height)?;
            assert!(resources.path_cache.len() <= 3);
            assert!(
                resources
                    .path_cache
                    .iter()
                    .map(PathIntermediate::bytes)
                    .sum::<u64>()
                    <= PATH_CACHE_BYTES
            );
        }
        Ok(())
    }

    fn check_layer_budget(renderer: &mut DirectXRenderer) -> Result<()> {
        renderer.layers.clear();
        for id in 100..105 {
            assert!(renderer.ensure_layer_texture(id, 2048, 2048)?);
        }
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            source: PaintSurfaceSource::Layer(LayerId(100)),
            bounds: rect(0., 0., 128., 128.),
            content_mask: ContentMask {
                bounds: rect(0., 0., 128., 128.),
                ..Default::default()
            },
            order: 0,
            stretch: false,
        });
        renderer.evict_stale_layers(&scene);
        assert!(
            renderer.layers.contains_key(&100),
            "live layers must survive budget eviction"
        );
        assert!(
            renderer
                .layers
                .values()
                .map(|t| t.width as u64 * t.height as u64 * 4)
                .sum::<u64>()
                <= LAYER_CACHE_BYTES
        );
        let evicted = (101..105)
            .find(|id| !renderer.layers.contains_key(id))
            .unwrap();
        assert!(
            renderer.ensure_layer_texture(evicted, 2048, 2048)?,
            "evicted layers must request a full render"
        );
        Ok(())
    }

    #[::core::prelude::v1::test]
    fn partial_frames_match_full_redraw() -> Result<()> {
        let devices = DirectXDevices::new()?;
        for disable_composition in [true, false] {
            let hwnd = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE::default(),
                    w!("STATIC"),
                    w!("GPUI partial redraw test"),
                    WS_POPUP,
                    0,
                    0,
                    160,
                    160,
                    None,
                    None,
                    None,
                    None,
                )?
            };
            let result = (|| -> Result<()> {
                let mut renderer = DirectXRenderer::new(hwnd, &devices, disable_composition)?;
                renderer.resize(size(DevicePixels(128), DevicePixels(128)))?;
                anyhow::ensure!(
                    renderer
                        .devices
                        .as_ref()
                        .and_then(|d| d.partial_clear_context.as_ref())
                        .is_some(),
                    "This GPU fixture requires ClearView support"
                );
                for appearance in [
                    WindowBackgroundAppearance::Opaque,
                    WindowBackgroundAppearance::Transparent,
                ] {
                    for layered in [false, true] {
                        let mut previous = fixture(Some(8.), layered);
                        renderer.render_to_image(&previous, appearance)?;
                        for (index, left) in [Some(16.), Some(24.), None, Some(8.)]
                            .into_iter()
                            .enumerate()
                        {
                            let mut scene = fixture(left, layered);
                            scene.update_damage(&previous, false);
                            let damage = renderer.effective_damage(scene.damage);
                            assert_ne!(damage, SceneDamage::Full);
                            renderer.render_layers(&scene, false)?;
                            renderer.render(&scene, appearance, damage)?;
                            assert_eq!(
                                renderer
                                    .resources
                                    .as_ref()
                                    .context("resources missing")?
                                    .path_intermediate
                                    .as_ref()
                                    .unwrap()
                                    .size,
                                (64, 64)
                            );
                            let partial = renderer.readback_target()?;
                            let full = renderer.render_to_image(&scene, appearance)?;
                            assert_same_pixels(
                                &partial,
                                &full,
                                &format!(
                                    "composition={}, {appearance:?}, layered={layered}, frame={index}",
                                    !disable_composition
                                ),
                            );
                            if let Some(directory) = std::env::var_os("GPUI_DAMAGE_TEST_OUTPUT") {
                                std::fs::create_dir_all(&directory)?;
                                partial.save(std::path::Path::new(&directory).join(format!(
                                    "directx-{disable_composition}-{appearance:?}-{layered}-{index}.png"
                                )))?;
                            }
                            previous = scene;
                        }
                    }
                }
                let clipped = |left| {
                    let mut scene = fixture(Some(left), true);
                    scene.surfaces[0].content_mask.bounds = rect(16., 16., 96., 4.);
                    scene
                };
                let hidden = clipped(8.);
                renderer.draw(&hidden, WindowBackgroundAppearance::Opaque)?;
                let mut changed = clipped(24.);
                changed.update_damage(&hidden, false);
                assert_eq!(changed.damage, SceneDamage::None);
                renderer.draw(&changed, WindowBackgroundAppearance::Opaque)?;
                let mut revealed = fixture(Some(24.), true);
                revealed.update_damage(&changed, false);
                renderer.render_layers(&revealed, false)?;
                renderer.render(
                    &revealed,
                    WindowBackgroundAppearance::Opaque,
                    SceneDamage::Full,
                )?;
                let partial = renderer.readback_target()?;
                assert_same_pixels(
                    &partial,
                    &renderer.render_to_image(&revealed, WindowBackgroundAppearance::Opaque)?,
                    "offscreen layer update was lost before reveal",
                );
                let mut previous = two_layer_fixture(8.);
                renderer.draw(&previous, WindowBackgroundAppearance::Opaque)?;
                let swap_chain = renderer
                    .resources
                    .as_ref()
                    .context("resources missing")?
                    .swap_chain
                    .clone();
                for frame in 1..12 {
                    let mut scene = two_layer_fixture(8. + (frame % 4) as f32 * 8.);
                    scene.update_damage(&previous, false);
                    assert!(
                        scene
                            .damage
                            .pixel_rects(size(DevicePixels(128), DevicePixels(128)))
                            .len()
                            >= 2
                    );
                    assert!(scene.layers.iter().all(|layer| matches!(
                        layer.scene.as_ref().unwrap().damage,
                        SceneDamage::Partial(_) | SceneDamage::Regions(_)
                    )));
                    let before = unsafe { swap_chain.GetLastPresentCount()? };
                    renderer.scene_uploads = 0;
                    renderer.uploaded_primitive_bytes = 0;
                    renderer.draw(&scene, WindowBackgroundAppearance::Opaque)?;
                    assert_eq!(
                        renderer.scene_uploads, 3,
                        "two layers and root upload once each"
                    );
                    assert_eq!(
                        renderer.uploaded_primitive_bytes,
                        4 * std::mem::size_of::<Quad>()
                    );
                    assert_eq!(
                        unsafe { swap_chain.GetLastPresentCount()? },
                        before + 1,
                        "two damaged layers must share one presentation: composition={}, frame={frame}",
                        !disable_composition,
                    );
                    previous = scene;
                }
                let before = unsafe { swap_chain.GetLastPresentCount()? };
                let mut unchanged = two_layer_fixture(32.);
                unchanged.update_damage(&previous, false);
                assert_eq!(unchanged.damage, SceneDamage::None);
                renderer.draw(&unchanged, WindowBackgroundAppearance::Opaque)?;
                assert_eq!(before, unsafe { swap_chain.GetLastPresentCount()? });
                unchanged.damage = SceneDamage::Partial(rect(256., 256., 8., 8.));
                renderer.draw(&unchanged, WindowBackgroundAppearance::Opaque)?;
                assert_eq!(before, unsafe { swap_chain.GetLastPresentCount()? });
                unchanged.damage = SceneDamage::None;
                renderer.resize(size(DevicePixels(160), DevicePixels(160)))?;
                renderer.draw(&unchanged, WindowBackgroundAppearance::Opaque)?;
                assert_eq!(unsafe { swap_chain.GetLastPresentCount()? }, before + 1);
                renderer.resize(size(DevicePixels(128), DevicePixels(128)))?;
                check_sparse_uploads(&mut renderer)?;
                check_scratch_reuse(&mut renderer)?;
                check_layer_budget(&mut renderer)?;
                // Legacy drivers must redraw fully even when the shared scene reports a small change.
                renderer
                    .devices
                    .as_mut()
                    .context("devices missing")?
                    .partial_clear_context = None;
                assert_eq!(
                    renderer.effective_damage(SceneDamage::Partial(rect(1., 1., 2., 2.))),
                    SceneDamage::Full
                );
                renderer.draw(&fixture(None, true), WindowBackgroundAppearance::Opaque)?;
                Ok(())
            })();
            unsafe { DestroyWindow(hwnd)? };
            result?;
        }
        Ok(())
    }
}

pub(crate) mod shader_resources {
    use anyhow::Result;

    #[cfg(debug_assertions)]
    use windows::{
        Win32::Graphics::Direct3D::{
            Fxc::{D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompileFromFile},
            ID3DBlob,
        },
        core::{HSTRING, PCSTR},
    };

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        Shadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        Surface,
        EmojiRasterization,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderTarget {
        Vertex,
        Fragment,
    }

    pub(crate) struct RawShaderBytes<'t> {
        inner: &'t [u8],

        #[cfg(debug_assertions)]
        _blob: ID3DBlob,
    }

    impl<'t> RawShaderBytes<'t> {
        pub(crate) fn new(module: ShaderModule, target: ShaderTarget) -> Result<Self> {
            #[cfg(not(debug_assertions))]
            {
                Ok(Self::from_bytes(module, target))
            }
            #[cfg(debug_assertions)]
            {
                let blob = build_shader_blob(module, target)?;
                let inner = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    )
                };
                Ok(Self { inner, _blob: blob })
            }
        }

        pub(crate) fn as_bytes(&'t self) -> &'t [u8] {
            self.inner
        }

        #[cfg(not(debug_assertions))]
        fn from_bytes(module: ShaderModule, target: ShaderTarget) -> Self {
            let bytes = match module {
                ShaderModule::Quad => match target {
                    ShaderTarget::Vertex => QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::Shadow => match target {
                    ShaderTarget::Vertex => SHADOW_VERTEX_BYTES,
                    ShaderTarget::Fragment => SHADOW_FRAGMENT_BYTES,
                },
                ShaderModule::Underline => match target {
                    ShaderTarget::Vertex => UNDERLINE_VERTEX_BYTES,
                    ShaderTarget::Fragment => UNDERLINE_FRAGMENT_BYTES,
                },
                ShaderModule::PathRasterization => match target {
                    ShaderTarget::Vertex => PATH_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::PathSprite => match target {
                    ShaderTarget::Vertex => PATH_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::MonochromeSprite => match target {
                    ShaderTarget::Vertex => MONOCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MONOCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::SubpixelSprite => match target {
                    ShaderTarget::Vertex => SUBPIXEL_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SUBPIXEL_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::PolychromeSprite => match target {
                    ShaderTarget::Vertex => POLYCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => POLYCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::Surface => match target {
                    ShaderTarget::Vertex => SURFACE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SURFACE_FRAGMENT_BYTES,
                },
                ShaderModule::EmojiRasterization => match target {
                    ShaderTarget::Vertex => EMOJI_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => EMOJI_RASTERIZATION_FRAGMENT_BYTES,
                },
            };
            Self { inner: bytes }
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn build_shader_blob(entry: ShaderModule, target: ShaderTarget) -> Result<ID3DBlob> {
        unsafe {
            use windows::Win32::Graphics::{
                Direct3D::ID3DInclude, Hlsl::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            };

            let shader_name = if matches!(entry, ShaderModule::EmojiRasterization) {
                "color_text_raster.hlsl"
            } else {
                "shaders.hlsl"
            };

            let entry = format!(
                "{}_{}\0",
                entry.as_str(),
                match target {
                    ShaderTarget::Vertex => "vertex",
                    ShaderTarget::Fragment => "fragment",
                }
            );
            let target = match target {
                ShaderTarget::Vertex => "vs_4_1\0",
                ShaderTarget::Fragment => "ps_4_1\0",
            };

            let mut compile_blob = None;
            let mut error_blob = None;
            let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(&format!("src/{}", shader_name))
                .canonicalize()?;

            let entry_point = PCSTR::from_raw(entry.as_ptr());
            let target_cstr = PCSTR::from_raw(target.as_ptr());

            // really dirty trick because winapi bindings are unhappy otherwise
            let include_handler = &std::mem::transmute::<usize, ID3DInclude>(
                D3D_COMPILE_STANDARD_FILE_INCLUDE as usize,
            );

            let ret = D3DCompileFromFile(
                &HSTRING::from(shader_path.to_str().unwrap()),
                None,
                include_handler,
                entry_point,
                target_cstr,
                D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
                0,
                &mut compile_blob,
                Some(&mut error_blob),
            );
            if ret.is_err() {
                let Some(error_blob) = error_blob else {
                    return Err(anyhow::anyhow!("{ret:?}"));
                };

                let error_string =
                    std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8)
                        .to_string_lossy();
                log::error!("Shader compile error: {}", error_string);
                return Err(anyhow::anyhow!("Compile error: {}", error_string));
            }
            Ok(compile_blob.unwrap())
        }
    }

    #[cfg(not(debug_assertions))]
    include!(concat!(env!("OUT_DIR"), "/shaders_bytes.rs"));

    #[cfg(debug_assertions)]
    impl ShaderModule {
        pub fn as_str(self) -> &'static str {
            match self {
                ShaderModule::Quad => "quad",
                ShaderModule::Shadow => "shadow",
                ShaderModule::Underline => "underline",
                ShaderModule::PathRasterization => "path_rasterization",
                ShaderModule::PathSprite => "path_sprite",
                ShaderModule::MonochromeSprite => "monochrome_sprite",
                ShaderModule::SubpixelSprite => "subpixel_sprite",
                ShaderModule::PolychromeSprite => "polychrome_sprite",
                ShaderModule::Surface => "surface",
                ShaderModule::EmojiRasterization => "emoji_rasterization",
            }
        }
    }
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}
