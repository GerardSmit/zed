use crate::{CompositorGpuHint, WgpuAtlas, WgpuContext};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use gpui::{
    AtlasTextureId, BackdropBlur, Background, BorderStyle, Bounds, ContentFade, DevicePixels,
    GpuSpecs, LayerId, PaintSurface, PaintSurfaceSource, Path, Point, PrimitiveBatch, Quad,
    ScaledPixels, Scene, SceneDamage, Size, get_gamma_correction_ratios,
};
use log::warn;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

const MAX_INSTANCE_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

const INSTANCE_TEXTURE_TEXEL_SIZE: u64 = 16;

/// Shader variant for backends with storage buffer support: the shared shader
/// logic plus the storage-buffer instance transport.
const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
);

/// Shader variant for WebGL2, which has no storage buffers: the shared shader
/// logic plus the texture-based instance transport.
const WEBGL_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_webgl.wgsl"),
);

/// Subpixel text rendering requires dual-source blending, which WebGL2 lacks, so
/// this variant only ever runs with the storage-buffer transport. The `enable`
/// directive must precede all declarations.
const SUBPIXEL_SHADERS: &str = concat!(
    "enable dual_source_blending;\n",
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_subpixel.wgsl"),
);

fn least_common_multiple(left: u64, right: u64) -> u64 {
    let mut first = left;
    let mut second = right;
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    left / first * right
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
    /// The mask's `ContentFade`: top, top length, bottom, bottom length.
    content_fade: [f32; 4],
}

/// Uniform block for the layer composite pipeline. `tex_size` is the actual
/// pixel size of the offscreen texture (nonzero) so the shader can compute
/// 1:1 texcoords and discard fragments outside the texture area (matching
/// DirectX's culling/alpha behaviour). When `tex_size == [0,0]` the shader
/// would stretch-to-fill, but we always set it for layer composites.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerSurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
    /// The mask's `ContentFade`: top, top length, bottom, bottom length.
    content_fade: [f32; 4],
    tex_size: [f32; 2],
    _pad: [f32; 2],
}

impl LayerSurfaceParams {
    fn new(surface: &gpui::PaintSurface, width: u32, height: u32) -> Self {
        let fade = surface.content_mask.fade;
        Self {
            bounds: surface.bounds.into(),
            content_mask: surface.content_mask.bounds.into(),
            content_fade: [fade.top.0, fade.top_len.0, fade.bottom.0, fade.bottom_len.0],
            tex_size: if surface.stretch {
                [surface.bounds.size.width.0, surface.bounds.size.height.0]
            } else { [width as f32, height as f32] },
            _pad: [0.0; 2],
        }
    }
}

/// Uniform block for the backdrop blur passes and composite (`BackdropParams` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BackdropParams {
    bounds: PodBounds,
    content_mask: PodBounds,
    content_fade: [f32; 4],
    /// Top left, top right, bottom right, bottom left.
    corner_radii: [f32; 4],
    /// The copied part of the frame, in device pixels.
    region: PodBounds,
    direction: [f32; 2],
    /// The standard deviation at the blur textures' resolution.
    sigma: f32,
    downscale: f32,
    blur_size: [f32; 2],
    opacity: f32,
    _pad: f32,
}

/// The largest factor a backdrop is downsampled by before it is blurred. The blur textures are
/// sized for the smallest, 2.
const BACKDROP_MAX_DOWNSCALE: u32 = 16;

/// A backdrop is downsampled until its standard deviation is at most this many blur texels, which
/// keeps the Gaussian passes to a dozen taps a side whatever the radius.
const BACKDROP_MAX_LOW_SIGMA: f32 = 4.0;

fn backdrop_downscale(sigma: f32) -> u32 {
    let mut downscale = 2;
    while downscale < BACKDROP_MAX_DOWNSCALE && sigma / downscale as f32 > BACKDROP_MAX_LOW_SIGMA {
        downscale *= 2;
    }
    downscale
}

/// Window-sized scratch for backdrop blurs: the copy of the frame, and two textures at half its
/// size the blur passes ping-pong between. 1.5 × the surface's bytes; created on the first frame
/// that blurs and dropped with the other intermediates on resize.
struct BackdropTextures {
    copy: wgpu::Texture,
    copy_view: wgpu::TextureView,
    _blur_a: wgpu::Texture,
    blur_a_view: wgpu::TextureView,
    _blur_b: wgpu::Texture,
    blur_b_view: wgpu::TextureView,
    blur_width: u32,
    blur_height: u32,
    bindings: RefCell<Vec<BackdropBinding>>,
    next_binding: Cell<usize>,
}

struct BackdropBinding {
    buffer: wgpu::Buffer,
    binding: wgpu::BindGroup,
    source: wgpu::TextureView,
    params: Option<BackdropParams>,
}

/// Number of frames a layer texture is kept alive after last being referenced.
/// Mirrors DirectX's `LAYER_EVICT_FRAMES`.
const LAYER_EVICT_FRAMES: u32 = 3;

/// One cached offscreen texture for a layered view.
struct LayerTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Frames since this layer was last seen in the scene.
    unseen: u32,
    valid: bool,
    globals: wgpu::BindGroup,
    composite: RefCell<Vec<LayerCompositeBinding>>,
    next_composite: Cell<usize>,
}

struct LayerCompositeBinding {
    params: Option<LayerSurfaceParams>,
    buffer: wgpu::Buffer,
    binding: wgpu::BindGroup,
}

/// One surface-sized retained image, replaced on resize (four bytes/pixel for BGRA8/RGBA8).
struct RetainedFrame {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    valid: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
    texture_bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
    fade: ContentFade<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

const BUFFER_COPY_MAX_GAP: u64 = 256;

struct BufferCopy {
    buffer: wgpu::Buffer,
    source: u64,
    target: u64,
    length: u64,
}

/// GPU cost trade-offs of a renderer; [`WgpuRenderer::set_tuning`] changes them on a live renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererTuning {
    /// Multisample count for path rasterization. The highest count the target format supports,
    /// up to this one, is used; 1 leaves only the shader's analytic curve antialiasing.
    pub path_sample_count: u32,
    /// Keep Naga's shader runtime checks (buffer index clamping and loop bounding).
    pub shader_runtime_checks: bool,
    /// Let shaders read instance data straight from the mapped upload ring instead of a copy of
    /// it. Takes effect only when the device has `Features::MAPPABLE_PRIMARY_BUFFERS`, which
    /// unified-memory adapters are given.
    pub direct_instance_uploads: bool,
    /// Draw all damage rectangles in one scene pass, rasterizing every path batch of the scene
    /// into one atlas pass beforehand, instead of splitting the scene pass per rectangle and
    /// around each path batch.
    pub merged_scene_passes: bool,
    /// Which primitives to draw; anything but [`DrawFilter::All`] renders wrong pixels.
    pub draw_filter: DrawFilter,
    /// Draw runs of quads with a solid background, no border and no fade with a smaller shader
    /// than the general quad shader. Both write the same pixels.
    pub split_quads: bool,
    /// Draw the interior of large opaque simple quads without blending, and the rest of them
    /// blended, so that tile-based GPUs can skip the fragments those interiors hide. Implies
    /// [`RendererTuning::split_quads`]; writes the same pixels. See [`OpaqueQuadPlan`].
    pub opaque_quads: bool,
    /// Redraw each damage rectangle of a partial frame in its own pass that clears and renders
    /// only that rectangle, instead of loading the whole target and clearing the rectangle with
    /// a draw. Tile-based GPUs then neither load nor store tiles outside the damage. Only for
    /// backends that honour the pass label's render area (Bitnest's Vulkan backend): elsewhere
    /// the clear would erase the whole target.
    pub area_passes: bool,
    /// Give scene passes a transient depth attachment that every scene pipeline writes with
    /// compare `Always`, which changes no pixel. PowerVR merges the two triangles of a quad into
    /// one object only when depth writes are on; this measures what that is worth.
    pub depth_buffer: bool,
    /// Build unblended pipelines with blending enabled as a plain replace (`One`, `Zero`),
    /// which changes no pixel. Bitnest's PowerVR driver then keeps the per-fragment coverage
    /// epilogue and the punch-through pass type it otherwise drops for opaque pipelines.
    pub coverage_epilogue: bool,
    /// Merge each frame's atlas tile uploads into few copy regions per atlas texture; see
    /// [`WgpuAtlas::set_merged_uploads`]. Writes the same texels.
    pub merged_atlas_uploads: bool,
    /// Pipelines are rebuilt whenever this changes, so that a driver hook watching pipeline
    /// creation sees all of them.
    pub pipeline_epoch: u32,
}

impl Default for RendererTuning {
    fn default() -> Self {
        Self {
            path_sample_count: 4,
            shader_runtime_checks: true,
            direct_instance_uploads: false,
            merged_scene_passes: true,
            draw_filter: DrawFilter::All,
            split_quads: false,
            opaque_quads: false,
            area_passes: false,
            depth_buffer: false,
            coverage_epilogue: false,
            merged_atlas_uploads: false,
            pipeline_epoch: 0,
        }
    }
}

/// Primitive filter for measuring the GPU cost of each kind of primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DrawFilter {
    #[default]
    All,
    /// Only the clear of the damaged area.
    None,
    Quads,
    /// Monochrome, subpixel and polychrome sprites.
    Text,
    NoShadows,
    /// Everything, with quads drawn unblended.
    NoBlend,
    /// Only quads, with the quad pipeline's varyings but a constant colour.
    QuadConst,
    /// Only quads, with a minimal rounded-rectangle shader that ignores borders and gradients.
    QuadLean,
    /// Only quads, clipped to their content mask by the rasterizer, filled with their background
    /// colour without blending.
    QuadOpaque,
    /// [`DrawFilter::QuadOpaque`] with blending.
    QuadClamp,
    /// Instead of the scene, this many viewport-sized rectangles, each in one flat colour, drawn
    /// by `fs_quad_opaque` without blending: the fill rate of the simplest opaque shader and
    /// whether hidden-surface removal drops the covered layers.
    Fill(u8),
    /// [`DrawFilter::Fill`] with blending.
    FillBlend(u8),
    /// [`DrawFilter::Fill`] with a constant colour and no varying.
    FillConst(u8),
}

impl DrawFilter {
    fn allows(self, batch: &PrimitiveBatch) -> bool {
        match self {
            Self::All | Self::NoBlend => true,
            Self::None | Self::Fill(_) | Self::FillBlend(_) | Self::FillConst(_) => false,
            Self::Quads | Self::QuadConst | Self::QuadLean | Self::QuadOpaque | Self::QuadClamp => {
                matches!(batch, PrimitiveBatch::Quads(_))
            }
            Self::Text => matches!(
                batch,
                PrimitiveBatch::MonochromeSprites { .. }
                    | PrimitiveBatch::SubpixelSprites { .. }
                    | PrimitiveBatch::PolychromeSprites { .. }
            ),
            Self::NoShadows => !matches!(batch, PrimitiveBatch::Shadows(_)),
        }
    }

    fn allows_paths(self) -> bool {
        matches!(self, Self::All | Self::NoShadows | Self::NoBlend)
    }

    /// The measurement pipeline that replaces the quad pipeline: its vertex and fragment
    /// entry points, and whether it blends.
    fn quad_variant(self) -> Option<(&'static str, &'static str, bool)> {
        match self {
            Self::NoBlend => Some(("vs_quad", "fs_quad", false)),
            Self::QuadConst => Some(("vs_quad", "fs_quad_const", true)),
            Self::QuadLean => Some(("vs_quad_lean", "fs_quad_lean", true)),
            Self::QuadOpaque => Some(("vs_quad_flat", "fs_quad_opaque", false)),
            Self::QuadClamp => Some(("vs_quad_flat", "fs_quad_opaque", true)),
            Self::Fill(_) => Some(("vs_fill", "fs_quad_opaque", false)),
            Self::FillBlend(_) => Some(("vs_fill", "fs_quad_opaque", true)),
            Self::FillConst(_) => Some(("vs_fill", "fs_fill_const", false)),
            _ => None,
        }
    }

    /// The number of viewport-sized rectangles a fill filter draws.
    fn fill_layers(self) -> u32 {
        match self {
            Self::Fill(layers) | Self::FillBlend(layers) | Self::FillConst(layers) => layers.into(),
            _ => 0,
        }
    }
}

/// One path batch of a merged frame and its place in the path atlas.
struct PathJob {
    range: Range<usize>,
    /// Visible device-pixel footprint of the batch within its damage rectangle.
    bounds: Option<Bounds<DevicePixels>>,
    extent: [u32; 2],
    atlas_origin: [u32; 2],
    vertices: Range<u32>,
    sprites: Range<u32>,
}

struct OpaqueQuadPipelines {
    /// Unblended interiors.
    interiors: wgpu::RenderPipeline,
    /// The rest of each quad, blended.
    clamped: wgpu::RenderPipeline,
}

struct WgpuPipelines {
    frame_clear: wgpu::RenderPipeline,
    frame_present: wgpu::RenderPipeline,
    quads: wgpu::RenderPipeline,
    /// Quads with a solid background, no border and no fade; see [`RendererTuning::split_quads`].
    simple_quads: wgpu::RenderPipeline,
    /// Replaces `quads` for the filters that have a [`DrawFilter::quad_variant`].
    quad_variant: Option<wgpu::RenderPipeline>,
    /// For [`RendererTuning::opaque_quads`].
    opaque_quads: Option<OpaqueQuadPipelines>,
    shadows: wgpu::RenderPipeline,
    shapes: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
    /// Pipeline for compositing a cached layer texture onto the frame.
    layer_composite: wgpu::RenderPipeline,
    backdrop_downsample: wgpu::RenderPipeline,
    backdrop_blur: wgpu::RenderPipeline,
    backdrop_composite: wgpu::RenderPipeline,
}

/// One frame allocation of instance data, ready to bind.
struct InstanceBinding {
    bind_group: wgpu::BindGroup,
    /// Index of the allocation's first instance within the bound data. Always
    /// zero on the storage-buffer path, where the binding offset already
    /// positions the array; on the WebGL texture path the shader indexes the
    /// shared instance texture absolutely, so draws must offset their
    /// instance (or vertex) ranges by this value.
    first_instance: u32,
}

struct InstanceBindings {
    /// For [`RendererTuning::opaque_quads`].
    opaque: Option<OpaqueQuadBindings>,
    quads: InstanceBinding,
    shadows: InstanceBinding,
    shapes: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    subpixel_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
    /// Bind group layout for the layer composite pipeline: uniform params +
    /// single RGBA texture + sampler.
    layer_surfaces: wgpu::BindGroupLayout,
    /// Backdrop blur passes and composite: `BackdropParams` + source texture + sampler.
    backdrop: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

enum InstanceData {
    Storage(wgpu::Buffer),
    // WebGL2 has no storage buffers. A uint texture keeps the records available to both shader
    // stages while preserving integer and floating-point bit patterns exactly.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        binding: std::sync::OnceLock<wgpu::BindGroup>,
        width: u32,
        height: u32,
    },
}

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    pipeline_cache: Option<wgpu::PipelineCache>,
    pipeline_cache_revision: Arc<std::sync::atomic::AtomicU64>,
    queue: Arc<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    instance_data: InstanceData,
    /// Keyed by source (0 for `instance_data`, otherwise the upload ring slot plus one), offset
    /// and size.
    instance_bindings: HashMap<(usize, u64, u64), wgpu::BindGroup>,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    path_scratch_globals: Option<wgpu::BindGroup>,
    path_scratch_binding: Option<wgpu::BindGroup>,
    /// Cached offscreen textures for layered views, keyed by [`LayerId`] raw value.
    layer_textures: HashMap<u64, LayerTexture>,
    backdrop_textures: Option<BackdropTextures>,
    retained_frame: Option<RetainedFrame>,
    /// [`RendererTuning::depth_buffer`] attachments by size.
    scene_depth: HashMap<(u32, u32), wgpu::TextureView>,
}

impl WgpuResources {
    fn configure_surface(&self, configuration: &wgpu::SurfaceConfiguration) {
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, configuration);
        }
    }

    /// Drop the retained frame and scratch textures so they're recreated at the new size.
    /// Does NOT touch `layer_textures`: those are per-layer (not surface-sized) and are the cached
    /// textures the resize cull composites — clearing them every resize frame made tool windows go
    /// invisible mid-resize. Per-layer textures are resized by `ensure_layer_texture` and dropped by
    /// `evict_stale_layers`; only a real device loss invalidates them (handled at the call site).
    fn invalidate_intermediate_textures(&mut self) {
        self.retained_frame = None;
        self.invalidate_path_intermediates();
        self.backdrop_textures = None;
    }

    fn invalidate_path_intermediates(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
        self.path_scratch_globals = None;
        self.path_scratch_binding = None;
    }
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    resources: Option<WgpuResources>,
    staging_scene: Scene,
    opaque_plan: OpaqueQuadPlan,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    gamma_offset: u64,
    instance_data_capacity: u64,
    max_instance_data_size: u64,
    instance_data_alignment: u64,
    uses_webgl_instance_data: bool,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    last_error: Arc<Mutex<Option<String>>>,
    failed_frame_count: u32,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    surface_configured: bool,
    needs_redraw: bool,
    retained_generation: u64,
    /// The surface can be copied from, so backdrop blur surfaces are drawn.
    backdrop_blur_supported: bool,
    frame_timings: Option<crate::frame_timings::FrameTimings>,
    instance_uploads: Option<crate::upload_ring::UploadRing>,
    instance_upload_encoder: Option<wgpu::CommandEncoder>,
    instance_upload_slot: Option<usize>,
    instance_upload_offset: u64,
    instance_upload_capacity: u64,
    /// Ring-to-buffer copies of the frame being recorded; the last one may still grow.
    buffer_copies: Vec<BufferCopy>,
    path_vertices_scratch: Vec<PathRasterizationVertex>,
    path_sprites_scratch: Vec<PathSprite>,
    path_jobs_scratch: Vec<PathJob>,
    layer_ids_scratch: std::collections::HashSet<u64>,
    tuning: RendererTuning,
    /// Frame uniforms the globals buffer holds, and those an unsubmitted frame is uploading.
    frame_uniforms: Option<[u8; FRAME_UNIFORMS_SIZE]>,
    pending_frame_uniforms: Option<[u8; FRAME_UNIFORMS_SIZE]>,
}

const FRAME_UNIFORMS_SIZE: usize =
    std::mem::size_of::<GlobalParams>() + std::mem::size_of::<GammaParams>();

#[derive(Debug)]
pub struct UploadUnavailable;
impl std::fmt::Display for UploadUnavailable {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str("GPU upload buffers are still in flight")
    }
}
impl std::error::Error for UploadUnavailable {}

impl WgpuRenderer {
    /// Warm caller-owned target pipelines on the device's submission owner
    /// before publishing the context to windows.
    pub async fn precompile_external(
        context: &WgpuContext,
        format: wgpu::TextureFormat,
        tuning: RendererTuning,
    ) -> anyhow::Result<()> {
        let memory_scope = context.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = context.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = context.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        let layouts = Self::create_bind_group_layouts(&context.device, uses_webgl_instance_data);
        let parameters = RenderingParameters::new(&context.adapter, format, tuning.path_sample_count);
        for alpha_mode in [wgpu::CompositeAlphaMode::Opaque, wgpu::CompositeAlphaMode::PreMultiplied] {
            let pipelines = Self::create_pipelines(
                &context.device,
                context.pipeline_cache.as_ref(),
                &layouts,
                format,
                alpha_mode,
                parameters.path_sample_count,
                context.supports_dual_source_blending(),
                uses_webgl_instance_data,
                tuning.shader_runtime_checks,
                tuning.draw_filter.quad_variant(),
                tuning.opaque_quads,
                tuning.depth_buffer,
                tuning.coverage_epilogue,
            );
            drop(pipelines);
        }
        let validation = validation_scope.pop().await;
        let internal = internal_scope.pop().await;
        let memory = memory_scope.pop().await;
        if let Some(error) = validation.or(internal).or(memory) {
            anyhow::bail!("External pipeline precompile failed: {error}");
        }
        context.pipeline_cache_revision.fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            // Fall back to the display handle already provided via InstanceDescriptor::display.
            raw_display_handle: None,
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(|| WgpuContext::instance(Box::new(window.clone())));

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(instance, &surface, compositor_gpu)?),
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            atlas,
            RendererTuning::default(),
        )
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_from_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_from_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(None, context, surface, config, None, atlas, RendererTuning::default())
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
        tuning: RendererTuning,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        Self::new_with_target(
            gpu_context,
            context,
            Some(surface),
            surface_caps,
            config,
            compositor_gpu,
            atlas,
            tuning,
        )
    }

    /// The caller owns target allocation, synchronization, and presentation.
    pub fn new_external(
        context: &WgpuContext,
        config: WgpuSurfaceConfig,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        Self::new_external_tuned(context, config, format, RendererTuning::default())
    }

    /// [`Self::new_external`] with explicit cost trade-offs, so no pipelines are built twice.
    pub fn new_external_tuned(
        context: &WgpuContext,
        config: WgpuSurfaceConfig,
        format: wgpu::TextureFormat,
        tuning: RendererTuning,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(
                format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            ),
            "External targets require RGBA8 or BGRA8 unorm"
        );
        anyhow::ensure!(
            config.size.width.0 > 0 && config.size.height.0 > 0,
            "Empty external target"
        );
        anyhow::ensure!(
            config.size.width.0 as u32 <= context.device.limits().max_texture_dimension_2d
                && config.size.height.0 as u32 <= context.device.limits().max_texture_dimension_2d,
            "External target exceeds device limits"
        );
        let capabilities = wgpu::SurfaceCapabilities {
            formats: vec![format],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        };
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        atlas.set_merged_uploads(tuning.merged_atlas_uploads);
        Self::new_with_target(None, context, None, capabilities, config, None, atlas, tuning)
    }

    fn new_with_target(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: Option<wgpu::Surface<'static>>,
        surface_caps: wgpu::SurfaceCapabilities,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
        tuning: RendererTuning,
    ) -> anyhow::Result<Self> {
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        let backdrop_blur_supported = !uses_webgl_instance_data
            && surface_caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: if backdrop_blur_supported {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            },
            format: surface_format,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: select_present_mode(
                config.preferred_present_mode,
                &surface_caps.present_modes,
            ),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        // Configure the surface immediately. The adapter selection process already validated
        // that this adapter can successfully configure this surface.
        if let Some(surface) = &surface {
            surface.configure(&context.device, &surface_config);
        }

        let queue = Arc::clone(&context.queue);
        let rendering_params =
            RenderingParameters::new(&context.adapter, surface_format, tuning.path_sample_count);
        let dual_source_blending =
            context.supports_dual_source_blending() && !uses_webgl_instance_data;
        let bind_group_layouts = Self::create_bind_group_layouts(&device, uses_webgl_instance_data);
        let pipelines = Self::create_pipelines(
            &device,
            context.pipeline_cache.as_ref(),
            &bind_group_layouts,
            surface_format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
            uses_webgl_instance_data,
            tuning.shader_runtime_checks,
            tuning.draw_filter.quad_variant(),
            tuning.opaque_quads,
            tuning.depth_buffer,
            tuning.coverage_epilogue,
        );
        context.pipeline_cache_revision.fetch_add(1, std::sync::atomic::Ordering::Release);

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let gamma_offset = globals_size.next_multiple_of(uniform_alignment);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (
            instance_data,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
        ) = if uses_webgl_instance_data {
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            let max_instance_data_size = (u64::from(max_texture_dimension).pow(2)
                * INSTANCE_TEXTURE_TEXEL_SIZE)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_instance_data_size);
            let (instance_data, capacity) =
                Self::create_instance_texture(&device, initial_capacity, max_texture_dimension);
            (
                instance_data,
                capacity,
                max_instance_data_size,
                INSTANCE_TEXTURE_TEXEL_SIZE,
            )
        } else {
            // Every frame allocation is exposed as one storage-buffer binding, so
            // its backing buffer must satisfy both the allocation and binding limits.
            let max_buffer_size = device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_buffer_size);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: initial_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                InstanceData::Storage(buffer),
                initial_capacity,
                max_buffer_size,
                device.limits().min_storage_buffer_offset_alignment as u64,
            )
        };

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = context.adapter.get_info();

        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let last_error_clone = Arc::clone(&last_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            let mut guard = last_error_clone.lock().unwrap();
            *guard = Some(error.to_string());
        }));

        let resources = WgpuResources {
            device,
            pipeline_cache: context.pipeline_cache.clone(),
            pipeline_cache_revision: context.pipeline_cache_revision.clone(),
            queue,
            surface,
            pipelines,
            bind_group_layouts,
            atlas_sampler,
            globals_buffer,
            globals_bind_group,
            instance_data,
            instance_bindings: HashMap::new(),
            // Defer intermediate texture creation until a path batch needs it.
            // This avoids panics when the device/surface is in an invalid state during initialization.
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
            path_scratch_globals: None,
            path_scratch_binding: None,
            layer_textures: HashMap::new(),
            backdrop_textures: None,
            retained_frame: None,
            scene_depth: HashMap::new(),
        };

        let frame_timings = if resources.surface.is_none() {
            crate::frame_timings::FrameTimings::new(&resources.device, &resources.queue)
        } else { None };
        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            resources: Some(resources),
            staging_scene: Scene::default(),
            opaque_plan: OpaqueQuadPlan::default(),
            surface_config,
            atlas,
            gamma_offset,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
            uses_webgl_instance_data,
            rendering_params,
            is_bgr: false,
            dual_source_blending,
            adapter_info,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            last_error,
            failed_frame_count: 0,
            device_lost: context.device_lost_flag(),
            surface_configured: true,
            needs_redraw: false,
            retained_generation: 0,
            backdrop_blur_supported,
            frame_timings,
            instance_uploads: None,
            instance_upload_encoder: None,
            instance_upload_slot: None,
            instance_upload_offset: 0,
            buffer_copies: Vec::new(),
            instance_upload_capacity: 512 * 1024,
            path_vertices_scratch: Vec::new(),
            path_sprites_scratch: Vec::new(),
            path_jobs_scratch: Vec::new(),
            layer_ids_scratch: std::collections::HashSet::new(),
            tuning,
            frame_uniforms: None,
            pending_frame_uniforms: None,
        })
    }

    fn create_bind_group_layouts(
        device: &wgpu::Device,
        uses_webgl_instance_data: bool,
    ) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let instance_data_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: if uses_webgl_instance_data {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[instance_data_entry],
        });

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layer_surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<LayerSurfaceParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let backdrop = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<BackdropParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            texture,
            surfaces,
            layer_surfaces,
            backdrop,
        }
    }

    fn create_instance_texture(
        device: &wgpu::Device,
        requested_capacity: u64,
        max_texture_dimension: u32,
    ) -> (InstanceData, u64) {
        let texel_count = requested_capacity.div_ceil(INSTANCE_TEXTURE_TEXEL_SIZE);
        let width = texel_count.min(u64::from(max_texture_dimension)).max(1) as u32;
        let height = texel_count
            .div_ceil(u64::from(width))
            .min(u64::from(max_texture_dimension))
            .max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("instance_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let capacity = u64::from(width) * u64::from(height) * INSTANCE_TEXTURE_TEXEL_SIZE;
        (
            InstanceData::Texture {
                texture,
                view,
                binding: Default::default(),
                width,
                height,
            },
            capacity,
        )
    }

    fn create_pipelines(
        device: &wgpu::Device,
        pipeline_cache: Option<&wgpu::PipelineCache>,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
        uses_webgl_instance_data: bool,
        shader_runtime_checks: bool,
        quad_variant: Option<(&str, &str, bool)>,
        opaque_quads: bool,
        scene_depth: bool,
        coverage_epilogue: bool,
    ) -> WgpuPipelines {
        // Every pipeline that draws in scene passes; the others have passes of their own.
        let depth_stencil = |name: &str| {
            let scene = !matches!(name, "path_rasterization" | "backdrop_downsample" | "backdrop_blur" | "fs_frame");
            (scene_depth && scene).then(|| wgpu::DepthStencilState {
                format: SCENE_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending =
            dual_source_blending && device_has_feature && !uses_webgl_instance_data;

        let shader_source = if uses_webgl_instance_data {
            WEBGL_SHADERS
        } else {
            STORAGE_BUFFER_SHADERS
        };
        let create_module = |descriptor: wgpu::ShaderModuleDescriptor<'_>| {
            if shader_runtime_checks {
                device.create_shader_module(descriptor)
            } else {
                // SAFETY: storage indices come from draw ranges sized to the renderer's own
                // instance arrays, and the shaders' loops have constant or data-bounded trip
                // counts.
                unsafe {
                    device.create_shader_module_trusted(
                        descriptor,
                        wgpu::ShaderRuntimeChecks::unchecked(),
                    )
                }
            }
        };
        let shader_module = create_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let frame_pipeline = |fragment: &str, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(fragment),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_frame"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some(fragment),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: depth_stencil(fragment),
                multisample: Default::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };
        let frame_clear = frame_pipeline("fs_clear_frame", &[]);
        let frame_present = frame_pipeline("fs_frame", &[Some(&layouts.texture)]);

        let subpixel_shader_module = if dual_source_blending {
            Some(create_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(SUBPIXEL_SHADERS.into()),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               texture_layout: Option<&wgpu::BindGroupLayout>,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let mut bind_group_layouts = vec![Some(globals_layout), Some(data_layout)];
            bind_group_layouts.extend(texture_layout.map(Some));
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &bind_group_layouts,
                immediate_size: 0,
            });

            let replace_targets: Vec<_>;
            let color_targets = if coverage_epilogue {
                replace_targets = color_targets
                    .iter()
                    .map(|target| {
                        target.clone().map(|target| wgpu::ColorTargetState {
                            blend: target.blend.or(Some(wgpu::BlendState::REPLACE)),
                            ..target
                        })
                    })
                    .collect();
                &replace_targets[..]
            } else {
                color_targets
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_stencil(name),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };

        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let simple_quads = create_pipeline(
            "simple_quads",
            "vs_quad_simple",
            "fs_quad_simple",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let quad_variant = quad_variant.map(|(vertex, fragment, blend)| {
            let target = wgpu::ColorTargetState {
                blend: if blend { color_target.blend } else { None },
                ..color_target.clone()
            };
            create_pipeline(
                fragment,
                vertex,
                fragment,
                &layouts.globals,
                &layouts.instances,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(target)],
                1,
                &shader_module,
            )
        });

        let opaque_quads = opaque_quads.then(|| OpaqueQuadPipelines {
            interiors: create_pipeline(
                "opaque_quads",
                "vs_quad_opaque",
                "fs_quad_opaque",
                &layouts.globals,
                &layouts.instances,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    blend: None,
                    ..color_target.clone()
                })],
                1,
                &shader_module,
            ),
            clamped: create_pipeline(
                "clamped_quads",
                "vs_quad_clamped",
                "fs_quad_simple",
                &layouts.globals,
                &layouts.instances,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(color_target.clone())],
                1,
                &shader_module,
            ),
        });

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shapes = create_pipeline(
            "shapes",
            "vs_shape",
            "fs_shape",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances,
                Some(&layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let layer_composite = create_pipeline(
            "layer_composite",
            "vs_layer_composite",
            "fs_layer_composite",
            &layouts.globals,
            &layouts.layer_surfaces,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let backdrop_pass_target = [Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let backdrop_downsample = create_pipeline(
            "backdrop_downsample",
            "vs_backdrop_pass",
            "fs_backdrop_downsample",
            &layouts.globals,
            &layouts.backdrop,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &backdrop_pass_target,
            1,
            &shader_module,
        );
        let backdrop_blur = create_pipeline(
            "backdrop_blur",
            "vs_backdrop_pass",
            "fs_backdrop_blur",
            &layouts.globals,
            &layouts.backdrop,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &backdrop_pass_target,
            1,
            &shader_module,
        );
        let backdrop_composite = create_pipeline(
            "backdrop_composite",
            "vs_backdrop_composite",
            "fs_backdrop_composite",
            &layouts.globals,
            &layouts.backdrop,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        WgpuPipelines {
            frame_clear,
            frame_present,
            quads,
            simple_quads,
            quad_variant,
            opaque_quads,
            shadows,
            shapes,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
            layer_composite,
            backdrop_downsample,
            backdrop_blur,
            backdrop_composite,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Samples are cleared, resolved and discarded within one pass, so
            // tile-based GPUs can keep them on chip without backing memory.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width != self.surface_config.width || height != self.surface_config.height {
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            self.surface_config.width = clamped_width.max(1);
            self.surface_config.height = clamped_height.max(1);
            let surface_config = self.surface_config.clone();

            let Some(resources) = self.resources.as_mut() else {
                return;
            };

            // Wait for any in-flight GPU work to complete before destroying textures
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                warn!("Failed to poll device during resize: {e:?}");
                self.device_lost.store(true, std::sync::atomic::Ordering::Release);
                return;
            }

            // Destroy old textures before allocating new ones to avoid GPU memory spikes
            if let Some(ref texture) = resources.path_intermediate_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.path_msaa_texture {
                texture.destroy();
            }

            resources.configure_surface(&surface_config);

            // Invalidate intermediate textures - they will be lazily recreated
            // in draw() after we confirm the surface is healthy. This avoids
            // panics when the device/surface is in an invalid state during resize.
            resources.invalidate_intermediate_textures();
        }
    }

    /// Returns the size of the path intermediate, which may exceed the request.
    fn ensure_intermediate_textures(&mut self, width: u32, height: u32) -> [u32; 2] {
        let current = self
            .resources()
            .path_intermediate_texture
            .as_ref()
            .map(|texture| [texture.width(), texture.height()]);
        let [width, height] = path_intermediate_size(current, [width, height]);
        if current == Some([width, height]) && self.resources().path_scratch_binding.is_some() {
            return [width, height];
        }

        let format = self.surface_config.format;
        let path_sample_count = self.rendering_params.path_sample_count;
        let gamma_offset = self.gamma_offset;
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(t);
        resources.path_intermediate_view = Some(v);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;

        resources.path_scratch_globals = Some(Self::viewport_globals(
            resources,
            gamma_offset,
            width,
            height,
            false,
        ));
        self.resources_mut().path_scratch_binding = Some(self.create_texture_bind_group(
            "path_scratch_binding",
            self.resources().path_intermediate_view.as_ref().unwrap(),
        ));
        [width, height]
    }

    fn viewport_globals(
        resources: &WgpuResources,
        gamma_offset: u64,
        width: u32,
        height: u32,
        premultiplied_alpha: bool,
    ) -> wgpu::BindGroup {
        let globals = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport_globals"),
            size: std::mem::size_of::<GlobalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        resources.queue.write_buffer(
            &globals,
            0,
            bytemuck::bytes_of(&GlobalParams {
                viewport_size: [width as f32, height as f32],
                premultiplied_alpha: u32::from(premultiplied_alpha),
                pad: 0,
            }),
        );
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewport_globals"),
                layout: &resources.bind_group_layouts.globals,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: globals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &resources.globals_buffer,
                            offset: gamma_offset,
                            size: NonZeroU64::new(std::mem::size_of::<GammaParams>() as u64),
                        }),
                    },
                ],
            })
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        if self.is_bgr != is_bgr {
            self.invalidate_retained_pixels();
        }
        self.is_bgr = is_bgr;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        if new_alpha_mode != self.surface_config.alpha_mode {
            self.invalidate_retained_pixels();
            self.surface_config.alpha_mode = new_alpha_mode;
            let surface_config = self.surface_config.clone();
            let Some(resources) = self.resources.as_ref() else {
                return;
            };
            resources.configure_surface(&surface_config);
            self.rebuild_pipelines();
        }
    }

    pub fn tuning(&self) -> RendererTuning {
        self.tuning
    }

    /// Applies new trade-offs from the next frame on. Pipelines, path intermediates and the
    /// upload ring are rebuilt only for the settings that changed.
    pub fn set_tuning(&mut self, tuning: RendererTuning) {
        if tuning == self.tuning {
            return;
        }
        let previous = std::mem::replace(&mut self.tuning, tuning);
        self.atlas.set_merged_uploads(tuning.merged_atlas_uploads);
        let path_sample_count = self.rendering_params.sample_count_for(tuning.path_sample_count);
        let samples_changed = path_sample_count != self.rendering_params.path_sample_count;
        self.rendering_params.path_sample_count = path_sample_count;
        if tuning.direct_instance_uploads != previous.direct_instance_uploads {
            if tuning.direct_instance_uploads && !self.supports_direct_instance_uploads() {
                warn!("Direct instance uploads need MAPPABLE_PRIMARY_BUFFERS; copying instead");
            }
            self.instance_uploads = None;
            if let Some(resources) = self.resources.as_mut() {
                resources.instance_bindings.clear();
            }
        }
        if samples_changed {
            if let Some(resources) = self.resources.as_mut() {
                resources.invalidate_path_intermediates();
            }
        }
        if samples_changed
            || tuning.shader_runtime_checks != previous.shader_runtime_checks
            || tuning.draw_filter.quad_variant() != previous.draw_filter.quad_variant()
            || tuning.opaque_quads != previous.opaque_quads
            || tuning.depth_buffer != previous.depth_buffer
            || tuning.coverage_epilogue != previous.coverage_epilogue
            || tuning.pipeline_epoch != previous.pipeline_epoch
        {
            self.rebuild_pipelines();
        }
    }

    fn rebuild_pipelines(&mut self) {
        let surface_config = &self.surface_config;
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        resources.pipelines = Self::create_pipelines(
            &resources.device,
            resources.pipeline_cache.as_ref(),
            &resources.bind_group_layouts,
            surface_config.format,
            surface_config.alpha_mode,
            self.rendering_params.path_sample_count,
            self.dual_source_blending,
            self.uses_webgl_instance_data,
            self.tuning.shader_runtime_checks,
            self.tuning.draw_filter.quad_variant(),
            self.tuning.opaque_quads,
            self.tuning.depth_buffer,
            self.tuning.coverage_epilogue,
        );
        resources.pipeline_cache_revision.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    fn supports_direct_instance_uploads(&self) -> bool {
        !self.uses_webgl_instance_data
            && self.resources.as_ref().is_some_and(|resources| {
                resources.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            })
    }

    fn direct_instance_uploads(&self) -> bool {
        self.tuning.direct_instance_uploads && self.supports_direct_instance_uploads()
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) -> bool {
        #[cfg(target_family = "wasm")]
        if self.device_lost() {
            if self.surface_configured {
                log::error!(
                    "Browser graphics context was lost; rendering has stopped. Reload the page to recover."
                );
                self.surface_configured = false;
            }
            return false;
        }

        // Bail out early if the surface has been unconfigured (e.g. during
        // Android background/rotation transitions).  Attempting to acquire
        // a texture from an unconfigured surface can block indefinitely on
        // some drivers (Adreno).
        if !self.surface_configured {
            return false;
        }

        let last_error = self.last_error.lock().unwrap().take();
        if let Some(error) = last_error {
            self.invalidate_retained_pixels();
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );

            // TBD. Does retrying more actually help?
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            } else if self.failed_frame_count > 5 {
                if let Some(res) = self.resources.as_mut() {
                    res.invalidate_intermediate_textures();
                    // A real GPU error may have lost the device — drop the cached layer textures
                    // too so they're recaptured fresh (resize alone must not reach here).
                    res.layer_textures.clear();
                }
                self.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        self.atlas.before_frame();

        let Some(surface) = &self.resources().surface else {
            return false;
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.invalidate_retained_pixels();
                // Textures must be destroyed before the surface can be reconfigured.
                drop(frame);
                let surface_config = self.surface_config.clone();
                let resources = self.resources_mut();
                resources.configure_surface(&surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.invalidate_retained_pixels();
                let surface_config = self.surface_config.clone();
                let resources = self.resources_mut();
                resources.configure_surface(&surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.invalidate_retained_pixels();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.invalidate_retained_pixels();
                *self.last_error.lock().unwrap() =
                    Some("Surface texture validation error".to_string());
                return false;
            }
        };

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(error) = self.write_frame_uniforms() {
            self.invalidate_retained_pixels();
            log::error!("Frame uniforms failed: {error:#}");
            return false;
        }
        if let Err(error) = self.record_frame(scene, &frame_view) {
            self.invalidate_retained_pixels();
            log::error!("{error:#}");
            self.resources().queue.submit(std::iter::empty());
            return false;
        }

        frame.present();
        self.evict_stale_layers(scene);
        true
    }

    fn invalidate_retained_pixels(&mut self) {
        self.needs_redraw = true;
        self.retained_generation = self.retained_generation.wrapping_add(1);
        if let Some(resources) = self.resources.as_mut() {
            if let Some(frame) = resources.retained_frame.as_mut() {
                frame.valid = false;
            }
            for layer in resources.layer_textures.values_mut() {
                layer.valid = false;
                for cached in layer.composite.borrow_mut().iter_mut() { cached.params = None; }
            }
            if let Some(textures) = &resources.backdrop_textures {
                for cached in textures.bindings.borrow_mut().iter_mut() { cached.params = None; }
            }
        }
    }

    fn ensure_retained_frame(&mut self) {
        if self.resources().retained_frame.is_some() {
            return;
        }
        let texture = self
            .resources()
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("retained_frame"),
                size: wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let view = texture.create_view(&Default::default());
        let bind_group = self.create_texture_bind_group("retained_frame", &view);
        self.resources_mut().retained_frame = Some(RetainedFrame {
            texture,
            view,
            bind_group,
            valid: false,
        });
    }

    fn present_retained_frame(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let resources = self.resources();
        let Some(frame) = resources.retained_frame.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present_retained_frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        pass.set_pipeline(&resources.pipelines.frame_present);
        pass.set_bind_group(0, &frame.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    /// Submission completion does not imply that the display has released a target.
    pub fn draw_external(
        &mut self,
        scene: &Scene,
        target: &wgpu::Texture,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.draw_external_with_view(scene, target, &view)
    }

    /// Reuse a default view owned by the scanout allocation through GPU/display retirement.
    pub fn draw_external_with_view(
        &mut self,
        scene: &Scene,
        target: &wgpu::Texture,
        view: &wgpu::TextureView,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        self.draw_external_damage(scene, target, view, SceneDamage::Full)
    }

    /// `damage` covers every pixel that differs between `scene` and the
    /// contents `target` already holds, for example the union of the frames
    /// drawn since this target was last rendered. Pixels outside it are kept.
    pub fn draw_external_damage(
        &mut self,
        scene: &Scene,
        target: &wgpu::Texture,
        view: &wgpu::TextureView,
        damage: SceneDamage,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        anyhow::ensure!(view.texture() == target, "External view belongs to another target");
        anyhow::ensure!(
            self.resources().surface.is_none(),
            "Renderer owns a window surface"
        );
        anyhow::ensure!(!self.device_lost(), "External-target device lost");
        anyhow::ensure!(
            target.format() == self.surface_config.format
                && target.width() == self.surface_config.width
                && target.height() == self.surface_config.height
                && target.depth_or_array_layers() == 1
                && target.sample_count() == 1
                && target.dimension() == wgpu::TextureDimension::D2
                && target
                    .usage()
                    .contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
            "External target does not match renderer configuration"
        );
        if let Some(error) = self.last_error.lock().unwrap().take() {
            anyhow::bail!("GPU error during external frame: {error}");
        }
        if !self.atlas.before_external_frame()? { return Err(anyhow::Error::new(UploadUnavailable)); }
        self.begin_instance_uploads()?;
        if let Err(error) = self.write_frame_uniforms() {
            self.instance_upload_encoder = None;
            self.instance_upload_slot = None;
            return Err(error);
        }
        let submission = match self.record_external_frame(scene, target, view, damage) {
            Ok(submission) => submission,
            Err(error) => {
                self.instance_upload_encoder = None;
                self.instance_upload_slot = None;
                if let Some(timings) = &self.frame_timings { timings.cancel(); }
                self.invalidate_retained_pixels();
                return Err(error);
            }
        };
        self.evict_stale_layers(scene);
        Ok(submission)
    }

    fn write_frame_uniforms(&mut self) -> Result<()> {
        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };

        let globals = GlobalParams {
            viewport_size: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            premultiplied_alpha: if self.surface_config.alpha_mode
                == wgpu::CompositeAlphaMode::PreMultiplied
            {
                1
            } else {
                0
            },
            pad: 0,
        };

        let mut uniforms = [0; FRAME_UNIFORMS_SIZE];
        let (global_bytes, gamma_bytes) = uniforms.split_at_mut(std::mem::size_of::<GlobalParams>());
        global_bytes.copy_from_slice(bytemuck::bytes_of(&globals));
        gamma_bytes.copy_from_slice(bytemuck::bytes_of(&gamma_params));
        self.pending_frame_uniforms = None;
        if self.frame_uniforms == Some(uniforms) {
            return Ok(());
        }
        let buffer = self.resources().globals_buffer.clone();
        self.upload_buffer(&buffer, 0, bytemuck::bytes_of(&globals))?;
        self.upload_buffer(&buffer, self.gamma_offset, bytemuck::bytes_of(&gamma_params))?;
        if self.instance_upload_slot.is_some() {
            // Ring copies only land if this frame is submitted.
            self.frame_uniforms = None;
            self.pending_frame_uniforms = Some(uniforms);
        } else {
            self.frame_uniforms = Some(uniforms);
        }
        Ok(())
    }

    fn record_external_frame(&mut self, scene: &Scene, target: &wgpu::Texture,
        view: &wgpu::TextureView, damage: SceneDamage) -> Result<wgpu::SubmissionIndex> {
        if let Some(textures) = &self.resources().backdrop_textures { textures.next_binding.set(0); }
        for layer in self.resources().layer_textures.values() { layer.next_composite.set(0); }
        let blurs_backdrop = self.backdrop_blur_supported && scene.surfaces.iter()
            .any(|surface| matches!(surface.source, PaintSurfaceSource::BackdropBlur(_)));
        anyhow::ensure!(!blurs_backdrop || target.usage().contains(wgpu::TextureUsages::COPY_SRC),
            "Backdrop blur requires a readable external target");
        if blurs_backdrop { self.ensure_backdrop_textures(); }
        let mut instance_offset = 0;
        let mut encoder = self.resources().device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpui_external_frame"),
        });
        let timing = self.frame_timings.as_mut().and_then(|timings| timings.begin(&mut encoder));
        self.render_layers(scene, &mut instance_offset, &mut encoder)?;
        if let (Some(timings), Some(index)) = (&self.frame_timings, timing) {
            timings.intermediate_end(index, &mut encoder);
        }
        let globals = self.resources().globals_bind_group.clone();
        // Paint directly into the scanout image; retaining the root would add a
        // second full-screen color pass. Backdrop blur samples pixels outside
        // the damage, so it repaints everything.
        let damage = if blurs_backdrop { SceneDamage::Full } else { damage };
        self.encode_scene(scene, damage, self.viewport_size(), target, view,
            &globals, blurs_backdrop, &mut instance_offset, &mut encoder)?;
        if let (Some(timings), Some(index)) = (&self.frame_timings, timing) {
            timings.end(index, &mut encoder);
        }
        let slot = self.instance_upload_slot.context("External frame upload slot missing")?;
        let mut uploads = self.instance_upload_encoder.take().context("External frame upload encoder missing")?;
        self.instance_upload_slot = None;
        let ring = self.instance_uploads.as_ref().context("External instance ring missing")?;
        for copy in self.buffer_copies.drain(..) {
            crate::pass_trace::note(|| format!("buffer_copy bytes={}", copy.length));
            crate::pass_trace::transfer(crate::pass_trace::Transfer::Buffer, 1, copy.length);
            uploads.copy_buffer_to_buffer(ring.buffer(slot), copy.source, &copy.buffer, copy.target, copy.length);
        }
        ring.unmap(slot);
        crate::pass_trace::note(|| format!("instance_uploads bytes={} direct={}",
            self.instance_upload_offset, ring.storage()));
        let submission = self.resources().queue.submit([uploads.finish(), encoder.finish()]);
        ring.submitted(slot);
        if let Some(uniforms) = self.pending_frame_uniforms.take() {
            self.frame_uniforms = Some(uniforms);
        }
        if let (Some(timings), Some(index)) = (&self.frame_timings, timing) {
            timings.submitted(index);
        }
        Ok(submission)
    }

    pub fn take_gpu_frame_timings(&mut self) -> crate::GpuFrameTimings {
        self.frame_timings.as_mut().map(|timings| timings.take()).unwrap_or_default()
    }

    fn begin_instance_uploads(&mut self) -> Result<()> {
        if self.resources().instance_bindings.len() > 4096 {
            self.resources_mut().instance_bindings.clear();
        }
        let direct = self.direct_instance_uploads();
        if self.instance_uploads.as_ref().is_none_or(|ring| {
            ring.capacity() < self.instance_upload_capacity || ring.storage() != direct
        }) {
            anyhow::ensure!(self.instance_upload_capacity <= self.resources().device.limits().max_buffer_size,
                "Instance uploads exceed device limit");
            // Cached bindings may name the replaced ring's buffers.
            self.resources_mut().instance_bindings.clear();
            self.instance_uploads = Some(crate::upload_ring::UploadRing::new(
                &self.resources().device, self.instance_upload_capacity, direct));
        }
        let ring = self.instance_uploads.as_ref().context("Instance upload ring missing")?;
        anyhow::ensure!(!ring.failed(), "Instance upload mapping failed");
        let slot = ring.acquire().ok_or_else(|| anyhow::Error::new(UploadUnavailable))?;
        self.instance_upload_slot = Some(slot);
        self.instance_upload_offset = 0;
        self.buffer_copies.clear();
        self.instance_upload_encoder = Some(self.resources().device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("gpui_instance_uploads") }));
        Ok(())
    }

    /// Stages `bytes` in the upload ring for a copy into `buffer` at `offset`. A copy that
    /// continues the previous one into the same buffer extends it, across an alignment gap,
    /// since PowerVR runs a transfer job per copy region.
    fn upload_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) -> Result<()> {
        let Some(slot) = self.instance_upload_slot else {
            crate::pass_trace::transfer(crate::pass_trace::Transfer::Buffer, 1, bytes.len() as u64);
            self.resources().queue.write_buffer(buffer, offset, bytes);
            return Ok(());
        };
        let length = bytes.len() as u64;
        anyhow::ensure!(length & 3 == 0 && offset & 3 == 0, "Unaligned instance upload");
        let gap = self.buffer_copies.last()
            .filter(|last| last.buffer == *buffer && last.source + last.length == self.instance_upload_offset)
            .and_then(|last| offset.checked_sub(last.target + last.length))
            .filter(|gap| *gap <= BUFFER_COPY_MAX_GAP)
            // The gap's bytes come from the ring; they must not hide an earlier copy.
            .filter(|gap| {
                let start = offset - gap;
                !self.buffer_copies.iter().any(|copy| copy.buffer == *buffer
                    && copy.target < offset && start < copy.target + copy.length)
            });
        let source = self.instance_upload_offset + gap.unwrap_or(0);
        let end = source.checked_add(length).context("Instance staging overflow")?;
        let ring = self.instance_uploads.as_ref().context("Instance upload ring missing")?;
        if end > ring.capacity() {
            // No commands from this attempted frame have been submitted. Retry
            // after replacing the ring; older submissions retain their buffers.
            self.instance_upload_capacity = end.checked_next_power_of_two().context("Instance ring size overflow")?;
            return Err(anyhow::Error::new(UploadUnavailable));
        }
        ring.write(slot, source, bytes)?;
        match gap {
            Some(gap) => self.buffer_copies.last_mut().context("Buffer copy missing")?.length += gap + length,
            None => self.buffer_copies.push(BufferCopy { buffer: buffer.clone(), source, target: offset, length }),
        }
        self.instance_upload_offset = end;
        Ok(())
    }

    fn record_frame(
        &mut self,
        scene: &Scene,
        surface_view: &wgpu::TextureView,
    ) -> Result<wgpu::SubmissionIndex> {
        if let Some(textures) = &self.resources().backdrop_textures { textures.next_binding.set(0); }
        for layer in self.resources().layer_textures.values() { layer.next_composite.set(0); }
        let mut instance_offset = 0;
        self.ensure_retained_frame();
        let frame = self
            .resources()
            .retained_frame
            .as_ref()
            .expect("frame was ensured");
        let frame_texture = frame.texture.clone();
        let frame_view = frame.view.clone();
        let mut damage = if frame.valid {
            scene.damage
        } else {
            SceneDamage::Full
        };
        let blurs_backdrop = self.backdrop_blur_supported
            && scene
                .surfaces
                .iter()
                .any(|surface| matches!(surface.source, PaintSurfaceSource::BackdropBlur(_)));
        if blurs_backdrop {
            self.ensure_backdrop_textures();
            damage = SceneDamage::Full;
        }
        let mut encoder = self
            .resources()
            .device
            .create_command_encoder(&Default::default());
        // Dirty children must upload even when their composite is clipped out of root damage.
        self.render_layers(scene, &mut instance_offset, &mut encoder)?;
        let globals = self.resources().globals_bind_group.clone();
        self.encode_scene(
            scene,
            damage,
            self.viewport_size(),
            &frame_texture,
            &frame_view,
            &globals,
            blurs_backdrop,
            &mut instance_offset,
            &mut encoder,
        )?;
        self.present_retained_frame(&mut encoder, surface_view);
        let submission = self.resources().queue.submit([encoder.finish()]);
        if let Some(frame) = self.resources_mut().retained_frame.as_mut() {
            frame.valid = true;
        }
        Ok(submission)
    }

    fn encode_scene(
        &mut self,
        source: &Scene,
        damage: SceneDamage,
        viewport: Size<DevicePixels>,
        frame_texture: &wgpu::Texture,
        frame_view: &wgpu::TextureView,
        globals: &wgpu::BindGroup,
        blurs_backdrop: bool,
        instance_offset: &mut u64,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        if damage.pixel_bounds(viewport).is_none() {
            return Ok(());
        }
        let mut staged = std::mem::take(&mut self.staging_scene);
        let scene = if matches!(damage, SceneDamage::Full) {
            source
        } else {
            source.copy_primitives_for_damage(damage, &mut staged);
            &staged
        };
        let mut jobs = std::mem::take(&mut self.path_jobs_scratch);
        let depth = self.scene_depth_view(viewport);
        let result = (|| {
            crate::pass_trace::note(|| {
                format!("{}x{} {}", viewport.width.0, viewport.height.0,
                    overdraw_summary(scene, damage.pixel_rects(viewport)))
            });
            let opaque_quads = self.tuning.opaque_quads
                && self.tuning.draw_filter.quad_variant().is_none()
                && self.resources().pipelines.opaque_quads.is_some();
            if opaque_quads {
                let partial = !matches!(damage, SceneDamage::Full);
                self.opaque_plan.clear();
                for rect in damage.pixel_rects(viewport) {
                    let batches = scene.batches_for_damage(damage_region(rect)).filter_map(|batch| match batch {
                        PrimitiveBatch::Quads(range) => Some(range),
                        _ => None,
                    });
                    let clip = partial.then(|| DeviceRect::of_pixels(rect));
                    self.opaque_plan.add_rect(&scene.quads, batches, OPAQUE_INTERIOR_MIN_AREA, clip);
                }
                crate::pass_trace::note(|| self.opaque_plan.summary());
            }
            let instance_bindings = self.write_instances(scene, instance_offset, opaque_quads)?;
            self.prepare_layer_surfaces(&scene.surfaces)?;
            // Backdrop blurs copy the frame between draws, which needs the split encoding.
            if self.tuning.merged_scene_passes && !blurs_backdrop {
                if let Some(atlas) = self.rasterize_scene_paths(
                    scene,
                    damage,
                    viewport,
                    &mut jobs,
                    instance_offset,
                    encoder,
                )? {
                    self.encode_merged_scene(
                        scene,
                        damage,
                        viewport,
                        frame_view,
                        depth.as_ref(),
                        globals,
                        &instance_bindings,
                        &jobs,
                        &atlas,
                        encoder,
                    );
                    return Ok(());
                }
            }
            self.encode_split_scene(
                scene,
                damage,
                viewport,
                frame_texture,
                frame_view,
                depth.as_ref(),
                globals,
                blurs_backdrop,
                &instance_bindings,
                instance_offset,
                encoder,
            )
        })();
        self.path_jobs_scratch = jobs;
        self.staging_scene = staged;
        result
    }

    /// The [`RendererTuning::depth_buffer`] attachment for scene passes of `size`.
    fn scene_depth_view(&mut self, size: Size<DevicePixels>) -> Option<wgpu::TextureView> {
        let enabled = self.tuning.depth_buffer;
        let resources = self.resources_mut();
        if !enabled {
            resources.scene_depth.clear();
            return None;
        }
        let key = (size.width.0.max(1) as u32, size.height.0.max(1) as u32);
        if resources.scene_depth.len() > 8 {
            resources.scene_depth.clear();
        }
        let device = &resources.device;
        let view = resources.scene_depth.entry(key).or_insert_with(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("scene_depth"),
                    size: wgpu::Extent3d { width: key.0, height: key.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: SCENE_DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        Some(view.clone())
    }

    /// Draws a batch that needs no pass of its own; paths and surfaces are handed back.
    /// `rect` is the index of the damage rectangle being drawn.
    fn draw_primitive_batch(
        &self,
        batch: PrimitiveBatch,
        rect: usize,
        quads: &[Quad],
        instance_bindings: &InstanceBindings,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Option<PrimitiveBatch> {
        if !self.tuning.draw_filter.allows(&batch) {
            return None;
        }
        let pipelines = &self.resources().pipelines;
        match batch {
            PrimitiveBatch::Quads(range) => match (
                &pipelines.quad_variant,
                &instance_bindings.opaque,
                &pipelines.opaque_quads,
            ) {
                (Some(variant), _, _) => self.draw_instances(
                    &instance_bindings.quads,
                    variant,
                    instance_range(range),
                    globals,
                    pass,
                ),
                (None, Some(bindings), Some(opaque)) => draw_opaque_quad_batch(
                    quads,
                    range,
                    rect,
                    &self.opaque_plan,
                    pipelines,
                    opaque,
                    &instance_bindings.quads,
                    bindings,
                    globals,
                    pass,
                ),
                _ if self.tuning.split_quads => {
                    for (run, simple) in simple_quad_runs(quads, range) {
                        self.draw_instances(
                            &instance_bindings.quads,
                            if simple { &pipelines.simple_quads } else { &pipelines.quads },
                            instance_range(run),
                            globals,
                            pass,
                        );
                    }
                }
                _ => self.draw_instances(
                    &instance_bindings.quads,
                    &pipelines.quads,
                    instance_range(range),
                    globals,
                    pass,
                ),
            },
            PrimitiveBatch::Shadows(range) => self.draw_instances(
                &instance_bindings.shadows,
                &pipelines.shadows,
                instance_range(range),
                globals,
                pass,
            ),
            PrimitiveBatch::Shapes(range) => self.draw_instances(
                &instance_bindings.shapes,
                &pipelines.shapes,
                instance_range(range),
                globals,
                pass,
            ),
            PrimitiveBatch::Underlines(range) => self.draw_instances(
                &instance_bindings.underlines,
                &pipelines.underlines,
                instance_range(range),
                globals,
                pass,
            ),
            PrimitiveBatch::MonochromeSprites { texture_id, range } => self.draw_sprites(
                &instance_bindings.monochrome_sprites,
                texture_id,
                &pipelines.mono_sprites,
                instance_range(range),
                globals,
                pass,
            ),
            PrimitiveBatch::SubpixelSprites { texture_id, range } => self.draw_sprites(
                &instance_bindings.subpixel_sprites,
                texture_id,
                pipelines
                    .subpixel_sprites
                    .as_ref()
                    .unwrap_or(&pipelines.mono_sprites),
                instance_range(range),
                globals,
                pass,
            ),
            PrimitiveBatch::PolychromeSprites { texture_id, range } => self.draw_sprites(
                &instance_bindings.polychrome_sprites,
                texture_id,
                &pipelines.poly_sprites,
                instance_range(range),
                globals,
                pass,
            ),
            batch @ (PrimitiveBatch::Paths(_) | PrimitiveBatch::Surfaces(_)) => return Some(batch),
        }
        None
    }

    /// The viewport-sized rectangles of [`DrawFilter::Fill`] and its variants.
    fn draw_fill(&self, instance_bindings: &InstanceBindings, globals: &wgpu::BindGroup, pass: &mut wgpu::RenderPass<'_>) {
        let layers = self.tuning.draw_filter.fill_layers();
        if let Some(pipeline) = self.resources().pipelines.quad_variant.as_ref().filter(|_| layers > 0) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, globals, &[]);
            pass.set_bind_group(1, &instance_bindings.quads.bind_group, &[]);
            pass.draw(0..4, 0..layers);
        }
    }

    /// One scene pass per damage rectangle, split again around every path batch and backdrop
    /// blur. Each split stores and reloads the whole target on tile-based GPUs.
    fn encode_split_scene(
        &mut self,
        scene: &Scene,
        damage: SceneDamage,
        viewport: Size<DevicePixels>,
        frame_texture: &wgpu::Texture,
        frame_view: &wgpu::TextureView,
        depth: Option<&wgpu::TextureView>,
        globals: &wgpu::BindGroup,
        blurs_backdrop: bool,
        instance_bindings: &InstanceBindings,
        instance_offset: &mut u64,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let partial = !matches!(damage, SceneDamage::Full);
        let area_passes = partial && self.tuning.area_passes;
        for (rect, bounds) in damage.pixel_rects(viewport).enumerate() {
            let region = damage_region(bounds);
            let area = area_passes.then_some(bounds);
            let mut pass = scene_pass(encoder, frame_view, !partial || area_passes, area, depth);
            set_damage_scissor(&mut pass, bounds);
            if partial && !area_passes {
                pass.set_pipeline(&self.resources().pipelines.frame_clear);
                pass.draw(0..4, 0..1);
            }
            self.draw_fill(instance_bindings, globals, &mut pass);
            for batch in scene.batches_for_damage(region) {
                let Some(batch) = self.draw_primitive_batch(batch, rect, &scene.quads, instance_bindings, globals, &mut pass) else {
                    continue;
                };
                match batch {
                    PrimitiveBatch::Paths(range) => {
                        let path_bounds = scene
                            .path_bounds_for_damage(region, range.clone())
                            .and_then(|bounds| SceneDamage::Partial(bounds).pixel_bounds(viewport));
                        let paths = &scene.paths[range];
                        drop(pass);
                        let rasterized = self.draw_paths_to_intermediate(
                            encoder,
                            paths,
                            instance_offset,
                            path_bounds,
                            partial,
                        )?;
                        pass = scene_pass(encoder, frame_view, false, area, depth);
                        set_damage_scissor(&mut pass, bounds);
                        if let Some((clip, texture_bounds)) = rasterized {
                            self.draw_paths_from_intermediate(
                                paths,
                                clip,
                                texture_bounds,
                                instance_offset,
                                globals,
                                &mut pass,
                            )?;
                        }
                    }
                    PrimitiveBatch::Surfaces(range) => {
                        for surface in &scene.surfaces[range] {
                            let PaintSurfaceSource::BackdropBlur(blur) = &surface.source else {
                                self.draw_surfaces(std::slice::from_ref(surface), globals, &mut pass);
                                continue;
                            };
                            if !blurs_backdrop {
                                continue;
                            }
                            drop(pass);
                            let composite = self.blur_backdrop(encoder, frame_texture, surface, blur)?;
                            pass = scene_pass(encoder, frame_view, false, area, depth);
                            set_damage_scissor(&mut pass, bounds);
                            if let Some(binding) = composite {
                                pass.set_pipeline(&self.resources().pipelines.backdrop_composite);
                                pass.set_bind_group(0, globals, &[]);
                                pass.set_bind_group(1, &binding, &[]);
                                pass.draw(0..3, 0..1);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Rasterizes every path batch that the damage replays into its own region of one path
    /// intermediate, in a single pass. Returns `None` without recording anything when the
    /// regions do not fit one texture.
    fn rasterize_scene_paths(
        &mut self,
        scene: &Scene,
        damage: SceneDamage,
        viewport: Size<DevicePixels>,
        jobs: &mut Vec<PathJob>,
        instance_offset: &mut u64,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Option<PathAtlas>> {
        let partial = !matches!(damage, SceneDamage::Full);
        jobs.clear();
        if !self.tuning.draw_filter.allows_paths() {
            return Ok(Some(PathAtlas { sprites: None }));
        }
        for bounds in damage.pixel_rects(viewport) {
            let region = damage_region(bounds);
            for batch in scene.batches_for_damage(region) {
                let PrimitiveBatch::Paths(range) = batch else {
                    continue;
                };
                let footprint = scene
                    .path_bounds_for_damage(region, range.clone())
                    .and_then(|bounds| SceneDamage::Partial(bounds).pixel_bounds(viewport));
                let extent = footprint.map_or([0, 0], |bounds| {
                    path_scratch_size(
                        [bounds.size.width.0 as u32, bounds.size.height.0 as u32],
                        partial,
                        self.max_texture_size,
                    )
                });
                jobs.push(PathJob {
                    range,
                    bounds: footprint,
                    extent,
                    atlas_origin: [0, 0],
                    vertices: 0..0,
                    sprites: 0..0,
                });
            }
        }
        let row_width = (viewport.width.0.max(1) as u32).min(self.max_texture_size);
        let Some([atlas_width, atlas_height]) = pack_path_atlas(jobs, row_width, self.max_texture_size)
        else {
            return Ok(None);
        };
        if atlas_width == 0 || atlas_height == 0 {
            return Ok(Some(PathAtlas { sprites: None }));
        }

        let mut vertices = std::mem::take(&mut self.path_vertices_scratch);
        vertices.clear();
        for job in jobs.iter_mut() {
            let Some(bounds) = job.bounds else { continue };
            let start = vertices.len() as u32;
            append_path_vertices(
                &scene.paths[job.range.clone()],
                path_clip_bounds(bounds, job.extent),
                gpui::point(
                    ScaledPixels(job.atlas_origin[0] as f32),
                    ScaledPixels(job.atlas_origin[1] as f32),
                ),
                &mut vertices,
            );
            job.vertices = start..vertices.len() as u32;
        }
        if vertices.is_empty() {
            self.path_vertices_scratch = vertices;
            return Ok(Some(PathAtlas { sprites: None }));
        }
        let vertex_binding =
            self.write_instance_binding("path_rasterization_bind_group", instance_offset, &vertices);
        self.path_vertices_scratch = vertices;
        let vertex_binding = vertex_binding?;
        let [texture_width, texture_height] =
            self.ensure_intermediate_textures(atlas_width, atlas_height);

        let mut sprites = std::mem::take(&mut self.path_sprites_scratch);
        sprites.clear();
        for job in jobs.iter_mut() {
            let Some(bounds) = job.bounds.filter(|_| !job.vertices.is_empty()) else {
                continue;
            };
            let clip = path_clip_bounds(bounds, job.extent);
            let texture_bounds = Bounds::new(
                gpui::point(
                    clip.origin.x - ScaledPixels(job.atlas_origin[0] as f32),
                    clip.origin.y - ScaledPixels(job.atlas_origin[1] as f32),
                ),
                gpui::size(
                    ScaledPixels(texture_width as f32),
                    ScaledPixels(texture_height as f32),
                ),
            );
            let start = sprites.len() as u32;
            push_path_sprites(&scene.paths[job.range.clone()], clip, texture_bounds, &mut sprites);
            job.sprites = start..sprites.len() as u32;
        }
        let sprite_binding =
            self.write_instance_binding("path_sprites_bind_group", instance_offset, &sprites);
        self.path_sprites_scratch = sprites;
        let sprite_binding = sprite_binding?;

        let resources = self.resources();
        let intermediate_view = resources
            .path_intermediate_view
            .as_ref()
            .context("Path intermediate missing")?;
        let path_globals = resources
            .path_scratch_globals
            .as_ref()
            .context("Path intermediate globals missing")?;
        let (target_view, resolve_target) = match resources.path_msaa_view.as_ref() {
            Some(msaa_view) => (msaa_view, Some(intermediate_view)),
            None => (intermediate_view, None),
        };
        let timing = self.frame_timings.as_ref().and_then(|timings| {
            timings.begin_pass(crate::frame_timings::PassKind::Path, encoder)
        });
        crate::pass_trace::note(|| format!("path_atlas_pass {atlas_width}x{atlas_height}"));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_atlas_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: if resolve_target.is_some() {
                            wgpu::StoreOp::Discard
                        } else {
                            wgpu::StoreOp::Store
                        },
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, path_globals, &[]);
            pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
            for job in jobs.iter() {
                let Some(bounds) = job.bounds.filter(|_| !job.vertices.is_empty()) else {
                    continue;
                };
                pass.set_scissor_rect(
                    job.atlas_origin[0],
                    job.atlas_origin[1],
                    bounds.size.width.0 as u32,
                    bounds.size.height.0 as u32,
                );
                // Records are loaded by vertex index, so the allocation's base shifts the range.
                pass.draw(
                    vertex_binding.first_instance + job.vertices.start
                        ..vertex_binding.first_instance + job.vertices.end,
                    0..1,
                );
            }
        }
        if let (Some(timings), Some(query)) = (&self.frame_timings, timing) {
            timings.end_pass(query, encoder);
        }
        Ok(Some(PathAtlas {
            sprites: Some(sprite_binding),
        }))
    }

    /// One scene pass for all damage rectangles, switching the scissor between them. Path
    /// batches composite their region of the atlas `rasterize_scene_paths` filled.
    fn encode_merged_scene(
        &self,
        scene: &Scene,
        damage: SceneDamage,
        viewport: Size<DevicePixels>,
        frame_view: &wgpu::TextureView,
        depth: Option<&wgpu::TextureView>,
        globals: &wgpu::BindGroup,
        instance_bindings: &InstanceBindings,
        jobs: &[PathJob],
        atlas: &PathAtlas,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let partial = !matches!(damage, SceneDamage::Full);
        let mut jobs = jobs.iter();
        let mut draw = |rect, bounds, pass: &mut wgpu::RenderPass<'_>| {
            self.draw_merged_rect(scene, rect, bounds, globals, instance_bindings, &mut jobs, atlas, pass)
        };
        if partial && self.tuning.area_passes {
            for (rect, bounds) in damage.pixel_rects(viewport).enumerate() {
                draw(rect, bounds, &mut scene_pass(encoder, frame_view, true, Some(bounds), depth));
            }
        } else {
            let mut pass = scene_pass(encoder, frame_view, !partial, None, depth);
            for (rect, bounds) in damage.pixel_rects(viewport).enumerate() {
                if partial {
                    set_damage_scissor(&mut pass, bounds);
                    pass.set_pipeline(&self.resources().pipelines.frame_clear);
                    pass.draw(0..4, 0..1);
                }
                draw(rect, bounds, &mut pass);
            }
        }
    }

    /// Draws damage rectangle `rect` of a merged scene pass.
    fn draw_merged_rect<'a>(
        &self,
        scene: &Scene,
        rect: usize,
        bounds: Bounds<DevicePixels>,
        globals: &wgpu::BindGroup,
        instance_bindings: &InstanceBindings,
        jobs: &mut impl Iterator<Item = &'a PathJob>,
        atlas: &PathAtlas,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let resources = self.resources();
        set_damage_scissor(pass, bounds);
        self.draw_fill(instance_bindings, globals, pass);
        for batch in scene.batches_for_damage(damage_region(bounds)) {
            let Some(batch) = self.draw_primitive_batch(batch, rect, &scene.quads, instance_bindings, globals, pass) else {
                continue;
            };
            match batch {
                PrimitiveBatch::Paths(_) => {
                    let job = jobs.next();
                    let (Some(job), Some(sprites), Some(texture)) = (
                        job,
                        atlas.sprites.as_ref(),
                        resources.path_scratch_binding.as_ref(),
                    ) else {
                        continue;
                    };
                    if job.sprites.is_empty() {
                        continue;
                    }
                    pass.set_pipeline(&resources.pipelines.paths);
                    pass.set_bind_group(0, globals, &[]);
                    pass.set_bind_group(1, &sprites.bind_group, &[]);
                    pass.set_bind_group(2, texture, &[]);
                    pass.draw(
                        0..4,
                        sprites.first_instance + job.sprites.start
                            ..sprites.first_instance + job.sprites.end,
                    );
                }
                PrimitiveBatch::Surfaces(range) => {
                    self.draw_surfaces(&scene.surfaces[range], globals, pass);
                }
                _ => {}
            }
        }
    }

    fn ensure_backdrop_textures(&mut self) {
        if self.resources().backdrop_textures.is_some() {
            return;
        }
        let format = self.surface_config.format;
        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);
        let blur_width = width.div_ceil(2);
        let blur_height = height.div_ceil(2);
        let resources = self.resources_mut();
        let create = |label: &str, width: u32, height: u32, usage: wgpu::TextureUsages| {
            let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (copy, copy_view) = create(
            "backdrop_copy",
            width,
            height,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let blur_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let (blur_a, blur_a_view) = create("backdrop_blur_a", blur_width, blur_height, blur_usage);
        let (blur_b, blur_b_view) = create("backdrop_blur_b", blur_width, blur_height, blur_usage);
        resources.backdrop_textures = Some(BackdropTextures {
            copy,
            copy_view,
            _blur_a: blur_a,
            blur_a_view,
            _blur_b: blur_b,
            blur_b_view,
            blur_width,
            blur_height,
            bindings: RefCell::new(Vec::new()), next_binding: Cell::new(0),
        });
    }

    /// Copy the part of the frame under `surface` (grown by three standard deviations), then
    /// downsample and blur it into the backdrop textures. Returns the bind group the composite
    /// draws with, or `None` when nothing of it is on screen.
    fn blur_backdrop(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame_texture: &wgpu::Texture,
        surface: &PaintSurface,
        blur: &BackdropBlur,
    ) -> Result<Option<wgpu::BindGroup>> {
        let resources = self.resources();
        let Some(textures) = resources.backdrop_textures.as_ref() else { return Ok(None); };
        let visible = surface.bounds.intersect(&surface.content_mask.bounds);
        if visible.is_empty() {
            return Ok(None);
        }

        let sigma = blur.radius.0.max(0.);
        let downscale = backdrop_downscale(sigma);
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let margin = (sigma * 3.).ceil() + downscale as f32;
        let floor_to = |value: f32| (value.max(0.) as u32) / downscale * downscale;
        let ceil_to = |value: f32, limit: u32| {
            ((value.max(0.).ceil() as u32).div_ceil(downscale) * downscale).min(limit)
        };
        let x0 = floor_to(visible.origin.x.0 - margin).min(width);
        let y0 = floor_to(visible.origin.y.0 - margin).min(height);
        let x1 = ceil_to(visible.right().0 + margin, width);
        let y1 = ceil_to(visible.bottom().0 + margin, height);
        if x1 <= x0 || y1 <= y0 {
            return Ok(None);
        }
        let scissor_x = x0 / downscale;
        let scissor_y = y0 / downscale;
        let scissor_width = x1
            .div_ceil(downscale)
            .min(textures.blur_width)
            .saturating_sub(scissor_x);
        let scissor_height = y1
            .div_ceil(downscale)
            .min(textures.blur_height)
            .saturating_sub(scissor_y);
        if scissor_width == 0 || scissor_height == 0 {
            return Ok(None);
        }

        crate::pass_trace::transfer(crate::pass_trace::Transfer::Other, 1,
            u64::from(x1 - x0) * u64::from(y1 - y0) * 4);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: frame_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &textures.copy,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: x1 - x0,
                height: y1 - y0,
                depth_or_array_layers: 1,
            },
        );

        let fade = surface.content_mask.fade;
        let radii = blur.corner_radii;
        let params = BackdropParams {
            bounds: surface.bounds.into(),
            content_mask: surface.content_mask.bounds.into(),
            content_fade: [fade.top.0, fade.top_len.0, fade.bottom.0, fade.bottom_len.0],
            corner_radii: [
                radii.top_left.0,
                radii.top_right.0,
                radii.bottom_right.0,
                radii.bottom_left.0,
            ],
            region: PodBounds {
                origin: [x0 as f32, y0 as f32],
                size: [(x1 - x0) as f32, (y1 - y0) as f32],
            },
            direction: [0., 0.],
            sigma: sigma / downscale as f32,
            downscale: downscale as f32,
            blur_size: [textures.blur_width as f32, textures.blur_height as f32],
            opacity: blur.opacity,
            _pad: 0.,
        };
        let copy_view = textures.copy_view.clone();
        let blur_a_view = textures.blur_a_view.clone();
        let blur_b_view = textures.blur_b_view.clone();
        let downsample = resources.pipelines.backdrop_downsample.clone();
        let blur_pipeline = resources.pipelines.backdrop_blur.clone();
        let globals = resources.globals_bind_group.clone();
        let horizontal = BackdropParams { direction: [1., 0.], ..params };
        let vertical = BackdropParams { direction: [0., 1.], ..params };
        let downsample_binding = self.backdrop_binding("backdrop_downsample", &params, &copy_view)?;
        let horizontal_binding = self.backdrop_binding("backdrop_blur_horizontal", &horizontal, &blur_a_view)?;
        let vertical_binding = self.backdrop_binding("backdrop_blur_vertical", &vertical, &blur_b_view)?;
        let composite = self.backdrop_binding("backdrop_composite", &params, &blur_a_view)?;
        let mut run = |label: &str,
                       pipeline: &wgpu::RenderPipeline,
                       bind_group: &wgpu::BindGroup,
                       target: &wgpu::TextureView| {
            let timing = self.frame_timings.as_ref().and_then(|timings| timings.begin_pass(crate::frame_timings::PassKind::Blur, encoder));
            crate::pass_trace::note(|| label.to_string());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &globals, &[]);
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            if let (Some(timings), Some(query)) = (&self.frame_timings, timing) { timings.end_pass(query, encoder); }
        };

        run(
            "backdrop_downsample",
            &downsample,
            &downsample_binding,
            &blur_a_view,
        );
        run(
            "backdrop_blur_horizontal",
            &blur_pipeline,
            &horizontal_binding,
            &blur_b_view,
        );
        run(
            "backdrop_blur_vertical",
            &blur_pipeline,
            &vertical_binding,
            &blur_a_view,
        );
        Ok(Some(composite))
    }

    fn backdrop_binding(&mut self, label: &str, params: &BackdropParams, source: &wgpu::TextureView) -> Result<wgpu::BindGroup> {
        let (index, buffer, binding, changed) = {
            let resources = self.resources();
            let textures = resources.backdrop_textures.as_ref().context("Backdrop textures missing")?;
            let index = textures.next_binding.get();
            textures.next_binding.set(index + 1);
            let mut bindings = textures.bindings.borrow_mut();
            if bindings.get(index).is_none_or(|cached| &cached.source != source) {
                let buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label), size: std::mem::size_of::<BackdropParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                });
                let binding = resources.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label), layout: &resources.bind_group_layouts.backdrop,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler) },
                    ],
                });
                let cached = BackdropBinding { buffer, binding, source: source.clone(), params: None };
                if index < bindings.len() { bindings[index] = cached; } else { bindings.push(cached); }
            }
            let cached = &bindings[index];
            let changed = cached.params.as_ref().is_none_or(|previous| bytemuck::bytes_of(previous) != bytemuck::bytes_of(params));
            (index, cached.buffer.clone(), cached.binding.clone(), changed)
        };
        if changed {
            self.upload_buffer(&buffer, 0, bytemuck::bytes_of(params))?;
            // Failed frame recording invalidates these values: staging bytes do not
            // become GPU-visible until that frame's upload encoder is submitted.
            self.resources().backdrop_textures.as_ref().context("Backdrop textures missing")?
                .bindings.borrow_mut()[index].params = Some(*params);
        }
        Ok(binding)
    }

    /// Whether this renderer draws backdrop blur surfaces: its surface can be copied from.
    pub fn supports_backdrop_blur(&self) -> bool {
        self.backdrop_blur_supported
    }

    fn write_instances(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
        opaque_quads: bool,
    ) -> Result<InstanceBindings> {
        let opaque = if opaque_quads {
            let plan = std::mem::take(&mut self.opaque_plan);
            let bindings = self
                .write_instance_binding("opaque_interiors_bind_group", instance_offset, &plan.interiors)
                .and_then(|interiors| {
                    let blended = self.write_instance_binding(
                        "opaque_blended_bind_group",
                        instance_offset,
                        &plan.blended,
                    )?;
                    Ok(OpaqueQuadBindings { interiors, blended })
                });
            self.opaque_plan = plan;
            Some(bindings?)
        } else {
            None
        };
        Ok(InstanceBindings {
            opaque,
            quads: self.write_instance_binding(
                "quads_bind_group",
                instance_offset,
                &scene.quads,
            )?,
            shadows: self.write_instance_binding(
                "shadows_bind_group",
                instance_offset,
                &scene.shadows,
            )?,
            shapes: self.write_instance_binding(
                "shapes_bind_group",
                instance_offset,
                &scene.shapes,
            )?,
            underlines: self.write_instance_binding(
                "underlines_bind_group",
                instance_offset,
                &scene.underlines,
            )?,
            monochrome_sprites: self.write_instance_binding(
                "monochrome_sprites_bind_group",
                instance_offset,
                &scene.monochrome_sprites,
            )?,
            subpixel_sprites: self.write_instance_binding(
                "subpixel_sprites_bind_group",
                instance_offset,
                &scene.subpixel_sprites,
            )?,
            polychrome_sprites: self.write_instance_binding(
                "polychrome_sprites_bind_group",
                instance_offset,
                &scene.polychrome_sprites,
            )?,
        })
    }

    fn create_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            })
    }

    fn prepare_layer_surfaces(&mut self, surfaces: &[gpui::PaintSurface]) -> Result<()> {
        for surface in surfaces {
            let PaintSurfaceSource::Layer(id) = &surface.source else { continue; };
            let (index, params, buffer, changed) = {
                let resources = self.resources();
                let layer = resources.layer_textures.get(&id.0).context("Layer composite texture missing")?;
                let params = LayerSurfaceParams::new(surface, layer.width, layer.height);
                let index = layer.next_composite.get();
                let mut cached = layer.composite.borrow_mut();
                if index == cached.len() {
                    cached.try_reserve(1).context("Layer composite binding allocation failed")?;
                    let buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("layer_surface_params"),
                        size: std::mem::size_of::<LayerSurfaceParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let binding = resources.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("layer_surface_bg"),
                        layout: &resources.bind_group_layouts.layer_surfaces,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&layer.view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler) },
                        ],
                    });
                    cached.push(LayerCompositeBinding { params: None, buffer, binding });
                }
                let entry = &cached[index];
                let changed = entry.params.as_ref().is_none_or(|previous| bytemuck::bytes_of(previous) != bytemuck::bytes_of(&params));
                (index, params, entry.buffer.clone(), changed)
            };
            if changed { self.upload_buffer(&buffer, 0, bytemuck::bytes_of(&params))?; }
            let layer = self.resources().layer_textures.get(&id.0).context("Layer composite texture missing")?;
            // Failed external recordings invalidate these values before reuse.
            layer.composite.borrow_mut()[index].params = Some(params);
            layer.next_composite.set(index + 1);
        }
        Ok(())
    }

    /// Composite layer surfaces. Each `PaintSurface` whose source is `Layer(id)` is drawn
    /// using the layer composite pipeline, sampling the offscreen texture produced by
    /// `render_layers`. Sources of other kinds (macOS video, etc.) are ignored on wgpu.
    fn draw_surfaces(
        &self,
        surfaces: &[gpui::PaintSurface],
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        for surface in surfaces {
            let PaintSurfaceSource::Layer(layer_id) = &surface.source else {
                continue;
            };

            let resources = self.resources();
            let Some(layer_tex) = resources.layer_textures.get(&layer_id.0) else {
                continue; // texture not yet rendered; skip silently
            };

            let params = LayerSurfaceParams::new(surface, layer_tex.width, layer_tex.height);
            let cached = layer_tex.composite.borrow();
            let bind_group = &cached[..layer_tex.next_composite.get()].iter()
                .find(|entry| entry.params.as_ref().is_some_and(|previous| bytemuck::bytes_of(previous) == bytemuck::bytes_of(&params)))
                .expect("layer surface binding was prepared").binding;
            pass.set_pipeline(&resources.pipelines.layer_composite);
            pass.set_bind_group(0, globals, &[]);
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
    }

    /// Render each `SceneLayer` that needs rendering into its offscreen texture.
    /// This must be called before the main render pass so the textures are ready for compositing.
    fn render_layers(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        for layer in &scene.layers {
            let Some(sub_scene) = layer.scene.as_deref() else {
                continue;
            };
            self.render_layers(sub_scene, instance_offset, encoder)?;
            let width = layer.size.width.0.max(1) as u32;
            let height = layer.size.height.0.max(1) as u32;
            self.ensure_layer_texture(layer.id, width, height);
            let texture = &self.resources().layer_textures[&layer.id.0];
            if !layer.needs_render && texture.valid {
                continue;
            }
            let damage = if texture.valid {
                sub_scene.damage
            } else {
                SceneDamage::Full
            };
            let target = texture._texture.clone();
            let view = texture.view.clone();
            let globals = texture.globals.clone();
            let timing = self.frame_timings.as_ref().and_then(|timings| timings.begin_pass(crate::frame_timings::PassKind::Layer, encoder));
            self.encode_scene(
                sub_scene,
                damage,
                layer.size,
                &target,
                &view,
                &globals,
                false,
                instance_offset,
                encoder,
            )?;
            if let (Some(timings), Some(query)) = (&self.frame_timings, timing) { timings.end_pass(query, encoder); }
            self.resources_mut()
                .layer_textures
                .get_mut(&layer.id.0)
                .unwrap()
                .valid = true;
        }
        Ok(())
    }

    /// Ensure a layer texture of the given size exists; (re)creates it if missing or wrong size.
    fn ensure_layer_texture(&mut self, id: LayerId, width: u32, height: u32) {
        let needs_create = self
            .resources()
            .layer_textures
            .get(&id.0)
            .map(|t| t.width != width || t.height != height)
            .unwrap_or(true);

        if !needs_create {
            // Reset unseen counter since we're using it this frame.
            if let Some(t) = self.resources_mut().layer_textures.get_mut(&id.0) {
                t.unseen = 0;
            }
            return;
        }

        let format = self.surface_config.format;
        let gamma_offset = self.gamma_offset;
        let premultiplied_alpha =
            self.surface_config.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied;
        let resources = self.resources_mut();
        let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        resources.layer_textures.insert(
            id.0,
            LayerTexture {
                _texture: texture,
                view,
                width,
                height,
                unseen: 0,
                valid: false,
                globals: Self::viewport_globals(
                    resources,
                    gamma_offset,
                    width,
                    height,
                    premultiplied_alpha,
                ),
                composite: RefCell::new(Vec::new()), next_composite: Cell::new(0),
            },
        );
    }

    /// Drop layer textures that haven't been referenced for `LAYER_EVICT_FRAMES` frames.
    fn evict_stale_layers(&mut self, scene: &Scene) {
        // Build the set of layer ids referenced this frame.
        fn collect(scene: &Scene, referenced: &mut std::collections::HashSet<u64>) {
            for surface in &scene.surfaces {
                if let PaintSurfaceSource::Layer(id) = surface.source {
                    referenced.insert(id.0);
                }
            }
            for layer in &scene.layers {
                referenced.insert(layer.id.0);
                if let Some(scene) = layer.scene.as_deref() {
                    collect(scene, referenced);
                }
            }
        }
        let mut referenced = std::mem::take(&mut self.layer_ids_scratch);
        referenced.clear();
        collect(scene, &mut referenced);

        let resources = self.resources_mut();
        resources.layer_textures.retain(|id, tex| {
            if referenced.contains(id) {
                tex.unseen = 0;
                true
            } else {
                tex.unseen += 1;
                tex.unseen < LAYER_EVICT_FRAMES
            }
        });
        self.layer_ids_scratch = referenced;
    }

    fn draw_instances(
        &self,
        instances: &InstanceBinding,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        draw_instance_range(instances, pipeline, range, globals, pass);
    }

    fn draw_sprites(
        &self,
        sprite_instances: &InstanceBinding,
        texture_id: AtlasTextureId,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        let texture = self.atlas.texture_bind_group(texture_id, |view| {
            self.create_texture_bind_group("atlas_texture_bind_group", view)
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_bind_group(1, &sprite_instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            sprite_instances.first_instance + range.start
                ..sprite_instances.first_instance + range.end,
        );
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        clip: Bounds<ScaledPixels>,
        texture_bounds: Bounds<ScaledPixels>,
        instance_offset: &mut u64,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let mut sprites = std::mem::take(&mut self.path_sprites_scratch);
        sprites.clear();
        push_path_sprites(paths, clip, texture_bounds, &mut sprites);

        let Some(texture) = self.resources().path_scratch_binding.clone() else {
            self.path_sprites_scratch = sprites;
            return Ok(());
        };
        let count = sprites.len() as u32;
        let instances = self.write_instance_binding("path_sprites_bind_group", instance_offset, &sprites);
        self.path_sprites_scratch = sprites;
        let instances = instances?;
        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.paths);
        pass.set_bind_group(0, globals, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            instances.first_instance..instances.first_instance + count,
        );
        Ok(())
    }

    fn draw_paths_to_intermediate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        damage_bounds: Option<Bounds<DevicePixels>>,
        partial: bool,
    ) -> Result<Option<(Bounds<ScaledPixels>, Bounds<ScaledPixels>)>> {
        let Some(damage_bounds) = damage_bounds else {
            return Ok(None);
        };
        let [width, height] = path_scratch_size(
            [
                damage_bounds.size.width.0 as u32,
                damage_bounds.size.height.0 as u32,
            ],
            partial,
            self.max_texture_size,
        );
        // Small animated bounds changes share one bucket, while clears and resolves stay local.
        let clip = path_clip_bounds(damage_bounds, [width, height]);
        let mut vertices = std::mem::take(&mut self.path_vertices_scratch);
        path_vertices_in_scratch(paths, clip, &mut vertices);

        if vertices.is_empty() {
            self.path_vertices_scratch = vertices;
            return Ok(None);
        }

        let count = vertices.len() as u32;
        let vertex_binding = self.write_instance_binding(
            "path_rasterization_bind_group",
            instance_offset,
            &vertices,
        );
        self.path_vertices_scratch = vertices;
        let vertex_binding = vertex_binding?;

        let [texture_width, texture_height] = self.ensure_intermediate_textures(width, height);
        let texture_bounds = Bounds::new(
            clip.origin,
            gpui::size(ScaledPixels(texture_width as f32), ScaledPixels(texture_height as f32)),
        );
        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Ok(None);
        };

        let globals = resources
            .path_scratch_globals
            .as_ref()
            .expect("scratch globals were ensured");

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        let timing = self.frame_timings.as_ref().and_then(|timings| timings.begin_pass(crate::frame_timings::PassKind::Path, encoder));
        {
            crate::pass_trace::note(|| "path_rasterization_pass".into());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: if resolve_target.is_some() {
                            wgpu::StoreOp::Discard
                        } else {
                            wgpu::StoreOp::Store
                        },
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_scissor_rect(
                0,
                0,
                damage_bounds.size.width.0 as u32,
                damage_bounds.size.height.0 as u32,
            );
            pass.set_bind_group(0, globals, &[]);
            pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
            // The path rasterization shader loads records by vertex index
            // rather than instance index, so the allocation's base shifts the
            // vertex range here.
            pass.draw(
                vertex_binding.first_instance
                    ..vertex_binding.first_instance + count,
                0..1,
            );
        }

        if let (Some(timings), Some(query)) = (&self.frame_timings, timing) { timings.end_pass(query, encoder); }
        Ok(Some((clip, texture_bounds)))
    }

    fn write_instance_binding<T>(
        &mut self,
        label: &str,
        instance_offset: &mut u64,
        instances: &[T],
    ) -> Result<InstanceBinding> {
        let data = unsafe { Self::instance_bytes(instances) };
        // wgpu rejects zero-sized bindings, so empty primitive arrays still
        // reserve the 16-byte minimum.
        let size = (data.len() as u64).max(16);
        let direct = self.direct_instance_uploads()
            && size <= u64::from(self.resources().device.limits().max_storage_buffer_binding_size);
        if let Some(slot) = self.instance_upload_slot.filter(|_| direct) {
            return self.write_direct_instance_binding(label, slot, data, size);
        }
        let stride = (std::mem::size_of::<T>() as u64).max(1);
        let (alignment, allocation_size) = if self.uses_webgl_instance_data {
            // The texture transport has no binding offset: the shader indexes
            // the instance texture absolutely, so each allocation must start on
            // a whole instance (a stride multiple) and a whole texel, and must
            // end on a texel boundary so the zero padding of its final partial
            // texel cannot overlap the next allocation.
            (
                least_common_multiple(self.instance_data_alignment, stride),
                size.next_multiple_of(INSTANCE_TEXTURE_TEXEL_SIZE),
            )
        } else {
            (self.instance_data_alignment.max(1), size)
        };
        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + allocation_size > self.instance_data_capacity {
            self.grow_instance_data(allocation_size)?;
            offset = 0;
        }
        *instance_offset = offset + allocation_size;

        let first_instance = if self.uses_webgl_instance_data {
            u32::try_from(offset / stride).context("instance index exceeds u32 range")?
        } else {
            0
        };

        if !data.is_empty() {
            match &self.resources().instance_data {
                InstanceData::Storage(buffer) => {
                    let buffer = buffer.clone();
                    self.upload_buffer(&buffer, offset, data)?;
                }
                InstanceData::Texture { .. } => {
                    Self::write_instance_texture(self.resources(), offset, data)
                }
            }
        }
        let resources = self.resources();
        let create = || {
            resources
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &resources.bind_group_layouts.instances,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match &resources.instance_data {
                            InstanceData::Storage(buffer) => {
                                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer,
                                    offset,
                                    size: NonZeroU64::new(size),
                                })
                            }
                            InstanceData::Texture { view, .. } => {
                                wgpu::BindingResource::TextureView(view)
                            }
                        },
                    }],
                })
        };
        let bind_group = match &resources.instance_data {
            InstanceData::Texture { binding, .. } => binding.get_or_init(create).clone(),
            InstanceData::Storage(_) => {
                if let Some(binding) = resources.instance_bindings.get(&(0, offset, size)) { binding.clone() }
                else {
                    let binding = create();
                    self.resources_mut().instance_bindings.insert((0, offset, size), binding.clone());
                    binding
                }
            }
        };
        Ok(InstanceBinding {
            bind_group,
            first_instance,
        })
    }

    /// Places instances in the frame's mapped upload slot, which the shaders then read in place.
    fn write_direct_instance_binding(
        &mut self,
        label: &str,
        slot: usize,
        data: &[u8],
        size: u64,
    ) -> Result<InstanceBinding> {
        let offset = self.instance_upload_offset.next_multiple_of(self.instance_data_alignment.max(1));
        let end = offset.checked_add(size).context("Instance staging overflow")?;
        let ring = self.instance_uploads.as_ref().context("Instance upload ring missing")?;
        if end > ring.capacity() {
            self.instance_upload_capacity = end.checked_next_power_of_two().context("Instance ring size overflow")?;
            return Err(anyhow::Error::new(UploadUnavailable));
        }
        if !data.is_empty() {
            ring.write(slot, offset, data)?;
        }
        self.instance_upload_offset = end;
        let key = (slot + 1, offset, size);
        let bind_group = match self.resources().instance_bindings.get(&key) {
            Some(binding) => binding.clone(),
            None => {
                let ring = self.instance_uploads.as_ref().context("Instance upload ring missing")?;
                let resources = self.resources();
                let binding = resources.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &resources.bind_group_layouts.instances,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: ring.buffer(slot),
                            offset,
                            size: NonZeroU64::new(size),
                        }),
                    }],
                });
                self.resources_mut().instance_bindings.insert(key, binding.clone());
                binding
            }
        };
        Ok(InstanceBinding {
            bind_group,
            first_instance: 0,
        })
    }

    fn write_instance_texture(resources: &WgpuResources, offset: u64, data: &[u8]) {
        let InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } = &resources.instance_data
        else {
            return;
        };
        let mut byte_offset = 0usize;
        let mut texel_offset = offset / INSTANCE_TEXTURE_TEXEL_SIZE;
        while byte_offset < data.len() {
            let x = (texel_offset % u64::from(*width)) as u32;
            let y = (texel_offset / u64::from(*width)) as u32;
            if y >= *height {
                // The capacity check in write_instance_binding should make this
                // unreachable. Truncating silently would leave stale bytes in the
                // texture and draw garbage for the remaining instances.
                debug_assert!(
                    false,
                    "instance texture write out of bounds: row {y} >= height {}",
                    *height
                );
                log::error!(
                    "instance texture write out of bounds; dropping {} bytes of instance data",
                    data.len() - byte_offset
                );
                return;
            }
            let remaining_bytes = data.len() - byte_offset;
            let complete_texels = remaining_bytes as u64 / INSTANCE_TEXTURE_TEXEL_SIZE;
            let [write_width, write_height] = instance_upload_extent(*width, x, complete_texels);
            let texels = u64::from(write_width) * u64::from(write_height);
            if texels > 0 {
                let byte_count = (texels * INSTANCE_TEXTURE_TEXEL_SIZE) as usize;
                resources.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[byte_offset..byte_offset + byte_count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(write_width * INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: write_width,
                        height: write_height,
                        depth_or_array_layers: 1,
                    },
                );
                byte_offset += byte_count;
                texel_offset += texels;
                continue;
            }

            let mut final_texel = [0; INSTANCE_TEXTURE_TEXEL_SIZE as usize];
            final_texel[..remaining_bytes].copy_from_slice(&data[byte_offset..]);
            resources.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &final_texel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            break;
        }
    }

    fn grow_instance_data(&mut self, required: u64) -> Result<()> {
        let capacity = (self.instance_data_capacity * 2)
            .max(required.next_power_of_two())
            .min(self.max_instance_data_size);
        anyhow::ensure!(
            capacity >= required,
            "instance data needs {required} bytes, above the maximum of {}",
            self.max_instance_data_size
        );
        anyhow::ensure!(
            capacity > self.instance_data_capacity,
            "frame instance data exceeds the {}-byte maximum",
            self.max_instance_data_size
        );
        log::debug!(
            "instance data grown from {} to {capacity}",
            self.instance_data_capacity
        );
        // Bind groups created earlier in the frame keep the previous buffer or
        // texture alive, so allocations written before the grow remain valid;
        // only subsequent writes land in the new allocation.
        let uses_webgl_instance_data = self.uses_webgl_instance_data;
        let resources = self.resources_mut();
        resources.instance_bindings.clear();
        if uses_webgl_instance_data {
            let max_texture_dimension = resources.device.limits().max_texture_dimension_2d;
            let (instance_data, actual_capacity) =
                Self::create_instance_texture(&resources.device, capacity, max_texture_dimension);
            resources.instance_data = instance_data;
            self.instance_data_capacity = actual_capacity;
        } else {
            resources.instance_data =
                InstanceData::Storage(resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance_buffer"),
                    size: capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instance_data_capacity = capacity;
        }
        Ok(())
    }

    /// Mark the surface as unconfigured so rendering is skipped until a new
    /// surface is provided via [`replace_surface`](Self::replace_surface).
    ///
    /// This does **not** drop the renderer — the device, queue, atlas, and
    /// pipelines stay alive.  Use this when the native window is destroyed
    /// (e.g. Android `TerminateWindow`) but you intend to re-create the
    /// surface later without losing cached atlas textures.
    pub fn unconfigure_surface(&mut self) {
        self.surface_configured = false;
        // Drop intermediate textures since they reference the old surface size.
        if let Some(res) = self.resources.as_mut() {
            res.invalidate_intermediate_textures();
        }
    }

    /// Replace the wgpu surface with a new one (e.g. after Android destroys
    /// and recreates the native window).  Keeps the device, queue, atlas, and
    /// all pipelines intact so cached `AtlasTextureId`s remain valid.
    ///
    /// The `instance` **must** be the same [`wgpu::Instance`] that was used to
    /// create the adapter and device (i.e. from the [`WgpuContext`]).  Using a
    /// different instance will cause a "Device does not exist" panic because
    /// the wgpu device is bound to its originating instance.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = create_surface(instance, window_handle.as_raw())?;
        let present_modes = {
            let context = self
                .context
                .as_ref()
                .context("surface replacement requires a GPU context")?
                .borrow();
            let context = context.as_ref().context("GPU context is unavailable")?;
            surface.get_capabilities(&context.adapter).present_modes
        };

        let width = (config.size.width.0 as u32).max(1);
        let height = (config.size.height.0 as u32).max(1);

        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface_config.alpha_mode = alpha_mode;
        self.surface_config.present_mode = select_present_mode(
            config
                .preferred_present_mode
                .or(Some(self.surface_config.present_mode)),
            &present_modes,
        );

        {
            let res = self
                .resources
                .as_mut()
                .expect("GPU resources not available");
            surface.configure(&res.device, &self.surface_config);
            res.surface = Some(surface);

            // Invalidate intermediate textures — they'll be recreated lazily.
            res.invalidate_intermediate_textures();
        }

        self.surface_configured = true;

        Ok(())
    }

    pub fn destroy(&mut self) {
        // Release surface-bound GPU resources eagerly so the underlying native
        // window can be destroyed before the renderer itself is dropped.
        self.resources.take();
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns true if a redraw is needed because GPU state was cleared.
    /// Calling this method clears the flag.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Changes whenever previously rendered pixels can no longer be trusted,
    /// including those in caller-owned external targets.
    pub fn retained_generation(&self) -> u64 {
        self.retained_generation
    }

    /// Recovers from a lost GPU device by recreating the renderer with a new context.
    ///
    /// Call this after detecting `device_lost()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        // Check if another window already recovered the context
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost());

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");

            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.resources = None;
            *gpu_context.borrow_mut() = None;

            // Wait briefly for the GPU driver to stabilize, then try to
            // recreate the context without software renderers. If this fails
            // the caller should request another frame and retry — the real GPU
            // may need more time to come back (e.g. after suspend/resume).
            std::thread::sleep(std::time::Duration::from_millis(350));

            let instance = WgpuContext::instance(Box::new(window.clone()));
            let surface = create_surface(&instance, window_handle.as_raw())?;
            let new_context =
                WgpuContext::new_rejecting_software(instance, &surface, self.compositor_gpu)?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, window_handle.as_raw())?
        };

        let config = WgpuSurfaceConfig {
            size: gpui::Size {
                width: gpui::DevicePixels(self.surface_config.width as i32),
                height: gpui::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let gpu_context = Rc::clone(gpu_context);
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.resources = None;
        self.atlas.handle_device_lost(context);

        *self = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            surface,
            config,
            self.compositor_gpu,
            self.atlas.clone(),
            self.tuning,
        )?;

        log::info!("GPU recovery complete");
        Ok(())
    }
}

/// Whether `fs_quad_simple` draws `quad` exactly as `fs_quad` does: a solid background, no
/// border and no fade. Mirrors the shaders' tests, so that `-0.0` is no border and a fade
/// needs a positive length.
fn is_simple_quad(quad: &Quad) -> bool {
    let fade = &quad.content_mask.fade;
    let borders = &quad.border_widths;
    quad.background.as_solid().is_some()
        && quad.border_style == BorderStyle::Solid
        && [borders.top, borders.right, borders.bottom, borders.left]
            .iter()
            .all(|width| width.0 == 0.0)
        && !(fade.top_len.0 > 0.0)
        && !(fade.bottom_len.0 > 0.0)
}

/// Splits `range` into maximal runs of quads that [`is_simple_quad`] accepts or rejects alike,
/// in draw order.
fn simple_quad_runs(
    quads: &[Quad],
    range: Range<usize>,
) -> impl Iterator<Item = (Range<usize>, bool)> + '_ {
    let mut start = range.start;
    std::iter::from_fn(move || {
        if start >= range.end {
            return None;
        }
        let simple = is_simple_quad(&quads[start]);
        let end = quads[start..range.end]
            .iter()
            .position(|quad| is_simple_quad(quad) != simple)
            .map_or(range.end, |length| start + length);
        let run = start..end;
        start = end;
        Some((run, simple))
    })
}

fn instance_range(range: Range<usize>) -> Range<u32> {
    range.start as u32..range.end as u32
}

fn draw_instance_range(
    instances: &InstanceBinding,
    pipeline: &wgpu::RenderPipeline,
    range: Range<u32>,
    globals: &wgpu::BindGroup,
    pass: &mut wgpu::RenderPass<'_>,
) {
    if range.is_empty() {
        return;
    }
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, globals, &[]);
    pass.set_bind_group(1, &instances.bind_group, &[]);
    pass.draw(
        0..4,
        instances.first_instance + range.start..instances.first_instance + range.end,
    );
}

/// A quad drawn over `raster` only: its left, top, right and bottom edges in device pixels.
/// `ClampedQuad` in the shaders.
#[repr(C)]
#[derive(Clone, Copy)]
struct ClampedQuad {
    raster: [f32; 4],
    quad: Quad,
}

// The shaders' `ClampedQuad` is 16-byte aligned, so its array stride matches only while `Quad`'s
// size is a multiple of 16.
const _: () = assert!(
    std::mem::size_of::<Quad>() % 16 == 0
        && std::mem::size_of::<ClampedQuad>() == 16 + std::mem::size_of::<Quad>()
);

/// Simple quads whose opaque interior covers fewer device pixels draw no separate interior: it
/// would save less hidden work than its extra primitives cost.
const OPAQUE_INTERIOR_MIN_AREA: f32 = 4096.;

/// [`RendererTuning::depth_buffer`] attachments.
const SCENE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
/// Opaque interiors keep at least this distance from every edge that changes what
/// `fs_quad_simple` draws, so that rounding in the shaders cannot move a pixel across one.
const OPAQUE_INTERIOR_MARGIN: f32 = 1. / 16.;
/// Blended parts of one split segment that a later interior is checked against.
const OPAQUE_SEGMENT_BLENDED_LIMIT: usize = 64;

/// An axis-aligned rectangle in device pixels, by its edges.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DeviceRect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl DeviceRect {
    /// Right and bottom edges as the shaders compute them, `size + origin`.
    fn of(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            left: bounds.origin.x.0,
            top: bounds.origin.y.0,
            right: bounds.size.width.0 + bounds.origin.x.0,
            bottom: bounds.size.height.0 + bounds.origin.y.0,
        }
    }

    fn of_pixels(bounds: Bounds<DevicePixels>) -> Self {
        Self::of(bounds.map(|pixel| ScaledPixels(pixel.0 as f32)))
    }

    fn intersect(self, other: Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }

    fn inflate(self, by: f32) -> Self {
        Self {
            left: self.left - by,
            top: self.top - by,
            right: self.right + by,
            bottom: self.bottom + by,
        }
    }

    fn is_empty(self) -> bool {
        !(self.left < self.right && self.top < self.bottom)
    }

    fn area(self) -> f64 {
        if self.is_empty() {
            0.
        } else {
            f64::from(self.right - self.left) * f64::from(self.bottom - self.top)
        }
    }

    fn overlaps(self, other: Self) -> bool {
        !self.intersect(other).is_empty()
    }

    fn edges(self) -> [f32; 4] {
        [self.left, self.top, self.right, self.bottom]
    }
}

/// The pixel-aligned rectangle of an opaque simple quad where `fs_quad_simple` returns its
/// background unchanged: pixels whose centres lie inside its bounds and content mask, beyond its
/// largest corner radius on each side. `None` when that covers fewer than `min_area` pixels.
fn opaque_interior(quad: &Quad, min_area: f32) -> Option<DeviceRect> {
    if quad.background.as_solid()?.a != 1. {
        return None;
    }
    let radii = &quad.corner_radii;
    let radius = |first: ScaledPixels, second: ScaledPixels| first.0.max(second.0).max(0.);
    let bounds = DeviceRect::of(quad.bounds);
    let inner = DeviceRect {
        left: bounds.left + radius(radii.top_left, radii.bottom_left),
        top: bounds.top + radius(radii.top_left, radii.top_right),
        right: bounds.right - radius(radii.top_right, radii.bottom_right),
        bottom: bounds.bottom - radius(radii.bottom_left, radii.bottom_right),
    }
    .intersect(DeviceRect::of(quad.content_mask.bounds));
    // The first pixel whose centre is at least the margin inside an edge, and one past the last.
    let first = |edge: f32| (edge + OPAQUE_INTERIOR_MARGIN - 0.5).ceil();
    let end = |edge: f32| (edge - OPAQUE_INTERIOR_MARGIN - 0.5).floor() + 1.;
    let interior = DeviceRect {
        left: first(inner.left),
        top: first(inner.top),
        right: end(inner.right),
        bottom: end(inner.bottom),
    };
    let finite = interior.edges().iter().all(|edge| edge.is_finite());
    (finite && !interior.is_empty() && interior.area() >= f64::from(min_area)).then_some(interior)
}

/// The parts of `bounds` around `interior`: above, below, left and right of it. Their edges
/// against the interior are whole pixels, so every pixel centre in `bounds` falls in exactly
/// one of them or in the interior.
fn rim(bounds: DeviceRect, interior: DeviceRect) -> [DeviceRect; 4] {
    [
        DeviceRect {
            bottom: interior.top,
            ..bounds
        },
        DeviceRect {
            top: interior.bottom,
            ..bounds
        },
        DeviceRect {
            left: bounds.left,
            right: interior.left,
            ..interior
        },
        DeviceRect {
            left: interior.right,
            right: bounds.right,
            ..interior
        },
    ]
}

/// How [`RendererTuning::opaque_quads`] draws the simple-quad runs of a scene. Each run becomes
/// segments, drawn in order. A split segment draws the opaque interiors of its quads first, then
/// the rest of every quad in it blended, in quad order. It ends before a quad whose interior
/// overlaps a blended part of an earlier quad in the segment, so drawing interiors early changes
/// no pixel: the blended parts never cover another quad's interior, and within one quad they do
/// not overlap its interior.
///
/// A partial frame is planned once per damage rectangle, with every part clipped to it, so that
/// the GPU receives no geometry outside the damage: quads whose interior does not qualify then
/// draw as blended parts too.
#[derive(Default)]
struct OpaqueQuadPlan {
    interiors: Vec<ClampedQuad>,
    blended: Vec<ClampedQuad>,
    segments: Vec<OpaqueSegment>,
    /// The damage rectangle and first quad of each simple run, and the run's segments.
    runs: Vec<(usize, usize, Range<usize>)>,
    /// Raster rectangles of the blended parts of the split segment being built.
    segment_blended: Vec<DeviceRect>,
    /// Device pixels of the quads drawn whole by `Quads` segments.
    quad_segment_px: f64,
}

#[derive(Clone, Debug, PartialEq)]
enum OpaqueSegment {
    /// Scene quads, drawn with `simple_quads`.
    Quads(Range<usize>),
    Split {
        interiors: Range<u32>,
        blended: Range<u32>,
    },
}

impl OpaqueQuadPlan {
    fn clear(&mut self) {
        self.interiors.clear();
        self.blended.clear();
        self.segments.clear();
        self.runs.clear();
        self.quad_segment_px = 0.;
    }

    /// Plans the quad batches `batches`, ranges of `quads` in draw order, for damage rectangle
    /// `rect`, whose index is the next one. With `clip`, every part is clipped to it.
    fn add_rect(
        &mut self,
        quads: &[Quad],
        batches: impl IntoIterator<Item = Range<usize>>,
        min_interior_area: f32,
        clip: Option<DeviceRect>,
    ) {
        let rect = self.runs.last().map_or(0, |(rect, _, _)| rect + 1);
        for batch in batches {
            for (run, simple) in simple_quad_runs(quads, batch) {
                if simple {
                    self.add_run(quads, rect, run, min_interior_area, clip);
                }
            }
        }
        // Keeps the rectangle's index even when it has no simple run.
        self.runs.push((rect, usize::MAX, self.segments.len()..self.segments.len()));
    }

    fn add_run(
        &mut self,
        quads: &[Quad],
        rect: usize,
        run: Range<usize>,
        min_interior_area: f32,
        clip: Option<DeviceRect>,
    ) {
        let first_segment = self.segments.len();
        let clip_to = |raster: DeviceRect| clip.map_or(raster, |clip| raster.intersect(clip));
        for index in run.clone() {
            let quad = &quads[index];
            let bounds = DeviceRect::of(quad.bounds);
            // Pixels whose centres lie on the mask's edge still reach the shader's clip test.
            let mask = DeviceRect::of(quad.content_mask.bounds).inflate(1.);
            let current = self.segments[first_segment..].last_mut();
            let interior = opaque_interior(quad, min_interior_area)
                .filter(|interior| !clip_to(*interior).is_empty());
            match interior {
                Some(interior) => {
                    let drawn = clip_to(interior);
                    let continues = matches!(current, Some(OpaqueSegment::Split { .. }))
                        && self.segment_blended.len() < OPAQUE_SEGMENT_BLENDED_LIMIT
                        && !self.segment_blended.iter().any(|part| part.overlaps(drawn));
                    if !continues {
                        let interiors = self.interiors.len() as u32;
                        let blended = self.blended.len() as u32;
                        self.segments.push(OpaqueSegment::Split {
                            interiors: interiors..interiors,
                            blended: blended..blended,
                        });
                        self.segment_blended.clear();
                    }
                    self.interiors.push(ClampedQuad {
                        raster: drawn.edges(),
                        quad: *quad,
                    });
                    for part in rim(bounds, interior) {
                        self.push_blended(quad, clip_to(part.intersect(mask)));
                    }
                }
                None => match current {
                    Some(OpaqueSegment::Quads(range)) => {
                        range.end = index + 1;
                        self.quad_segment_px += bounds.area();
                    }
                    Some(OpaqueSegment::Split { .. }) => {
                        self.push_blended(quad, clip_to(bounds.intersect(mask)))
                    }
                    None if clip.is_some() => {
                        let at = (self.interiors.len() as u32, self.blended.len() as u32);
                        self.segments.push(OpaqueSegment::Split {
                            interiors: at.0..at.0,
                            blended: at.1..at.1,
                        });
                        self.segment_blended.clear();
                        self.push_blended(quad, clip_to(bounds.intersect(mask)));
                    }
                    None => {
                        self.segments.push(OpaqueSegment::Quads(index..index + 1));
                        self.quad_segment_px += bounds.area();
                    }
                },
            }
            if let Some(OpaqueSegment::Split { interiors, blended }) = self.segments.last_mut() {
                interiors.end = self.interiors.len() as u32;
                blended.end = self.blended.len() as u32;
            }
        }
        self.runs.push((rect, run.start, first_segment..self.segments.len()));
    }

    fn push_blended(&mut self, quad: &Quad, raster: DeviceRect) {
        if raster.is_empty() {
            return;
        }
        self.blended.push(ClampedQuad {
            raster: raster.edges(),
            quad: *quad,
        });
        self.segment_blended.push(raster);
    }

    /// The segments of the simple run that starts at quad `start` in damage rectangle `rect`.
    fn run_segments(&self, rect: usize, start: usize) -> &[OpaqueSegment] {
        match self
            .runs
            .binary_search_by_key(&(rect, start), |(rect, first, _)| (*rect, *first))
        {
            Ok(index) => &self.segments[self.runs[index].2.clone()],
            Err(_) => &[],
        }
    }

    /// `drawn_px` is the area of all quad geometry the plan submits, after clipping to damage.
    fn summary(&self) -> String {
        let area = |parts: &[ClampedQuad]| -> f64 {
            parts
                .iter()
                .map(|part| {
                    let [left, top, right, bottom] = part.raster;
                    DeviceRect { left, top, right, bottom }.area()
                })
                .sum()
        };
        let interior_px = area(&self.interiors);
        let blended_px = area(&self.blended);
        let split = self
            .segments
            .iter()
            .filter(|segment| matches!(segment, OpaqueSegment::Split { .. }))
            .count();
        format!(
            "opaque_quads interiors={} interior_px={interior_px:.0} blended={} blended_px={blended_px:.0} \
             segments={} split_segments={split} drawn_px={:.0}",
            self.interiors.len(),
            self.blended.len(),
            self.segments.len(),
            interior_px + blended_px + self.quad_segment_px,
        )
    }
}

struct OpaqueQuadBindings {
    interiors: InstanceBinding,
    blended: InstanceBinding,
}

/// Draws the quads `range` of a batch as [`RendererTuning::opaque_quads`] does, from the plan
/// built for its scene.
fn draw_opaque_quad_batch(
    quads: &[Quad],
    range: Range<usize>,
    rect: usize,
    plan: &OpaqueQuadPlan,
    pipelines: &WgpuPipelines,
    opaque: &OpaqueQuadPipelines,
    quad_binding: &InstanceBinding,
    bindings: &OpaqueQuadBindings,
    globals: &wgpu::BindGroup,
    pass: &mut wgpu::RenderPass<'_>,
) {
    for (run, simple) in simple_quad_runs(quads, range) {
        if !simple {
            draw_instance_range(quad_binding, &pipelines.quads, instance_range(run), globals, pass);
            continue;
        }
        for segment in plan.run_segments(rect, run.start) {
            match segment {
                OpaqueSegment::Quads(range) => draw_instance_range(
                    quad_binding,
                    &pipelines.simple_quads,
                    instance_range(range.clone()),
                    globals,
                    pass,
                ),
                OpaqueSegment::Split { interiors, blended } => {
                    draw_instance_range(&bindings.interiors, &opaque.interiors, interiors.clone(), globals, pass);
                    draw_instance_range(&bindings.blended, &opaque.clamped, blended.clone(), globals, pass);
                }
            }
        }
    }
}

/// Primitive counts and covered areas of what a frame draws within its damage rectangles, in
/// device pixels, for `pass_trace`. `quad_raster_px` is what quads rasterize (their bounds),
/// `quad_px` what survives their content masks; `opaque_px` is the part of that from quads with
/// an opaque solid background and no fade, and `interior_px` its part beyond corner radii.
fn overdraw_summary(scene: &Scene, rects: impl Iterator<Item = Bounds<DevicePixels>>) -> String {
    #[derive(Default)]
    struct Kind {
        count: usize,
        area: f64,
    }
    impl Kind {
        fn add(&mut self, bounds: DeviceRect, mask: Bounds<ScaledPixels>, damage: DeviceRect) {
            self.count += 1;
            self.area += bounds.intersect(DeviceRect::of(mask)).intersect(damage).area();
        }
    }
    let mut damage_px = 0.;
    let (mut quads, mut quad_raster_px, mut opaque_px, mut interior_px) = (Kind::default(), 0., 0., 0.);
    let mut others: [(&str, Kind); 8] = [
        ("shadows", Kind::default()),
        ("shapes", Kind::default()),
        ("paths", Kind::default()),
        ("underlines", Kind::default()),
        ("mono", Kind::default()),
        ("subpixel", Kind::default()),
        ("poly", Kind::default()),
        ("surfaces", Kind::default()),
    ];
    for rect in rects {
        let damage = DeviceRect::of_pixels(rect);
        damage_px += damage.area();
        for batch in scene.batches_for_damage(damage_region(rect)) {
            match batch {
                PrimitiveBatch::Quads(range) => {
                    for quad in &scene.quads[range] {
                        let bounds = DeviceRect::of(quad.bounds);
                        let mask = DeviceRect::of(quad.content_mask.bounds);
                        quads.add(bounds, quad.content_mask.bounds, damage);
                        quad_raster_px += bounds.intersect(damage).area();
                        let fade = &quad.content_mask.fade;
                        if fade.top_len.0 > 0. || fade.bottom_len.0 > 0. {
                            continue;
                        }
                        if quad.background.as_solid().is_some_and(|color| color.a == 1.) {
                            opaque_px += bounds.intersect(mask).intersect(damage).area();
                        }
                        if let Some(interior) = opaque_interior(quad, 0.) {
                            interior_px += interior.intersect(damage).area();
                        }
                    }
                }
                PrimitiveBatch::Shadows(range) => {
                    for shadow in &scene.shadows[range] {
                        let margin = if shadow.inset != 0 { 0. } else { 3. * shadow.blur_radius.0 };
                        let bounds = DeviceRect::of(if shadow.inset != 0 { shadow.element_bounds } else { shadow.bounds });
                        others[0].1.add(bounds.inflate(margin), shadow.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::Shapes(range) => {
                    for shape in &scene.shapes[range] {
                        others[1].1.add(DeviceRect::of(shape.bounds), shape.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::Paths(range) => {
                    for path in &scene.paths[range] {
                        others[2].1.add(DeviceRect::of(path.bounds), path.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    for underline in &scene.underlines[range] {
                        others[3].1.add(DeviceRect::of(underline.bounds), underline.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    for sprite in &scene.monochrome_sprites[range] {
                        others[4].1.add(DeviceRect::of(sprite.bounds), sprite.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    for sprite in &scene.subpixel_sprites[range] {
                        others[5].1.add(DeviceRect::of(sprite.bounds), sprite.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    for sprite in &scene.polychrome_sprites[range] {
                        others[6].1.add(DeviceRect::of(sprite.bounds), sprite.content_mask.bounds, damage);
                    }
                }
                PrimitiveBatch::Surfaces(range) => {
                    for surface in &scene.surfaces[range] {
                        others[7].1.add(DeviceRect::of(surface.bounds), surface.content_mask.bounds, damage);
                    }
                }
            }
        }
    }
    let damage = damage_px.max(1.);
    let total = quads.area + others.iter().map(|(_, kind)| kind.area).sum::<f64>();
    let mut line = format!(
        "overdraw damage_px={damage_px:.0} factor={:.2} quads={} quad_raster_px={quad_raster_px:.0} \
         quad_px={:.0} opaque_px={opaque_px:.0} interior_px={interior_px:.0} quad_factor={:.2} \
         raster_factor={:.2} opaque_factor={:.2} interior_factor={:.2}",
        total / damage,
        quads.count,
        quads.area,
        quads.area / damage,
        quad_raster_px / damage,
        opaque_px / damage,
        interior_px / damage,
    );
    for (name, kind) in &others {
        line.push_str(&format!(" {name}={}/{:.0}", kind.count, kind.area));
    }
    line
}

#[cfg(not(target_family = "wasm"))]
fn create_surface(
    instance: &wgpu::Instance,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                // Fall back to the display handle already provided via InstanceDescriptor::display.
                raw_display_handle: None,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

fn select_present_mode(
    preferred: Option<wgpu::PresentMode>,
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    preferred
        .filter(|mode| supported.contains(mode))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

fn instance_upload_extent(width: u32, x: u32, texels: u64) -> [u32; 2] {
    if x == 0 && texels >= u64::from(width) {
        [width, (texels / u64::from(width)) as u32]
    } else {
        [texels.min(u64::from(width - x)) as u32, 1]
    }
}

/// With `area`, the pass renders only that rectangle, when the backend supports it: Bitnest's
/// Vulkan backend reads it from the label; see [`RendererTuning::area_passes`].
fn scene_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    clear: bool,
    area: Option<Bounds<DevicePixels>>,
    depth: Option<&wgpu::TextureView>,
) -> wgpu::RenderPass<'a> {
    let label = area.map(|area| {
        format!(
            "scene_pass@area={},{},{},{}",
            area.origin.x.0, area.origin.y.0, area.size.width.0, area.size.height.0
        )
    });
    let label = label.as_deref().unwrap_or("scene_pass");
    crate::pass_trace::note(|| format!("{label} {}", if clear { "clear" } else { "load" }));
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    })
}

fn path_scratch_size(size: [u32; 2], partial: bool, max_texture_size: u32) -> [u32; 2] {
    size.map(|extent| {
        if partial {
            extent.div_ceil(64).saturating_mul(64).min(max_texture_size)
        } else {
            extent
        }
    })
}

/// Size for a path intermediate that must hold `requested`. Tilers clear and resolve the whole
/// texture, so a larger existing one is kept, or grown to the union of both sizes, only while it
/// stays within twice the requested area; otherwise the request's exact size is allocated.
fn path_intermediate_size(current: Option<[u32; 2]>, requested: [u32; 2]) -> [u32; 2] {
    let area = |[width, height]: [u32; 2]| u64::from(width) * u64::from(height);
    let limit = 2 * area(requested);
    match current {
        Some(current) if current[0] >= requested[0] && current[1] >= requested[1] && area(current) <= limit => current,
        Some(current) => {
            let union = [current[0].max(requested[0]), current[1].max(requested[1])];
            if area(union) <= limit { union } else { requested }
        }
        None => requested,
    }
}

/// The frame's path atlas, ready for the merged scene pass to composite.
struct PathAtlas {
    sprites: Option<InstanceBinding>,
}

fn damage_region(bounds: Bounds<DevicePixels>) -> SceneDamage {
    SceneDamage::Partial(bounds.map(|pixel| ScaledPixels(pixel.0 as f32)))
}

/// The device-pixel region a path batch rasterizes, with its scratch extent.
fn path_clip_bounds(bounds: Bounds<DevicePixels>, extent: [u32; 2]) -> Bounds<ScaledPixels> {
    Bounds::new(
        gpui::point(
            ScaledPixels(bounds.origin.x.0 as f32),
            ScaledPixels(bounds.origin.y.0 as f32),
        ),
        gpui::size(ScaledPixels(extent[0] as f32), ScaledPixels(extent[1] as f32)),
    )
}

/// Shelf-packs the jobs' extents left to right in rows at most `row_width` wide (a wider job
/// gets a row of its own) and returns the atlas size, or `None` when it exceeds `max_size`.
fn pack_path_atlas(jobs: &mut [PathJob], row_width: u32, max_size: u32) -> Option<[u32; 2]> {
    let (mut x, mut y, mut row_height, mut width) = (0u32, 0u32, 0u32, 0u32);
    for job in jobs.iter_mut() {
        let [job_width, job_height] = job.extent;
        if job_width == 0 || job_height == 0 {
            continue;
        }
        if x > 0 && x.saturating_add(job_width) > row_width {
            y = y.checked_add(row_height)?;
            x = 0;
            row_height = 0;
        }
        job.atlas_origin = [x, y];
        x = x.checked_add(job_width)?;
        row_height = row_height.max(job_height);
        width = width.max(x);
    }
    let height = y.checked_add(row_height)?;
    (width <= max_size && height <= max_size).then_some([width, height])
}

/// One sprite per path when the batch shares a draw order, else one covering them all.
fn push_path_sprites(
    paths: &[Path<ScaledPixels>],
    clip: Bounds<ScaledPixels>,
    texture_bounds: Bounds<ScaledPixels>,
    sprites: &mut Vec<PathSprite>,
) {
    let Some(first_path) = paths.first() else {
        return;
    };
    if paths.last().map(|path| &path.order) == Some(&first_path.order) {
        sprites.extend(paths.iter().map(|path| PathSprite {
            bounds: path.clipped_bounds().intersect(&clip),
            texture_bounds,
        }));
    } else {
        let mut bounds = first_path.clipped_bounds();
        for path in paths.iter().skip(1) {
            bounds = bounds.union(&path.clipped_bounds());
        }
        sprites.push(PathSprite {
            bounds: bounds.intersect(&clip),
            texture_bounds,
        });
    }
}

fn path_vertices_in_scratch(
    paths: &[Path<ScaledPixels>],
    texture_bounds: Bounds<ScaledPixels>,
    vertices: &mut Vec<PathRasterizationVertex>,
) {
    vertices.clear();
    append_path_vertices(paths, texture_bounds, Point::default(), vertices);
}

/// Appends the vertices of paths touching `texture_bounds`, moved so that its origin lands on
/// `target_origin` in the intermediate.
fn append_path_vertices(
    paths: &[Path<ScaledPixels>],
    texture_bounds: Bounds<ScaledPixels>,
    target_origin: Point<ScaledPixels>,
    vertices: &mut Vec<PathRasterizationVertex>,
) {
    let origin = texture_bounds.origin - target_origin;
    for path in paths {
        let mut bounds = path.clipped_bounds();
        if bounds.intersect(&texture_bounds).is_empty() {
            continue;
        }
        bounds.origin -= origin;
        let mut fade = path.content_mask.fade;
        fade.top -= origin.y;
        fade.bottom -= origin.y;
        vertices.extend(path.vertices.iter().map(|vertex| PathRasterizationVertex {
            xy_position: vertex.xy_position - origin,
            st_position: vertex.st_position,
            color: path.color,
            bounds,
            fade,
        }));
    }
}

fn set_damage_scissor(pass: &mut wgpu::RenderPass<'_>, bounds: Bounds<DevicePixels>) {
    pass.set_scissor_rect(
        bounds.origin.x.0 as u32,
        bounds.origin.y.0 as u32,
        bounds.size.width.0 as u32,
        bounds.size.height.0 as u32,
    );
}

struct RenderingParameters {
    path_sample_count: u32,
    format_flags: wgpu::TextureFormatFeatureFlags,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        requested_samples: u32,
    ) -> Self {
        use std::env;

        let format_flags = adapter.get_texture_format_features(surface_format).flags;
        let path_sample_count = path_sample_count(format_flags, requested_samples);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            format_flags,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

impl RenderingParameters {
    fn sample_count_for(&self, requested: u32) -> u32 {
        path_sample_count(self.format_flags, requested)
    }
}

fn path_sample_count(flags: wgpu::TextureFormatFeatureFlags, requested: u32) -> u32 {
    [4, 2, 1]
        .into_iter()
        .filter(|&count| count <= requested)
        .find(|&count| flags.sample_count_supported(count))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_mode_falls_back_when_a_replacement_surface_lacks_mailbox() {
        use wgpu::PresentMode::{Fifo, Mailbox};
        assert_eq!(
            select_present_mode(Some(Mailbox), &[Fifo, Mailbox]),
            Mailbox
        );
        assert_eq!(select_present_mode(Some(Mailbox), &[Fifo]), Fifo);
        assert_eq!(select_present_mode(Some(Fifo), &[Fifo]), Fifo);
        assert_eq!(select_present_mode(None, &[Fifo, Mailbox]), Fifo);
    }

    #[test]
    fn instance_texture_uploads_coalesce_complete_rows() {
        assert_eq!(instance_upload_extent(8, 3, 31), [5, 1]);
        assert_eq!(instance_upload_extent(8, 0, 26), [8, 3]);
        assert_eq!(instance_upload_extent(8, 0, 2), [2, 1]);
        assert_eq!(instance_upload_extent(8, 7, 0), [0, 1]);
        // 31 texels beginning at column 3 require an edge, three full rows, and an edge.
        let mut remaining = 31;
        let mut offset = 3;
        let mut calls = 0;
        while remaining > 0 {
            let [width, height] = instance_upload_extent(8, offset % 8, remaining);
            let count = u64::from(width) * u64::from(height);
            remaining -= count;
            offset += count as u32;
            calls += 1;
        }
        assert_eq!(calls, 3);
        assert_eq!(offset, 34);
    }
    use gpui::{
        MonochromeSprite, PolychromeSprite, Quad, Shadow, Shape, SubpixelSprite, Underline,
    };

    #[test]
    fn animated_path_sizes_share_buckets_within_device_limits() {
        for size in [[1, 1], [17, 31], [63, 42], [64, 64]] {
            assert_eq!(path_scratch_size(size, true, 4096), [64, 64]);
        }
        assert_eq!(path_scratch_size([65, 129], true, 4096), [128, 192]);
        assert_eq!(path_scratch_size([99, 63], true, 100), [100, 64]);
        assert_eq!(path_scratch_size([17, 31], false, 4096), [17, 31]);
    }

    #[test]
    fn webgl_shader_is_valid_wgsl_without_storage_buffers() {
        assert!(!WEBGL_SHADERS.contains("var<storage"));
        validate_wgsl(WEBGL_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn storage_buffer_shader_is_valid_wgsl() {
        validate_wgsl(STORAGE_BUFFER_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn subpixel_shader_is_valid_wgsl() {
        validate_wgsl(
            SUBPIXEL_SHADERS,
            naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn disjoint_partial_clears_present_together_and_preserve_the_gap() {
        let instance = wgpu::Instance::default();
        let adapter = gpui::block_on(instance.request_adapter(&Default::default()))
            .expect("a GPU adapter is available");
        let (device, queue) = gpui::block_on(adapter.request_device(&Default::default()))
            .expect("a GPU device is available");
        for webgl in [false, true] {
            let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
            let pipelines = WgpuRenderer::create_pipelines(
                &device,
                None,
                &layouts,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::CompositeAlphaMode::PreMultiplied,
                1,
                false,
                webgl,
                true,
                None,
                false,
                false,
                false,
            );
            let descriptor = wgpu::TextureDescriptor {
                label: Some("partial_update_test"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };
            let retained = device.create_texture(&descriptor);
            let retained_view = retained.create_view(&Default::default());
            let output = device.create_texture(&descriptor);
            let output_view = output.create_view(&Default::default());
            let sampler = device.create_sampler(&Default::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&retained_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &retained_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                drop(pass);
            }
            let damage = SceneDamage::Partial(Bounds::new(
                gpui::point(ScaledPixels(0.), ScaledPixels(0.)),
                gpui::size(ScaledPixels(1.), ScaledPixels(2.)),
            ))
            .union(SceneDamage::Partial(Bounds::new(
                gpui::point(ScaledPixels(3.), ScaledPixels(2.)),
                gpui::size(ScaledPixels(1.), ScaledPixels(2.)),
            )));
            assert_eq!(
                damage
                    .pixel_rects(gpui::size(DevicePixels(4), DevicePixels(4)))
                    .len(),
                2
            );
            for bounds in damage.pixel_rects(gpui::size(DevicePixels(4), DevicePixels(4))) {
                let mut pass = scene_pass(&mut encoder, &retained_view, false, None, None);
                set_damage_scissor(&mut pass, bounds);
                pass.set_pipeline(&pipelines.frame_clear);
                pass.draw(0..4, 0..1);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&pipelines.frame_present);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..4, 0..1);
            }
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4 * 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                output.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                descriptor.size,
            );
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap()
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range();
            for y in 0..4 {
                for x in 0..4 {
                    let expected = if (x == 0 && y < 2) || (x == 3 && y >= 2) {
                        [0, 0, 0, 0]
                    } else {
                        [255, 0, 0, 255]
                    };
                    assert_eq!(
                        &bytes[y * 256 + x * 4..][..4],
                        &expected,
                        "pixel ({x}, {y}), webgl={webgl}"
                    );
                }
            }
        }
    }

    fn test_quad(seed: &mut u32) -> Quad {
        use gpui::{hsla, point, size};
        let mut next = || {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let origin = point(ScaledPixels(next() * 56. - 4.), ScaledPixels(next() * 56. - 4.));
        let extent = size(ScaledPixels(0.3 + next() * 30.), ScaledPixels(0.3 + next() * 30.));
        let radius = |choice: f32, value: f32| {
            ScaledPixels(if choice < 0.3 { 0. } else { value * 20. })
        };
        let rounded = next() < 0.75;
        let corner_radii = if rounded {
            gpui::Corners {
                top_left: radius(next(), next()),
                top_right: radius(next(), next()),
                bottom_right: radius(next(), next()),
                bottom_left: radius(next(), next()),
            }
        } else {
            Default::default()
        };
        let clip = Bounds::new(
            point(ScaledPixels(next() * 20. - 2.), ScaledPixels(next() * 20. - 2.)),
            size(ScaledPixels(30. + next() * 40.), ScaledPixels(30. + next() * 40.)),
        );
        Quad {
            order: Default::default(),
            border_style: Default::default(),
            bounds: Bounds::new(origin, extent),
            content_mask: gpui::ContentMask {
                bounds: clip,
                fade: Default::default(),
            },
            background: hsla(next(), next(), next(), 0.2 + next() * 0.8).into(),
            border_color: hsla(next(), next(), next(), next()),
            corner_radii,
            border_widths: Default::default(),
        }
    }

    #[test]
    fn simple_quad_runs_follow_draw_order() {
        use gpui::{BorderStyle, hsla, linear_color_stop, linear_gradient};
        let mut seed = 7;
        let simple = test_quad(&mut seed);
        let mut bordered = simple.clone();
        bordered.border_widths.left = ScaledPixels(1.);
        let mut dashed = simple.clone();
        dashed.border_style = BorderStyle::Dashed;
        let mut faded = simple.clone();
        faded.content_mask.fade.bottom_len = ScaledPixels(2.);
        let mut gradient = simple.clone();
        gradient.background = linear_gradient(
            0.,
            linear_color_stop(hsla(0., 1., 0.5, 1.), 0.),
            linear_color_stop(hsla(0.5, 1., 0.5, 1.), 1.),
        );
        let mut negative_zero = simple.clone();
        negative_zero.border_widths.top = ScaledPixels(-0.);
        negative_zero.content_mask.fade.top_len = ScaledPixels(-1.);
        assert!(is_simple_quad(&negative_zero));
        for quad in [&bordered, &dashed, &faded, &gradient] {
            assert!(!is_simple_quad(quad));
        }

        let quads = [
            simple.clone(),
            negative_zero,
            bordered,
            gradient,
            simple.clone(),
            faded,
            dashed,
            simple,
        ];
        let runs: Vec<_> = simple_quad_runs(&quads, 1..8).collect();
        assert_eq!(runs, [(1..2, true), (2..4, false), (4..5, true), (5..7, false), (7..8, true)]);
        assert_eq!(simple_quad_runs(&quads, 3..3).count(), 0);
    }

    fn test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = gpui::block_on(instance.request_adapter(&Default::default())).unwrap();
        gpui::block_on(adapter.request_device(&Default::default())).unwrap()
    }

    fn test_globals(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        size: u32,
        alpha_mode: wgpu::CompositeAlphaMode,
    ) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;
        let viewport = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&GlobalParams {
                viewport_size: [size as f32, size as f32],
                premultiplied_alpha: u32::from(alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied),
                pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&GammaParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.as_entire_binding(),
                },
            ],
        })
    }

    /// Instance data at index zero, through either transport.
    fn test_instances(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layouts: &WgpuBindGroupLayouts,
        webgl: bool,
        bytes: &[u8],
    ) -> InstanceBinding {
        use wgpu::util::DeviceExt;
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().next_multiple_of(16).max(16), 0);
        let buffer;
        let view;
        let resource = if webgl {
            let width = padded.len() as u32 / 16;
            assert!(width <= device.limits().max_texture_dimension_2d);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 16),
                    rows_per_image: None,
                },
                texture.size(),
            );
            view = texture.create_view(&Default::default());
            wgpu::BindingResource::TextureView(&view)
        } else {
            buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE,
            });
            buffer.as_entire_binding()
        };
        InstanceBinding {
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource,
                }],
            }),
            first_instance: 0,
        }
    }

    /// Renders into a `size`² RGBA8 target cleared to a translucent colour; returns its pixels.
    fn render_test_target(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) -> Vec<u8> {
        render_test_target_with_depth(device, queue, size, false, draw)
    }

    /// [`render_test_target`], with a [`SCENE_DEPTH_FORMAT`] attachment when `depth` is set.
    fn render_test_target_with_depth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        depth: bool,
        draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) -> Vec<u8> {
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output.create_view(&Default::default());
        let depth_view = depth.then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: SCENE_DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 0.4,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }
                }),
                ..Default::default()
            });
            draw(&mut pass);
        }
        let row = (size * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            extent,
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        let bytes = readback.slice(..).get_mapped_range();
        bytes
            .chunks(row as usize)
            .flat_map(|line| &line[..size as usize * 4])
            .copied()
            .collect()
    }

    /// Asserts that no channel of `actual` differs from `reference` by more than `tolerance`.
    fn assert_same_pixels(reference: &[u8], actual: &[u8], size: u32, tolerance: u8, context: &str) {
        let differing = reference.chunks(4).zip(actual.chunks(4)).position(|(reference, actual)| {
            reference.iter().zip(actual).any(|(reference, actual)| reference.abs_diff(*actual) > tolerance)
        });
        if let Some(pixel) = differing {
            panic!(
                "pixel ({}, {}) differs: {:?} != {:?}, {context}",
                pixel as u32 % size,
                pixel as u32 / size,
                &reference[pixel * 4..][..4],
                &actual[pixel * 4..][..4]
            );
        }
        assert!(reference.chunks(4).any(|pixel| pixel[3] != 102), "quads were drawn");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn simple_quad_shader_matches_quad_shader() {
        let (device, queue) = test_device();
        let mut seed = 1;
        let quads: Vec<Quad> = (0..400).map(|_| test_quad(&mut seed)).collect();
        assert!(quads.iter().all(is_simple_quad));
        let quad_bytes = unsafe { WgpuRenderer::instance_bytes(&quads) };

        for webgl in [false, true] {
            for alpha_mode in [
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ] {
                let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
                let pipelines = WgpuRenderer::create_pipelines(
                    &device,
                    None,
                    &layouts,
                    wgpu::TextureFormat::Rgba8Unorm,
                    alpha_mode,
                    1,
                    false,
                    webgl,
                    true,
                    None,
                    false,
                    false,
                    false,
                );
                let globals = test_globals(&device, &layouts, 64, alpha_mode);
                let instances = test_instances(&device, &queue, &layouts, webgl, quad_bytes);
                let render = |pipeline: &wgpu::RenderPipeline| {
                    render_test_target(&device, &queue, 64, |pass| {
                        draw_instance_range(&instances, pipeline, 0..quads.len() as u32, &globals, pass)
                    })
                };
                assert_same_pixels(
                    &render(&pipelines.quads),
                    &render(&pipelines.simple_quads),
                    64,
                    0,
                    &format!("webgl={webgl}, {alpha_mode:?}"),
                );
            }
        }
    }

    /// [`test_quad`] at twice the size, more often opaque, sometimes on half-pixel edges and
    /// sometimes bordered, which makes it a quad that `fs_quad_simple` does not draw.
    /// External frames copy instance data and uniforms from the upload ring, or let shaders read
    /// the ring directly; both draw the same pixels, and consecutive copies share a region.
    #[test]
    fn external_frames_match_across_instance_transports() {
        const SIZE: u32 = 64;
        let context = gpui::block_on(WgpuContext::new_external(wgpu::Instance::default())).unwrap();
        let mut seed = 11;
        let mut scene = Scene::default();
        for _ in 0..80 {
            scene.insert_primitive(test_quad(&mut seed));
        }
        scene.finish();
        let render = |direct: bool| {
            let tuning = RendererTuning { direct_instance_uploads: direct, ..Default::default() };
            let config = WgpuSurfaceConfig {
                size: gpui::size(DevicePixels(SIZE as i32), DevicePixels(SIZE as i32)),
                transparent: false,
                preferred_present_mode: None,
            };
            let mut renderer = WgpuRenderer::new_external_tuned(
                &context, config, wgpu::TextureFormat::Bgra8Unorm, tuning).unwrap();
            let device = &context.device;
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            crate::pass_trace::begin();
            renderer.draw_external_damage(&scene, &target, &view, SceneDamage::Full).unwrap();
            let trace = crate::pass_trace::finish();
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            renderer.draw_external_damage(&scene, &target, &view, SceneDamage::Full).unwrap();
            let copies = trace.iter().filter(|entry| entry.starts_with("buffer_copy")).count();
            // Without direct reads: the uniforms (two writes across an alignment gap) and the quads.
            assert_eq!(copies, if renderer.direct_instance_uploads() { 1 } else { 2 }, "{trace:?}");
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(256 * SIZE),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None },
                },
                target.size(),
            );
            context.queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let pixels = readback.slice(..).get_mapped_range().to_vec();
            pixels
        };
        let copied = render(false);
        assert!(copied.chunks_exact(4).any(|pixel| pixel != [0, 0, 0, 0]), "the scene draws something");
        assert!(copied == render(true), "copied and direct instance data draw different pixels");
    }

    fn opaque_test_quad(seed: &mut u32) -> Quad {
        use gpui::{point, size};
        let mut quad = test_quad(seed);
        let mut next = || {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let half_pixels = next() < 0.25;
        let scale = |value: ScaledPixels| {
            let value = value.0 * 2.;
            ScaledPixels(if half_pixels { (value * 2.).round() / 2. } else { value })
        };
        let scale_bounds = |bounds: Bounds<ScaledPixels>| {
            Bounds::new(
                point(scale(bounds.origin.x), scale(bounds.origin.y)),
                size(scale(bounds.size.width), scale(bounds.size.height)),
            )
        };
        quad.bounds = scale_bounds(quad.bounds);
        quad.content_mask.bounds = scale_bounds(quad.content_mask.bounds);
        let radii = &mut quad.corner_radii;
        for radius in [
            &mut radii.top_left,
            &mut radii.top_right,
            &mut radii.bottom_right,
            &mut radii.bottom_left,
        ] {
            *radius = scale(*radius);
        }
        if next() < 0.6 {
            let mut color = quad.background.as_solid().unwrap();
            color.a = 1.;
            quad.background = color.into();
        }
        if next() < 0.1 {
            quad.border_widths.left = ScaledPixels(2.);
        }
        quad
    }

    #[test]
    fn opaque_interiors_and_rims_partition_quads() {
        let mut seed = 3;
        let mut split = 0;
        for _ in 0..500 {
            let quad = opaque_test_quad(&mut seed);
            let Some(interior) = opaque_interior(&quad, 1.) else {
                continue;
            };
            split += 1;
            let bounds = DeviceRect::of(quad.bounds);
            let mask = DeviceRect::of(quad.content_mask.bounds);
            let parts = rim(bounds, interior);
            let covers = |rect: DeviceRect, x: f32, y: f32| {
                rect.left <= x && x < rect.right && rect.top <= y && y < rect.bottom
            };
            for y in (bounds.top.floor() as i32 - 1)..(bounds.bottom.ceil() as i32 + 1) {
                for x in (bounds.left.floor() as i32 - 1)..(bounds.right.ceil() as i32 + 1) {
                    let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
                    let in_interior = covers(interior, x, y);
                    let in_rim = parts.iter().filter(|part| covers(**part, x, y)).count();
                    assert_eq!(
                        usize::from(in_interior) + in_rim,
                        usize::from(covers(bounds, x, y)),
                        "pixel centre ({x}, {y}) of {quad:?}"
                    );
                    if in_interior {
                        let inside = |rect: DeviceRect| {
                            x - rect.left >= OPAQUE_INTERIOR_MARGIN
                                && rect.right - x >= OPAQUE_INTERIOR_MARGIN
                                && y - rect.top >= OPAQUE_INTERIOR_MARGIN
                                && rect.bottom - y >= OPAQUE_INTERIOR_MARGIN
                        };
                        assert!(inside(bounds) && inside(mask), "({x}, {y}) of {quad:?}");
                    }
                }
            }
        }
        assert!(split > 50, "{split}");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn opaque_quad_plan_matches_quad_shader() {
        const SIZE: u32 = 128;
        let (device, queue) = test_device();
        for scene in 1..=4 {
            let mut seed = scene;
            let quads: Vec<Quad> = (0..300).map(|_| opaque_test_quad(&mut seed)).collect();
            let batches = [0..120, 120..quads.len()];
            let quad_bytes = unsafe { WgpuRenderer::instance_bytes(&quads) };
            // Full redraws, and partial ones whose plan is clipped to non-overlapping damage.
            let damages = [
                None,
                Some(vec![
                    DeviceRect { left: 5., top: 7., right: 61., bottom: 50. },
                    DeviceRect { left: 70., top: 20., right: 123., bottom: 90. },
                    DeviceRect { left: 10., top: 100., right: 40., bottom: 128. },
                ]),
            ];
            for (min_area, damage) in [1., 300.].into_iter().flat_map(|area| damages.iter().map(move |damage| (area, damage))) {
                let mut plan = OpaqueQuadPlan::default();
                match damage {
                    None => plan.add_rect(&quads, batches.iter().cloned(), min_area, None),
                    Some(rects) => {
                        for rect in rects {
                            plan.add_rect(&quads, batches.iter().cloned(), min_area, Some(*rect));
                        }
                    }
                }
                let split_segments = plan
                    .segments
                    .iter()
                    .filter(|segment| matches!(segment, OpaqueSegment::Split { .. }))
                    .count();
                assert!(plan.interiors.len() > 10 && split_segments > 1, "{}", plan.summary());
                if let Some(rects) = damage {
                    let inside = |raster: [f32; 4], rect: DeviceRect| {
                        let [left, top, right, bottom] = raster;
                        rect.left <= left && rect.top <= top && right <= rect.right && bottom <= rect.bottom
                    };
                    for (rect, _, segments) in &plan.runs {
                        for segment in &plan.segments[segments.clone()] {
                            let OpaqueSegment::Split { interiors, blended } = segment else {
                                panic!("partial plans draw every quad clipped: {segment:?}");
                            };
                            let parts = plan.interiors[interiors.start as usize..interiors.end as usize]
                                .iter()
                                .chain(&plan.blended[blended.start as usize..blended.end as usize]);
                            for part in parts {
                                assert!(inside(part.raster, rects[*rect]), "{:?} outside {:?}", part.raster, rects[*rect]);
                            }
                        }
                    }
                }
                for webgl in [false, true] {
                    for alpha_mode in [
                        wgpu::CompositeAlphaMode::Opaque,
                        wgpu::CompositeAlphaMode::PreMultiplied,
                    ] {
                        let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
                        let pipelines = WgpuRenderer::create_pipelines(
                            &device,
                            None,
                            &layouts,
                            wgpu::TextureFormat::Rgba8Unorm,
                            alpha_mode,
                            1,
                            false,
                            webgl,
                            true,
                            None,
                            true,
                            false,
                            false,
                        );
                        let globals = test_globals(&device, &layouts, SIZE, alpha_mode);
                        let instances = |bytes| test_instances(&device, &queue, &layouts, webgl, bytes);
                        let quad_instances = instances(quad_bytes);
                        let bindings = OpaqueQuadBindings {
                            interiors: instances(unsafe { WgpuRenderer::instance_bytes(&plan.interiors) }),
                            blended: instances(unsafe { WgpuRenderer::instance_bytes(&plan.blended) }),
                        };
                        let reference = render_test_target(&device, &queue, SIZE, |pass| {
                            draw_instance_range(
                                &quad_instances,
                                &pipelines.quads,
                                0..quads.len() as u32,
                                &globals,
                                pass,
                            )
                        });
                        let split = render_test_target(&device, &queue, SIZE, |pass| {
                            for batch in batches.iter().cloned() {
                                for (run, simple) in simple_quad_runs(&quads, batch) {
                                    draw_instance_range(
                                        &quad_instances,
                                        if simple { &pipelines.simple_quads } else { &pipelines.quads },
                                        instance_range(run),
                                        &globals,
                                        pass,
                                    );
                                }
                            }
                        });
                        let full = [DeviceRect { left: 0., top: 0., right: SIZE as f32, bottom: SIZE as f32 }];
                        let rects = damage.as_deref().unwrap_or(&full);
                        let opaque = render_test_target(&device, &queue, SIZE, |pass| {
                            for (rect, bounds) in rects.iter().enumerate() {
                                let [left, top, right, bottom] = bounds.edges().map(|edge| edge as u32);
                                pass.set_scissor_rect(left, top, right - left, bottom - top);
                                for batch in batches.iter().cloned() {
                                    draw_opaque_quad_batch(
                                        &quads,
                                        batch,
                                        rect,
                                        &plan,
                                        &pipelines,
                                        pipelines.opaque_quads.as_ref().unwrap(),
                                        &quad_instances,
                                        &bindings,
                                        &globals,
                                        pass,
                                    );
                                }
                            }
                        });
                        // Within the damage the frame matches a full redraw; outside it nothing
                        // is drawn.
                        let background = render_test_target(&device, &queue, SIZE, |_| {});
                        let damaged = |pixel: usize| {
                            let (x, y) = ((pixel as u32 % SIZE) as f32 + 0.5, (pixel as u32 / SIZE) as f32 + 0.5);
                            rects.iter().any(|rect| rect.left <= x && x < rect.right && rect.top <= y && y < rect.bottom)
                        };
                        let within = |drawn: &[u8]| -> Vec<u8> {
                            drawn
                                .chunks(4)
                                .zip(background.chunks(4))
                                .enumerate()
                                .flat_map(|(pixel, (drawn, background))| {
                                    if damaged(pixel) { drawn } else { background }.to_vec()
                                })
                                .collect()
                        };
                        let context = format!(
                            "scene={scene}, min_area={min_area}, damage={}, webgl={webgl}, {alpha_mode:?}",
                            damage.is_some()
                        );
                        assert_same_pixels(&within(&split), &opaque, SIZE, 0, &context);
                        // `fs_quad_simple` repeats `fs_quad`'s arithmetic, but compilers may
                        // round it differently: at antialiased corners of these larger, often
                        // over-rounded quads, and through blending, the split path is up to one
                        // step off the general shader.
                        assert_same_pixels(&within(&reference), &opaque, SIZE, 1, &context);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn depth_buffer_and_coverage_pipelines_draw_the_same_pixels() {
        const SIZE: u32 = 128;
        let (device, queue) = test_device();
        let mut seed = 7;
        let quads: Vec<Quad> = (0..300).map(|_| opaque_test_quad(&mut seed)).collect();
        let quad_bytes = unsafe { WgpuRenderer::instance_bytes(&quads) };
        let mut plan = OpaqueQuadPlan::default();
        plan.add_rect(&quads, [0..quads.len()], 1., None);
        let layouts = WgpuRenderer::create_bind_group_layouts(&device, false);
        let alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
        let globals = test_globals(&device, &layouts, SIZE, alpha_mode);
        let instances = |bytes| test_instances(&device, &queue, &layouts, false, bytes);
        let quad_instances = instances(quad_bytes);
        let bindings = OpaqueQuadBindings {
            interiors: instances(unsafe { WgpuRenderer::instance_bytes(&plan.interiors) }),
            blended: instances(unsafe { WgpuRenderer::instance_bytes(&plan.blended) }),
        };
        let render = |depth: bool, coverage_epilogue: bool| {
            let pipelines = WgpuRenderer::create_pipelines(
                &device,
                None,
                &layouts,
                wgpu::TextureFormat::Rgba8Unorm,
                alpha_mode,
                1,
                false,
                false,
                true,
                None,
                true,
                depth,
                coverage_epilogue,
            );
            render_test_target_with_depth(&device, &queue, SIZE, depth, |pass| {
                draw_opaque_quad_batch(
                    &quads,
                    0..quads.len(),
                    0,
                    &plan,
                    &pipelines,
                    pipelines.opaque_quads.as_ref().unwrap(),
                    &quad_instances,
                    &bindings,
                    &globals,
                    pass,
                );
                draw_instance_range(&quad_instances, &pipelines.quads, 0..40, &globals, pass);
            })
        };
        let reference = render(false, false);
        assert_same_pixels(&reference, &render(true, false), SIZE, 0, "depth buffer");
        assert_same_pixels(&reference, &render(false, true), SIZE, 0, "coverage epilogue");
    }

    fn validate_wgsl(source: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .expect("shader should validate");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn cropped_msaa_path_matches_full_render_with_gradient_fade_and_quad() {
        use gpui::{
            ContentMask, PathBuilder, hsla, linear_color_stop, linear_gradient, point, px, size,
        };
        use wgpu::util::DeviceExt;

        let instance = wgpu::Instance::default();
        let adapter = gpui::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = gpui::block_on(adapter.request_device(&Default::default())).unwrap();
        let samples = [4, 2]
            .into_iter()
            .find(|count| {
                adapter
                    .get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm)
                    .flags
                    .sample_count_supported(*count)
            })
            .expect("MSAA is supported");
        let viewport = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(32.), ScaledPixels(32.)),
        );
        let crop = Bounds::new(
            point(ScaledPixels(7.), ScaledPixels(6.)),
            size(ScaledPixels(13.), ScaledPixels(17.)),
        );
        let mut builder = PathBuilder::fill();
        builder.move_to(point(px(2.4), px(4.3)));
        builder.curve_to(point(px(28.7), px(15.1)), point(px(17.2), px(-1.8)));
        builder.line_to(point(px(9.6), px(28.4)));
        builder.close();
        let mut path = builder.build().unwrap().scale(1.);
        path.content_mask = ContentMask {
            bounds: viewport,
            fade: ContentFade {
                top: ScaledPixels(4.),
                top_len: ScaledPixels(9.),
                bottom: ScaledPixels(29.),
                bottom_len: ScaledPixels(12.),
            },
        };
        path.color = linear_gradient(
            37.,
            linear_color_stop(hsla(0.03, 0.9, 0.5, 0.8), 0.),
            linear_color_stop(hsla(0.62, 0.8, 0.6, 0.6), 1.),
        );
        let quad = Quad {
            order: Default::default(),
            border_style: Default::default(),
            bounds: Bounds::new(
                point(ScaledPixels(10.3), ScaledPixels(9.1)),
                size(ScaledPixels(7.5), ScaledPixels(8.2)),
            ),
            content_mask: ContentMask {
                bounds: viewport,
                fade: Default::default(),
            },
            background: hsla(0.3, 0.8, 0.5, 0.35).into(),
            border_color: Default::default(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
        };

        for webgl in [false, true] {
            for alpha_mode in [
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ] {
                let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
                let pipelines = WgpuRenderer::create_pipelines(
                    &device,
                    None,
                    &layouts,
                    wgpu::TextureFormat::Rgba8Unorm,
                    alpha_mode,
                    samples,
                    false,
                    webgl,
                    true,
                    None,
                    false,
                    false,
                    false,
                );
                let globals = |width, height| {
                    let viewport = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&GlobalParams {
                            viewport_size: [width, height],
                            premultiplied_alpha: u32::from(
                                alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied,
                            ),
                            pad: 0,
                        }),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&GammaParams::zeroed()),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layouts.globals,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: viewport.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: gamma.as_entire_binding(),
                            },
                        ],
                    })
                };
                let instances = |data: &[u8]| {
                    let buffer;
                    let view;
                    let resource = if webgl {
                        let width = data.len().div_ceil(16) as u32;
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width,
                                height: 1,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba32Uint,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        let mut padded = data.to_vec();
                        padded.resize(width as usize * 16, 0);
                        queue.write_texture(
                            texture.as_image_copy(),
                            &padded,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(width * 16),
                                rows_per_image: None,
                            },
                            texture.size(),
                        );
                        view = texture.create_view(&Default::default());
                        wgpu::BindingResource::TextureView(&view)
                    } else {
                        buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: data,
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                        buffer.as_entire_binding()
                    };
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layouts.instances,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource,
                        }],
                    })
                };
                let render = |bounds: Bounds<ScaledPixels>, layered: bool, clear: wgpu::Color| {
                    let [width, height] = path_scratch_size(
                        [bounds.size.width.0 as u32, bounds.size.height.0 as u32],
                        bounds != viewport,
                        device.limits().max_texture_dimension_2d,
                    );
                    let texture_bounds = Bounds::new(
                        bounds.origin,
                        size(ScaledPixels(width as f32), ScaledPixels(height as f32)),
                    );
                    let (scratch, scratch_view) = WgpuRenderer::create_path_intermediate(
                        &device,
                        wgpu::TextureFormat::Rgba8Unorm,
                        width,
                        height,
                    );
                    let (_msaa, msaa_view) = WgpuRenderer::create_msaa_if_needed(
                        &device,
                        wgpu::TextureFormat::Rgba8Unorm,
                        width,
                        height,
                        samples,
                    )
                    .unwrap();
                    let output_descriptor = wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d {
                            width: 32,
                            height: 32,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    };
                    let output = device.create_texture(&output_descriptor);
                    let child = layered.then(|| device.create_texture(&output_descriptor));
                    let child_view = child
                        .as_ref()
                        .map(|texture| texture.create_view(&Default::default()));
                    let output_view = output.create_view(&Default::default());
                    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        ..Default::default()
                    });
                    let texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layouts.texture,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&scratch_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    });
                    let mut vertices = Vec::new();
                    path_vertices_in_scratch(std::slice::from_ref(&path), texture_bounds, &mut vertices);
                    let vertex_data = instances(unsafe { WgpuRenderer::instance_bytes(&vertices) });
                    let sprite = [PathSprite {
                        bounds: path.clipped_bounds().intersect(&bounds),
                        texture_bounds,
                    }];
                    let sprite_data = instances(unsafe { WgpuRenderer::instance_bytes(&sprite) });
                    let quad_data = instances(unsafe {
                        WgpuRenderer::instance_bytes(std::slice::from_ref(&quad))
                    });
                    let scratch_globals = globals(width as f32, height as f32);
                    let frame_globals = globals(32., 32.);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &msaa_view,
                                resolve_target: Some(&scratch_view),
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Discard,
                                },
                            })],
                            ..Default::default()
                        });
                        pass.set_pipeline(&pipelines.path_rasterization);
                        pass.set_scissor_rect(
                            0,
                            0,
                            bounds.size.width.0 as u32,
                            bounds.size.height.0 as u32,
                        );
                        pass.set_bind_group(0, &scratch_globals, &[]);
                        pass.set_bind_group(1, &vertex_data, &[]);
                        pass.draw(0..vertices.len() as u32, 0..1);
                    }
                    for composite_path in [true, false] {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: child_view.as_ref().unwrap_or(&output_view),
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: if composite_path {
                                        wgpu::LoadOp::Clear(if layered {
                                            wgpu::Color::TRANSPARENT
                                        } else {
                                            clear
                                        })
                                    } else {
                                        wgpu::LoadOp::Load
                                    },
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        set_damage_scissor(
                            &mut pass,
                            Bounds::new(
                                point(
                                    DevicePixels(bounds.origin.x.0 as i32),
                                    DevicePixels(bounds.origin.y.0 as i32),
                                ),
                                size(
                                    DevicePixels(bounds.size.width.0 as i32),
                                    DevicePixels(bounds.size.height.0 as i32),
                                ),
                            ),
                        );
                        pass.set_pipeline(if composite_path {
                            &pipelines.paths
                        } else {
                            &pipelines.quads
                        });
                        pass.set_bind_group(0, &frame_globals, &[]);
                        pass.set_bind_group(
                            1,
                            if composite_path {
                                &sprite_data
                            } else {
                                &quad_data
                            },
                            &[],
                        );
                        if composite_path {
                            pass.set_bind_group(2, &texture, &[]);
                        }
                        pass.draw(0..4, 0..1);
                    }
                    if let Some(view) = child_view.as_ref() {
                        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::bytes_of(&LayerSurfaceParams {
                                bounds: viewport.into(),
                                content_mask: viewport.into(),
                                content_fade: [0.; 4],
                                tex_size: [32.; 2],
                                _pad: [0.; 2],
                            }),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layouts.layer_surfaces,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: params.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&sampler),
                                },
                            ],
                        });
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &output_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(clear),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        pass.set_pipeline(&pipelines.layer_composite);
                        pass.set_bind_group(0, &frame_globals, &[]);
                        pass.set_bind_group(1, &binding, &[]);
                        pass.draw(0..4, 0..1);
                    }
                    let scratch_copy = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: scratch.size(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let scratch_copy_view = scratch_copy.create_view(&Default::default());
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &scratch_copy_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        pass.set_pipeline(&pipelines.frame_present);
                        pass.set_bind_group(0, &texture, &[]);
                        pass.draw(0..4, 0..1);
                    }
                    let readback = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: (32 + u64::from(height)) * 256,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    encoder.copy_texture_to_buffer(
                        output.as_image_copy(),
                        wgpu::TexelCopyBufferInfo {
                            buffer: &readback,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(256),
                                rows_per_image: None,
                            },
                        },
                        output.size(),
                    );
                    encoder.copy_texture_to_buffer(
                        scratch_copy.as_image_copy(),
                        wgpu::TexelCopyBufferInfo {
                            buffer: &readback,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 32 * 256,
                                bytes_per_row: Some(256),
                                rows_per_image: None,
                            },
                        },
                        scratch_copy.size(),
                    );
                    queue.submit([encoder.finish()]);
                    let (sender, receiver) = std::sync::mpsc::channel();
                    readback
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            sender.send(result).unwrap()
                        });
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: None,
                        })
                        .unwrap();
                    receiver.recv().unwrap().unwrap();
                    let bytes = readback.slice(..).get_mapped_range().to_vec();
                    assert_eq!(scratch.size().width, width);
                    for y in 0..height as usize {
                        for x in 0..width as usize {
                            if x >= bounds.size.width.0 as usize
                                || y >= bounds.size.height.0 as usize
                            {
                                assert_eq!(
                                    &bytes[(32 + y) * 256 + x * 4..][..4],
                                    &[0, 0, 0, 0],
                                    "scratch padding ({x}, {y}), webgl={webgl}"
                                );
                            }
                        }
                    }
                    bytes
                };
                let full = render(viewport, false, wgpu::Color::BLACK);
                assert!(
                    full[8 * 256 + 12 * 4..][..3]
                        .iter()
                        .any(|channel| *channel > 0),
                    "the reference contains the faded path above the overlapping quad"
                );
                for crop in [
                    crop,
                    Bounds::new(
                        point(ScaledPixels(5.), ScaledPixels(3.)),
                        size(ScaledPixels(17.), ScaledPixels(21.)),
                    ),
                ] {
                    let partial = render(crop, false, wgpu::Color::BLACK);
                    for y in 0..32 {
                        for x in 0..32 {
                            let offset = y * 256 + x * 4;
                            let inside = (crop.origin.x.0 as usize..crop.right().0 as usize)
                                .contains(&x)
                                && (crop.origin.y.0 as usize..crop.bottom().0 as usize)
                                    .contains(&y);
                            for channel in 0..4 {
                                let expected = if inside {
                                    full[offset + channel]
                                } else if channel == 3 {
                                    255
                                } else {
                                    0
                                };
                                assert!(
                                    partial[offset + channel].abs_diff(expected) <= 1,
                                    "pixel ({x}, {y}) channel {channel}, webgl={webgl}: {} != {expected}",
                                    partial[offset + channel]
                                );
                            }
                        }
                    }
                }
                for clear in [
                    wgpu::Color::BLACK,
                    wgpu::Color {
                        r: 0.05,
                        g: 0.1,
                        b: 0.2,
                        a: 0.25,
                    },
                ] {
                    let inline = render(viewport, false, clear);
                    let layered = render(viewport, true, clear);
                    for (index, (actual, expected)) in layered[..32 * 256]
                        .iter()
                        .zip(&inline[..32 * 256])
                        .enumerate()
                    {
                        assert!(
                            actual.abs_diff(*expected) <= 2,
                            "layer channel {index}: {actual} != {expected}, webgl={webgl}, mode={alpha_mode:?}, clear={clear:?}"
                        );
                    }
                    assert_eq!(layered[3], (clear.a * 255.).round() as u8);
                }
            }
        }
    }

    #[test]
    fn path_atlas_packs_rows_and_rejects_oversized_frames() {
        let job = |extent: [u32; 2]| PathJob {
            range: 0..0,
            bounds: None,
            extent,
            atlas_origin: [0, 0],
            vertices: 0..0,
            sprites: 0..0,
        };
        let mut jobs = vec![job([64, 128]), job([0, 0]), job([64, 64]), job([128, 64]), job([256, 32])];
        assert_eq!(pack_path_atlas(&mut jobs, 200, 4096), Some([256, 224]));
        let origins: Vec<_> = jobs.iter().map(|job| job.atlas_origin).collect();
        assert_eq!(origins, [[0, 0], [0, 0], [64, 0], [0, 128], [0, 192]]);
        assert_eq!(pack_path_atlas(&mut jobs, 200, 255), None);
        assert_eq!(pack_path_atlas(&mut [job([0, 0])], 200, 4096), Some([0, 0]));
    }

    #[test]
    fn path_intermediates_are_reused_within_twice_the_requested_area() {
        assert_eq!(path_intermediate_size(None, [64, 64]), [64, 64]);
        assert_eq!(path_intermediate_size(Some([128, 64]), [64, 64]), [128, 64]);
        assert_eq!(path_intermediate_size(Some([128, 128]), [64, 64]), [64, 64]);
        assert_eq!(path_intermediate_size(Some([128, 64]), [64, 128]), [128, 128]);
        assert_eq!(path_intermediate_size(Some([256, 64]), [64, 128]), [64, 128]);
        assert_eq!(path_intermediate_size(Some([128, 64]), [128, 128]), [128, 128]);
        assert_eq!(path_intermediate_size(Some([192, 128]), [128, 192]), [192, 192]);
    }

    #[test]
    fn path_sample_count_falls_back_to_supported_counts() {
        use wgpu::TextureFormatFeatureFlags as Flags;
        let four = Flags::MULTISAMPLE_X2 | Flags::MULTISAMPLE_X4;
        assert_eq!(path_sample_count(four, 4), 4);
        assert_eq!(path_sample_count(four, 2), 2);
        assert_eq!(path_sample_count(four, 1), 1);
        assert_eq!(path_sample_count(four, 3), 2);
        assert_eq!(path_sample_count(Flags::MULTISAMPLE_X2, 4), 2);
        assert_eq!(path_sample_count(Flags::empty(), 4), 1);
    }

    #[test]
    fn webgl_record_sizes_match_shader_word_strides() {
        assert_eq!(std::mem::size_of::<Quad>(), 44 * 4);
        assert_eq!(std::mem::size_of::<Shadow>(), 32 * 4);
        assert_eq!(std::mem::size_of::<Shape>(), 48 * 4);
        assert_eq!(std::mem::size_of::<PathRasterizationVertex>(), 30 * 4);
        assert_eq!(std::mem::size_of::<PathSprite>(), 8 * 4);
        assert_eq!(std::mem::size_of::<Underline>(), 20 * 4);
        assert_eq!(std::mem::size_of::<MonochromeSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<SubpixelSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<PolychromeSprite>(), 28 * 4);
    }
}
