// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    Pixels, Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[path = "scene_damage.rs"]
mod damage;
pub use damage::{SceneDamage, SceneDamageRegions};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    /// Pixels changed since the last presentation. Fresh render targets must still redraw fully.
    pub damage: SceneDamage,
    rounded_clips: Vec<Vec<Bounds<ScaledPixels>>>,
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub shapes: Vec<Shape>,
    pub paths: Vec<Path<ScaledPixels>>,
    // Upload staging alternates between targets with and without paths. Keep a bounded spare pool.
    staged_path_pool: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// Cached render layers captured this frame. Each is a view subtree painted into its own
    /// sub-scene (in layer-local coordinates); the platform renderer draws each into an offscreen
    /// texture, then the main scene composites it via a `PaintSurface` with
    /// `SurfaceSource::Layer(id)`. See `Window::paint_layer`.
    pub layers: Vec<SceneLayer>,
    damage_scratch: damage::DamageScratch,
}

/// One cached render layer captured during a frame's paint (see [`Scene::layers`]).
pub struct SceneLayer {
    /// Stable id linking this layer to its `SurfaceSource::Layer(id)` composite.
    pub id: LayerId,
    /// Device-pixel size of the offscreen texture (the layered view's bounds at the current scale).
    pub size: Size<DevicePixels>,
    /// When true the texture must be (re)rendered from `scene` this frame (content changed / first
    /// paint / scale changed). Otherwise reuse the existing texture unless its GPU contents were
    /// lost. The last scene snapshot is retained for recovery.
    pub needs_render: bool,
    /// The layer's last painted primitives in layer-local coordinates, also retained during reuse.
    pub scene: Option<Box<Scene>>,
}

#[expect(missing_docs)]
impl Scene {
    pub(crate) fn push_rounded_clip(&mut self, bands: Vec<Bounds<ScaledPixels>>) {
        self.rounded_clips.push(bands);
    }

    pub(crate) fn pop_rounded_clip(&mut self) {
        self.rounded_clips.pop();
    }

    pub fn clear(&mut self) {
        self.damage = SceneDamage::Full;
        self.rounded_clips.clear();
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.shapes.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.layers.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.insert_clipped_primitive(primitive.into(), 0);
    }

    fn insert_clipped_primitive(&mut self, primitive: Primitive, depth: usize) {
        if depth == self.rounded_clips.len() {
            self.insert_unclipped_primitive(primitive);
            return;
        }
        let visible = primitive
            .raster_bounds()
            .intersect(&primitive.content_mask().bounds);
        if visible.is_empty() { return; }
        for index in 0..self.rounded_clips[depth].len() {
            let band = self.rounded_clips[depth][index];
            let bounds = visible.intersect(&band);
            if bounds.is_empty() { continue; }
            let mut clipped = primitive.clone();
            clipped.content_mask_mut().bounds = bounds;
            self.insert_clipped_primitive(clipped, depth + 1);
        }
    }

    fn insert_unclipped_primitive(&mut self, mut primitive: Primitive) {
        let clipped_bounds = primitive
            .raster_bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Shape(shape) => {
                shape.order = order;
                self.shapes.push(*shape);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Shift every primitive in this scene by `offset`. Used to convert a captured layer sub-scene
    /// from absolute window coordinates into layer-local coordinates (offset = `-view_origin`)
    /// before rendering it into its own offscreen texture, whose top-left is local `(0, 0)`.
    pub fn translate(&mut self, offset: Point<ScaledPixels>) {
        #[inline]
        fn shift(bounds: &mut Bounds<ScaledPixels>, offset: Point<ScaledPixels>) {
            bounds.origin.x = bounds.origin.x + offset.x;
            bounds.origin.y = bounds.origin.y + offset.y;
        }
        #[inline]
        fn shift_mask(mask: &mut ContentMask<ScaledPixels>, offset: Point<ScaledPixels>) {
            shift(&mut mask.bounds, offset);
            mask.fade.shift(offset.y);
        }
        for s in &mut self.shadows {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
        for q in &mut self.quads {
            shift(&mut q.bounds, offset);
            shift_mask(&mut q.content_mask, offset);
        }
        for s in &mut self.shapes {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
        for p in &mut self.paths {
            shift(&mut p.bounds, offset);
            shift_mask(&mut p.content_mask, offset);
            for v in &mut p.vertices {
                v.xy_position.x = v.xy_position.x + offset.x;
                v.xy_position.y = v.xy_position.y + offset.y;
                shift_mask(&mut v.content_mask, offset);
            }
        }
        for u in &mut self.underlines {
            shift(&mut u.bounds, offset);
            shift_mask(&mut u.content_mask, offset);
        }
        for s in &mut self.monochrome_sprites {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
        for s in &mut self.subpixel_sprites {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
        for s in &mut self.polychrome_sprites {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
        for s in &mut self.surfaces {
            shift(&mut s.bounds, offset);
            shift_mask(&mut s.content_mask, offset);
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.shapes.sort_by_key(|shape| shape.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            shapes_start: 0,
            shapes_iter: self.shapes.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Shape,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Shape(Shape),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    fn raster_bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Self::Shadow(shadow) => shadow.raster_bounds(),
            _ => *self.bounds(),
        }
    }

    fn content_mask_mut(&mut self) -> &mut ContentMask<ScaledPixels> {
        match self {
            Self::Shadow(value) => &mut value.content_mask,
            Self::Quad(value) => &mut value.content_mask,
            Self::Shape(value) => &mut value.content_mask,
            Self::Path(value) => &mut value.content_mask,
            Self::Underline(value) => &mut value.content_mask,
            Self::MonochromeSprite(value) => &mut value.content_mask,
            Self::SubpixelSprite(value) => &mut value.content_mask,
            Self::PolychromeSprite(value) => &mut value.content_mask,
            Self::Surface(value) => &mut value.content_mask,
        }
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Shape(shape) => &shape.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Shape(shape) => &shape.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    shapes_start: usize,
    shapes_iter: Peekable<slice::Iter<'a, Shape>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.shapes_iter.peek().map(|s| s.order), PrimitiveKind::Shape),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Shape => {
                let shapes_start = self.shapes_start;
                let mut shapes_end = shapes_start + 1;
                self.shapes_iter.next();
                while self
                    .shapes_iter
                    .next_if(|shape| (shape.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shapes_end += 1;
                }
                self.shapes_start = shapes_end;
                Some(PrimitiveBatch::Shapes(shapes_start..shapes_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Shapes(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Shapes(range) => format!("shapes ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl Shadow {
    // Match the renderer's three-sigma Gaussian quad without changing the shape it blurs.
    // Clipping and draw ordering must include the tail; inset shadows stay inside the element.
    pub(crate) fn raster_bounds(&self) -> Bounds<ScaledPixels> {
        if self.inset != 0 {
            self.element_bounds
        } else {
            self.bounds.dilate(ScaledPixels(3. * self.blur_radius.0))
        }
    }
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// An analytic shape, as the renderers receive it: an instanced quad whose fragment shader
/// evaluates the outline's field, estimates a signed distance from the field and its gradient,
/// and shades it. Built by [`Window::paint_shape`](crate::Window::paint_shape) from a
/// [`PaintShape`](crate::PaintShape); the shaders mirror this layout field for field.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub struct Shape {
    /// Draw order within the scene.
    pub order: DrawOrder,
    /// Which [`ShapeOutline`] field to evaluate.
    pub outline: u32,
    /// Which [`ShapeMaterial`] shades it.
    pub material: u32,
    /// [`Shape::MIRROR`] and [`Shape::SHADOW_HALO`].
    pub flags: u32,
    /// The quad drawn. Bounds space spans its largest centred square from -1 to 1, y down.
    pub bounds: Bounds<ScaledPixels>,
    /// Clip and fade.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The outline's parameters, packed by [`ShapeOutline`].
    pub params: [f32; 4],
    /// Row-major map from bounds space, after `placement`'s offset, to outline space.
    pub transform: [f32; 4],
    /// The outline's centre in bounds space, then the liquid warp's amount and phase.
    pub placement: [f32; 4],
    /// Key light angle, inner light angle, rim strength and halo strength.
    pub lighting: [f32; 4],
    /// Glass: primary, secondary, deep and rim, with `colors[0].a` as the whole shape's opacity.
    /// Glow: the colour, then unused.
    pub colors: [Hsla; 4],
}

const _: () = assert!(std::mem::size_of::<Shape>() == 192);
// The WGSL `Shape` puts every `vec4` on a 16-byte boundary.
const _: () = {
    assert!(std::mem::offset_of!(Shape, bounds) == 16);
    assert!(std::mem::offset_of!(Shape, content_mask) == 32);
    assert!(std::mem::offset_of!(Shape, params) == 64);
    assert!(std::mem::offset_of!(Shape, transform) == 80);
    assert!(std::mem::offset_of!(Shape, placement) == 96);
    assert!(std::mem::offset_of!(Shape, lighting) == 112);
    assert!(std::mem::offset_of!(Shape, colors) == 128);
};

impl Shape {
    /// Draw the outline mirrored across the bounds' vertical centre line as well.
    pub const MIRROR: u32 = 1;
    /// Shade a glass halo as a soft shadow below the outline instead of a glow around it.
    pub const SHADOW_HALO: u32 = 2;
}

impl From<Shape> for Primitive {
    fn from(shape: Shape) -> Self {
        Primitive::Shape(shape)
    }
}

/// The outline a [`PaintShape`](crate::PaintShape) fills, in an outline space where the shape's
/// radius is 1 and y points down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeOutline {
    /// The unit circle.
    Circle,
    /// A star, flower or cookie with `lobes` tips, one pointing up, whose radius swings between
    /// `inner` (0 to 1) and 1. `sharpness` above 1 narrows the tips and rounds the valleys; below
    /// 1 swells the lobes and pinches the valleys.
    Polar {
        /// Tip count, at least 1.
        lobes: u32,
        /// Valley radius.
        inner: f32,
        /// Tip narrowing exponent.
        sharpness: f32,
    },
    /// A regular polygon with a vertex pointing up, its corners rounded by `rounding` (0 to 1 of
    /// the radius) without growing past the unit circle.
    Polygon {
        /// Side count, at least 3.
        sides: u32,
        /// Corner radius.
        rounding: f32,
    },
    /// The superellipse `|x|^exponent + |y|^exponent = 1`: a circle at 2, a squircle near 4.
    Superellipse {
        /// The exponent, at least 1.
        exponent: f32,
    },
    /// A heart with its point down, its outline rounded by `rounding`.
    Heart {
        /// How far the outline is rounded, in radii.
        rounding: f32,
    },
    /// A vertical capsule of radius 1 whose straight part reaches `half_length` above and below
    /// its centre.
    Capsule {
        /// Half the straight part's length, in radii.
        half_length: f32,
    },
    /// The unit circle drawn out to a rounded point above it, as a drop or a speech balloon's
    /// tail: the point is a circle of radius `tip` (0 to 1) `reach` radii above the centre, joined
    /// to the unit circle by their common tangents.
    Drop {
        /// How far above the centre the point's circle sits, in radii.
        reach: f32,
        /// The point's radius.
        tip: f32,
    },
}

impl ShapeOutline {
    pub(crate) fn encode(self) -> (u32, [f32; 4]) {
        match self {
            Self::Circle => (0, [0.; 4]),
            Self::Polar {
                lobes,
                inner,
                sharpness,
            } => (
                1,
                [
                    lobes.max(1) as f32,
                    inner.clamp(0.05, 1.),
                    sharpness.clamp(0.1, 8.),
                    0.,
                ],
            ),
            Self::Polygon { sides, rounding } => {
                (2, [sides.max(3) as f32, rounding.clamp(0., 0.9), 0., 0.])
            }
            Self::Superellipse { exponent } => (3, [exponent.clamp(1., 32.), 0., 0., 0.]),
            Self::Heart { rounding } => (4, [rounding.clamp(0., 0.5), 0., 0., 0.]),
            Self::Capsule { half_length } => (5, [half_length.max(0.), 0., 0., 0.]),
            Self::Drop { reach, tip } => (6, [reach.max(0.), tip.clamp(0., 1.), 0., 0.]),
        }
    }
}

/// Whether a glass shape's halo glows around it or falls below it as a soft shadow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ShapeHalo {
    /// Light spilling out around the rim; reads on dark surfaces.
    #[default]
    Glow,
    /// A soft tinted shadow below the shape; reads on light surfaces.
    Shadow,
}

/// Lit glass: a `deep` body with `primary` glowing inside toward the top left and `secondary`
/// toward the bottom right, a frosted rim lit by a key light and by a back light opposite it, and
/// a halo outside the outline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassMaterial {
    /// The light inside the glass toward the top left.
    pub primary: Hsla,
    /// The light inside the glass toward the bottom right.
    pub secondary: Hsla,
    /// The body the lights glow in.
    pub deep: Hsla,
    /// The rim light's colour.
    pub rim: Hsla,
    /// Direction toward the key light, in radians from +x toward +y (down). The back light sits
    /// roughly opposite it.
    pub light_angle: f32,
    /// How far the inner lights are turned about the centre, in radians.
    pub flow_angle: f32,
    /// Rim brightness; 1 is the designed look.
    pub rim_strength: f32,
    /// Halo strength; 0 for none.
    pub halo: f32,
    /// How the halo is drawn.
    pub halo_style: ShapeHalo,
}

impl GlassMaterial {
    /// The key light toward the top left: the designed default.
    pub const TOP_LEFT_LIGHT: f32 = -2.58;

    /// Glass of these colours, lit from the top left with a glowing halo.
    pub fn new(primary: Hsla, secondary: Hsla, deep: Hsla, rim: Hsla) -> Self {
        Self {
            primary,
            secondary,
            deep,
            rim,
            light_angle: Self::TOP_LEFT_LIGHT,
            flow_angle: 0.,
            rim_strength: 1.,
            halo: 1.,
            halo_style: ShapeHalo::Glow,
        }
    }
}

/// How a [`PaintShape`](crate::PaintShape) is shaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeMaterial {
    /// Lit glass.
    Glass(GlassMaterial),
    /// A flat emissive fill with a soft halo of the same colour; `halo` 0 for none.
    Glow {
        /// The fill.
        color: Hsla,
        /// Halo strength.
        halo: f32,
    },
}

/// Places a [`ShapeOutline`] in its bounds: a map from outline space to bounds space (the bounds'
/// largest centred square, -1 to 1, y down). Each step applies after the ones before it, so
/// `ShapeTransform::default().scale(0.7, 0.7).translate(0., -0.1)` shrinks the unit outline and
/// then raises it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeTransform {
    matrix: [[f32; 2]; 2],
    offset: [f32; 2],
}

impl Default for ShapeTransform {
    fn default() -> Self {
        Self {
            matrix: [[1., 0.], [0., 1.]],
            offset: [0., 0.],
        }
    }
}

impl ShapeTransform {
    /// Rotate by `radians`, turning +x toward +y (clockwise on screen).
    pub fn rotate(self, radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        self.then([[cos, -sin], [sin, cos]])
    }

    /// Scale along the bounds' axes.
    pub fn scale(self, x: f32, y: f32) -> Self {
        self.then([[x, 0.], [0., y]])
    }

    /// Move the outline by `(x, y)` in bounds space.
    pub fn translate(mut self, x: f32, y: f32) -> Self {
        self.offset[0] += x;
        self.offset[1] += y;
        self
    }

    fn then(self, m: [[f32; 2]; 2]) -> Self {
        let a = self.matrix;
        let o = self.offset;
        Self {
            matrix: [
                [
                    m[0][0] * a[0][0] + m[0][1] * a[1][0],
                    m[0][0] * a[0][1] + m[0][1] * a[1][1],
                ],
                [
                    m[1][0] * a[0][0] + m[1][1] * a[1][0],
                    m[1][0] * a[0][1] + m[1][1] * a[1][1],
                ],
            ],
            offset: [
                m[0][0] * o[0] + m[0][1] * o[1],
                m[1][0] * o[0] + m[1][1] * o[1],
            ],
        }
    }

    /// The row-major inverse of the linear part and the offset, or `None` when the outline has
    /// been scaled to nothing.
    pub(crate) fn inverse(&self) -> Option<([f32; 4], [f32; 2])> {
        let [[a, b], [c, d]] = self.matrix;
        let determinant = a * d - b * c;
        if !determinant.is_finite() || determinant.abs() < 1e-6 {
            return None;
        }
        let inverse = 1. / determinant;
        Some((
            [d * inverse, -b * inverse, -c * inverse, a * inverse],
            self.offset,
        ))
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

/// Identifies a cached render layer: a view subtree rendered into its own offscreen GPU texture so
/// it can be re-composited (cheaply, stretched) on window resize without re-running the view's
/// paint. Stable for the lifetime of the layered view (derived from its `EntityId`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u64);

/// Where a [`PaintSurface`] composite samples from.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum PaintSurfaceSource {
    /// A cached render layer (a view rendered to an offscreen GPU texture). The platform renderer
    /// looks the texture up by id from the frame's layer set and draws it stretched to `bounds`.
    Layer(LayerId),
    /// Whatever the frame holds under `bounds` at this point in the draw order, blurred. See
    /// [`crate::Window::paint_backdrop_blur`]. Renderers that cannot read their own target back
    /// skip it.
    BackdropBlur(BackdropBlur),
    /// A platform pixel buffer (e.g. a decoded video frame). macOS only.
    #[cfg(target_os = "macos")]
    Image(core_video::pixel_buffer::CVPixelBuffer),
}

/// A live backdrop blur: the Gaussian standard deviation and the shape it is clipped to.
#[derive(Clone, Copy, Debug)]
pub struct BackdropBlur {
    /// Standard deviation of the blur, in device pixels (CSS `blur()` semantics).
    pub radius: ScaledPixels,
    /// Corner radii of the blurred shape, in device pixels.
    pub corner_radii: Corners<ScaledPixels>,
    /// The element opacity the blur was painted at.
    pub opacity: f32,
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub source: PaintSurfaceSource,
    /// Sample a layer texture stretched over `bounds` instead of 1:1 from its top-left: the
    /// composite of [`crate::Window::capture_layer_scaled`], which draws a subtree smaller or
    /// larger than it was laid out.
    pub stretch: bool,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug, PartialEq)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_transform_inverse_maps_bounds_space_back_to_outline_space() {
        let transform = ShapeTransform::default()
            .rotate(std::f32::consts::FRAC_PI_2)
            .scale(0.5, 0.25)
            .translate(0.1, -0.2);
        let (inverse, offset) = transform.inverse().expect("an invertible transform");
        // The outline's top, (0, -1), turns to (1, 0), shrinks to (0.5, 0) and then moves.
        let moved = [0.6 - offset[0], -0.2 - offset[1]];
        let outline = [
            inverse[0] * moved[0] + inverse[1] * moved[1],
            inverse[2] * moved[0] + inverse[3] * moved[1],
        ];
        assert!(outline[0].abs() < 1e-5, "{outline:?}");
        assert!((outline[1] + 1.).abs() < 1e-5, "{outline:?}");

        assert!(ShapeTransform::default().scale(0., 1.).inverse().is_none());
    }

    #[test]
    fn shape_outline_parameters_are_clamped_into_their_documented_ranges() {
        assert_eq!(
            ShapeOutline::Polar {
                lobes: 0,
                inner: 2.,
                sharpness: 100.,
            }
            .encode(),
            (1, [1., 1., 8., 0.])
        );
        assert_eq!(
            ShapeOutline::Polygon {
                sides: 1,
                rounding: -1.,
            }
            .encode(),
            (2, [3., 0., 0., 0.])
        );
        assert_eq!(
            ShapeOutline::Superellipse { exponent: 0.5 }.encode(),
            (3, [1., 0., 0., 0.])
        );
        assert_eq!(
            ShapeOutline::Drop {
                reach: -1.,
                tip: 2.,
            }
            .encode(),
            (6, [0., 1., 0., 0.])
        );
    }
}
