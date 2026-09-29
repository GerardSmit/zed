use super::*;

const MAX_DAMAGE_REGIONS: usize = 8;
const MAX_STAGED_PATHS: usize = 16;
const MAX_STAGED_PATH_BYTES: usize = 1024 * 1024;

/// A bounded set of non-overlapping damage rectangles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneDamageRegions {
    bounds: [Bounds<ScaledPixels>; MAX_DAMAGE_REGIONS],
    len: usize,
}

/// The portion of a retained render target that must be cleared and painted again.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SceneDamage {
    /// Target contents are unavailable, or the scene depends on pixels outside its own bounds.
    #[default]
    Full,
    /// The retained target already contains this scene.
    None,
    /// A conservative union of changed pixels, in device-pixel coordinates.
    Partial(Bounds<ScaledPixels>),
    /// Separate changed regions, coalesced when they overlap.
    Regions(SceneDamageRegions),
}

impl SceneDamage {
    /// Convert damage to an outward-rounded, viewport-clamped integer rectangle.
    pub fn pixel_bounds(self, viewport: Size<DevicePixels>) -> Option<Bounds<DevicePixels>> {
        self.pixel_rects(viewport)
            .reduce(|left, right| left.union(&right))
    }

    /// Outward-rounded, clipped, non-overlapping regions for one frame's render passes.
    pub fn pixel_rects(
        self,
        viewport: Size<DevicePixels>,
    ) -> impl ExactSizeIterator<Item = Bounds<DevicePixels>> {
        let full = Bounds::new(Point::default(), viewport);
        let mut output = [Bounds::default(); MAX_DAMAGE_REGIONS];
        let mut len = 0;
        if !full.is_empty() {
            let mut clipped = Self::None;
            if self == Self::Full || self.partial_rects().iter().any(|bounds| !finite(*bounds)) {
                output[0] = full;
                len = 1;
            } else {
                for bounds in self.partial_rects() {
                    let clamp =
                        |value: f32, max: DevicePixels| ScaledPixels(value.clamp(0., max.0 as f32));
                    clipped.include(Bounds::from_corners(
                        point(
                            clamp(bounds.left().0.floor(), viewport.width),
                            clamp(bounds.top().0.floor(), viewport.height),
                        ),
                        point(
                            clamp(bounds.right().0.ceil(), viewport.width),
                            clamp(bounds.bottom().0.ceil(), viewport.height),
                        ),
                    ));
                }
                for bounds in clipped.partial_rects() {
                    output[len] = Bounds::new(
                        bounds.origin.map(|value| DevicePixels(value.0 as i32)),
                        bounds.size.map(|value| DevicePixels(value.0 as i32)),
                    );
                    len += 1;
                }
            }
        }
        output.into_iter().take(len)
    }

    fn partial_rects(&self) -> &[Bounds<ScaledPixels>] {
        match self {
            Self::None | Self::Full => &[],
            Self::Partial(bounds) => std::slice::from_ref(bounds),
            Self::Regions(regions) => &regions.bounds[..regions.len],
        }
    }

    /// Whether any damaged pixels can intersect a primitive's conservative footprint.
    pub fn intersects(&self, bounds: Bounds<ScaledPixels>) -> bool {
        match self {
            Self::Full => true,
            Self::None => false,
            _ => {
                !finite(bounds)
                    || self
                        .partial_rects()
                        .iter()
                        .any(|region| !finite(*region) || !region.intersect(&bounds).is_empty())
            }
        }
    }

    /// Combine changes without creating another presentation.
    pub fn union(mut self, other: Self) -> Self {
        self.extend(other);
        self
    }

    fn include(&mut self, mut bounds: Bounds<ScaledPixels>) {
        if !finite(bounds) {
            *self = Self::Full;
        } else if !bounds.is_empty() {
            if matches!(self, Self::Full) {
                return;
            }
            let mut regions = SceneDamageRegions {
                bounds: [Bounds::default(); MAX_DAMAGE_REGIONS],
                len: 0,
            };
            for previous in self.partial_rects() {
                regions.bounds[regions.len] = *previous;
                regions.len += 1;
            }
            let mut index = 0;
            while index < regions.len {
                let previous = regions.bounds[index];
                if previous.left() <= bounds.right()
                    && bounds.left() <= previous.right()
                    && previous.top() <= bounds.bottom()
                    && bounds.top() <= previous.bottom()
                {
                    bounds = bounds.union(&previous);
                    regions.len -= 1;
                    regions.bounds[index] = regions.bounds[regions.len];
                    index = 0;
                } else {
                    index += 1;
                }
            }
            if regions.len == MAX_DAMAGE_REGIONS {
                for previous in &regions.bounds {
                    bounds = bounds.union(previous);
                }
                *self = Self::Partial(bounds);
            } else if regions.len == 0 {
                *self = Self::Partial(bounds);
            } else {
                regions.bounds[regions.len] = bounds;
                regions.len += 1;
                *self = Self::Regions(regions);
            }
        }
    }

    fn extend(&mut self, other: Self) {
        match other {
            Self::Full => *self = Self::Full,
            Self::None => (),
            Self::Partial(bounds) => self.include(bounds),
            Self::Regions(regions) => {
                for bounds in &regions.bounds[..regions.len] {
                    self.include(*bounds);
                }
            }
        }
    }
}

fn finite(bounds: Bounds<ScaledPixels>) -> bool {
    [
        bounds.left().0,
        bounds.top().0,
        bounds.right().0,
        bounds.bottom().0,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn footprint(
    bounds: Bounds<ScaledPixels>,
    mask: ContentMask<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    if !finite(bounds) || !finite(mask.bounds) {
        return Bounds::new(
            point(ScaledPixels(f32::NAN), ScaledPixels(f32::NAN)),
            Size::default(),
        );
    }
    // Cover fractional edges and antialiasing, including MSAA resolve/filtering at primitive edges.
    bounds
        .dilate(ScaledPixels(1.))
        .intersect(&mask.bounds.dilate(ScaledPixels(1.)))
}

fn shadow_footprint(shadow: &Shadow) -> Bounds<ScaledPixels> {
    let bounds = if shadow.inset != 0 {
        shadow.element_bounds
    } else {
        shadow
            .bounds
            .dilate(ScaledPixels(3. * shadow.blur_radius.0))
    };
    footprint(bounds, shadow.content_mask)
}

fn sprite_footprint(
    bounds: Bounds<ScaledPixels>,
    mask: ContentMask<ScaledPixels>,
    transform: TransformationMatrix,
) -> Bounds<ScaledPixels> {
    let transform_point = |x: ScaledPixels, y: ScaledPixels| {
        let point = transform.apply(point(Pixels(x.0), Pixels(y.0)));
        point.map(|value| ScaledPixels(value.0))
    };
    let corners = [
        transform_point(bounds.left(), bounds.top()),
        transform_point(bounds.right(), bounds.top()),
        transform_point(bounds.left(), bounds.bottom()),
        transform_point(bounds.right(), bounds.bottom()),
    ];
    if corners
        .iter()
        .any(|point| !point.x.0.is_finite() || !point.y.0.is_finite())
    {
        return Bounds::new(
            point(ScaledPixels(f32::NAN), ScaledPixels(f32::NAN)),
            Size::default(),
        );
    }
    let mut minimum = corners[0];
    let mut maximum = corners[0];
    for corner in &corners[1..] {
        minimum.x = minimum.x.min(corner.x);
        minimum.y = minimum.y.min(corner.y);
        maximum.x = maximum.x.max(corner.x);
        maximum.y = maximum.y.max(corner.y);
    }
    footprint(Bounds::from_corners(minimum, maximum), mask)
}

fn changed<T>(
    previous: &[T],
    current: &[T],
    equal: impl Fn(&T, &T) -> bool,
    bounds: impl Fn(&T) -> Bounds<ScaledPixels>,
) -> SceneDamage {
    let mut damage = SceneDamage::None;
    let prefix = previous
        .iter()
        .zip(current)
        .take_while(|(old, new)| equal(old, new))
        .count();
    let previous = &previous[prefix..];
    let current = &current[prefix..];
    let suffix = previous
        .iter()
        .rev()
        .zip(current.iter().rev())
        .take_while(|(old, new)| equal(old, new))
        .count();
    let previous = &previous[..previous.len() - suffix];
    let current = &current[..current.len() - suffix];
    if previous.len() == current.len() {
        for (old, new) in previous.iter().zip(current) {
            if !equal(old, new) {
                damage.include(bounds(old));
                damage.include(bounds(new));
            }
        }
    } else {
        for value in previous.iter().chain(current) {
            damage.include(bounds(value));
        }
    }
    damage
}

impl Scene {
    pub(crate) fn clear_damage(&mut self) {
        self.damage = SceneDamage::None;
        for layer in &mut self.layers {
            if let Some(scene) = &mut layer.scene {
                scene.clear_damage();
            }
        }
    }

    pub(crate) fn reuse_layers(&mut self, previous: &mut Scene, pending: bool) {
        // Cached paint replays the composite alone. Move its existing snapshot forward so a
        // skipped presentation or lost GPU texture can recover without repainting the view.
        // Retention is bounded to layer ids referenced by the current scene, with no scene copy.
        for layer in &mut self.layers {
            if let Some(scene) = &mut layer.scene {
                if !layer.needs_render {
                    scene.clear_damage();
                    continue;
                }
                if let Some(old) = previous
                    .layers
                    .iter_mut()
                    .find(|old| old.id == layer.id)
                    .and_then(|old| old.scene.as_mut())
                {
                    scene.reuse_layers(old, pending);
                }
            }
        }
        for surface in &self.surfaces {
            let PaintSurfaceSource::Layer(id) = surface.source else {
                continue;
            };
            let current = self.layers.iter().position(|layer| layer.id == id);
            if current.is_some_and(|index| self.layers[index].scene.is_some()) {
                continue;
            }
            if let Some(index) = previous.layers.iter().position(|layer| layer.id == id) {
                let mut layer = previous.layers.swap_remove(index);
                layer.needs_render &= pending;
                if let Some(index) = current {
                    self.layers[index] = layer;
                } else {
                    self.layers.push(layer);
                }
            }
        }
    }

    /// Compare final primitive values, including draw order, without retaining another scene copy.
    /// `pending` preserves changes from frames drawn but not yet presented.
    pub fn update_damage(&mut self, previous: &Scene, pending: bool) {
        for layer in &mut self.layers {
            if let Some(scene) = &mut layer.scene {
                if !layer.needs_render {
                    scene.clear_damage();
                    continue;
                }
                if let Some(old) = previous
                    .layers
                    .iter()
                    .find(|old| old.id == layer.id && old.size == layer.size)
                    .and_then(|old| old.scene.as_ref())
                {
                    scene.update_damage(old, pending);
                } else {
                    scene.damage = SceneDamage::Full;
                }
            }
        }
        // Backdrop blur reads neighboring pixels from earlier draw order; live pixel buffers can
        // mutate without changing identity. Neither is safe to infer from primitive equality.
        if self
            .surfaces
            .iter()
            .chain(&previous.surfaces)
            .any(|surface| !matches!(surface.source, PaintSurfaceSource::Layer(_)))
        {
            self.damage = SceneDamage::Full;
            return;
        }
        let mut damage = SceneDamage::None;
        macro_rules! compare {
            ($field:ident, $bounds:expr) => {
                damage.extend(changed(
                    &previous.$field,
                    &self.$field,
                    |old, new| old == new,
                    $bounds,
                ));
            };
            ($field:ident) => {
                compare!($field, |value| footprint(value.bounds, value.content_mask));
            };
        }
        compare!(shadows, shadow_footprint);
        compare!(quads);
        compare!(paths);
        compare!(underlines);
        compare!(monochrome_sprites, |value| sprite_footprint(
            value.bounds,
            value.content_mask,
            value.transformation
        ));
        compare!(subpixel_sprites, |value| sprite_footprint(
            value.bounds,
            value.content_mask,
            value.transformation
        ));
        compare!(polychrome_sprites);
        damage.extend(changed(&previous.surfaces, &self.surfaces, |old, new| {
            old.order == new.order && old.bounds == new.bounds && old.content_mask == new.content_mask && old.stretch == new.stretch
                && matches!((&old.source, &new.source), (PaintSurfaceSource::Layer(old), PaintSurfaceSource::Layer(new)) if old == new)
        }, |surface| footprint(surface.bounds, surface.content_mask)));
        for surface in &self.surfaces {
            let PaintSurfaceSource::Layer(id) = surface.source else {
                continue;
            };
            let Some(layer) = self
                .layers
                .iter()
                .find(|layer| layer.id == id && layer.needs_render)
            else {
                continue;
            };
            match layer
                .scene
                .as_ref()
                .map_or(SceneDamage::Full, |scene| scene.damage)
            {
                SceneDamage::None => (),
                SceneDamage::Full => {
                    damage.include(footprint(surface.bounds, surface.content_mask))
                }
                partial @ (SceneDamage::Partial(_) | SceneDamage::Regions(_)) => {
                    let scale = if surface.stretch {
                        point(
                            surface.bounds.size.width.0 / layer.size.width.0.max(1) as f32,
                            surface.bounds.size.height.0 / layer.size.height.0.max(1) as f32,
                        )
                    } else {
                        point(1., 1.)
                    };
                    let map = |value: Point<ScaledPixels>| {
                        surface.bounds.origin
                            + point(
                                ScaledPixels(value.x.0 * scale.x),
                                ScaledPixels(value.y.0 * scale.y),
                            )
                    };
                    for bounds in partial.partial_rects() {
                        let bounds =
                            Bounds::from_corners(map(bounds.origin), map(bounds.bottom_right()));
                        damage.include(footprint(
                            bounds.intersect(&surface.bounds),
                            surface.content_mask,
                        ));
                    }
                }
            }
        }
        if pending {
            damage.extend(previous.damage);
        }
        self.damage = damage;
    }

    /// Copy only primitives touching damage into reusable upload staging, retaining paint order.
    /// Child layers must be processed from the original scene before using this primitive-only copy.
    pub fn copy_primitives_for_damage(&self, damage: SceneDamage, output: &mut Scene) {
        output.damage = damage;
        output.layers.clear();
        output.paint_operations.clear();
        output.primitive_bounds.clear();
        output.layer_stack.clear();
        output.rounded_clips.clear();
        macro_rules! copy {
            ($field:ident, $bounds:expr) => {
                output.$field.clear();
                output.$field.extend(
                    self.$field
                        .iter()
                        .filter(|value| damage.intersects(($bounds)(value)))
                        .cloned(),
                );
            };
            ($field:ident: $value:ty) => {
                copy!($field, |value: &$value| footprint(
                    value.bounds,
                    value.content_mask
                ));
            };
        }
        copy!(shadows, shadow_footprint);
        copy!(quads: Quad);
        copy!(underlines: Underline);
        copy!(monochrome_sprites, |value: &MonochromeSprite| {
            sprite_footprint(value.bounds, value.content_mask, value.transformation)
        });
        copy!(subpixel_sprites, |value: &SubpixelSprite| sprite_footprint(
            value.bounds,
            value.content_mask,
            value.transformation
        ));
        copy!(polychrome_sprites: PolychromeSprite);
        copy!(surfaces: PaintSurface);

        let mut count = 0;
        for path in &self.paths {
            if !damage.intersects(footprint(path.bounds, path.content_mask)) {
                continue;
            }
            if output.paths.len() == count {
                output.paths.push(
                    output
                        .staged_path_pool
                        .pop()
                        .unwrap_or_else(|| Path::new(Point::default()).scale(1.)),
                );
            }
            if let Some(target) = output.paths.get_mut(count) {
                target.vertices.clone_from(&path.vertices);
                target.order = path.order;
                target.bounds = path.bounds;
                target.content_mask = path.content_mask;
                target.color = path.color;
                target.start = path.start;
                target.current = path.current;
                target.contour_count = path.contour_count;
            }
            output.paths[count].id = PathId(count);
            count += 1;
        }
        let vertex_bytes = |path: &Path<ScaledPixels>| {
            path.vertices.capacity() * std::mem::size_of::<PathVertex<ScaledPixels>>()
        };
        let mut spare_bytes: usize = output.staged_path_pool.iter().map(vertex_bytes).sum();
        while output.paths.len() > count {
            if let Some(path) = output.paths.pop() {
                let bytes = vertex_bytes(&path);
                if output.staged_path_pool.len() < MAX_STAGED_PATHS
                    && spare_bytes + bytes <= MAX_STAGED_PATH_BYTES
                {
                    spare_bytes += bytes;
                    output.staged_path_pool.push(path);
                }
            }
        }
    }

    /// Visible footprint of a path batch, intersected with the damage being replayed.
    pub fn path_bounds_for_damage(
        &self,
        damage: SceneDamage,
        range: Range<usize>,
    ) -> Option<Bounds<ScaledPixels>> {
        let mut bounds: Option<Bounds<ScaledPixels>> = None;
        let full = damage == SceneDamage::Full
            || damage.partial_rects().iter().any(|region| !finite(*region));
        for path in &self.paths[range] {
            let visible = footprint(path.bounds, path.content_mask);
            if damage == SceneDamage::None {
                continue;
            }
            if !finite(visible) {
                return Some(visible);
            }
            if full {
                bounds = Some(bounds.map_or(visible, |bounds| bounds.union(&visible)));
            } else {
                for region in damage.partial_rects() {
                    let clipped = visible.intersect(region);
                    if !clipped.is_empty() {
                        bounds = Some(bounds.map_or(clipped, |bounds| bounds.union(&clipped)));
                    }
                }
            }
        }
        bounds
    }

    /// Batches touching damaged pixels, in their original compositing order. Renderers must also
    /// scissor writes to damage and replay unchanged primitives overlapping it after clearing.
    pub fn batches_for_damage(
        &self,
        damage: SceneDamage,
    ) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        self.batches().filter(move |batch| {
            if damage == SceneDamage::Full {
                return true;
            }
            if damage == SceneDamage::None {
                return false;
            }
            let touches = |other: Bounds<ScaledPixels>| damage.intersects(other);
            macro_rules! intersects {
                ($field:ident, $range:expr) => {
                    self.$field[$range.clone()]
                        .iter()
                        .any(|value| touches(footprint(value.bounds, value.content_mask)))
                };
            }
            match batch {
                PrimitiveBatch::Shadows(range) => self.shadows[range.clone()]
                    .iter()
                    .any(|value| touches(shadow_footprint(value))),
                PrimitiveBatch::Quads(range) => intersects!(quads, range),
                PrimitiveBatch::Paths(range) => intersects!(paths, range),
                PrimitiveBatch::Underlines(range) => intersects!(underlines, range),
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    self.monochrome_sprites[range.clone()].iter().any(|value| {
                        touches(sprite_footprint(
                            value.bounds,
                            value.content_mask,
                            value.transformation,
                        ))
                    })
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    self.subpixel_sprites[range.clone()].iter().any(|value| {
                        touches(sprite_footprint(
                            value.bounds,
                            value.content_mask,
                            value.transformation,
                        ))
                    })
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    intersects!(polychrome_sprites, range)
                }
                PrimitiveBatch::Surfaces(range) => intersects!(surfaces, range),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rgba, size};

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: rect(x, 20., 10., 10.),
            content_mask: ContentMask {
                bounds: rect(0., 0., 200., 200.),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn separated_damage_stays_separate_and_merges_after_pixel_rounding() {
        let damage = SceneDamage::Partial(rect(0., 0., 10., 10.))
            .union(SceneDamage::Partial(rect(90., 90., 10., 10.)));
        let rectangles: Vec<_> = damage
            .pixel_rects(size(DevicePixels(100), DevicePixels(100)))
            .collect();
        assert_eq!(rectangles.len(), 2);
        assert_eq!(
            rectangles
                .iter()
                .map(|rect| rect.size.width.0 * rect.size.height.0)
                .sum::<i32>(),
            200
        );
        assert!(!damage.intersects(rect(40., 40., 10., 10.)));

        let rounded = SceneDamage::Partial(rect(0., 0., 0.6, 2.))
            .union(SceneDamage::Partial(rect(0.8, 0., 0.6, 2.)));
        let rectangles: Vec<_> = rounded
            .pixel_rects(size(DevicePixels(10), DevicePixels(10)))
            .collect();
        assert_eq!(
            rectangles,
            vec![Bounds::new(
                Point::default(),
                size(DevicePixels(2), DevicePixels(2))
            )]
        );
    }

    #[test]
    fn damage_capacity_is_bounded_and_overflow_keeps_every_pixel() {
        let mut damage = SceneDamage::None;
        for index in 0..20 {
            damage.include(rect(index as f32 * 10., 0., 2., 2.));
        }
        let rectangles: Vec<_> = damage
            .pixel_rects(size(DevicePixels(200), DevicePixels(20)))
            .collect();
        assert!(rectangles.len() <= MAX_DAMAGE_REGIONS);
        for index in 0..20 {
            assert!(damage.intersects(rect(index as f32 * 10., 0., 2., 2.)));
        }
        for (index, bounds) in rectangles.iter().enumerate() {
            assert!(
                rectangles[index + 1..]
                    .iter()
                    .all(|other| bounds.intersect(other).is_empty())
            );
        }
    }

    #[test]
    fn compaction_culls_individual_primitives_and_reuses_path_vertices() {
        let mut scene = Scene::default();
        for index in 0..1000 {
            let mut value = quad(index as f32 * 20.);
            value.content_mask.bounds = rect(0., 0., 20000., 100.);
            scene.quads.push(value);
        }
        for x in [20., 100.] {
            let mut path = Path::new(point(Pixels(x), Pixels(20.))).scale(1.);
            path.bounds = rect(x, 20., 10., 10.);
            path.content_mask = quad(x).content_mask;
            path.vertices.push(PathVertex {
                xy_position: point(ScaledPixels(x), ScaledPixels(20.)),
                st_position: point(0., 0.),
                content_mask: path.content_mask,
            });
            path.id = PathId(scene.paths.len());
            scene.paths.push(path);
        }
        let damage = SceneDamage::Partial(rect(100., 20., 10., 10.));
        let mut staged = Scene::default();
        scene.copy_primitives_for_damage(damage, &mut staged);
        assert_eq!(staged.quads.len(), 1);
        assert_eq!(staged.quads[0], scene.quads[5]);
        assert_eq!(staged.paths.len(), 1);
        assert_eq!(staged.paths[0].id, PathId(0));
        assert_eq!(staged.paths[0].vertices, scene.paths[1].vertices);
        let allocation = staged.paths[0].vertices.as_ptr();
        scene.copy_primitives_for_damage(damage, &mut staged);
        assert_eq!(staged.paths[0].vertices.as_ptr(), allocation);
        assert_eq!(
            staged.path_bounds_for_damage(damage, 0..1),
            Some(rect(100., 20., 10., 10.))
        );
        assert_eq!(scene.quads.len(), 1000);
        scene.copy_primitives_for_damage(SceneDamage::None, &mut staged);
        assert!(staged.paths.is_empty());
        scene.copy_primitives_for_damage(damage, &mut staged);
        assert_eq!(staged.paths[0].vertices.as_ptr(), allocation);
    }

    #[test]
    fn invalid_damage_keeps_primitives_for_full_recovery() {
        let mut scene = Scene::default();
        scene.quads = vec![quad(20.), quad(120.)];
        let damage = SceneDamage::Partial(rect(f32::NAN, 0., 1., 1.));
        let mut staged = Scene::default();
        scene.copy_primitives_for_damage(damage, &mut staged);
        assert_eq!(staged.quads, scene.quads);
        let viewport = size(DevicePixels(200), DevicePixels(200));
        assert_eq!(
            damage.pixel_bounds(viewport),
            SceneDamage::Full.pixel_bounds(viewport)
        );
    }

    #[test]
    fn invalid_path_footprint_cannot_be_hidden_by_a_finite_path() {
        let mut scene = Scene::default();
        for x in [f32::NAN, 20.] {
            let mut path = Path::new(Point::default()).scale(1.);
            path.bounds = rect(x, 20., 10., 10.);
            path.content_mask = quad(20.).content_mask;
            scene.paths.push(path);
        }
        for damage in [
            SceneDamage::Full,
            SceneDamage::Partial(rect(20., 20., 10., 10.)),
        ] {
            let bounds = scene.path_bounds_for_damage(damage, 0..2).unwrap();
            assert!(!finite(bounds));
            let viewport = size(DevicePixels(200), DevicePixels(200));
            assert_eq!(
                SceneDamage::Partial(bounds).pixel_bounds(viewport),
                SceneDamage::Full.pixel_bounds(viewport)
            );
        }
    }

    #[test]
    fn spare_path_storage_has_a_count_and_byte_limit() {
        let mut source = Scene::default();
        let mut path = Path::new(Point::default()).scale(1.);
        path.bounds = rect(0., 0., 10., 10.);
        path.content_mask = quad(0.).content_mask;
        path.vertices =
            vec![
                PathVertex {
                    xy_position: point(ScaledPixels(1.), ScaledPixels(1.)),
                    st_position: point(0., 0.),
                    content_mask: path.content_mask,
                };
                MAX_STAGED_PATH_BYTES / std::mem::size_of::<PathVertex<ScaledPixels>>() / 4
            ];
        source.paths = vec![path; MAX_STAGED_PATHS + 1];
        let mut staged = Scene::default();
        source.copy_primitives_for_damage(SceneDamage::Full, &mut staged);
        source.copy_primitives_for_damage(SceneDamage::None, &mut staged);
        assert!(staged.paths.is_empty());
        assert!(staged.staged_path_pool.len() <= MAX_STAGED_PATHS);
        assert!(
            staged
                .staged_path_pool
                .iter()
                .map(|path| {
                    path.vertices.capacity() * std::mem::size_of::<PathVertex<ScaledPixels>>()
                })
                .sum::<usize>()
                <= MAX_STAGED_PATH_BYTES
        );
    }

    #[test]
    fn movement_removal_and_order_include_old_pixels() {
        let mut old = Scene::default();
        old.quads.push(quad(20.));
        let mut new = Scene::default();
        new.quads.push(quad(30.));
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 22., 12.)));
        new.quads.clear();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 12., 12.)));
        new.quads.push(Quad {
            order: 1,
            ..quad(20.)
        });
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 12., 12.)));
    }

    #[test]
    fn equality_and_pending_damage_preserve_presentation_baseline() {
        let mut old = Scene::default();
        old.quads.push(quad(20.));
        old.damage = SceneDamage::Partial(rect(70., 70., 5., 5.));
        let mut new = Scene::default();
        new.quads = old.quads.clone();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::None);
        new.update_damage(&old, true);
        assert_eq!(new.damage, old.damage);
        new.quads[0].background = rgba(0xff0000ff).into();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 12., 12.)));
    }

    #[test]
    fn independent_changes_share_damage_until_the_frame_is_presented() {
        let mut previous = Scene::default();
        previous.quads = vec![quad(20.), quad(120.)];
        previous.clear_damage();

        let mut current = Scene::default();
        current.quads = vec![quad(40.), quad(140.)];
        current.update_damage(&previous, false);
        let viewport = size(DevicePixels(200), DevicePixels(200));
        let combined = SceneDamage::Partial(rect(19., 19., 132., 12.)).pixel_bounds(viewport);
        assert_eq!(current.damage.pixel_bounds(viewport), combined);
        assert_eq!(current.damage.pixel_rects(viewport).count(), 4);

        for x in [60., 80.] {
            previous = current;
            current = Scene::default();
            current.quads = vec![quad(x), quad(140.)];
            current.update_damage(&previous, true);
            assert_eq!(current.damage.pixel_bounds(viewport), combined);
            assert!(current.damage.intersects(rect(20., 20., 10., 10.)));
            assert!(!current.damage.intersects(rect(105., 20., 5., 5.)));
        }

        current.clear_damage();
        previous = current;
        current = Scene::default();
        current.quads = vec![quad(90.), quad(140.)];
        current.update_damage(&previous, false);
        assert_eq!(
            current.damage,
            SceneDamage::Partial(rect(79., 19., 22., 12.))
        );
    }

    #[test]
    fn clipping_transforms_and_shadow_tails_bound_rasterization() {
        let mask = ContentMask {
            bounds: rect(0., 0., 200., 200.),
            ..Default::default()
        };
        let transform = TransformationMatrix {
            rotation_scale: [[0., -1.], [1., 0.]],
            translation: [80., 0.],
        };
        assert_eq!(
            sprite_footprint(rect(10., 20., 10., 20.), mask, transform),
            rect(39., 9., 22., 12.)
        );
        let shadow = Shadow {
            order: 0,
            blur_radius: ScaledPixels(4.),
            bounds: rect(30., 30., 10., 10.),
            corner_radii: Corners::default(),
            content_mask: mask,
            color: Hsla::default(),
            element_bounds: rect(30., 30., 10., 10.),
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
        };
        assert_eq!(shadow_footprint(&shadow), rect(17., 17., 36., 36.));
        assert_eq!(
            footprint(
                rect(20., 20., 20., 20.),
                ContentMask {
                    bounds: rect(25., 25., 5., 5.),
                    ..mask
                }
            ),
            rect(24., 24., 7., 7.)
        );
    }

    #[test]
    fn pixel_damage_rounds_outward_and_clamps() {
        let viewport = size(DevicePixels(100), DevicePixels(80));
        assert_eq!(
            SceneDamage::Partial(rect(-1.5, 2.4, 20.1, 4.2)).pixel_bounds(viewport),
            Some(Bounds::new(
                point(DevicePixels(0), DevicePixels(2)),
                size(DevicePixels(19), DevicePixels(5))
            ))
        );
        assert_eq!(
            SceneDamage::Partial(rect(120., 0., 5., 5.)).pixel_bounds(viewport),
            None
        );
        assert_eq!(SceneDamage::None.pixel_bounds(viewport), None);
        assert_eq!(
            SceneDamage::Partial(rect(f32::NAN, 0., 1., 1.)).pixel_bounds(viewport),
            SceneDamage::Full.pixel_bounds(viewport)
        );
    }

    fn layered_scene(x: f32) -> Scene {
        let mut content = Scene::default();
        content.quads.push(quad(x));
        let mut scene = Scene::default();
        scene.layers.push(SceneLayer {
            id: LayerId(1),
            size: size(DevicePixels(100), DevicePixels(100)),
            needs_render: true,
            scene: Some(Box::new(content)),
        });
        scene.surfaces.push(PaintSurface {
            order: 0,
            bounds: rect(100., 50., 200., 200.),
            content_mask: ContentMask {
                bounds: rect(0., 0., 500., 500.),
                ..Default::default()
            },
            source: PaintSurfaceSource::Layer(LayerId(1)),
            stretch: true,
        });
        scene
    }

    #[test]
    fn layer_damage_maps_scaled_content_and_accumulates_pending_changes() {
        let mut old = layered_scene(20.);
        old.damage = SceneDamage::None;
        let mut new = layered_scene(30.);
        new.update_damage(&old, false);
        assert_eq!(
            new.layers[0].scene.as_ref().unwrap().damage,
            SceneDamage::Partial(rect(19., 19., 22., 12.))
        );
        assert_eq!(new.damage, SceneDamage::Partial(rect(137., 87., 46., 26.)));
        let mut next = layered_scene(30.);
        next.update_damage(&new, true);
        assert_eq!(
            next.layers[0].scene.as_ref().unwrap().damage,
            new.layers[0].scene.as_ref().unwrap().damage
        );
        assert_eq!(next.damage, new.damage);
        next.update_damage(&new, false);
        assert_eq!(next.damage, SceneDamage::None);
    }

    #[test]
    fn separate_child_damage_stays_separate_in_parent_coordinates() {
        let mut old = layered_scene(20.);
        old.layers[0].scene.as_mut().unwrap().quads.push(quad(70.));
        old.clear_damage();
        let mut new = layered_scene(20.);
        new.layers[0].scene.as_mut().unwrap().quads.push(quad(70.));
        for quad in &mut new.layers[0].scene.as_mut().unwrap().quads {
            quad.background = rgba(0xff0000ff).into();
        }
        new.update_damage(&old, false);
        let viewport = size(DevicePixels(500), DevicePixels(500));
        assert_eq!(new.damage.pixel_rects(viewport).count(), 2);
        assert!(new.damage.intersects(rect(140., 90., 20., 20.)));
        assert!(new.damage.intersects(rect(240., 90., 20., 20.)));
        assert!(!new.damage.intersects(rect(190., 90., 20., 20.)));
    }

    #[test]
    fn nested_layers_keep_damage_local_and_recover_cached_children() {
        let wrap = |content: Scene| {
            let mut scene = Scene::default();
            scene.layers.push(SceneLayer {
                id: LayerId(2),
                size: size(DevicePixels(500), DevicePixels(500)),
                needs_render: true,
                scene: Some(Box::new(content)),
            });
            scene.surfaces.push(PaintSurface {
                order: 0,
                bounds: rect(0., 0., 500., 500.),
                content_mask: ContentMask {
                    bounds: rect(0., 0., 500., 500.),
                    ..Default::default()
                },
                source: PaintSurfaceSource::Layer(LayerId(2)),
                stretch: false,
            });
            scene
        };
        let mut old = wrap(layered_scene(20.));
        let mut new = wrap(layered_scene(30.));
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(136., 86., 48., 28.)));
        old.clear_damage();
        let mut replayed = Scene::default();
        replayed.surfaces = old.layers[0].scene.as_ref().unwrap().surfaces.clone();
        let mut replayed = wrap(replayed);
        replayed.reuse_layers(&mut old, false);
        replayed.update_damage(&old, false);
        let content = replayed.layers[0].scene.as_ref().unwrap();
        assert!(content.layers[0].scene.is_some());
        assert!(!content.layers[0].needs_render);
        assert_eq!(replayed.damage, SceneDamage::None);
    }

    #[test]
    fn clipped_layer_changes_still_require_upload_before_presentation() {
        let mut old = layered_scene(20.);
        old.surfaces[0].content_mask.bounds = rect(0., 0., 500., 60.);
        let mut new = layered_scene(30.);
        new.surfaces[0].content_mask = old.surfaces[0].content_mask;
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::None);
        assert!(new.layers[0].needs_render);
        assert!(matches!(
            new.layers[0].scene.as_ref().unwrap().damage,
            SceneDamage::Partial(_)
        ));

        // Presenters must process the dirty layer even when its composite is wholly clipped:
        // after presentation, a later reveal legitimately reuses that layer's existing pixels.
        new.clear_damage();
        let mut revealed = Scene::default();
        revealed.surfaces = new.surfaces.clone();
        revealed.surfaces[0].content_mask.bounds = rect(0., 0., 500., 500.);
        revealed.reuse_layers(&mut new, false);
        revealed.update_damage(&new, false);
        assert!(matches!(revealed.damage, SceneDamage::Partial(_)));
        assert!(!revealed.layers[0].needs_render);
        assert_eq!(
            revealed.layers[0].scene.as_ref().unwrap().damage,
            SceneDamage::None
        );
    }

    #[test]
    fn replay_retains_recovery_snapshot_without_copying_and_preserves_pending_upload() {
        for pending in [false, true] {
            let mut old = layered_scene(20.);
            let snapshot = old.layers[0].scene.as_deref().unwrap() as *const Scene;
            let mut next = Scene::default();
            next.surfaces = old.surfaces.clone();
            next.reuse_layers(&mut old, pending);
            assert!(old.layers.is_empty());
            assert_eq!(
                next.layers[0].scene.as_deref().unwrap() as *const Scene,
                snapshot
            );
            assert_eq!(next.layers[0].needs_render, pending);
            let mut resized = Scene::default();
            resized.surfaces = next.surfaces.clone();
            resized.layers.push(SceneLayer {
                id: LayerId(1),
                size: size(DevicePixels(300), DevicePixels(300)),
                needs_render: false,
                scene: None,
            });
            resized.reuse_layers(&mut next, pending);
            assert_eq!(
                resized.layers[0].scene.as_deref().unwrap() as *const Scene,
                snapshot
            );
            assert_eq!(
                resized.layers[0].size,
                size(DevicePixels(100), DevicePixels(100))
            );
        }
    }

    #[test]
    fn backdrop_forces_full_and_batch_culling_keeps_overlapping_background() {
        let mut scene = Scene::default();
        scene.quads.push(Quad {
            order: 0,
            bounds: rect(0., 0., 200., 200.),
            ..quad(0.)
        });
        scene.quads.push(Quad {
            order: 2,
            ..quad(100.)
        });
        scene.underlines.push(Underline {
            order: 1,
            pad: 0,
            bounds: rect(20., 20., 10., 10.),
            content_mask: quad(0.).content_mask,
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        });
        let batches: Vec<_> = scene
            .batches_for_damage(SceneDamage::Partial(rect(20., 20., 10., 10.)))
            .collect();
        assert_eq!(batches.len(), 2);
        assert!(matches!(batches[0], PrimitiveBatch::Quads(_)));
        assert!(matches!(batches[1], PrimitiveBatch::Underlines(_)));
        scene.surfaces.push(PaintSurface {
            order: 3,
            bounds: rect(10., 10., 20., 20.),
            content_mask: quad(0.).content_mask,
            source: PaintSurfaceSource::BackdropBlur(BackdropBlur {
                radius: ScaledPixels(5.),
                corner_radii: Corners::default(),
                opacity: 1.,
            }),
            stretch: false,
        });
        scene.update_damage(&Scene::default(), false);
        assert_eq!(scene.damage, SceneDamage::Full);
    }

    #[test]
    fn path_vertex_change_with_stable_bounds_is_detected() {
        let mut path = Path::new(point(Pixels(20.), Pixels(20.))).scale(1.);
        path.bounds = rect(20., 20., 10., 10.);
        path.content_mask = quad(0.).content_mask;
        path.vertices.push(PathVertex {
            xy_position: point(ScaledPixels(20.), ScaledPixels(20.)),
            st_position: point(0., 0.),
            content_mask: path.content_mask,
        });
        let mut old = Scene::default();
        old.paths.push(path.clone());
        path.vertices[0].xy_position.x = ScaledPixels(21.);
        let mut new = Scene::default();
        new.paths.push(path);
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 12., 12.)));
    }
}
