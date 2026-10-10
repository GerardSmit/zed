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
        return invalid_bounds();
    }
    // Cover fractional edges and antialiasing, including MSAA resolve/filtering at primitive edges.
    bounds
        .dilate(ScaledPixels(1.))
        .intersect(&mask.bounds.dilate(ScaledPixels(1.)))
}

fn shadow_footprint(shadow: &Shadow) -> Bounds<ScaledPixels> {
    footprint(shadow.raster_bounds(), shadow.content_mask)
}

fn sprite_footprint(
    bounds: Bounds<ScaledPixels>,
    mask: ContentMask<ScaledPixels>,
    transform: TransformationMatrix,
) -> Bounds<ScaledPixels> {
    match transform.transformed_aabb(bounds) {
        Some(transformed) => footprint(transformed, mask),
        None => invalid_bounds(),
    }
}

const MAX_CONTENT_COMPARISONS: usize = 1 << 20;
const MAX_ORDER_CHECKS: usize = 1 << 21;

fn invalid_bounds() -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(f32::NAN), ScaledPixels(f32::NAN)),
        Size::default(),
    )
}

/// Position in the renderer's paint sequence: batches interleave kinds by `(order, kind)` and
/// draw each kind's instances in array order.
type DrawKey = (DrawOrder, PrimitiveKind, usize);

#[derive(Clone, Copy)]
struct ContentEntry {
    hash: u64,
    draw: DrawKey,
}

#[derive(Clone, Copy)]
struct MatchedPrimitive {
    previous: DrawKey,
    current: DrawKey,
    footprint: Bounds<ScaledPixels>,
}

/// Buffers reused across frames by content-based damage, sized by the largest scene compared.
#[derive(Default)]
pub(super) struct DamageScratch {
    previous: Vec<ContentEntry>,
    current: Vec<ContentEntry>,
    previous_matched: Vec<bool>,
    matched: Vec<MatchedPrimitive>,
    tails: Vec<usize>,
    predecessors: Vec<usize>,
    in_order: Vec<bool>,
}

trait DamagePrimitive {
    const KIND: PrimitiveKind;
    fn order(&self) -> DrawOrder;
    /// Must agree with `same_content`: equal content implies an equal hash.
    fn content_hash(&self) -> u64;
    /// Whether two primitives rasterize identically, ignoring only their draw order.
    fn same_content(&self, other: &Self) -> bool;
    fn damage_footprint(&self) -> Bounds<ScaledPixels>;
}

fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn geometry_hash(
    kind: PrimitiveKind,
    bounds: Bounds<ScaledPixels>,
    mask: ContentMask<ScaledPixels>,
) -> u64 {
    [
        bounds.origin.x,
        bounds.origin.y,
        bounds.size.width,
        bounds.size.height,
        mask.bounds.origin.x,
        mask.bounds.origin.y,
        mask.bounds.size.width,
        mask.bounds.size.height,
    ]
    .iter()
    // Positive and negative zero compare equal, so they must hash equally.
    .fold(kind as u64, |hash, value| {
        mix(
            hash,
            if value.0 == 0. {
                0
            } else {
                value.0.to_bits() as u64
            },
        )
    })
}

macro_rules! damage_primitive {
    ($type:ty, $kind:ident, |$value:ident| $footprint:expr) => {
        impl DamagePrimitive for $type {
            const KIND: PrimitiveKind = PrimitiveKind::$kind;

            fn order(&self) -> DrawOrder {
                self.order
            }

            fn content_hash(&self) -> u64 {
                geometry_hash(Self::KIND, self.bounds, self.content_mask)
            }

            fn same_content(&self, other: &Self) -> bool {
                Self { order: 0, ..*self } == Self { order: 0, ..*other }
            }

            fn damage_footprint(&self) -> Bounds<ScaledPixels> {
                let $value = self;
                $footprint
            }
        }
    };
}

damage_primitive!(Shadow, Shadow, |value| shadow_footprint(value));
damage_primitive!(Quad, Quad, |value| footprint(
    value.bounds,
    value.content_mask
));
damage_primitive!(Shape, Shape, |value| footprint(
    value.bounds,
    value.content_mask
));
damage_primitive!(Underline, Underline, |value| footprint(
    value.bounds,
    value.content_mask
));
damage_primitive!(
    MonochromeSprite,
    MonochromeSprite,
    |value| sprite_footprint(value.bounds, value.content_mask, value.transformation)
);
damage_primitive!(SubpixelSprite, SubpixelSprite, |value| sprite_footprint(
    value.bounds,
    value.content_mask,
    value.transformation
));
damage_primitive!(PolychromeSprite, PolychromeSprite, |value| footprint(
    value.bounds,
    value.content_mask
));

impl DamagePrimitive for Path<ScaledPixels> {
    const KIND: PrimitiveKind = PrimitiveKind::Path;

    fn order(&self) -> DrawOrder {
        self.order
    }

    fn content_hash(&self) -> u64 {
        geometry_hash(Self::KIND, self.bounds, self.content_mask)
    }

    // `id` is the path's index in this scene, so it changes whenever earlier paths do.
    fn same_content(&self, other: &Self) -> bool {
        self.bounds == other.bounds
            && self.content_mask == other.content_mask
            && self.color == other.color
            && self.start == other.start
            && self.current == other.current
            && self.contour_count == other.contour_count
            && self.vertices == other.vertices
    }

    fn damage_footprint(&self) -> Bounds<ScaledPixels> {
        footprint(self.bounds, self.content_mask)
    }
}

impl DamagePrimitive for PaintSurface {
    const KIND: PrimitiveKind = PrimitiveKind::Surface;

    fn order(&self) -> DrawOrder {
        self.order
    }

    fn content_hash(&self) -> u64 {
        geometry_hash(Self::KIND, self.bounds, self.content_mask)
    }

    // Only layer composites are comparable; changes inside a layer are added separately.
    fn same_content(&self, other: &Self) -> bool {
        self.bounds == other.bounds
            && self.content_mask == other.content_mask
            && self.stretch == other.stretch
            && matches!(
                (&self.source, &other.source),
                (PaintSurfaceSource::Layer(old), PaintSurfaceSource::Layer(new)) if old == new
            )
    }

    fn damage_footprint(&self) -> Bounds<ScaledPixels> {
        footprint(self.bounds, self.content_mask)
    }
}

fn identical<T: DamagePrimitive>(previous: &[T], current: &[T]) -> bool {
    previous.len() == current.len()
        && previous
            .iter()
            .zip(current)
            .all(|(old, new)| old.order() == new.order() && old.same_content(new))
}

fn push_entries<T: DamagePrimitive>(entries: &mut Vec<ContentEntry>, primitives: &[T]) {
    entries.extend(
        primitives
            .iter()
            .enumerate()
            .map(|(index, primitive)| ContentEntry {
                hash: primitive.content_hash(),
                draw: (primitive.order(), T::KIND, index),
            }),
    );
}

fn collect_entries(scene: &Scene, entries: &mut Vec<ContentEntry>) {
    entries.clear();
    push_entries(entries, &scene.shadows);
    push_entries(entries, &scene.quads);
    push_entries(entries, &scene.shapes);
    push_entries(entries, &scene.paths);
    push_entries(entries, &scene.underlines);
    push_entries(entries, &scene.monochrome_sprites);
    push_entries(entries, &scene.subpixel_sprites);
    push_entries(entries, &scene.polychrome_sprites);
    push_entries(entries, &scene.surfaces);
    // Within a hash group, matching walks both frames in paint order so duplicates pair up
    // without reordering among themselves.
    entries.sort_unstable_by_key(|entry| (entry.hash, entry.draw));
}

fn primitive_footprint(scene: &Scene, (_, kind, index): DrawKey) -> Bounds<ScaledPixels> {
    fn get<T: DamagePrimitive>(primitives: &[T], index: usize) -> Bounds<ScaledPixels> {
        primitives
            .get(index)
            .map_or_else(invalid_bounds, DamagePrimitive::damage_footprint)
    }
    match kind {
        PrimitiveKind::Shadow => get(&scene.shadows, index),
        PrimitiveKind::Quad => get(&scene.quads, index),
        PrimitiveKind::Shape => get(&scene.shapes, index),
        PrimitiveKind::Path => get(&scene.paths, index),
        PrimitiveKind::Underline => get(&scene.underlines, index),
        PrimitiveKind::MonochromeSprite => get(&scene.monochrome_sprites, index),
        PrimitiveKind::SubpixelSprite => get(&scene.subpixel_sprites, index),
        PrimitiveKind::PolychromeSprite => get(&scene.polychrome_sprites, index),
        PrimitiveKind::Surface => get(&scene.surfaces, index),
    }
}

fn same_primitive(
    previous: &Scene,
    current: &Scene,
    (_, previous_kind, previous_index): DrawKey,
    (_, current_kind, current_index): DrawKey,
) -> bool {
    fn same<T: DamagePrimitive>(previous: &[T], current: &[T], old: usize, new: usize) -> bool {
        match (previous.get(old), current.get(new)) {
            (Some(old), Some(new)) => old.same_content(new),
            _ => false,
        }
    }
    let (old, new) = (previous_index, current_index);
    previous_kind == current_kind
        && match current_kind {
            PrimitiveKind::Shadow => same(&previous.shadows, &current.shadows, old, new),
            PrimitiveKind::Quad => same(&previous.quads, &current.quads, old, new),
            PrimitiveKind::Shape => same(&previous.shapes, &current.shapes, old, new),
            PrimitiveKind::Path => same(&previous.paths, &current.paths, old, new),
            PrimitiveKind::Underline => same(&previous.underlines, &current.underlines, old, new),
            PrimitiveKind::MonochromeSprite => same(
                &previous.monochrome_sprites,
                &current.monochrome_sprites,
                old,
                new,
            ),
            PrimitiveKind::SubpixelSprite => same(
                &previous.subpixel_sprites,
                &current.subpixel_sprites,
                old,
                new,
            ),
            PrimitiveKind::PolychromeSprite => same(
                &previous.polychrome_sprites,
                &current.polychrome_sprites,
                old,
                new,
            ),
            PrimitiveKind::Surface => same(&previous.surfaces, &current.surfaces, old, new),
        }
}

impl DamageScratch {
    /// Damage from primitives present in only one frame, plus overlaps whose paint order flipped.
    ///
    /// A pixel's value depends only on the primitives covering it and their relative paint
    /// order. Primitives drawn identically in both frames are paired up; unpaired ones damage
    /// their footprint, and paired ones whose order flipped damage the area they share.
    fn compare(&mut self, previous: &Scene, current: &Scene) -> SceneDamage {
        if identical(&previous.shadows, &current.shadows)
            && identical(&previous.quads, &current.quads)
            && identical(&previous.shapes, &current.shapes)
            && identical(&previous.paths, &current.paths)
            && identical(&previous.underlines, &current.underlines)
            && identical(&previous.monochrome_sprites, &current.monochrome_sprites)
            && identical(&previous.subpixel_sprites, &current.subpixel_sprites)
            && identical(&previous.polychrome_sprites, &current.polychrome_sprites)
            && identical(&previous.surfaces, &current.surfaces)
        {
            return SceneDamage::None;
        }

        let Self {
            previous: previous_entries,
            current: current_entries,
            previous_matched,
            matched,
            tails,
            predecessors,
            in_order,
        } = self;
        collect_entries(previous, previous_entries);
        collect_entries(current, current_entries);
        previous_matched.clear();
        previous_matched.resize(previous_entries.len(), false);
        matched.clear();

        let mut damage = SceneDamage::None;
        let mut comparisons = 0;
        let (mut previous_start, mut current_start) = (0, 0);
        loop {
            let hash = match (
                previous_entries.get(previous_start),
                current_entries.get(current_start),
            ) {
                (Some(old), Some(new)) => old.hash.min(new.hash),
                (Some(old), None) => old.hash,
                (None, Some(new)) => new.hash,
                (None, None) => break,
            };
            let previous_end = previous_start
                + previous_entries[previous_start..]
                    .iter()
                    .take_while(|entry| entry.hash == hash)
                    .count();
            let current_end = current_start
                + current_entries[current_start..]
                    .iter()
                    .take_while(|entry| entry.hash == hash)
                    .count();
            let mut first_unmatched = previous_start;
            for new in &current_entries[current_start..current_end] {
                while previous_matched
                    .get(first_unmatched)
                    .is_some_and(|matched| *matched)
                {
                    first_unmatched += 1;
                }
                let mut found = None;
                for candidate in first_unmatched..previous_end {
                    if previous_matched[candidate] {
                        continue;
                    }
                    // Past the budget, leaving primitives unpaired only over-reports damage.
                    if comparisons == MAX_CONTENT_COMPARISONS {
                        break;
                    }
                    comparisons += 1;
                    if same_primitive(
                        previous,
                        current,
                        previous_entries[candidate].draw,
                        new.draw,
                    ) {
                        found = Some(candidate);
                        break;
                    }
                }
                let footprint = primitive_footprint(current, new.draw);
                if let Some(candidate) = found {
                    previous_matched[candidate] = true;
                    matched.push(MatchedPrimitive {
                        previous: previous_entries[candidate].draw,
                        current: new.draw,
                        footprint,
                    });
                } else {
                    damage.include(footprint);
                }
            }
            previous_start = previous_end;
            current_start = current_end;
        }
        for (entry, matched) in previous_entries.iter().zip(previous_matched.iter()) {
            if !matched {
                damage.include(primitive_footprint(previous, entry.draw));
            }
        }

        matched.sort_unstable_by_key(|primitive| primitive.previous);
        if matched
            .windows(2)
            .all(|pair| pair[0].current < pair[1].current)
        {
            return damage;
        }
        // Every pair whose relative order flipped has at least one member outside a longest
        // order-preserving subsequence, so only those members need checking against the rest.
        mark_longest_in_order(matched, tails, predecessors, in_order);
        let moved = in_order.iter().filter(|kept| !**kept).count();
        let check_overlaps = moved.saturating_mul(matched.len()) <= MAX_ORDER_CHECKS;
        for (primitive, kept) in matched.iter().zip(in_order.iter()) {
            if *kept {
                continue;
            }
            if !check_overlaps {
                damage.include(primitive.footprint);
                continue;
            }
            for other in matched.iter() {
                if (primitive.previous < other.previous) == (primitive.current < other.current) {
                    continue;
                }
                if finite(primitive.footprint) && finite(other.footprint) {
                    damage.include(primitive.footprint.intersect(&other.footprint));
                } else {
                    damage.include(invalid_bounds());
                }
            }
        }
        damage
    }
}

/// Marks one longest subsequence whose current paint order is increasing, given primitives
/// sorted by their previous paint order.
fn mark_longest_in_order(
    matched: &[MatchedPrimitive],
    tails: &mut Vec<usize>,
    predecessors: &mut Vec<usize>,
    in_order: &mut Vec<bool>,
) {
    tails.clear();
    predecessors.clear();
    in_order.clear();
    in_order.resize(matched.len(), false);
    for (index, primitive) in matched.iter().enumerate() {
        let position = tails.partition_point(|tail| {
            matched
                .get(*tail)
                .is_some_and(|tail| tail.current < primitive.current)
        });
        predecessors.push(
            position
                .checked_sub(1)
                .and_then(|previous| tails.get(previous).copied())
                .unwrap_or(usize::MAX),
        );
        if let Some(tail) = tails.get_mut(position) {
            *tail = index;
        } else {
            tails.push(index);
        }
    }
    let mut index = tails.last().copied().unwrap_or(usize::MAX);
    while let Some(kept) = in_order.get_mut(index) {
        *kept = true;
        index = predecessors.get(index).copied().unwrap_or(usize::MAX);
    }
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

    /// Compare final primitive values and their overlapping paint order, without retaining another
    /// scene copy.
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
        let mut scratch = std::mem::take(&mut self.damage_scratch);
        let mut damage = scratch.compare(previous, self);
        self.damage_scratch = scratch;
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
        copy!(shapes: Shape);
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
                PrimitiveBatch::Shapes(range) => intersects!(shapes, range),
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
    use crate::{AtlasTextureKind, TileId, rgba, size};

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

    fn shape(x: f32) -> Shape {
        Shape {
            bounds: rect(x, 20., 10., 10.),
            content_mask: ContentMask {
                bounds: rect(0., 0., 200., 200.),
                ..Default::default()
            },
            transform: [1., 0., 0., 1.],
            ..Default::default()
        }
    }

    #[test]
    fn shapes_batch_in_draw_order_and_join_damage() {
        let mut old = Scene::default();
        old.insert_primitive(quad(20.));
        old.insert_primitive(shape(40.));
        old.insert_primitive(shape(60.));
        old.insert_primitive(quad(80.));
        old.finish();
        // None of the four overlap, so the bounds tree gives them one draw order and the two
        // quads share a batch; overlap is what would split them around the shapes.
        assert_eq!(
            old.batches().map(|batch| batch.label()).collect::<Vec<_>>(),
            ["quads (2)", "shapes (2)"]
        );

        let mut new = Scene::default();
        new.insert_primitive(quad(20.));
        new.insert_primitive(shape(40.));
        new.insert_primitive(Shape {
            placement: [0., 0., 0.05, 1.],
            ..shape(60.)
        });
        new.insert_primitive(quad(80.));
        new.finish();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(59., 19., 12., 12.)));
        assert_eq!(
            new.batches_for_damage(new.damage)
                .map(|batch| batch.label())
                .collect::<Vec<_>>(),
            ["shapes (2)"]
        );
        let mut staged = Scene::default();
        new.copy_primitives_for_damage(new.damage, &mut staged);
        assert_eq!(staged.shapes.len(), 1);
        assert!(staged.quads.is_empty());
    }

    #[test]
    fn movement_and_removal_include_old_pixels() {
        let mut old = Scene::default();
        old.quads.push(quad(20.));
        let mut new = Scene::default();
        new.quads.push(quad(30.));
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 22., 12.)));
        new.quads.clear();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(19., 19., 12., 12.)));
        // A draw order value alone changes no pixels unless another primitive overlaps.
        new.quads.push(Quad {
            order: 1,
            ..quad(20.)
        });
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::None);
    }

    fn glyph(x: f32, tile_id: u32) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: rect(x, 20., 8., 16.),
            content_mask: ContentMask {
                bounds: rect(0., 0., 1000., 200.),
                ..Default::default()
            },
            color: Hsla::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(tile_id),
                padding: 0,
                bounds: Bounds::default(),
            },
            transformation: TransformationMatrix::unit(),
        }
    }

    fn text_scene(glyphs: impl IntoIterator<Item = MonochromeSprite>) -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds: rect(0., 0., 1000., 100.),
            content_mask: glyph(0., 0).content_mask,
            ..quad(0.)
        });
        for glyph in glyphs {
            scene.insert_primitive(glyph);
        }
        scene.finish();
        scene
    }

    #[test]
    fn changing_one_glyph_damages_only_that_glyph() {
        // Glyph boxes overlap their neighbours, so a changed box would also renumber later orders.
        let tiles = |changed: u32| {
            (0..80).map(move |index| {
                let tile = if index == 40 { changed } else { index % 10 };
                glyph(index as f32 * 7., tile)
            })
        };
        let old = text_scene(tiles(3));
        let mut new = text_scene(tiles(4));
        new.update_damage(&old, false);
        assert_eq!(
            new.damage,
            SceneDamage::Partial(footprint(
                glyph(280., 0).bounds,
                glyph(280., 0).content_mask
            ))
        );

        let narrower = |index: usize| {
            let mut value = glyph(index as f32 * 7., index as u32 % 10);
            if index == 40 {
                value.bounds.size.width = ScaledPixels(5.);
            }
            value
        };
        let mut new = text_scene((0..80).map(narrower));
        assert_ne!(
            new.monochrome_sprites
                .iter()
                .map(|sprite| sprite.order)
                .collect::<Vec<_>>(),
            old.monochrome_sprites
                .iter()
                .map(|sprite| sprite.order)
                .collect::<Vec<_>>()
        );
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(279., 19., 10., 18.)));
    }

    fn wide_quad(x: f32) -> Quad {
        Quad {
            content_mask: glyph(0., 0).content_mask,
            ..quad(x)
        }
    }

    #[test]
    fn inserting_a_primitive_damages_only_its_footprint() {
        let row = |inserted: bool| {
            let mut scene = Scene::default();
            for index in 0..20 {
                scene.insert_primitive(wide_quad(index as f32 * 40.));
                if inserted && index == 9 {
                    scene.insert_primitive(Underline {
                        order: 0,
                        pad: 0,
                        bounds: rect(380., 50., 30., 2.),
                        content_mask: wide_quad(0.).content_mask,
                        color: Hsla::default(),
                        thickness: ScaledPixels(1.),
                        wavy: false.into(),
                    });
                    scene.insert_primitive(Quad {
                        bounds: rect(380., 60., 10., 10.),
                        ..wide_quad(0.)
                    });
                }
            }
            scene.finish();
            scene
        };
        let old = row(false);
        let mut new = row(true);
        new.update_damage(&old, false);
        let viewport = size(DevicePixels(1000), DevicePixels(200));
        let mut rects: Vec<_> = new
            .damage
            .pixel_rects(viewport)
            .map(|bounds| (bounds.origin.y.0, bounds.size.width.0, bounds.size.height.0))
            .collect();
        rects.sort();
        assert_eq!(rects, [(49, 32, 4), (59, 12, 12)]);
        assert!(
            new.damage
                .pixel_rects(viewport)
                .all(|bounds| bounds.origin.x.0 == 379)
        );
    }

    #[test]
    fn moving_a_primitive_damages_old_and_new_bounds_separately() {
        let scene = |moved: f32| {
            let mut scene = Scene::default();
            for x in [0., 40., 80., 400.] {
                scene.insert_primitive(wide_quad(x));
            }
            scene.insert_primitive(wide_quad(moved));
            scene.finish();
            scene
        };
        let old = scene(150.);
        let mut new = scene(300.);
        new.update_damage(&old, false);
        let viewport = size(DevicePixels(1000), DevicePixels(200));
        let rects: Vec<_> = new.damage.pixel_rects(viewport).collect();
        assert_eq!(rects.len(), 2);
        assert!(new.damage.intersects(rect(150., 20., 10., 10.)));
        assert!(new.damage.intersects(rect(300., 20., 10., 10.)));
        assert!(!new.damage.intersects(rect(200., 20., 10., 10.)));
        assert!(!new.damage.intersects(rect(80., 20., 10., 10.)));
    }

    #[test]
    fn swapping_overlapping_primitives_damages_their_overlap() {
        let first = Quad {
            bounds: rect(20., 20., 20., 20.),
            background: rgba(0xff0000ff).into(),
            ..quad(0.)
        };
        let second = Quad {
            bounds: rect(30., 30., 20., 20.),
            background: rgba(0x0000ffff).into(),
            ..quad(0.)
        };
        let far = |order| Quad {
            order,
            bounds: rect(150., 150., 10., 10.),
            ..quad(0.)
        };
        let mut old = Scene::default();
        old.quads = vec![
            far(0),
            Quad { order: 1, ..first },
            Quad { order: 2, ..second },
        ];
        let mut new = Scene::default();
        // The distant quad also changes relative order, but overlaps neither.
        new.quads = vec![
            Quad { order: 1, ..second },
            Quad { order: 2, ..first },
            far(3),
        ];
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(29., 29., 12., 12.)));

        // Kinds interleave by draw order, so a sprite moving under a quad is a swap too.
        let sprite = MonochromeSprite {
            bounds: rect(25., 25., 8., 8.),
            ..glyph(0., 1)
        };
        let mut old = Scene::default();
        old.quads = vec![Quad { order: 1, ..first }];
        old.monochrome_sprites = vec![MonochromeSprite { order: 2, ..sprite }];
        let mut new = Scene::default();
        new.monochrome_sprites = vec![MonochromeSprite { order: 1, ..sprite }];
        new.quads = vec![Quad { order: 2, ..first }];
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::Partial(rect(24., 24., 10., 10.)));
    }

    #[test]
    fn identical_frames_have_no_damage() {
        let build = || text_scene((0..50).map(|index| glyph(index as f32 * 7., index % 7)));
        let old = build();
        let mut new = build();
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::None);

        // Duplicates pair up, and a frame re-sorted only among non-overlapping primitives is
        // unchanged on screen.
        let mut old = Scene::default();
        old.quads = vec![
            quad(20.),
            quad(20.),
            Quad {
                order: 1,
                ..quad(60.)
            },
        ];
        let mut new = Scene::default();
        new.quads = vec![
            Quad {
                order: 0,
                ..quad(60.)
            },
            Quad {
                order: 4,
                ..quad(20.)
            },
            Quad {
                order: 4,
                ..quad(20.)
            },
        ];
        new.update_damage(&old, false);
        assert_eq!(new.damage, SceneDamage::None);
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
    fn rounded_clips_retain_shadow_tails_and_limit_inset_shadows() {
        let shadow = Shadow {
            order: 0,
            blur_radius: ScaledPixels(4.),
            bounds: rect(30., 30., 10., 10.),
            corner_radii: Corners::default(),
            content_mask: ContentMask {
                bounds: rect(0., 0., 200., 200.),
                ..Default::default()
            },
            color: rgba(0x00000040).into(),
            element_bounds: rect(28., 28., 14., 14.),
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
        };
        let mut scene = Scene::default();
        scene.push_rounded_clip(vec![rect(0., 0., 200., 200.)]);
        scene.insert_primitive(shadow);
        assert_eq!(
            scene.shadows[0].content_mask.bounds,
            rect(18., 18., 34., 34.)
        );
        // A nested clip can intersect only the blur, with the shape itself fully outside it.
        scene.push_rounded_clip(vec![rect(16., 16., 10., 10.)]);
        scene.insert_primitive(shadow);
        assert_eq!(scene.shadows.len(), 2);
        assert_eq!(scene.shadows[1].content_mask.bounds, rect(18., 18., 8., 8.));
        scene.pop_rounded_clip();
        scene.insert_primitive(Shadow {
            inset: 1,
            ..shadow
        });
        assert_eq!(scene.shadows[2].content_mask.bounds, shadow.element_bounds);

        // The tail also participates in ordering outside rounded containers.
        let mut scene = Scene::default();
        scene.insert_primitive(shadow);
        scene.insert_primitive(Quad {
            bounds: rect(19., 19., 5., 5.),
            ..quad(19.)
        });
        assert!(scene.quads[0].order > scene.shadows[0].order);
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
