/* Functions useful for debugging:

// A heat map color for debugging (blue -> cyan -> green -> yellow -> red).
fn heat_map_color(value: f32, minValue: f32, maxValue: f32, position: vec2<f32>) -> vec4<f32> {
    // Normalize value to 0-1 range
    let t = clamp((value - minValue) / (maxValue - minValue), 0.0, 1.0);

    // Heat map color calculation
    let r = t * t;
    let g = 4.0 * t * (1.0 - t);
    let b = (1.0 - t) * (1.0 - t);
    let heat_color = vec3<f32>(r, g, b);

    // Create a checkerboard pattern (black and white)
    let sum = floor(position.x / 3) + floor(position.y / 3);
    let is_odd = fract(sum * 0.5); // 0.0 for even, 0.5 for odd
    let checker_value = is_odd * 2.0; // 0.0 for even, 1.0 for odd
    let checker_color = vec3<f32>(checker_value);

    // Determine if value is in range (1.0 if in range, 0.0 if out of range)
    let in_range = step(minValue, value) * step(value, maxValue);

    // Mix checkerboard and heat map based on whether value is in range
    let final_color = mix(checker_color, heat_color, in_range);

    return vec4<f32>(final_color, 1.0);
}

*/

// Contrast and gamma correction adapted from https://github.com/microsoft/terminal/blob/1283c0f5b99a2961673249fa77c6b986efb5086c/src/renderer/atlas/dwrite.hlsl
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
fn color_brightness(color: vec3<f32>) -> f32 {
    // REC. 601 luminance coefficients for perceived brightness
    return dot(color, vec3<f32>(0.30, 0.59, 0.11));
}

fn light_on_dark_contrast(enhancedContrast: f32, color: vec3<f32>) -> f32 {
    let brightness = color_brightness(color);
    let multiplier = saturate(4.0 * (0.75 - brightness));
    return enhancedContrast * multiplier;
}

fn enhance_contrast(alpha: f32, k: f32) -> f32 {
    return alpha * (k + 1.0) / (alpha * k + 1.0);
}

fn enhance_contrast3(alpha: vec3<f32>, k: f32) -> vec3<f32> {
    return alpha * (k + 1.0) / (alpha * k + 1.0);
}

fn apply_alpha_correction(a: f32, b: f32, g: vec4<f32>) -> f32 {
    let brightness_adjustment = g.x * b + g.y;
    let correction = brightness_adjustment * a + (g.z * b + g.w);
    return a + a * (1.0 - a) * correction;
}

fn apply_alpha_correction3(a: vec3<f32>, b: vec3<f32>, g: vec4<f32>) -> vec3<f32> {
    let brightness_adjustment = g.x * b + g.y;
    let correction = brightness_adjustment * a + (g.z * b + g.w);
    return a + a * (1.0 - a) * correction;
}

fn apply_contrast_and_gamma_correction(sample: f32, color: vec3<f32>, enhanced_contrast_factor: f32, gamma_ratios: vec4<f32>) -> f32 {
    let enhanced_contrast = light_on_dark_contrast(enhanced_contrast_factor, color);
    let brightness = color_brightness(color);

    let contrasted = enhance_contrast(sample, enhanced_contrast);
    return apply_alpha_correction(contrasted, brightness, gamma_ratios);
}

fn apply_contrast_and_gamma_correction3(sample: vec3<f32>, color: vec3<f32>, enhanced_contrast_factor: f32, gamma_ratios: vec4<f32>) -> vec3<f32> {
    let enhanced_contrast = light_on_dark_contrast(enhanced_contrast_factor, color);

    let contrasted = enhance_contrast3(sample, enhanced_contrast);
    return apply_alpha_correction3(contrasted, color, gamma_ratios);
}

struct GlobalParams {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

struct GammaParams {
    gamma_ratios: vec4<f32>,
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> gamma_params: GammaParams;
@group(2) @binding(0) var t_sprite: texture_2d<f32>;
@group(2) @binding(1) var s_sprite: sampler;

@group(0) @binding(0) var t_frame: texture_2d<f32>;

@vertex
fn vs_frame(@builtin(vertex_index) vertex_id: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vertex_id % 2u) * 2.0 - 1.0,
                     1.0 - f32(vertex_id / 2u) * 2.0, 0.0, 1.0);
}

@fragment
fn fs_frame(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(t_frame, vec2<i32>(position.xy), 0);
}

@fragment
fn fs_clear_frame() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

const M_PI_F: f32 = 3.1415926;
const GRAYSCALE_FACTORS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct Corners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

struct Edges {
    top: f32,
    right: f32,
    bottom: f32,
    left: f32,
}

struct Hsla {
    h: f32,
    s: f32,
    l: f32,
    a: f32,
}

struct LinearColorStop {
    color: Hsla,
    percentage: f32,
}

struct Background {
    // 0u is Solid
    // 1u is LinearGradient
    // 2u is PatternSlash
    // 3u is Checkerboard
    tag: u32,
    // 0u is sRGB linear color
    // 1u is Oklab color
    color_space: u32,
    solid: Hsla,
    gradient_angle_or_pattern_height: f32,
    colors: array<LinearColorStop, 2>,
    pad: u32,
}

struct AtlasTextureId {
    index: u32,
    kind: u32,
}

struct AtlasBounds {
    origin: vec2<i32>,
    size: vec2<i32>,
}

struct AtlasTile {
    texture_id: AtlasTextureId,
    tile_id: u32,
    padding: u32,
    bounds: AtlasBounds,
}

struct TransformationMatrix {
    rotation_scale: mat2x2<f32>,
    translation: vec2<f32>,
}

fn to_device_position_impl(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

fn to_device_position(unit_vertex: vec2<f32>, bounds: Bounds) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    return to_device_position_impl(position);
}

fn to_device_position_transformed(unit_vertex: vec2<f32>, bounds: Bounds, transform: TransformationMatrix) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    //Note: Rust side stores it as row-major, so transposing here
    let transformed = transpose(transform.rotation_scale) * position + transform.translation;
    return to_device_position_impl(transformed);
}

fn to_tile_position(unit_vertex: vec2<f32>, tile: AtlasTile) -> vec2<f32> {
  let atlas_size = vec2<f32>(textureDimensions(t_sprite, 0));
  return (vec2<f32>(tile.bounds.origin) + unit_vertex * vec2<f32>(tile.bounds.size)) / atlas_size;
}

fn distance_from_clip_rect_impl(position: vec2<f32>, clip_bounds: Bounds) -> vec4<f32> {
    let tl = position - clip_bounds.origin;
    let br = clip_bounds.origin + clip_bounds.size - position;
    return vec4<f32>(tl.x, br.x, tl.y, br.y);
}

fn distance_from_clip_rect(unit_vertex: vec2<f32>, bounds: Bounds, clip_bounds: Bounds) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    return distance_from_clip_rect_impl(position, clip_bounds);
}

// The content mask's vertical fade (`gpui::ContentFade`): absolute device-pixel edges, alpha 0 at
// `top`/`bottom` and 1 at `*_len` inside them. It follows the mask's `Bounds` in every record, as
// the Rust `ContentMask { bounds, fade }` does. Four scalars rather than a `vec4`, so it keeps the
// Rust struct's 4-byte alignment in a storage buffer.
struct ContentFade {
    top: f32,
    top_len: f32,
    bottom: f32,
    bottom_len: f32,
}

fn fade_vector(fade: ContentFade) -> vec4<f32> {
    return vec4<f32>(fade.top, fade.top_len, fade.bottom, fade.bottom_len);
}

// The alpha a fade leaves at window height `y`; `fade` is `fade_vector`'s packing.
fn fade_alpha(y: f32, fade: vec4<f32>) -> f32 {
    var alpha = 1.0;
    if (fade.y > 0.0) {
        alpha *= saturate((y - fade.x) / fade.y);
    }
    if (fade.w > 0.0) {
        alpha *= saturate((fade.z - y) / fade.w);
    }
    return alpha;
}

fn distance_from_clip_rect_transformed(unit_vertex: vec2<f32>, bounds: Bounds, clip_bounds: Bounds, transform: TransformationMatrix) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    let transformed = transpose(transform.rotation_scale) * position + transform.translation;
    return distance_from_clip_rect_impl(transformed, clip_bounds);
}

// https://gamedev.stackexchange.com/questions/92015/optimized-linear-to-srgb-glsl
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = srgb / vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

fn srgb_to_linear_component(a: f32) -> f32 {
    let cutoff = a < 0.04045;
    let higher = pow((a + 0.055) / 1.055, 2.4);
    let lower = a / 12.92;
    return select(higher, lower, cutoff);
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let lower = linear * vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

/// Convert a linear color to sRGBA space.
fn linear_to_srgba(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(linear_to_srgb(color.rgb), color.a);
}

/// Convert a sRGBA color to linear space.
fn srgba_to_linear(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(srgb_to_linear(color.rgb), color.a);
}

/// Hsla to linear RGBA conversion.
fn hsla_to_rgba(hsla: Hsla) -> vec4<f32> {
    let h = hsla.h * 6.0; // Now, it's an angle but scaled in [0, 6) range
    let s = hsla.s;
    let l = hsla.l;
    let a = hsla.a;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(h % 2.0 - 1.0));
    let m = l - c / 2.0;
    var color = vec3<f32>(m);

    if (h >= 0.0 && h < 1.0) {
        color.r += c;
        color.g += x;
    } else if (h >= 1.0 && h < 2.0) {
        color.r += x;
        color.g += c;
    } else if (h >= 2.0 && h < 3.0) {
        color.g += c;
        color.b += x;
    } else if (h >= 3.0 && h < 4.0) {
        color.g += x;
        color.b += c;
    } else if (h >= 4.0 && h < 5.0) {
        color.r += x;
        color.b += c;
    } else {
        color.r += c;
        color.b += x;
    }

    return vec4<f32>(color, a);
}

/// Convert a linear sRGB to Oklab space.
/// Reference: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
fn linear_srgb_to_oklab(color: vec4<f32>) -> vec4<f32> {
	let l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
	let m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
	let s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

	let l_ = pow(l, 1.0 / 3.0);
	let m_ = pow(m, 1.0 / 3.0);
	let s_ = pow(s, 1.0 / 3.0);

	return vec4<f32>(
		0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
		1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
		0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
		color.a
	);
}

/// Convert an Oklab color to linear sRGB space.
fn oklab_to_linear_srgb(color: vec4<f32>) -> vec4<f32> {
	let l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
	let m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
	let s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

	let l = l_ * l_ * l_;
	let m = m_ * m_ * m_;
	let s = s_ * s_ * s_;

	return vec4<f32>(
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
		color.a
	);
}

fn over(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let alpha = above.a + below.a * (1.0 - above.a);
    let color = (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
    return vec4<f32>(color, alpha);
}

// A standard gaussian function, used for weighting samples
fn gaussian(x: f32, sigma: f32) -> f32{
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI_F) * sigma);
}

// This approximates the error function, needed for the gaussian integral
fn erf(v: vec2<f32>) -> vec2<f32> {
    let s = sign(v);
    let a = abs(v);
    let r1 = 1.0 + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
    let r2 = r1 * r1;
    return s - s / (r2 * r2);
}

fn blur_along_x(x: f32, y: f32, sigma: f32, corner: f32, half_size: vec2<f32>) -> f32 {
  let delta = min(half_size.y - corner - abs(y), 0.0);
  let curved = half_size.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
  let integral = 0.5 + 0.5 * erf((x + vec2<f32>(-curved, curved)) * (sqrt(0.5) / sigma));
  return integral.y - integral.x;
}

// Selects corner radius based on quadrant.
fn pick_corner_radius(center_to_point: vec2<f32>, radii: Corners) -> f32 {
    if (center_to_point.x < 0.0) {
        if (center_to_point.y < 0.0) {
            return radii.top_left;
        } else {
            return radii.bottom_left;
        }
    } else {
        if (center_to_point.y < 0.0) {
            return radii.top_right;
        } else {
            return radii.bottom_right;
        }
    }
}

// Signed distance of the point to the quad's border - positive outside the
// border, and negative inside.
//
// See comments on similar code using `quad_sdf_impl` in `fs_quad` for
// explanation.
fn quad_sdf(point: vec2<f32>, bounds: Bounds, corner_radii: Corners) -> f32 {
    let half_size = bounds.size / 2.0;
    let center = bounds.origin + half_size;
    let center_to_point = point - center;
    let corner_radius = pick_corner_radius(center_to_point, corner_radii);
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

fn quad_sdf_impl(corner_center_to_point: vec2<f32>, corner_radius: f32) -> f32 {
    if (corner_radius == 0.0) {
        // Fast path for unrounded corners.
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        // Signed distance of the point from a quad that is inset by corner_radius.
        // It is negative inside this quad, and positive outside.
        let signed_distance_to_inset_quad =
            // 0 inside the inset quad, and positive outside.
            length(max(vec2<f32>(0.0), corner_center_to_point)) +
            // 0 outside the inset quad, and negative inside.
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));

        return signed_distance_to_inset_quad - corner_radius;
    }
}

// Abstract away the final color transformation based on the
// target alpha compositing mode.
fn blend_color(color: vec4<f32>, alpha_factor: f32) -> vec4<f32> {
    let alpha = color.a * alpha_factor;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    return vec4<f32>(color.rgb * multiplier, alpha);
}

// Scale an already-blended colour by a fade: its colour too when the target is premultiplied.
fn apply_fade(color: vec4<f32>, fade: f32) -> vec4<f32> {
    let multiplier = select(1.0, fade, globals.premultiplied_alpha != 0u);
    return vec4<f32>(color.rgb * multiplier, color.a * fade);
}


struct GradientColor {
    solid: vec4<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
}

fn prepare_gradient_color(tag: u32, color_space: u32,
    solid: Hsla, colors: array<LinearColorStop, 2>) -> GradientColor {
    var result = GradientColor();

    if (tag == 0u || tag == 2u || tag == 3u) {
        result.solid = hsla_to_rgba(solid);
    } else if (tag == 1u) {
        // The hsla_to_rgba is returns a linear sRGB color
        result.color0 = hsla_to_rgba(colors[0].color);
        result.color1 = hsla_to_rgba(colors[1].color);

        // Prepare color space in vertex for avoid conversion
        // in fragment shader for performance reasons
        if (color_space == 0u) {
            // sRGB
            result.color0 = linear_to_srgba(result.color0);
            result.color1 = linear_to_srgba(result.color1);
        } else if (color_space == 1u) {
            // Oklab
            result.color0 = linear_srgb_to_oklab(result.color0);
            result.color1 = linear_srgb_to_oklab(result.color1);
        }
    }

    return result;
}

fn gradient_color(background: Background, position: vec2<f32>, bounds: Bounds,
    solid_color: vec4<f32>, color0: vec4<f32>, color1: vec4<f32>) -> vec4<f32> {
    var background_color = vec4<f32>(0.0);

    switch (background.tag) {
        default: {
            return solid_color;
        }
        case 1u: {
            // Linear gradient background.
            // -90 degrees to match the CSS gradient angle.
            let angle = background.gradient_angle_or_pattern_height;
            let radians = (angle % 360.0 - 90.0) * M_PI_F / 180.0;
            var direction = vec2<f32>(cos(radians), sin(radians));
            let stop0_percentage = background.colors[0].percentage;
            let stop1_percentage = background.colors[1].percentage;

            // Expand the short side to be the same as the long side
            if (bounds.size.x > bounds.size.y) {
                direction.y *= bounds.size.y / bounds.size.x;
            } else {
                direction.x *= bounds.size.x / bounds.size.y;
            }

            // Get the t value for the linear gradient with the color stop percentages.
            let half_size = bounds.size / 2.0;
            let center = bounds.origin + half_size;
            let center_to_point = position - center;
            var t = dot(center_to_point, direction) / length(direction);
            // Check the direct to determine the use x or y
            if (abs(direction.x) > abs(direction.y)) {
                t = (t + half_size.x) / bounds.size.x;
            } else {
                t = (t + half_size.y) / bounds.size.y;
            }

            // Adjust t based on the stop percentages
            t = (t - stop0_percentage) / (stop1_percentage - stop0_percentage);
            t = clamp(t, 0.0, 1.0);

            switch (background.color_space) {
                default: {
                    background_color = srgba_to_linear(mix(color0, color1, t));
                }
                case 1u: {
                    let oklab_color = mix(color0, color1, t);
                    background_color = oklab_to_linear_srgb(oklab_color);
                }
            }
        }
        case 2u: {
            // pattern slash
            let gradient_angle_or_pattern_height = background.gradient_angle_or_pattern_height;
            let pattern_width = (gradient_angle_or_pattern_height / 65535.0f) / 255.0f;
            let pattern_interval = (gradient_angle_or_pattern_height % 65535.0f) / 255.0f;
            let pattern_height = pattern_width + pattern_interval;
            let stripe_angle = M_PI_F / 4.0;
            let pattern_period = pattern_height * sin(stripe_angle);
            let rotation = mat2x2<f32>(
                cos(stripe_angle), -sin(stripe_angle),
                sin(stripe_angle), cos(stripe_angle)
            );
            let relative_position = position - bounds.origin;
            let rotated_point = rotation * relative_position;
            let pattern = rotated_point.x % pattern_period;
            let distance = min(pattern, pattern_period - pattern) - pattern_period * (pattern_width / pattern_height) /  2.0f;
            background_color = solid_color;
            background_color.a *= saturate(0.5 - distance);
        }
        case 3u: {
            // checkerboard
            let size = background.gradient_angle_or_pattern_height;
            let relative_position = position - bounds.origin;

            let x_index = floor(relative_position.x / size);
            let y_index = floor(relative_position.y / size);
            let should_be_colored = (x_index + y_index) % 2.0;

            background_color = solid_color;
            background_color.a *= saturate(should_be_colored);
        }
    }

    return background_color;
}

// --- quads --- //

struct Quad {
    order: u32,
    border_style: u32,
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    background: Background,
    border_color: Hsla,
    corner_radii: Corners,
    border_widths: Edges,
}

struct QuadVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) border_color: vec4<f32>,
    @location(1) @interpolate(flat) quad_id: u32,
    @location(2) @interpolate(flat) clip_edges: vec4<f32>,
    @location(3) @interpolate(flat) background_solid: vec4<f32>,
    @location(4) @interpolate(flat) background_color0: vec4<f32>,
    @location(5) @interpolate(flat) background_color1: vec4<f32>,
    // Bounds origin and half size, and the largest corner radius and per-axis reduced border
    // width, for `quad_interior`. Quads that are not solid and unfaded never qualify.
    @location(6) @interpolate(flat) interior_frame: vec4<f32>,
    @location(7) @interpolate(flat) interior_insets: vec3<f32>,
}

@vertex
fn vs_quad(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> QuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = load_quad(instance_id);

    var out = QuadVarying();
    out.position = to_device_position(unit_vertex, quad.bounds);

    let gradient = prepare_gradient_color(
        quad.background.tag,
        quad.background.color_space,
        quad.background.solid,
        quad.background.colors
    );
    out.background_solid = gradient.solid;
    out.background_color0 = gradient.color0;
    out.background_color1 = gradient.color1;
    out.border_color = hsla_to_rgba(quad.border_color);
    out.quad_id = instance_id;
    out.clip_edges = edges_of(quad.content_mask);
    let radii = quad.corner_radii;
    let borders = quad.border_widths;
    // `!(x > 0.0)` matches `fade_alpha`, which skips a fade unless its length is positive.
    let eligible = quad.background.tag == 0u &&
        !(quad.content_fade.top_len > 0.0) && !(quad.content_fade.bottom_len > 0.0);
    out.interior_frame = vec4<f32>(quad.bounds.origin, quad.bounds.size / 2.0);
    out.interior_insets = vec3<f32>(
        max(max(radii.top_left, radii.top_right), max(radii.bottom_right, radii.bottom_left)),
        select(vec2<f32>(1e30),
            max(reduced_border(vec2<f32>(borders.left, borders.top)),
                reduced_border(vec2<f32>(borders.right, borders.bottom))),
            eligible));
    return out;
}

// `quad_color` replaces zero border widths by this, so that they draw no antialiasing.
fn reduced_border(border: vec2<f32>) -> vec2<f32> {
    return select(border, vec2<f32>(-0.5), border == vec2<f32>(0.0));
}

// Whether `quad_color` takes its background fast path at this fragment: within the inner edge
// of the widest border on each axis and outside the largest corner radius on some axis. Uses
// the same arithmetic on the same values, so a passing fragment gets exactly that result.
fn quad_interior(input: QuadVarying) -> bool {
    let center_to_point = input.position.xy - input.interior_frame.xy - input.interior_frame.zw;
    let corner_to_point = abs(center_to_point) - input.interior_frame.zw;
    let border = corner_to_point + input.interior_insets.yz;
    let corner = corner_to_point + input.interior_insets.x;
    return all(border < vec2<f32>(-0.5)) && any(corner < vec2<f32>(0.0));
}

// Left, top, right and bottom edges; the right and bottom ones as `to_device_position` computes
// them.
fn edges_of(bounds: Bounds) -> vec4<f32> {
    return vec4<f32>(bounds.origin, bounds.size + bounds.origin);
}

// Whether a fragment lies outside a content mask given by `edges_of`; one on an edge is inside.
// Quads test this exactly instead of interpolating clip distances, so that drawing one in parts
// (`OpaqueQuadPlan`) changes no pixel on the edge of its mask.
fn outside_clip(position: vec2<f32>, edges: vec4<f32>) -> bool {
    return any(position < edges.xy) || any(position > edges.zw);
}

@fragment
fn fs_quad(input: QuadVarying) -> @location(0) vec4<f32> {
    // Alpha clip first, since we don't have `clip_distance`.
    if (outside_clip(input.position.xy, input.clip_edges)) {
        return vec4<f32>(0.0);
    }
    if (quad_interior(input)) {
        return blend_color(input.background_solid, 1.0);
    }
    let quad = load_quad(input.quad_id);
    let fade = fade_alpha(input.position.y, fade_vector(quad.content_fade));
    return apply_fade(quad_color(input, quad), fade);
}

// Quads with a solid background, no border and no fade, drawn without `fs_quad`'s border,
// gradient and fade code. `fs_quad_simple` repeats `quad_color`'s arithmetic for zero border
// widths, so it writes the same pixels.
struct SimpleQuadVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) clip_edges: vec4<f32>,
    @location(1) @interpolate(flat) background: vec4<f32>,
    @location(2) @interpolate(flat) border_color: vec4<f32>,
    // Bounds origin and half size.
    @location(3) @interpolate(flat) frame: vec4<f32>,
    // Top left, top right, bottom right and bottom left corner radii.
    @location(4) @interpolate(flat) corner_radii: vec4<f32>,
}

@vertex
fn vs_quad_simple(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> SimpleQuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = load_quad(instance_id);
    return simple_quad_varying(unit_vertex * quad.bounds.size + quad.bounds.origin, quad);
}

// `vs_quad_simple`'s output at a vertex placed at `position`, in device pixels.
fn simple_quad_varying(position: vec2<f32>, quad: Quad) -> SimpleQuadVarying {
    var out = SimpleQuadVarying();
    out.position = to_device_position_impl(position);
    out.clip_edges = edges_of(quad.content_mask);
    out.background = hsla_to_rgba(quad.background.solid);
    out.border_color = hsla_to_rgba(quad.border_color);
    out.frame = vec4<f32>(quad.bounds.origin, quad.bounds.size / 2.0);
    let radii = quad.corner_radii;
    out.corner_radii = vec4<f32>(radii.top_left, radii.top_right, radii.bottom_right, radii.bottom_left);
    return out;
}

@fragment
fn fs_quad_simple(input: SimpleQuadVarying) -> @location(0) vec4<f32> {
    if (outside_clip(input.position.xy, input.clip_edges)) {
        return vec4<f32>(0.0);
    }
    let background_color = input.background;
    if (all(input.corner_radii == vec4<f32>(0.0))) {
        return blend_color(background_color, 1.0);
    }
    let antialias_threshold = 0.5;
    let half_size = input.frame.zw;
    let center_to_point = input.position.xy - input.frame.xy - half_size;
    let radii = input.corner_radii;
    let corner_radius = pick_corner_radius(center_to_point, Corners(radii.x, radii.y, radii.z, radii.w));
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;
    let is_near_rounded_corner =
            corner_center_to_point.x >= 0 &&
            corner_center_to_point.y >= 0;
    // `quad_color`'s `reduced_border` is `-antialias_threshold` on both axes.
    let reduced_border = -antialias_threshold;
    let straight_border_inner_corner_to_point = corner_to_point + reduced_border;
    let is_beyond_inner_straight_border =
            straight_border_inner_corner_to_point.x > 0 ||
            straight_border_inner_corner_to_point.y > 0;
    let is_within_inner_straight_border =
        straight_border_inner_corner_to_point.x < -antialias_threshold &&
        straight_border_inner_corner_to_point.y < -antialias_threshold;
    if (is_within_inner_straight_border && !is_near_rounded_corner) {
        return blend_color(background_color, 1.0);
    }
    let outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);
    var inner_sdf = 0.0;
    if (corner_center_to_point.x <= 0 || corner_center_to_point.y <= 0) {
        inner_sdf = -max(straight_border_inner_corner_to_point.x,
                         straight_border_inner_corner_to_point.y);
    } else if (is_beyond_inner_straight_border) {
        inner_sdf = -1.0;
    } else {
        inner_sdf = -(outer_sdf + reduced_border);
    }
    let border_sdf = max(inner_sdf, outer_sdf);
    var color = background_color;
    if (border_sdf < antialias_threshold) {
        let blended_border = over(background_color, input.border_color);
        color = mix(background_color, blended_border,
                    saturate(antialias_threshold - inner_sdf));
    }
    return blend_color(color, saturate(antialias_threshold - outer_sdf));
}

// A simple quad drawn over `raster` only, a part of its bounds given by its left, top, right and
// bottom edges in device pixels. Within it,
// `vs_quad_clamped` with `fs_quad_simple` and `vs_quad_opaque` with `fs_quad_opaque` draw what
// `vs_quad_simple` with `fs_quad_simple` would; see `OpaqueQuadPlan` in `wgpu_renderer.rs`.
struct ClampedQuad {
    raster: vec4<f32>,
    quad: Quad,
}

fn clamped_quad_corner(unit_vertex: vec2<f32>, raster: vec4<f32>) -> vec2<f32> {
    return select(raster.xy, raster.zw, unit_vertex > vec2<f32>(0.5));
}

@vertex
fn vs_quad_clamped(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> SimpleQuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let clamped = load_clamped_quad(instance_id);
    return simple_quad_varying(clamped_quad_corner(unit_vertex, clamped.raster), clamped.quad);
}

struct OpaqueQuadVarying {
    @builtin(position) position: vec4<f32>,
    // Already blended for the target's alpha mode, so the fragment shader reads no uniform.
    @location(0) @interpolate(flat) background: vec4<f32>,
}

// The interior of an opaque quad, where `fs_quad_simple` returns the background unchanged.
// Drawn without blending, so tile-based GPUs can drop the work it hides.
@vertex
fn vs_quad_opaque(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> OpaqueQuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let clamped = load_clamped_quad(instance_id);
    var out = OpaqueQuadVarying();
    out.position = to_device_position_impl(clamped_quad_corner(unit_vertex, clamped.raster));
    out.background = blend_color(hsla_to_rgba(clamped.quad.background.solid), 1.0);
    return out;
}

@fragment
fn fs_quad_opaque(input: OpaqueQuadVarying) -> @location(0) vec4<f32> {
    return input.background;
}

// Debug: viewport-sized rectangles, one per instance, in a colour that differs between
// neighbouring instances.
@vertex
fn vs_fill(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> OpaqueQuadVarying {
    var out = OpaqueQuadVarying();
    out.position = vec4<f32>(f32(vertex_id % 2u) * 2.0 - 1.0, 1.0 - f32(vertex_id / 2u) * 2.0, 0.0, 1.0);
    let shade = vec3<f32>(f32(instance_id & 1u), f32((instance_id >> 1u) & 1u), f32((instance_id >> 2u) & 1u));
    out.background = vec4<f32>(0.25 + 0.5 * shade, 1.0);
    return out;
}

@fragment
fn fs_fill_const() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.5, 0.75, 1.0);
}

// Debug: quads clipped to their content mask by the rasterizer and filled with their background
// colour; borders, corners, gradients and fades are drawn wrong.
@vertex
fn vs_quad_flat(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> OpaqueQuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = load_quad(instance_id);
    let low = max(quad.bounds.origin, quad.content_mask.origin);
    let high = max(low, min(quad.bounds.origin + quad.bounds.size,
        quad.content_mask.origin + quad.content_mask.size));
    var out = OpaqueQuadVarying();
    out.position = to_device_position_impl(clamped_quad_corner(unit_vertex, vec4<f32>(low, high)));
    out.background = blend_color(hsla_to_rgba(quad.background.solid), 1.0);
    return out;
}

// Debug: `fs_quad`'s pipeline layout and varyings with a constant colour, to separate the cost
// of its arithmetic from that of rasterization and varyings. Every varying is read, behind a
// condition the compiler cannot fold, so none is optimized away.
@fragment
fn fs_quad_const(input: QuadVarying) -> @location(0) vec4<f32> {
    if (outside_clip(input.position.xy, input.clip_edges)) {
        return vec4<f32>(0.0);
    }
    let inputs = input.border_color + input.background_color0 + input.background_color1 +
        input.interior_frame + vec4<f32>(input.interior_insets, f32(input.quad_id));
    return select(input.background_solid, inputs, globals.viewport_size.x < 0.0);
}

// Debug: the fewest varyings a rounded quad needs, and only background and corner arithmetic.
// Borders, gradients, fades and differing corner radii are drawn wrong.
struct LeanQuadVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_distances: vec4<f32>,
    @location(1) @interpolate(flat) background: vec4<f32>,
    // Bounds origin and half size.
    @location(2) @interpolate(flat) frame: vec4<f32>,
    @location(3) @interpolate(flat) corner_radius: f32,
}

@vertex
fn vs_quad_lean(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> LeanQuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = load_quad(instance_id);
    var out = LeanQuadVarying();
    out.position = to_device_position(unit_vertex, quad.bounds);
    out.clip_distances = distance_from_clip_rect(unit_vertex, quad.bounds, quad.content_mask);
    out.background = hsla_to_rgba(quad.background.solid);
    out.frame = vec4<f32>(quad.bounds.origin, quad.bounds.size / 2.0);
    let radii = quad.corner_radii;
    out.corner_radius =
        max(max(radii.top_left, radii.top_right), max(radii.bottom_right, radii.bottom_left));
    return out;
}

@fragment
fn fs_quad_lean(input: LeanQuadVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let center_to_point = input.position.xy - input.frame.xy - input.frame.zw;
    let corner_to_point = abs(center_to_point) - input.frame.zw;
    let corner_center_to_point = corner_to_point + input.corner_radius;
    if (all(corner_to_point < vec2<f32>(-0.5)) && any(corner_center_to_point < vec2<f32>(0.0))) {
        return blend_color(input.background, 1.0);
    }
    let outer_sdf = quad_sdf_impl(corner_center_to_point, input.corner_radius);
    return blend_color(input.background, saturate(0.5 - outer_sdf));
}

fn quad_color(input: QuadVarying, quad: Quad) -> vec4<f32> {
    let background_color = gradient_color(quad.background, input.position.xy, quad.bounds,
        input.background_solid, input.background_color0, input.background_color1);

    let unrounded = quad.corner_radii.top_left == 0.0 &&
        quad.corner_radii.bottom_left == 0.0 &&
        quad.corner_radii.top_right == 0.0 &&
        quad.corner_radii.bottom_right == 0.0;

    // Fast path when the quad is not rounded and doesn't have any border
    if (quad.border_widths.top == 0.0 &&
            quad.border_widths.left == 0.0 &&
            quad.border_widths.right == 0.0 &&
            quad.border_widths.bottom == 0.0 &&
            unrounded) {
        return blend_color(background_color, 1.0);
    }

    let size = quad.bounds.size;
    let half_size = size / 2.0;
    let point = input.position.xy - quad.bounds.origin;
    let center_to_point = point - half_size;

    // Signed distance field threshold for inclusion of pixels. 0.5 is the
    // minimum distance between the center of the pixel and the edge.
    let antialias_threshold = 0.5;

    // Radius of the nearest corner
    let corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

    // Width of the nearest borders
    let border = vec2<f32>(
        select(
            quad.border_widths.right,
            quad.border_widths.left,
            center_to_point.x < 0.0),
        select(
            quad.border_widths.bottom,
            quad.border_widths.top,
            center_to_point.y < 0.0));

    // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
    // The purpose of this is to not draw antialiasing pixels in this case.
    let reduced_border =
        vec2<f32>(select(border.x, -antialias_threshold, border.x == 0.0),
                  select(border.y, -antialias_threshold, border.y == 0.0));

    // Vector from the corner of the quad bounds to the point, after mirroring
    // the point into the bottom right quadrant. Both components are <= 0.
    let corner_to_point = abs(center_to_point) - half_size;

    // Vector from the point to the center of the rounded corner's circle, also
    // mirrored into bottom right quadrant.
    let corner_center_to_point = corner_to_point + corner_radius;

    // Whether the nearest point on the border is rounded
    let is_near_rounded_corner =
            corner_center_to_point.x >= 0 &&
            corner_center_to_point.y >= 0;

    // Vector from straight border inner corner to point.
    let straight_border_inner_corner_to_point = corner_to_point + reduced_border;

    // Whether the point is beyond the inner edge of the straight border.
    let is_beyond_inner_straight_border =
            straight_border_inner_corner_to_point.x > 0 ||
            straight_border_inner_corner_to_point.y > 0;

    // Whether the point is far enough inside the quad, such that the pixels are
    // not affected by the straight border.
    let is_within_inner_straight_border =
        straight_border_inner_corner_to_point.x < -antialias_threshold &&
        straight_border_inner_corner_to_point.y < -antialias_threshold;

    // Fast path for points that must be part of the background.
    //
    // This could be optimized further for large rounded corners by including
    // points in an inscribed rectangle, or some other quick linear check.
    // However, that might negatively impact performance in the case of
    // reasonable sizes for rounded corners.
    if (is_within_inner_straight_border && !is_near_rounded_corner) {
        return blend_color(background_color, 1.0);
    }

    // Signed distance of the point to the outside edge of the quad's border. It
    // is positive outside this edge, and negative inside.
    let outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

    // Approximate signed distance of the point to the inside edge of the quad's
    // border. It is negative outside this edge (within the border), and
    // positive inside.
    //
    // This is not always an accurate signed distance:
    // * The rounded portions with varying border width use an approximation of
    //   nearest-point-on-ellipse.
    // * When it is quickly known to be outside the edge, -1.0 is used.
    var inner_sdf = 0.0;
    if (corner_center_to_point.x <= 0 || corner_center_to_point.y <= 0) {
        // Fast paths for straight borders.
        inner_sdf = -max(straight_border_inner_corner_to_point.x,
                         straight_border_inner_corner_to_point.y);
    } else if (is_beyond_inner_straight_border) {
        // Fast path for points that must be outside the inner edge.
        inner_sdf = -1.0;
    } else if (reduced_border.x == reduced_border.y) {
        // Fast path for circular inner edge.
        inner_sdf = -(outer_sdf + reduced_border.x);
    } else {
        let ellipse_radii = max(vec2<f32>(0.0), corner_radius - reduced_border);
        inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
    }

    // Negative when inside the border
    let border_sdf = max(inner_sdf, outer_sdf);

    var color = background_color;
    if (border_sdf < antialias_threshold) {
        var border_color = input.border_color;

        // Dashed border logic when border_style == 1
        if (quad.border_style == 1) {
            // Position along the perimeter in "dash space", where each dash
            // period has length 1
            var t = 0.0;

            // Total number of dash periods, so that the dash spacing can be
            // adjusted to evenly divide it
            var max_t = 0.0;

            // Border width is proportional to dash size. This is the behavior
            // used by browsers, but also avoids dashes from different segments
            // overlapping when dash size is smaller than the border width.
            //
            // Dash pattern: (2 * border width) dash, (1 * border width) gap
            let dash_length_per_width = 2.0;
            let dash_gap_per_width = 1.0;
            let dash_period_per_width = dash_length_per_width + dash_gap_per_width;

            // Since the dash size is determined by border width, the density of
            // dashes varies. Multiplying a pixel distance by this returns a
            // position in dash space - it has units (dash period / pixels). So
            // a dash velocity of (1 / 10) is 1 dash every 10 pixels.
            var dash_velocity = 0.0;

            // Dividing this by the border width gives the dash velocity
            let dv_numerator = 1.0 / dash_period_per_width;

            if (unrounded) {
                // When corners aren't rounded, the dashes are separately laid
                // out on each straight line, rather than around the whole
                // perimeter. This way each line starts and ends with a dash.
                let is_horizontal =
                        corner_center_to_point.x <
                        corner_center_to_point.y;

                // When applying dashed borders to just some, not all, the sides.
                // The way we chose border widths above sometimes comes with a 0 width value.
                // So we choose again to avoid division by zero.
                // TODO: A better solution exists taking a look at the whole file.
                // this does not fix single dashed borders at the corners
                let dashed_border = vec2<f32>(
                        max(
                            quad.border_widths.bottom,
                            quad.border_widths.top,
                        ),
                        max(
                            quad.border_widths.right,
                            quad.border_widths.left,
                        )
                   );

                let border_width = select(dashed_border.y, dashed_border.x, is_horizontal);
                dash_velocity = dv_numerator / border_width;
                t = select(point.y, point.x, is_horizontal) * dash_velocity;
                max_t = select(size.y, size.x, is_horizontal) * dash_velocity;
            } else {
                // When corners are rounded, the dashes are laid out clockwise
                // around the whole perimeter.

                let r_tr = quad.corner_radii.top_right;
                let r_br = quad.corner_radii.bottom_right;
                let r_bl = quad.corner_radii.bottom_left;
                let r_tl = quad.corner_radii.top_left;

                let w_t = quad.border_widths.top;
                let w_r = quad.border_widths.right;
                let w_b = quad.border_widths.bottom;
                let w_l = quad.border_widths.left;

                // Straight side dash velocities
                let dv_t = select(dv_numerator / w_t, 0.0, w_t <= 0.0);
                let dv_r = select(dv_numerator / w_r, 0.0, w_r <= 0.0);
                let dv_b = select(dv_numerator / w_b, 0.0, w_b <= 0.0);
                let dv_l = select(dv_numerator / w_l, 0.0, w_l <= 0.0);

                // Straight side lengths in dash space
                let s_t = (size.x - r_tl - r_tr) * dv_t;
                let s_r = (size.y - r_tr - r_br) * dv_r;
                let s_b = (size.x - r_br - r_bl) * dv_b;
                let s_l = (size.y - r_bl - r_tl) * dv_l;

                let corner_dash_velocity_tr = corner_dash_velocity(dv_t, dv_r);
                let corner_dash_velocity_br = corner_dash_velocity(dv_b, dv_r);
                let corner_dash_velocity_bl = corner_dash_velocity(dv_b, dv_l);
                let corner_dash_velocity_tl = corner_dash_velocity(dv_t, dv_l);

                // Corner lengths in dash space
                let c_tr = r_tr * (M_PI_F / 2.0) * corner_dash_velocity_tr;
                let c_br = r_br * (M_PI_F / 2.0) * corner_dash_velocity_br;
                let c_bl = r_bl * (M_PI_F / 2.0) * corner_dash_velocity_bl;
                let c_tl = r_tl * (M_PI_F / 2.0) * corner_dash_velocity_tl;

                // Cumulative dash space upto each segment
                let upto_tr = s_t;
                let upto_r = upto_tr + c_tr;
                let upto_br = upto_r + s_r;
                let upto_b = upto_br + c_br;
                let upto_bl = upto_b + s_b;
                let upto_l = upto_bl + c_bl;
                let upto_tl = upto_l + s_l;
                max_t = upto_tl + c_tl;

                if (is_near_rounded_corner) {
                    let radians = atan2(corner_center_to_point.y,
                                        corner_center_to_point.x);
                    let corner_t = radians * corner_radius;

                    if (center_to_point.x >= 0.0) {
                        if (center_to_point.y < 0.0) {
                            dash_velocity = corner_dash_velocity_tr;
                            // Subtracted because radians is pi/2 to 0 when
                            // going clockwise around the top right corner,
                            // since the y axis has been flipped
                            t = upto_r - corner_t * dash_velocity;
                        } else {
                            dash_velocity = corner_dash_velocity_br;
                            // Added because radians is 0 to pi/2 when going
                            // clockwise around the bottom-right corner
                            t = upto_br + corner_t * dash_velocity;
                        }
                    } else {
                        if (center_to_point.y >= 0.0) {
                            dash_velocity = corner_dash_velocity_bl;
                            // Subtracted because radians is pi/2 to 0 when
                            // going clockwise around the bottom-left corner,
                            // since the x axis has been flipped
                            t = upto_l - corner_t * dash_velocity;
                        } else {
                            dash_velocity = corner_dash_velocity_tl;
                            // Added because radians is 0 to pi/2 when going
                            // clockwise around the top-left corner, since both
                            // axis were flipped
                            t = upto_tl + corner_t * dash_velocity;
                        }
                    }
                } else {
                    // Straight borders
                    let is_horizontal =
                            corner_center_to_point.x <
                            corner_center_to_point.y;
                    if (is_horizontal) {
                        if (center_to_point.y < 0.0) {
                            dash_velocity = dv_t;
                            t = (point.x - r_tl) * dash_velocity;
                        } else {
                            dash_velocity = dv_b;
                            t = upto_bl - (point.x - r_bl) * dash_velocity;
                        }
                    } else {
                        if (center_to_point.x < 0.0) {
                            dash_velocity = dv_l;
                            t = upto_tl - (point.y - r_tl) * dash_velocity;
                        } else {
                            dash_velocity = dv_r;
                            t = upto_r + (point.y - r_tr) * dash_velocity;
                        }
                    }
                }
            }

            let dash_length = dash_length_per_width / dash_period_per_width;
            let desired_dash_gap = dash_gap_per_width / dash_period_per_width;

            // Straight borders should start and end with a dash, so max_t is
            // reduced to cause this.
            max_t -= select(0.0, dash_length, unrounded);
            if (max_t >= 1.0) {
                // Adjust dash gap to evenly divide max_t.
                let dash_count = floor(max_t);
                let dash_period = max_t / dash_count;
                border_color.a *= dash_alpha(
                    t,
                    dash_period,
                    dash_length,
                    dash_velocity,
                    antialias_threshold);
            } else if (unrounded) {
                // When there isn't enough space for the full gap between the
                // two start / end dashes of a straight border, reduce gap to
                // make them fit.
                let dash_gap = max_t - dash_length;
                if (dash_gap > 0.0) {
                    let dash_period = dash_length + dash_gap;
                    border_color.a *= dash_alpha(
                        t,
                        dash_period,
                        dash_length,
                        dash_velocity,
                        antialias_threshold);
                }
            }
        }

        // Blend the border on top of the background and then linearly interpolate
        // between the two as we slide inside the background.
        let blended_border = over(background_color, border_color);
        color = mix(background_color, blended_border,
                    saturate(antialias_threshold - inner_sdf));
    }

    return blend_color(color, saturate(antialias_threshold - outer_sdf));
}

// Returns the dash velocity of a corner given the dash velocity of the two
// sides, by returning the slower velocity (larger dashes).
//
// Since 0 is used for dash velocity when the border width is 0 (instead of
// +inf), this returns the other dash velocity in that case.
//
// An alternative to this might be to appropriately interpolate the dash
// velocity around the corner, but that seems overcomplicated.
fn corner_dash_velocity(dv1: f32, dv2: f32) -> f32 {
    if (dv1 == 0.0) {
        return dv2;
    } else if (dv2 == 0.0) {
        return dv1;
    } else {
        return min(dv1, dv2);
    }
}

// Returns alpha used to render antialiased dashes.
// `t` is within the dash when `fmod(t, period) < length`.
fn dash_alpha(t: f32, period: f32, length: f32, dash_velocity: f32, antialias_threshold: f32) -> f32 {
    let half_period = period / 2;
    let half_length = length / 2;
    // Value in [-half_period, half_period].
    // The dash is in [-half_length, half_length].
    let centered = fmod(t + half_period - half_length, period) - half_period;
    // Signed distance for the dash, negative values are inside the dash.
    let signed_distance = abs(centered) - half_length;
    // Antialiased alpha based on the signed distance.
    return saturate(antialias_threshold - signed_distance / dash_velocity);
}

// This approximates distance to the nearest point to a quarter ellipse in a way
// that is sufficient for anti-aliasing when the ellipse is not very eccentric.
// The components of `point` are expected to be positive.
//
// Negative on the outside and positive on the inside.
fn quarter_ellipse_sdf(point: vec2<f32>, radii: vec2<f32>) -> f32 {
    // Scale the space to treat the ellipse like a unit circle.
    let circle_vec = point / radii;
    let unit_circle_sdf = length(circle_vec) - 1.0;
    // Approximate up-scaling of the length by using the average of the radii.
    //
    // TODO: A better solution would be to use the gradient of the implicit
    // function for an ellipse to approximate a scaling factor.
    return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

// Modulus that has the same sign as `a`.
fn fmod(a: f32, b: f32) -> f32 {
    return a - b * trunc(a / b);
}

// --- shadows --- //

struct Shadow {
    order: u32,
    blur_radius: f32,
    // The shadow rect for drop shadows; the "hole" rect for inset shadows.
    bounds: Bounds,
    corner_radii: Corners,
    content_mask: Bounds,
    content_fade: ContentFade,
    color: Hsla,
    // Only consulted when `inset == 1u`: the element's own bounds, used as a rounded-rect
    // clip so the shadow never escapes the element.
    element_bounds: Bounds,
    element_corner_radii: Corners,
    // 0 = drop shadow, 1 = inset shadow.
    inset: u32,
    pad: u32, // align to 8 bytes
}

struct ShadowVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) shadow_id: u32,
    //TODO: use `clip_distance` once Naga supports it
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_shadow(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> ShadowVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    var shadow = load_shadow(instance_id);

    var geometry: Bounds;
    if (shadow.inset != 0u) {
        geometry = shadow.element_bounds;
    } else {
        // Leave room for the gaussian tail outside the shadow rect.
        let margin = 3.0 * shadow.blur_radius;
        geometry = shadow.bounds;
        geometry.origin -= vec2<f32>(margin);
        geometry.size += 2.0 * vec2<f32>(margin);
    }

    var out = ShadowVarying();
    out.position = to_device_position(unit_vertex, geometry);
    out.color = hsla_to_rgba(shadow.color);
    out.shadow_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, geometry, shadow.content_mask);
    return out;
}

@fragment
fn fs_shadow(input: ShadowVarying) -> @location(0) vec4<f32> {
    // Alpha clip first, since we don't have `clip_distance`.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let shadow = load_shadow(input.shadow_id);
    let half_size = shadow.bounds.size / 2.0;
    let center = shadow.bounds.origin + half_size;
    let center_to_point = input.position.xy - center;

    let corner_radius = pick_corner_radius(center_to_point, shadow.corner_radii);

    var alpha: f32;
    if (shadow.blur_radius == 0.0) {
        let distance = quad_sdf(input.position.xy, shadow.bounds, shadow.corner_radii);
        alpha = saturate(0.5 - distance);
    } else {
        // The signal is only non-zero in a limited range, so don't waste samples
        let low = center_to_point.y - half_size.y;
        let high = center_to_point.y + half_size.y;
        let start = clamp(-3.0 * shadow.blur_radius, low, high);
        let end = clamp(3.0 * shadow.blur_radius, low, high);

        // Accumulate samples (we can get away with surprisingly few samples)
        let step = (end - start) / 4.0;
        var y = start + step * 0.5;
        alpha = 0.0;
        for (var i = 0; i < 4; i += 1) {
            let blur = blur_along_x(center_to_point.x, center_to_point.y - y,
                shadow.blur_radius, corner_radius, half_size);
            alpha +=  blur * gaussian(y, shadow.blur_radius) * step;
            y += step;
        }
    }

    if (shadow.inset != 0u) {
        // The inset shadow is the complement of the (blurred) hole rect, clipped to the element.
        // `saturate(0.5 - d)` gives a 1-pixel antialiased edge: d <= -0.5 -> 1, d >= 0.5 -> 0.
        alpha = 1.0 - alpha;
        let element_distance = quad_sdf(input.position.xy, shadow.element_bounds,
                                        shadow.element_corner_radii);
        alpha *= saturate(0.5 - element_distance);
    }

    alpha *= fade_alpha(input.position.y, fade_vector(shadow.content_fade));
    return blend_color(input.color, alpha);
}

// --- shapes --- //

// `gpui::Shape`. Every `vec4` sits on a 16-byte boundary, matching the Rust layout.
struct Shape {
    order: u32,
    outline: u32,
    material: u32,
    flags: u32,
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    params: vec4<f32>,
    transform: vec4<f32>,
    placement: vec4<f32>,
    lighting: vec4<f32>,
    colors: array<Hsla, 4>,
}

const SHAPE_MIRROR: u32 = 1u;
const SHAPE_SHADOW_HALO: u32 = 2u;

struct ShapeVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) shape_id: u32,
    //TODO: use `clip_distance` once Naga supports it
    @location(1) clip_distances: vec4<f32>,
    @location(2) @interpolate(flat) primary: vec4<f32>,
    @location(3) @interpolate(flat) secondary: vec4<f32>,
    @location(4) @interpolate(flat) deep: vec4<f32>,
    @location(5) @interpolate(flat) rim: vec4<f32>,
}

@vertex
fn vs_shape(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> ShapeVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let shape = load_shape(instance_id);

    var out = ShapeVarying();
    out.position = to_device_position(unit_vertex, shape.bounds);
    out.shape_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, shape.bounds, shape.content_mask);
    out.primary = hsla_to_rgba(shape.colors[0]);
    out.secondary = hsla_to_rgba(shape.colors[1]);
    out.deep = hsla_to_rgba(shape.colors[2]);
    out.rim = hsla_to_rgba(shape.colors[3]);
    return out;
}

fn shape_dot2(v: vec2<f32>) -> f32 {
    return dot(v, v);
}

// Turns +x toward +y: clockwise on screen.
fn shape_rotate(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// The angle of `q` from straight up. The tiny bias keeps `atan2(0, 0)` defined at the centre.
fn shape_angle_from_up(q: vec2<f32>) -> f32 {
    return atan2(q.x, -q.y - 1e-7);
}

fn shape_polar(q: vec2<f32>, lobes: f32, inner: f32, sharpness: f32) -> f32 {
    // Lifting the swing before the power rounds the bottom of each notch, so the outline turns
    // smoothly through it even when `sharpness` is below one. Without it the outline's slope jumps
    // there, and so does every distance estimated from it.
    let lift = 0.04;
    let swing = max(0.5 + 0.5 * cos(lobes * shape_angle_from_up(q)), 0.0) + lift;
    let low = pow(lift, sharpness);
    let profile = (pow(swing, sharpness) - low) / (pow(1.0 + lift, sharpness) - low);
    let radius = inner + (1.0 - inner) * profile;
    let reach = length(q);
    // Inside, dividing by the radius keeps the field smooth at the centre, where the angle changes
    // fastest. Outside, subtracting it keeps the angle's pull from growing with the distance, which
    // would streak halos out past the flanks.
    return select(reach - radius, reach / radius - 1.0, reach < radius);
}

// Inigo Quilez's regular polygon, turned so a vertex points up, shrunk by the rounding so the
// rounded corners stay inside the unit circle.
fn shape_polygon(q: vec2<f32>, sides: f32, rounding: f32) -> f32 {
    let half_sector = M_PI_F / sides;
    let corner = vec2<f32>(cos(half_sector), sin(half_sector));
    let angle = shape_angle_from_up(q);
    let sector = 2.0 * half_sector;
    let folded = angle - sector * floor(angle / sector) - half_sector;
    var point = length(q) * vec2<f32>(cos(folded), abs(sin(folded)));
    let radius = 1.0 - rounding;
    point -= radius * corner;
    point.y += clamp(-point.y, 0.0, radius * corner.y);
    return length(point) * sign(point.x) - rounding;
}

fn shape_superellipse(q: vec2<f32>, exponent: f32) -> f32 {
    let magnitude = abs(q) + vec2<f32>(1e-6);
    // Normalising by the larger axis keeps the powers finite far from the outline.
    let largest = max(magnitude.x, magnitude.y);
    let ratio = magnitude / largest;
    return largest * pow(pow(ratio.x, exponent) + pow(ratio.y, exponent), 1.0 / exponent) - 1.0;
}

// Inigo Quilez's heart, whose point sits at the origin with the lobes above it, flipped to point
// down and fitted to the unit circle.
fn shape_heart(q: vec2<f32>, rounding: f32) -> f32 {
    var h = vec2<f32>(q.x, -q.y) * 0.56 * (1.0 + rounding) + vec2<f32>(0.0, 0.56);
    h.x = abs(h.x);
    var distance: f32;
    if (h.y + h.x > 1.0) {
        distance = sqrt(shape_dot2(h - vec2<f32>(0.25, 0.75))) - sqrt(2.0) / 4.0;
    } else {
        distance = sqrt(min(shape_dot2(h - vec2<f32>(0.0, 1.0)),
                            shape_dot2(h - vec2<f32>(0.5 * max(h.x + h.y, 0.0)))))
            * sign(h.x - h.y);
    }
    return distance / 0.56 - rounding;
}

fn shape_capsule(q: vec2<f32>, half_length: f32) -> f32 {
    let along = q.y - clamp(q.y, -half_length, half_length);
    return length(vec2<f32>(q.x, along)) - 1.0;
}

// Inigo Quilez's uneven capsule, turned to point up: the unit circle and a circle of radius `tip`
// `reach` above it, joined by their common tangents. A point circle that fits inside the unit
// circle leaves just the circle.
fn shape_drop(q: vec2<f32>, reach: f32, tip: f32) -> f32 {
    let slope = (1.0 - tip) / max(reach, 1e-4);
    if (slope >= 1.0) {
        return length(q) - 1.0;
    }
    let rise = sqrt(1.0 - slope * slope);
    let p = vec2<f32>(abs(q.x), -q.y);
    let along = dot(p, vec2<f32>(-slope, rise));
    if (along < 0.0) {
        return length(p) - 1.0;
    }
    if (along > rise * reach) {
        return length(p - vec2<f32>(0.0, reach)) - tip;
    }
    return dot(p, vec2<f32>(rise, slope)) - 1.0;
}

// The outline's field at a bounds-space point: negative inside, zero on the outline. It is not a
// distance; callers divide by its gradient's length.
fn shape_field(shape: Shape, point: vec2<f32>) -> f32 {
    var mirrored = point;
    if ((shape.flags & SHAPE_MIRROR) != 0u) {
        mirrored.x = abs(mirrored.x);
    }
    let offset = mirrored - shape.placement.xy;
    var q = vec2<f32>(
        shape.transform.x * offset.x + shape.transform.y * offset.y,
        shape.transform.z * offset.x + shape.transform.w * offset.y,
    );
    let warp = shape.placement.z;
    if (warp > 0.0) {
        let phase = shape.placement.w;
        q += warp * vec2<f32>(
            sin(2.3 * q.y + phase) + 0.5 * sin(3.1 * q.x - 2.0 * phase + 1.0),
            cos(2.1 * q.x - phase) + 0.5 * sin(2.9 * q.y + 2.0 * phase + 2.0),
        );
    }

    let params = shape.params;
    var value: f32;
    switch (shape.outline) {
        case 1u: { value = shape_polar(q, params.x, params.y, params.z); }
        case 2u: { value = shape_polygon(q, params.x, params.y); }
        case 3u: { value = shape_superellipse(q, params.x); }
        case 4u: { value = shape_heart(q, params.x); }
        case 5u: { value = shape_capsule(q, params.x); }
        case 6u: { value = shape_drop(q, params.x, params.y); }
        default: { value = length(q) - 1.0; }
    }
    return value;
}

@fragment
fn fs_shape(input: ShapeVarying) -> @location(0) vec4<f32> {
    // Alpha clip first, since we don't have `clip_distance`.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let shape = load_shape(input.shape_id);
    let half_extent = max(0.5 * min(shape.bounds.size.x, shape.bounds.size.y), 1e-3);
    let center = shape.bounds.origin + 0.5 * shape.bounds.size;
    let point = (input.position.xy - center) / half_extent;
    let pixel = 1.0 / half_extent;

    // A signed distance estimate from the field and its gradient, which works for every outline
    // and survives the transform and the warp.
    let epsilon = 0.35 * pixel;
    let value = shape_field(shape, point);
    let gradient = vec2<f32>(
        shape_field(shape, point + vec2<f32>(epsilon, 0.0)) - value,
        shape_field(shape, point + vec2<f32>(0.0, epsilon)) - value,
    ) / epsilon;
    let gradient_length = max(length(gradient), 1e-3);
    let distance = value / gradient_length;
    let normal = gradient / gradient_length;
    let coverage = saturate(0.5 - distance / pixel);

    // The outline's drawn radius, for widths that grow with it.
    let determinant = shape.transform.x * shape.transform.w - shape.transform.y * shape.transform.z;
    let radius = inverseSqrt(max(abs(determinant), 1e-6));
    // Fades the halo out before the quad's edge would cut it off.
    let edge = 0.5 * shape.bounds.size / half_extent - abs(point);
    let edge_fade = saturate(min(edge.x, edge.y) / 0.08);

    var color: vec3<f32>;
    var alpha: f32;
    if (shape.material == 1u) {
        let halo = exp(-max(distance, 0.0) / 0.06) * 0.35 * shape.lighting.w * edge_fade
            * (1.0 - coverage);
        color = input.primary.rgb * (coverage + halo);
        alpha = coverage + halo;
    } else {
        let depth = max(-distance, 0.0);

        // Two coloured lights inside a deep body, fixed to the screen rather than the outline's
        // rotation, turned together by the flow angle.
        let local = (point - shape.placement.xy) / radius;
        let primary_light = shape_rotate(vec2<f32>(-0.525, -0.5), shape.lighting.y);
        let secondary_light = shape_rotate(vec2<f32>(0.5625, 0.525), shape.lighting.y);
        let primary_weight = exp(-shape_dot2(local - primary_light) / 0.47);
        let secondary_weight = exp(-shape_dot2(local - secondary_light) / 0.47);
        var body = input.deep.rgb;
        body = mix(body, input.primary.rgb, saturate(primary_weight * 0.8));
        body = mix(body, input.secondary.rgb, saturate(secondary_weight * 0.8));
        let hue = mix(input.secondary.rgb, input.primary.rgb,
                      primary_weight / (primary_weight + secondary_weight + 1e-4));

        // A key light and a back light roughly opposite it catch the rim.
        let key_direction = vec2<f32>(cos(shape.lighting.x), sin(shape.lighting.x));
        let back_direction = vec2<f32>(cos(shape.lighting.x + 3.65), sin(shape.lighting.x + 3.65));
        let shine = 0.08
            + pow(max(dot(normal, key_direction), 0.0), 1.8)
            + pow(max(dot(normal, back_direction), 0.0), 2.4);

        // A frosted band inside the edge, then a thin bright rim line.
        let band = exp(-depth / (0.075 * radius));
        body = mix(body, mix(hue, input.rim.rgb, 0.55), saturate(band * shine));
        let rim_line = exp(-depth / max(1.8 * pixel, 0.006));
        body = mix(body, input.rim.rgb, saturate(rim_line * shine * 1.3 * shape.lighting.z));

        var halo = exp(-max(distance, 0.0) / (0.07 * radius)) * 0.3 * clamp(shine, 0.0, 1.2);
        var halo_color = mix(hue, input.rim.rgb, 0.5);
        if ((shape.flags & SHAPE_SHADOW_HALO) != 0u) {
            let shadow_distance =
                shape_field(shape, point - vec2<f32>(0.0, 0.075 * radius)) / gradient_length;
            halo = exp(-max(shadow_distance, 0.0) / (0.112 * radius)) * 0.32;
            halo_color = input.deep.rgb * 0.6;
        }
        halo *= shape.lighting.w * edge_fade * (1.0 - coverage);
        color = body * coverage + halo_color * halo;
        alpha = coverage + halo;
    }

    alpha = saturate(alpha);
    let straight = color / max(alpha, 1e-4);
    let fade = fade_alpha(input.position.y, fade_vector(shape.content_fade));
    return blend_color(vec4<f32>(straight, alpha * input.primary.a), fade);
}

// --- path rasterization --- //

struct PathRasterizationVertex {
    xy_position: vec2<f32>,
    st_position: vec2<f32>,
    color: Background,
    bounds: Bounds,
    fade: ContentFade,
}



struct PathRasterizationVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) st_position: vec2<f32>,
    @location(1) @interpolate(flat) vertex_id: u32,
    //TODO: use `clip_distance` once Naga supports it
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_path_rasterization(@builtin(vertex_index) vertex_id: u32) -> PathRasterizationVarying {
    let v = load_path_vertex(vertex_id);

    var out = PathRasterizationVarying();
    out.position = to_device_position_impl(v.xy_position);
    out.st_position = v.st_position;
    out.vertex_id = vertex_id;
    out.clip_distances = distance_from_clip_rect_impl(v.xy_position, v.bounds);
    return out;
}

@fragment
fn fs_path_rasterization(input: PathRasterizationVarying) -> @location(0) vec4<f32> {
    let dx = dpdx(input.st_position);
    let dy = dpdy(input.st_position);
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let v = load_path_vertex(input.vertex_id);
    let background = v.color;
    let bounds = v.bounds;

    var alpha: f32;
    if (length(vec2<f32>(dx.x, dy.x)) < 0.001) {
        // If the gradient is too small, return a solid color.
        alpha = 1.0;
    } else {
        let gradient = 2.0 * input.st_position.xx * vec2<f32>(dx.x, dy.x) - vec2<f32>(dx.y, dy.y);
        let f = input.st_position.x * input.st_position.x - input.st_position.y;
        let distance = f / length(gradient);
        alpha = saturate(0.5 - distance);
    }
    let prepared_gradient = prepare_gradient_color(
        background.tag,
        background.color_space,
        background.solid,
        background.colors,
    );
    let color = gradient_color(background, input.position.xy, bounds,
        prepared_gradient.solid, prepared_gradient.color0, prepared_gradient.color1);
    // Premultiplied, so the fade scales colour and alpha alike; the sprite pass composites the
    // intermediate as is.
    alpha *= fade_alpha(input.position.y, fade_vector(v.fade));
    return vec4<f32>(color.rgb * color.a * alpha, color.a * alpha);
}

// --- paths --- //

struct PathSprite {
    bounds: Bounds,
    texture_bounds: Bounds,
}


struct PathVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_coords: vec2<f32>,
}

@vertex
fn vs_path(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> PathVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = load_path_sprite(instance_id);
    // Don't apply content mask because it was already accounted for when rasterizing the path.
    let device_position = to_device_position(unit_vertex, sprite.bounds);
    let screen_position = sprite.bounds.origin + unit_vertex * sprite.bounds.size;
    let texture_coords = (screen_position - sprite.texture_bounds.origin) / sprite.texture_bounds.size;

    var out = PathVarying();
    out.position = device_position;
    out.texture_coords = texture_coords;

    return out;
}

@fragment
fn fs_path(input: PathVarying) -> @location(0) vec4<f32> {
    let sample = textureSample(t_sprite, s_sprite, input.texture_coords);
    return sample;
}

// --- underlines --- //

struct Underline {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    color: Hsla,
    thickness: f32,
    wavy: u32,
}


struct UnderlineVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) underline_id: u32,
    //TODO: use `clip_distance` once Naga supports it
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_underline(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> UnderlineVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let underline = load_underline(instance_id);

    var out = UnderlineVarying();
    out.position = to_device_position(unit_vertex, underline.bounds);
    out.color = hsla_to_rgba(underline.color);
    out.underline_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, underline.bounds, underline.content_mask);
    return out;
}

@fragment
fn fs_underline(input: UnderlineVarying) -> @location(0) vec4<f32> {
    const WAVE_FREQUENCY: f32 = 2.0;
    const WAVE_HEIGHT_RATIO: f32 = 0.8;

    // Alpha clip first, since we don't have `clip_distance`.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let underline = load_underline(input.underline_id);
    let fade = fade_alpha(input.position.y, fade_vector(underline.content_fade));
    if (underline.wavy == 0u)
    {
        return blend_color(input.color, input.color.a * fade);
    }

    let half_thickness = underline.thickness * 0.5;

    let st = (input.position.xy - underline.bounds.origin) / underline.bounds.size.y - vec2<f32>(0.0, 0.5);
    let frequency = M_PI_F * WAVE_FREQUENCY * underline.thickness / underline.bounds.size.y;
    let amplitude = (underline.thickness * WAVE_HEIGHT_RATIO) / underline.bounds.size.y;

    let sine = sin(st.x * frequency) * amplitude;
    let dSine = cos(st.x * frequency) * amplitude * frequency;
    let distance = (st.y - sine) / sqrt(1.0 + dSine * dSine);
    let distance_in_pixels = distance * underline.bounds.size.y;
    let distance_from_top_border = distance_in_pixels - half_thickness;
    let distance_from_bottom_border = distance_in_pixels + half_thickness;
    let alpha = saturate(0.5 - max(-distance_from_bottom_border, distance_from_top_border));
    return blend_color(input.color, alpha * input.color.a * fade);
}

// --- monochrome sprites --- //

struct MonochromeSprite {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    color: Hsla,
    tile: AtlasTile,
    transformation: TransformationMatrix,
}


struct MonoSpriteVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(3) clip_distances: vec4<f32>,
    @location(4) @interpolate(flat) fade: vec4<f32>,
}

@vertex
fn vs_mono_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> MonoSpriteVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = load_mono_sprite(instance_id);

    var out = MonoSpriteVarying();
    out.position = to_device_position_transformed(unit_vertex, sprite.bounds, sprite.transformation);

    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.color = hsla_to_rgba(sprite.color);
    out.clip_distances = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds, sprite.content_mask, sprite.transformation);
    out.fade = fade_vector(sprite.content_fade);
    return out;
}

@fragment
fn fs_mono_sprite(input: MonoSpriteVarying) -> @location(0) vec4<f32> {
    let sample = textureSample(t_sprite, s_sprite, input.tile_position).r;
    let alpha_corrected = apply_contrast_and_gamma_correction(sample, input.color.rgb, gamma_params.grayscale_enhanced_contrast, gamma_params.gamma_ratios);

    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    return blend_color(input.color, alpha_corrected * fade_alpha(input.position.y, input.fade));
}

// --- polychrome sprites --- //

struct PolychromeSprite {
    order: u32,
    pad: u32,
    grayscale: u32,
    opacity: f32,
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    corner_radii: Corners,
    tile: AtlasTile,
}


struct PolySpriteVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) @interpolate(flat) sprite_id: u32,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_poly_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> PolySpriteVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = load_poly_sprite(instance_id);

    var out = PolySpriteVarying();
    out.position = to_device_position(unit_vertex, sprite.bounds);
    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.sprite_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, sprite.bounds, sprite.content_mask);
    return out;
}

@fragment
fn fs_poly_sprite(input: PolySpriteVarying) -> @location(0) vec4<f32> {
    let sample = textureSample(t_sprite, s_sprite, input.tile_position);
    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let sprite = load_poly_sprite(input.sprite_id);
    let distance = quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii);

    var color = sample;
    if (sprite.grayscale != 0u) {
        let grayscale = dot(color.rgb, GRAYSCALE_FACTORS);
        color = vec4<f32>(vec3<f32>(grayscale), sample.a);
    }
    let fade = fade_alpha(input.position.y, fade_vector(sprite.content_fade));
    return blend_color(color, sprite.opacity * saturate(0.5 - distance) * fade);
}

// --- surfaces --- //

struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
}

@group(1) @binding(0) var<uniform> surface_locals: SurfaceParams;
@group(1) @binding(1) var t_y: texture_2d<f32>;
@group(1) @binding(2) var t_cb_cr: texture_2d<f32>;
@group(1) @binding(3) var s_surface: sampler;

const ycbcr_to_RGB = mat4x4<f32>(
    vec4<f32>( 1.0000f,  1.0000f,  1.0000f, 0.0),
    vec4<f32>( 0.0000f, -0.3441f,  1.7720f, 0.0),
    vec4<f32>( 1.4020f, -0.7141f,  0.0000f, 0.0),
    vec4<f32>(-0.7010f,  0.5291f, -0.8860f, 1.0),
);

struct SurfaceVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_position: vec2<f32>,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_surface(@builtin(vertex_index) vertex_id: u32) -> SurfaceVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));

    var out = SurfaceVarying();
    out.position = to_device_position(unit_vertex, surface_locals.bounds);
    out.texture_position = unit_vertex;
    out.clip_distances = distance_from_clip_rect(unit_vertex, surface_locals.bounds, surface_locals.content_mask);
    return out;
}

@fragment
fn fs_surface(input: SurfaceVarying) -> @location(0) vec4<f32> {
    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let y_cb_cr = vec4<f32>(
        textureSampleLevel(t_y, s_surface, input.texture_position, 0.0).r,
        textureSampleLevel(t_cb_cr, s_surface, input.texture_position, 0.0).rg,
        1.0);

    let fade = fade_alpha(input.position.y, fade_vector(surface_locals.content_fade));
    return apply_fade(ycbcr_to_RGB * y_cb_cr, fade);
}

// --- layer composite --- //
//
// Composites a cached layer RGBA texture onto the frame. `tex_size` is the
// actual pixel size of the layer texture. The texcoord is computed so that
// unit_vertex=(0,0)..(1,1) maps exactly onto the portion of the texture that
// corresponds to the layer's bounds, clamped so fragments outside the texture
// area are discarded (matching DirectX's culling behaviour). The texture
// carries premultiplied pixels from the layer render pass.

struct LayerSurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    tex_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(1) @binding(0) var<uniform> layer_surface_locals: LayerSurfaceParams;
@group(1) @binding(1) var t_layer: texture_2d<f32>;
@group(1) @binding(2) var s_layer: sampler;

struct LayerSurfaceVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) unit_vertex: vec2<f32>,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_layer_composite(@builtin(vertex_index) vertex_id: u32) -> LayerSurfaceVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));

    var out = LayerSurfaceVarying();
    out.position = to_device_position(unit_vertex, layer_surface_locals.bounds);
    out.unit_vertex = unit_vertex;
    out.clip_distances = distance_from_clip_rect(
        unit_vertex,
        layer_surface_locals.bounds,
        layer_surface_locals.content_mask,
    );
    return out;
}

@fragment
fn fs_layer_composite(input: LayerSurfaceVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    // Cull (DirectX parity): sample the cached texture at its captured 1:1 size, anchored at the
    // surface origin. bounds.size / tex_size is the fraction of the new surface the old texture
    // covers; when they match (settled) tex_coord is exactly [0,1]. Mid-resize the surface is a
    // different size than the cached texture — keep the texture crisp at its own size and discard
    // past its extent (the window reflows around it) rather than stretching it to fill.
    let bounds_size = layer_surface_locals.bounds.size;
    let tex_size = layer_surface_locals.tex_size;
    let tex_coord = input.unit_vertex * bounds_size / tex_size;

    if (any(tex_coord > vec2<f32>(1.0))) {
        return vec4<f32>(0.0);
    }

    let sample = textureSampleLevel(t_layer, s_layer, tex_coord, 0.0);
    return sample * fade_alpha(input.position.y, fade_vector(layer_surface_locals.content_fade));
}

// --- backdrop blur --- //
//
// A live `backdrop-filter: blur()`. The renderer copies `region` of the frame (the element's
// visible bounds grown by three standard deviations) into `t_backdrop`, then runs three passes
// into two textures at 1/`downscale` resolution: a box downsample, and a separable Gaussian along
// `direction`. The composite draws the result through the element's rounded corners, content
// mask and fade. Every read is clamped to `region`, because the rest of the copy is stale.

struct BackdropParams {
    bounds: Bounds,
    content_mask: Bounds,
    content_fade: ContentFade,
    corner_radii: Corners,
    region: Bounds,
    direction: vec2<f32>,
    sigma: f32,
    downscale: f32,
    blur_size: vec2<f32>,
    opacity: f32,
    _pad: f32,
}

@group(1) @binding(0) var<uniform> backdrop: BackdropParams;
@group(1) @binding(1) var t_backdrop: texture_2d<f32>;
@group(1) @binding(2) var s_backdrop: sampler;

// One triangle over the whole target; the scissor limits it to the region.
@vertex
fn vs_backdrop_pass(@builtin(vertex_index) vertex_id: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn backdrop_low_region() -> vec4<f32> {
    let low_origin = backdrop.region.origin / backdrop.downscale;
    let low_size = backdrop.region.size / backdrop.downscale;
    return vec4<f32>(low_origin + 0.5, low_origin + low_size - 0.5);
}

@fragment
fn fs_backdrop_downsample(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let d = backdrop.downscale;
    let texture_size = vec2<f32>(textureDimensions(t_backdrop, 0));
    let lo = backdrop.region.origin + 1.0;
    let hi = backdrop.region.origin + backdrop.region.size - 1.0;
    // `position` is a low-resolution texel centre; it covers `d`×`d` full-resolution texels, and
    // each bilinear tap at a 2×2 block's shared corner averages four of them.
    let block_origin = (position.xy - 0.5) * d;
    let taps = u32(d * 0.5);
    var sum = vec4<f32>(0.0);
    for (var j = 0u; j < taps; j++) {
        for (var i = 0u; i < taps; i++) {
            let at = block_origin + vec2<f32>(f32(2u * i + 1u), f32(2u * j + 1u));
            let uv = clamp(at, lo, hi) / texture_size;
            sum += textureSampleLevel(t_backdrop, s_backdrop, uv, 0.0);
        }
    }
    return sum / f32(taps * taps);
}

@fragment
fn fs_backdrop_blur(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(t_backdrop, 0));
    let region = backdrop_low_region();
    let sigma = max(backdrop.sigma, 0.5);
    let taps = min(i32(ceil(sigma * 3.0)), 32);
    var sum = textureSampleLevel(t_backdrop, s_backdrop, position.xy / texture_size, 0.0);
    var total = 1.0;
    for (var k = 1; k <= taps; k++) {
        let weight = exp(-f32(k * k) / (2.0 * sigma * sigma));
        let offset = backdrop.direction * f32(k);
        let ahead = clamp(position.xy + offset, region.xy, region.zw) / texture_size;
        let behind = clamp(position.xy - offset, region.xy, region.zw) / texture_size;
        sum += (textureSampleLevel(t_backdrop, s_backdrop, ahead, 0.0)
            + textureSampleLevel(t_backdrop, s_backdrop, behind, 0.0)) * weight;
        total += 2.0 * weight;
    }
    return sum / total;
}

struct BackdropVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_distances: vec4<f32>,
}

@vertex
fn vs_backdrop_composite(@builtin(vertex_index) vertex_id: u32) -> BackdropVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    var out = BackdropVarying();
    out.position = to_device_position(unit_vertex, backdrop.bounds);
    out.clip_distances = distance_from_clip_rect(unit_vertex, backdrop.bounds, backdrop.content_mask);
    return out;
}

@fragment
fn fs_backdrop_composite(input: BackdropVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let region = backdrop_low_region();
    let low = clamp(input.position.xy / backdrop.downscale, region.xy, region.zw);
    let blurred = textureSampleLevel(t_backdrop, s_backdrop, low / backdrop.blur_size, 0.0);
    let distance = quad_sdf(input.position.xy, backdrop.bounds, backdrop.corner_radii);
    let coverage = saturate(0.5 - distance)
        * fade_alpha(input.position.y, fade_vector(backdrop.content_fade))
        * backdrop.opacity;
    // The copy holds the target's own encoding: premultiplied when the target is.
    if (globals.premultiplied_alpha != 0u) {
        return blurred * coverage;
    }
    return vec4<f32>(blurred.rgb, blurred.a * coverage);
}
