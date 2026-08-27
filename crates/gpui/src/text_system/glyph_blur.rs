//! Gaussian blur of a rasterised glyph mask, for [`crate::TextShadow`].
//!
//! A text shadow cannot be faked by painting the run again in a darker colour at an offset: every
//! copy is a fully hinted, fully antialiased glyph, so a "blur" assembled out of copies is a stack
//! of sharp ghost letters rather than a soft edge. The only thing that produces a soft edge is a
//! blur of the *coverage mask*, and the mask only exists inside the text system.
//!
//! So the blur happens exactly where the sharp mask is already produced: the platform rasteriser
//! hands back an 8-bit coverage bitmap, this module widens it and convolves it with a separable
//! Gaussian, and the result goes into the sprite atlas under a key that includes the blur radius
//! ([`crate::RenderGlyphParams::blur`]). From there it is an ordinary `MonochromeSprite` tinted
//! with the shadow colour, which every backend already knows how to draw — there is no renderer
//! change and no second blur implementation.
//!
//! Cost is paid once per distinct (glyph, size, subpixel variant, blur) and then cached like any
//! other glyph. See [`padding`] for what that costs in atlas bytes.

use crate::{DevicePixels, Size, size};

/// The largest blur radius, in device pixels, that a text shadow will actually blur at.
///
/// Not a rendering limit but a memory one: the padded tile grows as the square of the radius, so
/// an unclamped radius is a way to ask the atlas for a megabyte per letter. At the cap a glyph
/// gains 36 pixels of padding on every side — see [`padding`].
pub(crate) const MAX_BLUR_RADIUS: u8 = 24;

/// The standard deviation a blur radius means.
///
/// The CSS convention, which is also what [`crate::BoxShadow`] is authored against: the radius is
/// two standard deviations, so a `blur_radius` of 4 and a `box-shadow` blur of 4 look the same.
fn sigma(radius: u8) -> f32 {
    radius as f32 / 2.
}

/// How far a blur of this radius spreads ink past the edge of the glyph, in pixels.
///
/// Three sigma, where a Gaussian has given up 99.7% of its mass. This is the amount the tile grows
/// by on each side, so it is also the memory story: a 20x24 glyph mask is 480 bytes sharp, 1.2 kB
/// at the radius this project's overlay uses (2px logical, 3 or 4 device pixels), and 12 kB at
/// [`MAX_BLUR_RADIUS`]. One entry per distinct glyph and radius, in the same atlas and under the
/// same eviction as every sharp glyph.
pub(crate) fn padding(radius: u8) -> i32 {
    if radius == 0 {
        0
    } else {
        (sigma(radius) * 3.).ceil() as i32
    }
}

/// Blur an 8-bit coverage mask, returning the grown tile and its size.
///
/// `mask` is `size.width * size.height` bytes of coverage, as the platform rasteriser produces for
/// a monochrome glyph. The result is `padding(radius)` pixels larger on every side.
pub(crate) fn blur_mask(
    mask: &[u8],
    mask_size: Size<DevicePixels>,
    radius: u8,
) -> (Size<DevicePixels>, Vec<u8>) {
    let width = mask_size.width.0.max(0) as usize;
    let height = mask_size.height.0.max(0) as usize;
    let pad = padding(radius) as usize;
    if pad == 0 || width == 0 || height == 0 || mask.len() < width * height {
        return (mask_size, mask.to_vec());
    }

    let out_width = width + pad * 2;
    let out_height = height + pad * 2;

    // A normalised Gaussian sampled at integer offsets. Truncating at three sigma drops 0.3% of
    // the mass, which re-normalising here puts back rather than leaving the shadow slightly dim.
    let sigma = sigma(radius);
    let kernel: Vec<f32> = (0..=pad * 2)
        .map(|i| {
            let d = i as f32 - pad as f32;
            (-(d * d) / (2. * sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.into_iter().map(|k| k / total).collect();

    // Separable: a horizontal pass into a buffer that is already the output width, then a vertical
    // pass over that. O(n * kernel) rather than O(n * kernel^2).
    let mut horizontal = vec![0f32; out_width * height];
    for y in 0..height {
        let row = &mask[y * width..(y + 1) * width];
        for (out_x, slot) in horizontal[y * out_width..(y + 1) * out_width]
            .iter_mut()
            .enumerate()
        {
            let mut sum = 0.;
            for (tap, weight) in kernel.iter().enumerate() {
                let src_x = out_x as isize - pad as isize + tap as isize - pad as isize;
                if src_x >= 0 && (src_x as usize) < width {
                    sum += row[src_x as usize] as f32 * weight;
                }
            }
            *slot = sum;
        }
    }

    let mut out = vec![0u8; out_width * out_height];
    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let mut sum = 0.;
            for (tap, weight) in kernel.iter().enumerate() {
                let src_y = out_y as isize - pad as isize + tap as isize - pad as isize;
                if src_y >= 0 && (src_y as usize) < height {
                    sum += horizontal[src_y as usize * out_width + out_x] * weight;
                }
            }
            out[out_y * out_width + out_x] = sum.round().clamp(0., 255.) as u8;
        }
    }

    (
        size(
            DevicePixels(out_width as i32),
            DevicePixels(out_height as i32),
        ),
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: usize, height: usize) -> Vec<u8> {
        vec![255; width * height]
    }

    /// The whole point: a blurred mask has intermediate coverage. Painted copies of a glyph only
    /// ever produce 0 or 255, which is why they read as ghost letters rather than as a shadow.
    #[test]
    fn the_edge_is_a_ramp_rather_than_a_step() {
        let (size, blurred) = blur_mask(&solid(8, 8), size(8.into(), 8.into()), 4);
        let width = size.width.0 as usize;
        let mid = size.height.0 as usize / 2;
        let row = &blurred[mid * width..(mid + 1) * width];
        let intermediate = row.iter().filter(|v| **v > 0 && **v < 250).count();
        assert!(
            intermediate >= 4,
            "a Gaussian edge has to pass through partial coverage, got {row:?}"
        );
        assert!(
            row.iter().any(|v| *v > 200),
            "the middle of a solid block must stay solid, got {row:?}"
        );
        assert!(
            row[0] <= 2,
            "three sigma out there is all but nothing left to spread, got {}",
            row[0]
        );
    }

    /// A shadow that loses ink is a shadow that reads dimmer than it was asked for; the kernel is
    /// re-normalised for exactly this.
    #[test]
    fn the_blur_conserves_ink() {
        let sharp = solid(6, 6);
        let (_, blurred) = blur_mask(&sharp, size(6.into(), 6.into()), 3);
        let before: u64 = sharp.iter().map(|v| *v as u64).sum();
        let after: u64 = blurred.iter().map(|v| *v as u64).sum();
        let drift = (after as f64 - before as f64).abs() / before as f64;
        assert!(drift < 0.02, "lost {:.1}% of the coverage", drift * 100.);
    }

    /// The tile has to grow by exactly what [`padding`] promises, because the paint path offsets
    /// the sprite by the *unblurred* raster origin minus that padding.
    #[test]
    fn the_tile_grows_by_the_padding() {
        for radius in [1u8, 2, 5, MAX_BLUR_RADIUS] {
            let pad = padding(radius);
            let (grown, bytes) = blur_mask(&solid(4, 3), size(4.into(), 3.into()), radius);
            assert_eq!(grown.width.0, 4 + pad * 2, "radius {radius}");
            assert_eq!(grown.height.0, 3 + pad * 2, "radius {radius}");
            assert_eq!(bytes.len(), (grown.width.0 * grown.height.0) as usize);
        }
    }

    /// Zero is the identity, so a shadow with no blur costs no atlas growth at all.
    #[test]
    fn a_zero_radius_is_the_sharp_mask() {
        let sharp = solid(4, 4);
        let (grown, bytes) = blur_mask(&sharp, size(4.into(), 4.into()), 0);
        assert_eq!(grown.width.0, 4);
        assert_eq!(bytes, sharp);
    }

    /// A blur is symmetric, and an asymmetric one puts the shadow beside the glyph instead of
    /// under it.
    #[test]
    fn the_blur_is_symmetric() {
        let mut mask = vec![0u8; 9 * 9];
        mask[4 * 9 + 4] = 255;
        let (grown, blurred) = blur_mask(&mask, size(9.into(), 9.into()), 4);
        let width = grown.width.0 as usize;
        let height = grown.height.0 as usize;
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    blurred[y * width + x],
                    blurred[y * width + (width - 1 - x)],
                    "asymmetric at ({x}, {y})"
                );
                assert_eq!(
                    blurred[y * width + x],
                    blurred[(height - 1 - y) * width + x],
                    "asymmetric at ({x}, {y})"
                );
            }
        }
    }
}
