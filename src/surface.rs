//! Procedural terrain texture (no files): a tileable snow/dirt grain, as albedo and as normal map.
//! Mip levels are built here because runtime images get none, so the grain fades out with
//! distance instead of shimmering.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// Metres of terrain covered by one texture repeat.
pub(crate) const TILE: f32 = 5.0;
const SIZE: usize = 256;
/// Albedo of the flattest and the brightest grain.
const ALBEDO: (f32, f32) = (0.86, 0.98);
/// Normal map slope per unit of grain height (per texel).
const BUMP: f32 = 7.0;

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA6B)
        ^ seed.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Value noise on a lattice of `period` cells that wraps, so the texture tiles.
fn noise(x: f32, y: f32, period: i32, seed: u32) -> f32 {
    let (fx, fy) = (x * period as f32, y * period as f32);
    let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - ix as f32, fy - iy as f32);
    let (tx, ty) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let at = |dx: i32, dy: i32| {
        hash(
            (ix + dx).rem_euclid(period),
            (iy + dy).rem_euclid(period),
            seed,
        )
    };
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * tx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * tx;
    top + (bottom - top) * ty
}

/// Grain height in 0..1: drifts and crust at several scales plus per-texel sparkle.
fn height(x: usize, y: usize) -> f32 {
    let (u, v) = (x as f32 / SIZE as f32, y as f32 / SIZE as f32);
    let mut h = 0.0;
    let mut amp = 0.5;
    let mut sum = 0.0;
    for octave in 0..5u32 {
        h += amp * noise(u, v, 8 << octave, octave);
        sum += amp;
        amp *= 0.55;
    }
    0.8 * h / sum + 0.2 * hash(x as i32, y as i32, 99)
}

/// Averages 2x2 texels, level by level, down to 1x1.
fn with_mips(base: Vec<u8>) -> (Vec<u8>, u32) {
    let mut data = base;
    let (mut size, mut start, mut levels) = (SIZE, 0, 1);
    while size > 1 {
        let half = size / 2;
        let mut next = Vec::with_capacity(half * half * 4);
        for y in 0..half {
            for x in 0..half {
                for c in 0..4 {
                    let at = |dx: usize, dy: usize| {
                        data[start + ((2 * y + dy) * size + 2 * x + dx) * 4 + c] as u32
                    };
                    next.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
                }
            }
        }
        start = data.len();
        data.extend(next);
        size = half;
        levels += 1;
    }
    (data, levels)
}

fn image(base: Vec<u8>, format: TextureFormat) -> Image {
    let (data, levels) = with_mips(base.clone());
    let mut image = Image::new(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        base,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    });
    image
}

/// `(albedo, normal map)` of the terrain grain.
pub(crate) fn grain() -> (Image, Image) {
    let h: Vec<f32> = (0..SIZE * SIZE)
        .map(|i| height(i % SIZE, i / SIZE))
        .collect();
    let wrap = |i: usize, d: isize| (i as isize + d).rem_euclid(SIZE as isize) as usize;
    let mut albedo = Vec::with_capacity(SIZE * SIZE * 4);
    let mut normal = Vec::with_capacity(SIZE * SIZE * 4);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let g = ALBEDO.0 + (ALBEDO.1 - ALBEDO.0) * h[y * SIZE + x];
            let shade = (g * 255.0).round() as u8;
            albedo.extend([shade, shade, shade.saturating_add(2), 255]);
            let dx = h[y * SIZE + wrap(x, 1)] - h[y * SIZE + wrap(x, -1)];
            let dy = h[wrap(y, 1) * SIZE + x] - h[wrap(y, -1) * SIZE + x];
            let n = Vec3::new(-dx * BUMP, -dy * BUMP, 1.0).normalize();
            normal.extend([
                ((n.x * 0.5 + 0.5) * 255.0).round() as u8,
                ((n.y * 0.5 + 0.5) * 255.0).round() as u8,
                ((n.z * 0.5 + 0.5) * 255.0).round() as u8,
                255,
            ]);
        }
    }
    (
        image(albedo, TextureFormat::Rgba8UnormSrgb),
        image(normal, TextureFormat::Rgba8Unorm),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grain_tiles_and_mips_cover_every_level() {
        // Noise wraps at the lattice period, so opposite edges are one step apart.
        for s in 0..8 {
            let t = s as f32 / 8.0;
            assert!((noise(0.0, t, 4, 1) - noise(1.0, t, 4, 1)).abs() < 1e-5);
        }
        let (albedo, normal) = grain();
        for image in [&albedo, &normal] {
            let levels = image.texture_descriptor.mip_level_count;
            assert_eq!(levels, SIZE.ilog2() + 1);
            let texels: usize = (0..levels).map(|l| (SIZE >> l) * (SIZE >> l)).sum();
            assert_eq!(image.data.as_ref().unwrap().len(), texels * 4);
        }
        // Subtle: the grain stays within a few percent of the mean.
        let data = albedo.data.as_ref().unwrap();
        let (lo, hi) = data[..SIZE * SIZE * 4]
            .chunks(4)
            .fold((255u8, 0u8), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
        assert!(lo > 190 && hi < 255, "{lo}..{hi}");
    }
}
