//! Skinned rider body shared by the bike rider and the skier: one smooth procedural mesh per
//! material (one per material), all skinned to the same 17 bones. Each bone sits on a rig point and is
//! aimed at its child point, so the solved (or ragdoll) pose drives the skin directly; a bone
//! also stretches along its axis to the actual joint spacing of the rig that drives it.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use std::f32::consts::TAU;

/// Rig points a rider pose supplies, in this order.
#[derive(Clone, Copy)]
pub(crate) enum Pt {
    Pelvis,
    HipL,
    HipR,
    KneeL,
    KneeR,
    AnkleL,
    AnkleR,
    HeelL,
    HeelR,
    ToeL,
    ToeR,
    Waist,
    /// Shoulder line centre.
    Chest,
    Neck,
    /// Head centre.
    Head,
    ShoulderL,
    ShoulderR,
    ElbowL,
    ElbowR,
    WristL,
    WristR,
    /// Fist centre; the hand points from the wrist towards it.
    HandL,
    HandR,
    N,
}

const PN: usize = Pt::N as usize;

/// A rider pose in any one space (bike-local or world).
pub(crate) struct Pose {
    pub p: [Vec3; PN],
    /// Head orientation: identity looks along -Z with +Y up.
    pub head: Quat,
}

impl std::ops::Index<Pt> for Pose {
    type Output = Vec3;
    fn index(&self, i: Pt) -> &Vec3 {
        &self.p[i as usize]
    }
}

// Bone indices. The paired bones are `BASE + side`, side 0 = left.
const PELVIS: usize = 0;
const SPINE: usize = 1;
const CHEST: usize = 2;
const NECK: usize = 3;
const HEAD: usize = 4;
const THIGH: usize = 5;
const SHIN: usize = 7;
const FOOT: usize = 9;
const ARM: usize = 11;
const FOREARM: usize = 13;
const HAND: usize = 15;
const BONES: usize = 17;

/// Pelvis, spine, chest and neck joints along the bind pose's Y axis.
const CUTS: [f32; 3] = [0.22, 0.45, 0.57];
const HEAD_Y: f32 = 0.69;
const HIP_HALF: f32 = 0.1016;
const SHOULDER_HALF: f32 = 0.178;
const THIGH_LEN: f32 = 0.459;
const SHIN_LEN: f32 = 0.412;
const ARM_LEN: f32 = 0.31;
const FOREARM_LEN: f32 = 0.262;
/// Wrist to fist centre in the bind pose.
const HAND_LEN: f32 = 0.075;
/// Bind arms hang this far (rad) out from vertical.
const ARM_SPLAY: f32 = 0.436;

/// The rest pose the meshes are built in: standing, arms hanging a little out from vertical.
fn bind_points() -> [Vec3; PN] {
    let mut p = [Vec3::ZERO; PN];
    let mut set = |i: Pt, v: Vec3| p[i as usize] = v;
    set(Pt::Waist, Vec3::new(0.0, CUTS[0], 0.0));
    set(Pt::Chest, Vec3::new(0.0, CUTS[1], 0.0));
    set(Pt::Neck, Vec3::new(0.0, CUTS[2], 0.0));
    set(Pt::Head, Vec3::new(0.0, HEAD_Y, 0.0));
    let arm = Vec3::new(ARM_SPLAY.sin(), -ARM_SPLAY.cos(), 0.0);
    for (s, x) in [(0, -1.0), (1, 1.0)] {
        let at = |l: Pt, r: Pt| if s == 0 { l } else { r };
        let hip = Vec3::new(x * HIP_HALF, 0.0, 0.0);
        let knee = hip + Vec3::NEG_Y * THIGH_LEN;
        let ankle = knee + Vec3::NEG_Y * SHIN_LEN;
        let shoulder = Vec3::new(x * SHOULDER_HALF, CUTS[1], 0.0);
        let dir = arm * Vec3::new(x, 1.0, 1.0);
        let elbow = shoulder + dir * ARM_LEN;
        let wrist = elbow + dir * FOREARM_LEN;
        set(at(Pt::HipL, Pt::HipR), hip);
        set(at(Pt::KneeL, Pt::KneeR), knee);
        set(at(Pt::AnkleL, Pt::AnkleR), ankle);
        set(
            at(Pt::HeelL, Pt::HeelR),
            ankle + Vec3::new(0.0, -0.05, 0.04),
        );
        set(at(Pt::ToeL, Pt::ToeR), ankle + Vec3::new(0.0, -0.05, -0.2));
        set(at(Pt::ShoulderL, Pt::ShoulderR), shoulder);
        set(at(Pt::ElbowL, Pt::ElbowR), elbow);
        set(at(Pt::WristL, Pt::WristR), wrist);
        set(at(Pt::HandL, Pt::HandR), wrist + dir * HAND_LEN);
    }
    p
}

/// Bone `i`: origin point, aim-and-stretch child point (`None`: orientation comes from a frame).
fn bone_points(i: usize) -> (Pt, Option<Pt>) {
    let s = (i + 1) % 2; // meaningful for paired bones only (5 is left, 6 right, ...)
    let at = |l: Pt, r: Pt| if s == 0 { l } else { r };
    match i {
        PELVIS => (Pt::Pelvis, Some(Pt::Waist)),
        SPINE => (Pt::Waist, Some(Pt::Chest)),
        CHEST => (Pt::Chest, Some(Pt::Neck)),
        NECK => (Pt::Neck, Some(Pt::Head)),
        HEAD => (Pt::Head, None),
        THIGH | 6 => (at(Pt::HipL, Pt::HipR), Some(at(Pt::KneeL, Pt::KneeR))),
        SHIN | 8 => (at(Pt::KneeL, Pt::KneeR), Some(at(Pt::AnkleL, Pt::AnkleR))),
        FOOT | 10 => (at(Pt::AnkleL, Pt::AnkleR), None),
        ARM | 12 => (
            at(Pt::ShoulderL, Pt::ShoulderR),
            Some(at(Pt::ElbowL, Pt::ElbowR)),
        ),
        FOREARM | 14 => (at(Pt::ElbowL, Pt::ElbowR), Some(at(Pt::WristL, Pt::WristR))),
        _ => (at(Pt::WristL, Pt::WristR), Some(at(Pt::HandL, Pt::HandR))),
    }
}

/// Bones that aim at their child but keep their length (a hand is as big as it is).
fn rigid_length(i: usize) -> bool {
    i >= HAND
}

/// Rotation with Y along `y` and X as close to `x_hint` as orthogonality allows.
fn basis(x_hint: Vec3, y: Vec3) -> Quat {
    let y = y.normalize_or(Vec3::Y);
    let x = (x_hint - y * x_hint.dot(y))
        .try_normalize()
        .unwrap_or_else(|| y.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y)))
}

/// Bind rotation of bone `i`: local Y along the bind bone.
fn bind_rotation(i: usize, bind: &[Vec3; PN]) -> Quat {
    let (from, to) = bone_points(i);
    to.map_or(Quat::IDENTITY, |to| {
        let d = (bind[to as usize] - bind[from as usize]).normalize_or(Vec3::Y);
        Quat::from_rotation_arc(Vec3::Y, d)
    })
}

/// Joint transforms for `pose`, one per bone. Twist is stable: aimed limb bones take the minimal
/// arc from the bind direction, torso bones their pelvis / shoulder-line frames, the foot its
/// sole frame and the head the supplied orientation.
pub(crate) fn joints(pose: &Pose) -> [Transform; BONES] {
    let bind = bind_points();
    let pelvis = basis(
        pose[Pt::HipR] - pose[Pt::HipL],
        pose[Pt::Waist] - pose[Pt::Pelvis],
    );
    let chest = basis(
        pose[Pt::ShoulderR] - pose[Pt::ShoulderL],
        pose[Pt::Neck] - pose[Pt::Chest],
    );
    std::array::from_fn(|i| {
        let (from, to) = bone_points(i);
        let origin = pose[from];
        let mut scale = 1.0;
        if let Some(to) = to.filter(|_| !rigid_length(i)) {
            scale =
                (pose[to] - origin).length() / (bind[to as usize] - bind[from as usize]).length();
        }
        let rotation = match (i, to) {
            (PELVIS, _) => pelvis,
            (SPINE, _) => pelvis.slerp(chest, 0.5),
            (CHEST, _) => chest,
            (HEAD, _) => pose.head,
            (_, None) => {
                let (heel, toe) = if i == FOOT {
                    (Pt::HeelL, Pt::ToeL)
                } else {
                    (Pt::HeelR, Pt::ToeR)
                };
                let fwd = (pose[toe] - pose[heel]).normalize_or(Vec3::NEG_Z);
                let up = (origin - (pose[heel] + pose[toe]) * 0.5).reject_from(fwd);
                let y = up.normalize_or(fwd.any_orthonormal_vector());
                Quat::from_mat3(&Mat3::from_cols(y.cross(-fwd), y, -fwd))
            }
            (_, Some(to)) => {
                let bd = bind[to as usize] - bind[from as usize];
                Quat::from_rotation_arc(
                    bd.normalize_or(Vec3::Y),
                    (pose[to] - origin).normalize_or(Vec3::Y),
                )
            }
        };
        Transform {
            translation: origin,
            rotation: rotation * bind_rotation(i, &bind),
            scale: Vec3::new(1.0, scale.max(0.2), 1.0),
        }
    })
}

fn inverse_bindposes() -> Vec<Mat4> {
    let bind = bind_points();
    (0..BONES)
        .map(|i| {
            let origin = bind[bone_points(i).0 as usize];
            Mat4::from_rotation_translation(bind_rotation(i, &bind), origin).inverse()
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------------------------

/// Meshes, and the materials the caller supplies for them, in this order: top (jersey / jacket),
/// bottom (shorts / pants), skin, shoe, glove.
pub(crate) const REGIONS: usize = 5;

/// Footwear dimensions of the rig that drives the body, metres.
#[derive(Clone, Copy)]
pub(crate) struct Style {
    /// Ankle joint above the sole.
    pub sole: f32,
    /// Sole behind / ahead of the ankle.
    pub heel: f32,
    pub toe: f32,
    /// Stiff boot cuff above the ankle (0: low shoe).
    pub cuff: f32,
}

type Weights = [(usize, f32); 2];

#[derive(Default)]
struct Builder {
    pos: Vec<[f32; 3]>,
    joint: Vec<[u16; 4]>,
    weight: Vec<[f32; 4]>,
    idx: Vec<u32>,
}

/// Ring section: an ellipse spanned by `u` and `v` (already scaled by the radii).
#[derive(Clone, Copy)]
struct Ring {
    c: Vec3,
    u: Vec3,
    v: Vec3,
    w: Weights,
}

const SIDES: usize = 14;
/// Dome profile angles from the rim to the pole.
const DOME: [f32; 3] = [0.6, 1.1, 1.45];

/// Bone weights of a vertex at parameter `t` along a chain: bone `k` owns the span between
/// `cuts[k-1]` and `cuts[k]`, blending over `half` either side of each cut. Two strongest kept.
fn blend(t: f32, bones: &[usize], cuts: &[f32], half: f32) -> Weights {
    let s = |c: f32| {
        let x = ((t - (c - half)) / (2.0 * half)).clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let mut w: Vec<(usize, f32)> = bones
        .iter()
        .enumerate()
        .map(|(k, &b)| {
            let above = if k == 0 { 1.0 } else { s(cuts[k - 1]) };
            let below = if k == cuts.len() { 0.0 } else { s(cuts[k]) };
            (b, above - below)
        })
        .collect();
    w.sort_by(|a, b| b.1.total_cmp(&a.1));
    let sum = w[0].1 + w[1].1;
    [(w[0].0, w[0].1 / sum), (w[1].0, w[1].1 / sum)]
}

fn lookup(profile: &[(f32, f32, f32)], s: f32) -> (f32, f32) {
    let i = profile
        .windows(2)
        .position(|w| s <= w[1].0)
        .unwrap_or(profile.len() - 2);
    let (a, b) = (profile[i], profile[i + 1]);
    let t = ((s - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
    (a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
}

impl Builder {
    fn vertex(&mut self, p: Vec3, w: Weights) -> u32 {
        self.pos.push(p.to_array());
        self.joint.push([w[0].0 as u16, w[1].0 as u16, 0, 0]);
        self.weight.push([w[0].1, w[1].1, 0.0, 0.0]);
        self.pos.len() as u32 - 1
    }

    /// Skins `rings` into one closed surface; `dome` is the rounded cap depth at each end
    /// (0: flat).
    fn loft(&mut self, rings: &[Ring], dome: [f32; 2]) {
        let mut all = rings.to_vec();
        let cap = |a: Ring, b: Ring, depth: f32| {
            // Rings stepping from `a` outwards past `b`'s side: `b` is the neighbour inside.
            let axis = (a.c - b.c).normalize_or(Vec3::Y);
            DOME.map(|t| Ring {
                c: a.c + axis * depth * t.sin(),
                u: a.u * t.cos(),
                v: a.v * t.cos(),
                w: a.w,
            })
        };
        if dome[1] > 0.0 {
            all.extend(cap(rings[rings.len() - 1], rings[rings.len() - 2], dome[1]));
        }
        if dome[0] > 0.0 {
            let start = cap(rings[0], rings[1], dome[0]);
            all.splice(0..0, start.into_iter().rev());
        }
        let first = self.pos.len() as u32;
        for r in &all {
            for k in 0..SIDES {
                let a = TAU * k as f32 / SIDES as f32;
                self.vertex(r.c + r.u * a.cos() + r.v * a.sin(), r.w);
            }
        }
        let (a, z) = (all[0], all[all.len() - 1]);
        let ring = |i: usize, k: usize| first + (i * SIDES + k % SIDES) as u32;
        let mut tris = Vec::new();
        for i in 0..all.len() - 1 {
            for k in 0..SIDES {
                tris.push([ring(i, k), ring(i, k + 1), ring(i + 1, k)]);
                tris.push([ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k)]);
            }
        }
        for (end, last) in [(a, false), (z, true)] {
            let c = self.vertex(end.c, end.w);
            let i = if last { all.len() - 1 } else { 0 };
            for k in 0..SIDES {
                tris.push(if last {
                    [c, ring(i, k), ring(i, k + 1)]
                } else {
                    [c, ring(i, k + 1), ring(i, k)]
                });
            }
        }
        // Rings run with (u, v, axis) right-handed; mirrored rings wind the other way.
        let flip = a.u.cross(a.v).dot(z.c - a.c) < 0.0;
        for [x, y, w] in tris {
            self.idx.extend(if flip { [x, w, y] } else { [x, y, w] });
        }
    }

    /// A limb through bind joints `j` (two segments driven by `bones`): `profile` is
    /// `(s, width, depth)` with `s` 0..1 on the first segment and 1..2 on the second.
    fn limb(
        &mut self,
        bones: [usize; 2],
        j: [Vec3; 3],
        profile: &[(f32, f32, f32)],
        dome: [f32; 2],
    ) {
        const S: [f32; 13] = [
            0.0, 0.12, 0.35, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 1.9, 2.0,
        ];
        let d = [(j[1] - j[0]).normalize(), (j[2] - j[1]).normalize()];
        let rings: Vec<Ring> = S
            .iter()
            .map(|&s| {
                let c = if s <= 1.0 {
                    j[0].lerp(j[1], s)
                } else {
                    j[1].lerp(j[2], s - 1.0)
                };
                let k = ((s - 0.8) / 0.4).clamp(0.0, 1.0);
                let axis = d[0].lerp(d[1], k * k * (3.0 - 2.0 * k)).normalize();
                let (rx, rz) = lookup(profile, s);
                let v = Vec3::Z.reject_from(axis).normalize();
                Ring {
                    c,
                    u: v.cross(axis) * rx,
                    v: v * rz,
                    w: blend(s, &bones, &[1.0], 0.2),
                }
            })
            .collect();
        self.loft(&rings, dome);
    }

    /// A body of revolution-ish section stack along Y: `(y, width, depth)`.
    fn stack(
        &mut self,
        sections: &[(f32, f32, f32)],
        weights: impl Fn(f32) -> Weights,
        dome: [f32; 2],
    ) {
        let rings: Vec<Ring> = sections
            .iter()
            .map(|&(y, rx, rz)| Ring {
                c: Vec3::new(0.0, y, 0.0),
                u: Vec3::Z * rz,
                v: Vec3::X * rx,
                w: weights(y),
            })
            .collect();
        self.loft(&rings, dome);
    }

    fn into_mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.pos)
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(self.joint),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, self.weight)
        .with_inserted_indices(Indices::U32(self.idx));
        mesh.compute_smooth_normals();
        mesh
    }
}

/// Meshes in material order, in the bind pose.
fn build(style: Style) -> [Mesh; REGIONS] {
    let bind = bind_points();
    let [mut top, mut bottom, mut skin, mut shoe, mut glove] = <[Builder; REGIONS]>::default();
    let torso = |y: f32| blend(y, &[PELVIS, SPINE, CHEST, NECK], &CUTS, 0.05);

    bottom.stack(
        &[
            (-0.10, 0.06, 0.07),
            (-0.05, 0.14, 0.10),
            (0.03, 0.17, 0.12),
            (0.12, 0.165, 0.115),
            (0.21, 0.152, 0.108),
        ],
        |y| blend(y, &[PELVIS, SPINE], &CUTS[..1], 0.05),
        [0.03, 0.0],
    );
    top.stack(
        &[
            (0.14, 0.180, 0.132),
            (0.24, 0.168, 0.120),
            (0.34, 0.164, 0.114),
            (0.42, 0.185, 0.114),
            (0.49, 0.150, 0.100),
            (0.53, 0.085, 0.075),
            (0.56, 0.052, 0.052),
        ],
        torso,
        [0.0, 0.02],
    );

    for s in 0..2 {
        let at = |l: Pt, r: Pt| bind[if s == 0 { l } else { r } as usize];
        let (hip, knee, ankle) = (
            at(Pt::HipL, Pt::HipR),
            at(Pt::KneeL, Pt::KneeR),
            at(Pt::AnkleL, Pt::AnkleR),
        );
        bottom.limb(
            [THIGH + s, SHIN + s],
            [hip, knee, ankle],
            &[
                (0.0, 0.085, 0.09),
                (0.4, 0.084, 0.088),
                (0.8, 0.066, 0.068),
                (1.0, 0.058, 0.060),
                (1.15, 0.058, 0.062),
                (1.4, 0.060, 0.066),
                (1.8, 0.042, 0.045),
                (2.0, 0.040, 0.042),
            ],
            [0.07, 0.0],
        );
        let (shoulder, elbow, wrist, hand) = (
            at(Pt::ShoulderL, Pt::ShoulderR),
            at(Pt::ElbowL, Pt::ElbowR),
            at(Pt::WristL, Pt::WristR),
            at(Pt::HandL, Pt::HandR),
        );
        top.limb(
            [ARM + s, FOREARM + s],
            [shoulder, elbow, wrist],
            &[
                (0.0, 0.052, 0.054),
                (0.5, 0.047, 0.049),
                (0.9, 0.041, 0.041),
                (1.0, 0.039, 0.039),
                (1.3, 0.041, 0.043),
                (1.7, 0.034, 0.035),
                (2.0, 0.031, 0.031),
            ],
            [0.055, 0.0],
        );
        // Mitt: a short stack along the arm line, wrist to past the fist centre.
        let dir = (hand - wrist).normalize();
        let v = Vec3::Z.reject_from(dir).normalize();
        let mitt: Vec<Ring> = [
            (-0.03, 0.030, 0.028),
            (0.0, 0.034, 0.030),
            (0.04, 0.048, 0.036),
            (0.075, 0.052, 0.040),
            (0.10, 0.040, 0.032),
        ]
        .iter()
        .map(|&(t, rx, rz)| Ring {
            c: wrist + dir * t,
            u: v.cross(dir) * rx,
            v: v * rz,
            w: [(HAND + s, 1.0), (HAND + s, 0.0)],
        })
        .collect();
        glove.loft(&mitt, [0.0, 0.03]);

        shoe_stack(&mut shoe, &style, ankle, FOOT + s, SHIN + s);
    }

    // Neck and head share the skin.
    let neck = |y: f32| blend(y, &[CHEST, NECK, HEAD], &[CUTS[2], HEAD_Y - 0.06], 0.03);
    skin.stack(
        &[
            (CUTS[1] + 0.05, 0.05, 0.05),
            (CUTS[2], 0.048, 0.05),
            (HEAD_Y - 0.03, 0.046, 0.05),
        ],
        neck,
        [0.0, 0.0],
    );
    skin.stack(
        &[
            (HEAD_Y - 0.06, 0.056, 0.07),
            (HEAD_Y - 0.01, 0.076, 0.092),
            (HEAD_Y + 0.05, 0.080, 0.097),
            (HEAD_Y + 0.09, 0.062, 0.078),
        ],
        |_| [(HEAD, 1.0), (HEAD, 0.0)],
        [0.04, 0.025],
    );

    [top, bottom, skin, shoe, glove].map(Builder::into_mesh)
}

/// Shoe (and boot cuff) around the bind ankle `a`; the cuff follows the shin above the ankle.
fn shoe_stack(b: &mut Builder, st: &Style, a: Vec3, foot: usize, shin: usize) {
    let w = [(foot, 1.0), (foot, 0.0)];
    let (heel, toe) = (st.heel + 0.03, st.toe + 0.03);
    let height = 0.095;
    let rings: Vec<Ring> = [
        (0.0, 0.034, 0.80),
        (0.18, 0.046, 1.0),
        (0.45, 0.048, 0.95),
        (0.75, 0.048, 0.80),
        (0.95, 0.038, 0.55),
    ]
    .iter()
    .map(|&(f, rx, h)| {
        let ry = height * 0.5 * h;
        Ring {
            c: a + Vec3::new(0.0, ry - st.sole, heel - (heel + toe) * f),
            u: Vec3::Y * ry,
            v: Vec3::X * rx,
            w,
        }
    })
    .collect();
    b.loft(&rings, [0.03, 0.03]);
    if st.cuff > 0.0 {
        let cuff: Vec<Ring> = [-0.03, 0.03, 0.08, st.cuff]
            .iter()
            .map(|&y| Ring {
                c: a + Vec3::Y * y,
                u: Vec3::Z * 0.082,
                v: Vec3::X * 0.074,
                w: blend(y, &[foot, shin], &[0.05], 0.04),
            })
            .collect();
        b.loft(&cuff, [0.0, 0.01]);
    }
}

/// Marks a skin joint entity; the value is its bone index.
#[derive(Component)]
pub(crate) struct SkinJoint(pub usize);

/// Spawns the 17 joint entities and the five skinned meshes (top, bottom, skin, shoe, glove) under `root`.
/// `marker` is added to every joint so each rider's driver finds only its own.
pub(crate) fn spawn(
    c: &mut Commands,
    meshes: &mut Assets<Mesh>,
    bindposes: &mut Assets<SkinnedMeshInverseBindposes>,
    root: Entity,
    style: Style,
    mats: [Handle<StandardMaterial>; REGIONS],
    marker: impl Bundle + Clone,
) {
    let joints: Vec<Entity> = (0..BONES)
        .map(|i| {
            c.spawn((
                SkinJoint(i),
                marker.clone(),
                Transform::default(),
                ChildOf(root),
            ))
            .id()
        })
        .collect();
    let inverse_bindposes = bindposes.add(SkinnedMeshInverseBindposes::from(inverse_bindposes()));
    for (mesh, mat) in build(style).into_iter().zip(mats) {
        c.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(mat),
            SkinnedMesh {
                inverse_bindposes: inverse_bindposes.clone(),
                joints: joints.clone(),
            },
            // The skin bounds are the bind pose's, not the animated pose's.
            NoFrustumCulling,
            Transform::default(),
            ChildOf(root),
        ));
    }
}

/// Drives the joint entities of one rider.
pub(crate) fn pose_joints<'a>(
    pose: &Pose,
    joints: impl IntoIterator<Item = (&'a SkinJoint, Mut<'a, Transform>)>,
) {
    let t = self::joints(pose);
    for (j, mut tf) in joints {
        *tf = t[j.0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A standing pose equal to the bind pose.
    fn bind_pose() -> Pose {
        Pose {
            p: bind_points(),
            head: Quat::IDENTITY,
        }
    }

    #[test]
    fn the_bind_pose_is_the_identity_skin() {
        let ibp = inverse_bindposes();
        for (t, inv) in joints(&bind_pose()).iter().zip(&ibp) {
            let m = t.to_matrix() * *inv;
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-4), "{m:?}");
        }
    }

    #[test]
    fn a_bent_limb_keeps_its_joints_on_the_rig_points() {
        let mut pose = bind_pose();
        // Raise the left arm forward: elbow bent, wrist well in front.
        pose.p[Pt::ElbowL as usize] = pose[Pt::ShoulderL] + Vec3::new(0.0, 0.0, -0.31);
        pose.p[Pt::WristL as usize] = pose[Pt::ElbowL] + Vec3::new(0.0, 0.262, 0.0);
        pose.p[Pt::HandL as usize] = pose[Pt::WristL] + Vec3::new(0.0, 0.075, 0.0);
        let t = joints(&pose);
        let ibp = inverse_bindposes();
        let bind = bind_points();
        // The elbow ring is shared by both arm bones, so both carry the bind elbow to the pose elbow.
        for bone in [ARM, FOREARM] {
            let m = t[bone].to_matrix() * ibp[bone];
            let moved = m.transform_point3(bind[Pt::ElbowL as usize]);
            assert!(
                moved.distance(pose[Pt::ElbowL]) < 1e-4,
                "bone {bone}: {moved:?}"
            );
        }
        let m = t[FOREARM].to_matrix() * ibp[FOREARM];
        let wrist = m.transform_point3(bind[Pt::WristL as usize]);
        assert!(wrist.distance(pose[Pt::WristL]) < 1e-4);
    }

    #[test]
    fn every_vertex_has_two_normalised_weights_and_valid_bones() {
        let style = Style {
            sole: 0.13,
            heel: 0.12,
            toe: 0.17,
            cuff: 0.2,
        };
        for mesh in build(style) {
            let Some(VertexAttributeValues::Float32x4(w)) =
                mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
            else {
                panic!("no weights");
            };
            let Some(VertexAttributeValues::Uint16x4(j)) =
                mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX)
            else {
                panic!("no joints");
            };
            let Some(VertexAttributeValues::Float32x3(p)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("no positions");
            };
            assert!(!p.is_empty() && p.len() == w.len() && w.len() == j.len());
            for (w, j) in w.iter().zip(j) {
                assert!((w[0] + w[1] - 1.0).abs() < 1e-5 && w[0] >= 0.0 && w[1] >= 0.0);
                assert!((j[0] as usize) < BONES && (j[1] as usize) < BONES);
            }
            assert!(p.iter().flatten().all(|v| v.is_finite()));
        }
    }
}
