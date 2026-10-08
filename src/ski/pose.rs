//! Shared skier skeleton contract: named joints, bone tree, equipment dimensions and the
//! world-space pose consumed by the renderer, the debug skeleton and the crash ragdoll.

use bevy::math::Vec3;

/// Named skier points, world space. `N` must stay last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J {
    Pelvis,
    HipL,
    HipR,
    KneeL,
    KneeR,
    AnkleL,
    AnkleR,
    /// Boot sole points, resting on the ski top surface.
    HeelL,
    HeelR,
    ToeL,
    ToeR,
    Waist,
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
    /// Fist centre, on the pole shaft.
    HandL,
    HandR,
    /// Ski top surface under the boot centre.
    BindL,
    BindR,
    /// Front end of the upturned shovel.
    TipL,
    TipR,
    TailL,
    TailR,
    /// Pole grip top (above the fist) and basket tip.
    PoleTopL,
    PoleTopR,
    PoleTipL,
    PoleTipR,
    N,
}

pub(crate) const JOINTS: usize = J::N as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    Centre,
    Left,
    Right,
}

/// Skeleton tree (body) followed by equipment links. Every joint appears.
pub(crate) const BONES: &[(J, J)] = &[
    (J::Pelvis, J::Waist),
    (J::Waist, J::Chest),
    (J::Chest, J::Neck),
    (J::Neck, J::Head),
    (J::Pelvis, J::HipL),
    (J::HipL, J::KneeL),
    (J::KneeL, J::AnkleL),
    (J::AnkleL, J::HeelL),
    (J::AnkleL, J::ToeL),
    (J::HeelL, J::ToeL),
    (J::Pelvis, J::HipR),
    (J::HipR, J::KneeR),
    (J::KneeR, J::AnkleR),
    (J::AnkleR, J::HeelR),
    (J::AnkleR, J::ToeR),
    (J::HeelR, J::ToeR),
    (J::Chest, J::ShoulderL),
    (J::ShoulderL, J::ElbowL),
    (J::ElbowL, J::WristL),
    (J::WristL, J::HandL),
    (J::Chest, J::ShoulderR),
    (J::ShoulderR, J::ElbowR),
    (J::ElbowR, J::WristR),
    (J::WristR, J::HandR),
];
/// Equipment links: ski tail-binding-tip lines and pole shafts.
pub(crate) const GEAR: &[(J, J)] = &[
    (J::TailL, J::BindL),
    (J::BindL, J::TipL),
    (J::TailR, J::BindR),
    (J::BindR, J::TipR),
    (J::PoleTopL, J::PoleTipL),
    (J::PoleTopR, J::PoleTipR),
];

pub(crate) fn side(j: J) -> Side {
    use J::*;
    match j {
        HipL | KneeL | AnkleL | HeelL | ToeL | ShoulderL | ElbowL | WristL | HandL | BindL
        | TipL | TailL | PoleTopL | PoleTipL => Side::Left,
        HipR | KneeR | AnkleR | HeelR | ToeR | ShoulderR | ElbowR | WristR | HandR | BindR
        | TipR | TailR | PoleTopR | PoleTipR => Side::Right,
        _ => Side::Centre,
    }
}

// Body dimensions (metres), an adult of ~1.78 m, proportioned like the retail rider.
pub(crate) const THIGH: f32 = 0.459;
pub(crate) const SHIN: f32 = 0.412;
pub(crate) const UPPER_ARM: f32 = 0.31;
pub(crate) const FOREARM: f32 = 0.262;
pub(crate) const HAND: f32 = 0.08;
pub(crate) const HIP_HALF_WIDTH: f32 = 0.1016;
pub(crate) const SHOULDER_HALF_WIDTH: f32 = 0.178;
pub(crate) const PELVIS_TO_WAIST: f32 = 0.20;
pub(crate) const WAIST_TO_CHEST: f32 = 0.214;
pub(crate) const CHEST_TO_NECK: f32 = 0.174;
pub(crate) const NECK_TO_HEAD: f32 = 0.115;
/// Ankle joint above the ski top surface (boot + binding stack).
pub(crate) const ANKLE_ABOVE_SKI: f32 = 0.13;
/// Boot sole from the binding point: heel behind, toe ahead.
pub(crate) const HEEL_BACK: f32 = 0.12;
pub(crate) const TOE_FORWARD: f32 = 0.17;

// Equipment.
/// Binding point to tip / tail along the ski. The retail ski bones put the binding 46 % of the way
/// from the tail; the visible ski reaches past its end bones, so the ski is longer than their
/// 1.26 m spacing.
pub(crate) const SKI_FRONT: f32 = 0.81;
pub(crate) const SKI_BACK: f32 = 0.69;
pub(crate) const SKI_WIDTH: f32 = 0.09;
/// Ski base to top surface.
pub(crate) const SKI_THICKNESS: f32 = 0.025;
/// Shovel rise of the tip above the base plane.
pub(crate) const SKI_TIP_RISE: f32 = 0.07;
/// Centre-to-centre distance of the two skis in a neutral stance (retail boots sit 0.366 m apart).
pub(crate) const STANCE_WIDTH: f32 = 0.366;
pub(crate) const POLE_LENGTH: f32 = 1.22;
/// Fist centre below the pole top.
pub(crate) const POLE_GRIP: f32 = 0.07;

/// World-space skier pose.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkierPose {
    pub p: [Vec3; JOINTS],
    /// Unit top-surface normal of the left / right ski.
    pub ski_up: [Vec3; 2],
    /// Unit forward of the chest (torso slab orientation).
    pub facing: Vec3,
    /// Unit forward of the face (helmet / goggles orientation); the head holds its gaze while the
    /// chest twists.
    pub gaze: Vec3,
}

impl std::ops::Index<J> for SkierPose {
    type Output = Vec3;
    fn index(&self, j: J) -> &Vec3 {
        &self.p[j as usize]
    }
}

impl std::ops::IndexMut<J> for SkierPose {
    fn index_mut(&mut self, j: J) -> &mut Vec3 {
        &mut self.p[j as usize]
    }
}

impl SkierPose {
    /// Derives `ski_up`, `facing` and `gaze` from the points alone (used by the ragdoll).
    pub fn from_points(p: [Vec3; JOINTS]) -> Self {
        let up_of = |ankle: J, bind: J, tip: J, tail: J| {
            let along = (p[tip as usize] - p[tail as usize]).normalize_or_zero();
            let raw = p[ankle as usize] - p[bind as usize];
            (raw - along * raw.dot(along)).normalize_or(Vec3::Y)
        };
        let right = (p[J::ShoulderR as usize] - p[J::ShoulderL as usize]).normalize_or(Vec3::X);
        let up = (p[J::Neck as usize] - p[J::Waist as usize]).normalize_or(Vec3::Y);
        Self {
            p,
            ski_up: [
                up_of(J::AnkleL, J::BindL, J::TipL, J::TailL),
                up_of(J::AnkleR, J::BindR, J::TipR, J::TailR),
            ],
            facing: up.cross(right).normalize_or(Vec3::NEG_Z),
            gaze: up.cross(right).normalize_or(Vec3::NEG_Z),
        }
    }
}
