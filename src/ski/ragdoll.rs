//! Detached skier: an articulated rigid-body ragdoll (`crate::rigid`).
//!
//! Eighteen segments carry anthropometric masses and inertias: pelvis, abdomen, thorax, head, upper
//! arms, forearms, hands, thighs, shins, boot-and-ski and poles. They are pinned together at the
//! skeleton's joints. Every joint keeps an anatomical swing/twist range, measured from the upright
//! stance, and breaks (its range opens by `BROKEN_SLACK`) once the torque holding it at the limit
//! exceeds its strength. Every pair of segments that is not jointed collides as rendered-size
//! capsules, so no limb passes through another limb or the body, broken or not. The snow gives
//! Coulomb friction (skis glide along their length and bite across it), and the body sleeps at rest.

use bevy::math::{Quat, Vec3};
use bevy::prelude::Resource;

use super::pose::{J, JOINTS, SKI_THICKNESS, SKI_WIDTH, SkierPose};
use crate::rigid::*;

/// Seed velocities beyond these (pose-difference glitches) are clamped: m/s, rad/s.
const MAX_SEED_SPEED: f32 = 40.0;
const MAX_SEED_SPIN: f32 = 25.0;

/// Segments. Each has an anatomical frame: X right, Y up, Z back when standing with the arms
/// hanging; Y runs along the segment. The ski shares its boot's frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum B {
    Pelvis,
    Abdomen,
    Thorax,
    Head,
    UpperArmL,
    UpperArmR,
    ForearmL,
    ForearmR,
    HandL,
    HandR,
    ThighL,
    ThighR,
    ShinL,
    ShinR,
    BootL,
    BootR,
    SkiL,
    SkiR,
    PoleL,
    PoleR,
}

const BODIES: usize = 20;

/// Every joint in enum order, to map array indices back to `J`.
const ALL: [J; JOINTS] = {
    use J::*;
    [
        Pelvis, HipL, HipR, KneeL, KneeR, AnkleL, AnkleR, HeelL, HeelR, ToeL, ToeR, Waist, Chest,
        Neck, Head, ShoulderL, ShoulderR, ElbowL, ElbowR, WristL, WristR, HandL, HandR, BindL,
        BindR, TipL, TipR, TailL, TailR, PoleTopL, PoleTopR, PoleTipL, PoleTipR,
    ]
};

/// The segment a pose point is rigidly fixed to.
fn owner(j: J) -> B {
    use J::*;
    match j {
        Pelvis | HipL | HipR => B::Pelvis,
        Waist => B::Abdomen,
        Chest => B::Thorax,
        Neck | Head => B::Head,
        ShoulderL => B::UpperArmL,
        ShoulderR => B::UpperArmR,
        ElbowL => B::ForearmL,
        ElbowR => B::ForearmR,
        WristL | HandL => B::HandL,
        WristR | HandR => B::HandR,
        KneeL => B::ShinL,
        KneeR => B::ShinR,
        AnkleL | HeelL | ToeL => B::BootL,
        AnkleR | HeelR | ToeR => B::BootR,
        BindL | TipL | TailL => B::SkiL,
        BindR | TipR | TailR => B::SkiR,
        PoleTopL | PoleTipL => B::PoleL,
        PoleTopR | PoleTipR => B::PoleR,
        N => unreachable!(),
    }
}
/// Ankle inside a ski boot: the shell allows forward flex and little else; the boot top breaks
/// the shin under a forward or sideways lever.
const ANKLE: Limit = Limit {
    stiffness: 6000.0,
    ..hinged(
        [-10.0, 40.0],
        [-6.0, 6.0],
        [-6.0, 6.0],
        [500.0, 375.0, 190.0],
    )
};
/// Binding: holds the ski rigidly and releases it before the leg gives way: forward (the heel)
/// and twist (the toe), never sideways. The thresholds are about 2.5 times the DIN 7 torques for a
/// 75 kg skier (270 N m forward, 70 N m twist) because this limp body, with no working leg
/// muscles, levers the boot harder in an ordinary drop than a skier does.
const BINDING: Limit = Limit {
    stiffness: 20000.0,
    ..hinged([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [700.0, 1.0e4, 180.0])
};
/// Pull-out force of a pole from a gloved fist with its strap, N (about 90 kgf). The fist grips
/// about 400 N and the wrist strap, which on a modern pole opens in a fall rather than dragging the
/// arm after the pole, adds the rest. Calibrated, not a rating: a skier toppling onto a pole in the
/// snow loads the grip with up to 750 N and keeps it, a 22 m/s dive with 1400 N and loses it.
const GRIP_FORCE: f32 = 900.0;

impl From<B> for usize {
    fn from(b: B) -> usize {
        b as usize
    }
}

fn side_of(i: usize, l: B, r: B) -> B {
    if i == 0 { l } else { r }
}

/// Mass elements (kg, about 75 kg of skier plus boots, skis and poles) and collision capsules, sized
/// like the rendered skier.
fn parts(p: &SkierPose) -> Vec<Part> {
    let slab = |body: B, a: J, b: J, dx: f32, dz: f32, mass: f32| {
        let (a, b) = (p[a], p[b]);
        let y = unit(b - a);
        let z = unit(p.facing.reject_from(y));
        let x = y.cross(z) * (0.5 * (dx - dz));
        [
            part(body, a - x, b - x, 0.5 * dz, mass * 0.5),
            part(body, a + x, b + x, 0.5 * dz, mass * 0.5),
        ]
    };
    let mut out = Vec::new();
    out.extend(slab(B::Pelvis, J::Pelvis, J::Waist, 0.30, 0.20, 8.0));
    out.extend(slab(B::Abdomen, J::Waist, J::Chest, 0.34, 0.21, 10.4));
    out.extend(slab(B::Thorax, J::Chest, J::Neck, 0.40, 0.22, 16.0));
    out.push(part(B::Head, p[J::Head], p[J::Head], 0.11, 5.0));
    out.push(part(B::Head, p[J::Neck], p[J::Head], 0.04, 1.1));
    for i in 0..2 {
        let pick = |l: J, r: J| p[if i == 0 { l } else { r }];
        let [hip, knee, ankle, heel, toe] = [
            pick(J::HipL, J::HipR),
            pick(J::KneeL, J::KneeR),
            pick(J::AnkleL, J::AnkleR),
            pick(J::HeelL, J::HeelR),
            pick(J::ToeL, J::ToeR),
        ];
        let [shoulder, elbow, wrist, hand, top, tip] = [
            pick(J::ShoulderL, J::ShoulderR),
            pick(J::ElbowL, J::ElbowR),
            pick(J::WristL, J::WristR),
            pick(J::HandL, J::HandR),
            pick(J::PoleTopL, J::PoleTopR),
            pick(J::PoleTipL, J::PoleTipR),
        ];
        let [tail, bind, ski_tip] = [
            pick(J::TailL, J::TailR),
            pick(J::BindL, J::BindR),
            pick(J::TipL, J::TipR),
        ];
        let up = p.ski_up[i];
        out.push(part(B::Pelvis, p[J::Pelvis], hip, 0.07, 1.35));
        // The upper arm collides from below the shoulder, which sits inside the torso slab.
        let upper = side_of(i, B::UpperArmL, B::UpperArmR);
        out.push(Part {
            collide: false,
            ..part(upper, shoulder, elbow, 0.05, 2.1)
        });
        out.push(part(upper, shoulder.lerp(elbow, 0.45), elbow, 0.05, 0.0));
        out.push(part(
            side_of(i, B::ForearmL, B::ForearmR),
            elbow,
            wrist,
            0.04,
            1.2,
        ));
        let fist = side_of(i, B::HandL, B::HandR);
        out.push(part(fist, wrist, hand, 0.0375, 0.15));
        out.push(part(fist, hand, hand, 0.055, 0.3));
        out.push(part(
            side_of(i, B::ThighL, B::ThighR),
            hip,
            knee,
            0.075,
            7.5,
        ));
        out.push(part(
            side_of(i, B::ShinL, B::ShinR),
            knee,
            ankle,
            0.0575,
            3.5,
        ));
        let boot = side_of(i, B::BootL, B::BootR);
        out.push(part(
            boot,
            ankle,
            ankle + unit(knee - ankle) * 0.20,
            0.07,
            1.5,
        ));
        out.push(part(boot, heel + up * 0.05, toe + up * 0.05, 0.055, 1.4));
        // Ski (with its binding plate): two thin capsules along its edges, tail to binding and
        // binding to tip, spanning its width and thickness.
        let ski = side_of(i, B::SkiL, B::SkiR);
        let r = SKI_THICKNESS * 0.5;
        let edge = unit(up.cross(toe - heel)) * (SKI_WIDTH * 0.5 - r);
        for (a, b) in [(tail, bind), (bind, ski_tip)] {
            for e in [edge, -edge] {
                let off = e - up * r;
                out.push(Part {
                    ski: true,
                    ..part(ski, a + off, b + off, r, 0.875)
                });
            }
        }
        // The pole: its grip section runs through the fist along the wrist, so it may lie on its
        // own forearm; the rest of the shaft and the basket collide with everything.
        let pole = side_of(i, B::PoleL, B::PoleR);
        let grip_end = hand - unit(top - tip) * 0.25;
        out.push(Part {
            ignore: Some(side_of(i, B::ForearmL, B::ForearmR) as usize),
            ..part(pole, top, grip_end, 0.007, 0.06)
        });
        out.push(part(pole, grip_end, tip, 0.007, 0.14));
        out.push(part(
            pole,
            tip + unit(top - tip) * 0.08,
            tip + unit(top - tip) * 0.08,
            0.06,
            0.05,
        ));
    }
    out
}

/// Anatomical frame of every segment in pose `p`.
fn frames(p: &SkierPose) -> [Quat; BODIES] {
    let hips = unit(p[J::HipR] - p[J::HipL]);
    let shoulders = unit(p[J::ShoulderR] - p[J::ShoulderL]);
    let mut f = [Quat::IDENTITY; BODIES];
    f[B::Pelvis as usize] = basis(hips, p[J::Waist] - p[J::Pelvis]);
    f[B::Abdomen as usize] = basis(hips + shoulders, p[J::Chest] - p[J::Waist]);
    let thorax = basis(shoulders, p[J::Neck] - p[J::Chest]);
    f[B::Thorax as usize] = thorax;
    let up = p[J::Head] - p[J::Neck];
    f[B::Head as usize] = basis(up.cross(-p.gaze), up);
    for i in 0..2 {
        let pick = |l: J, r: J| p[if i == 0 { l } else { r }];
        let [hip, knee, ankle, heel, toe] = [
            pick(J::HipL, J::HipR),
            pick(J::KneeL, J::KneeR),
            pick(J::AnkleL, J::AnkleR),
            pick(J::HeelL, J::HeelR),
            pick(J::ToeL, J::ToeR),
        ];
        let [shoulder, elbow, wrist, hand, top, tip] = [
            pick(J::ShoulderL, J::ShoulderR),
            pick(J::ElbowL, J::ElbowR),
            pick(J::WristL, J::WristR),
            pick(J::HandL, J::HandR),
            pick(J::PoleTopL, J::PoleTopR),
            pick(J::PoleTipL, J::PoleTipR),
        ];
        let set = |f: &mut [Quat; BODIES], l: B, r: B, q: Quat| f[side_of(i, l, r) as usize] = q;
        let up = p.ski_up[i];
        let boot = basis(up.cross(heel - toe), up);
        // Knees bend backwards: their flexion axis is the elbow's with root and end swapped.
        let knee_x = hinge(ankle, knee, hip, boot * Vec3::X);
        set(&mut f, B::ThighL, B::ThighR, basis(knee_x, hip - knee));
        set(&mut f, B::ShinL, B::ShinR, basis(knee_x, knee - ankle));
        set(&mut f, B::BootL, B::BootR, boot);
        set(&mut f, B::SkiL, B::SkiR, boot);
        let elbow_x = hinge(shoulder, elbow, wrist, thorax * Vec3::X);
        set(
            &mut f,
            B::UpperArmL,
            B::UpperArmR,
            basis(elbow_x, shoulder - elbow),
        );
        set(
            &mut f,
            B::ForearmL,
            B::ForearmR,
            basis(elbow_x, elbow - wrist),
        );
        set(&mut f, B::HandL, B::HandR, basis(elbow_x, wrist - hand));
        set(&mut f, B::PoleL, B::PoleR, basis(elbow_x, top - tip));
    }
    f
}

/// `boot` (X right, Y up, Z back) turned into the ski boot's ankle range relative to the shin of
/// the leg `hip`-`knee`-`ankle`; the animation keeps airborne skis on the feet with it.
pub(crate) fn boot_in_ankle_range(
    hip: Vec3,
    knee: Vec3,
    ankle: Vec3,
    boot: Quat,
    left: bool,
) -> Quat {
    let shin = basis(hinge(ankle, knee, hip, boot * Vec3::X), knee - ankle);
    shin * clamp_rotation(shin.inverse() * boot, &ANKLE, left, 0.0)
}

#[derive(Resource, Default)]
pub(crate) struct SkiRagdoll {
    body: Option<Body>,
}

impl SkiRagdoll {
    pub(crate) fn reset(&mut self) {
        self.body = None;
    }

    pub(crate) fn active(&self) -> bool {
        self.body.is_some()
    }

    /// Detaches the skier from the rig at `current`; velocities come from the difference with
    /// `previous` over `dt`.
    pub(crate) fn activate(&mut self, previous: &SkierPose, current: &SkierPose, dt: f32) {
        let dt = dt.max(1e-4);
        let mut body = Body::new(current);
        let (x0, q0) = (centres(&parts(previous), BODIES), frames(previous));
        for (k, b) in body.solver.bodies.iter_mut().enumerate() {
            b.v = ((b.x - x0[k]) / dt).clamp_length_max(MAX_SEED_SPEED);
            let dq = b.q * q0[k].inverse();
            let dq = if dq.w < 0.0 { -dq } else { dq };
            b.w = (dq.to_scaled_axis() / dt).clamp_length_max(MAX_SEED_SPIN);
        }
        for k in 0..body.solver.joints.len() {
            body.solver.joints[k].allow = body.solver.excess(k) * 1.2 + DEG;
        }
        body.solver.settle();
        self.body = Some(body);
    }

    pub(crate) fn step(&mut self, dt: f32) {
        if let Some(body) = &mut self.body {
            body.solver.step(dt);
        }
    }

    pub(crate) fn pose(&self) -> Option<SkierPose> {
        self.body.as_ref().map(Body::pose)
    }

    /// Pelvis position and velocity (camera target).
    pub(crate) fn centre(&self) -> Option<(Vec3, Vec3)> {
        self.body.as_ref().map(|b| {
            let (k, local) = b.points[J::Pelvis as usize];
            (b.solver.bodies[k].point(local), b.solver.bodies[k].v)
        })
    }

    pub(crate) fn sleeping(&self) -> bool {
        self.body.as_ref().is_some_and(|b| b.solver.sleeping)
    }

    /// Joints broken so far, in the order they gave way.
    pub(crate) fn injuries(&self) -> &[&'static str] {
        self.body.as_ref().map_or(&[], |b| &b.solver.injuries)
    }

    /// Skis whose binding has released.
    pub(crate) fn skis_off(&self) -> usize {
        self.released(B::SkiL, B::SkiR)
    }

    /// Poles that have been pulled out of the hand.
    pub(crate) fn poles_off(&self) -> usize {
        self.released(B::PoleL, B::PoleR)
    }

    /// Released detachable joints holding `left` or `right`.
    fn released(&self, left: B, right: B) -> usize {
        self.body.as_ref().map_or(0, |b| {
            b.solver
                .joints
                .iter()
                .filter(|j| {
                    j.detachable
                        && j.broken
                        && (j.child == left as usize || j.child == right as usize)
                })
                .count()
        })
    }
}

struct Body {
    solver: Solver,
    /// Owner segment and local position of every pose point.
    points: [(usize, Vec3); JOINTS],
}

impl Body {
    fn new(p: &SkierPose) -> Self {
        let parts = parts(p);
        let f = frames(p);
        let com = centres(&parts, BODIES);
        let mut solver = Solver::new(&parts, &f, &com);
        let points = std::array::from_fn(|i| {
            let k = owner(ALL[i]) as usize;
            (k, f[k].inverse() * (p.p[i] - com[k]))
        });
        let mut join = |parent: B, child: B, at: J, limit: Option<Limit>, name: &'static str| {
            let mirror = matches!(crate::ski::pose::side(at), crate::ski::pose::Side::Left);
            let joint = Joint {
                limit,
                mirror,
                frame: limit.map_or(Quat::IDENTITY, |l| l.frame(mirror)),
                name,
                detachable: matches!(child, B::SkiL | B::SkiR | B::PoleL | B::PoleR),
                grip: if matches!(child, B::PoleL | B::PoleR) {
                    GRIP_FORCE
                } else {
                    f32::INFINITY
                },
                ..solver.pin(parent as usize, child as usize, p[at])
            };
            solver.joints.push(joint);
        };
        join(B::Pelvis, B::Abdomen, J::Waist, Some(LUMBAR), "lower back");
        join(
            B::Abdomen,
            B::Thorax,
            J::Chest,
            Some(THORACIC),
            "upper back",
        );
        join(B::Thorax, B::Head, J::Neck, Some(NECK), "neck");
        for i in 0..2 {
            let s = |l: B, r: B| side_of(i, l, r);
            let j = |l: J, r: J| if i == 0 { l } else { r };
            let n = |l: &'static str, r: &'static str| if i == 0 { l } else { r };
            join(
                B::Thorax,
                s(B::UpperArmL, B::UpperArmR),
                j(J::ShoulderL, J::ShoulderR),
                Some(SHOULDER),
                n("left shoulder", "right shoulder"),
            );
            join(
                s(B::UpperArmL, B::UpperArmR),
                s(B::ForearmL, B::ForearmR),
                j(J::ElbowL, J::ElbowR),
                Some(ELBOW),
                n("left elbow", "right elbow"),
            );
            join(
                s(B::ForearmL, B::ForearmR),
                s(B::HandL, B::HandR),
                j(J::WristL, J::WristR),
                Some(WRIST),
                n("left wrist", "right wrist"),
            );
            join(
                B::Pelvis,
                s(B::ThighL, B::ThighR),
                j(J::HipL, J::HipR),
                Some(HIP),
                n("left hip", "right hip"),
            );
            join(
                s(B::ThighL, B::ThighR),
                s(B::ShinL, B::ShinR),
                j(J::KneeL, J::KneeR),
                Some(KNEE),
                n("left knee", "right knee"),
            );
            join(
                s(B::ShinL, B::ShinR),
                s(B::BootL, B::BootR),
                j(J::AnkleL, J::AnkleR),
                Some(ANKLE),
                n("left ankle", "right ankle"),
            );
            join(
                s(B::BootL, B::BootR),
                s(B::SkiL, B::SkiR),
                j(J::BindL, J::BindR),
                Some(BINDING),
                n("left binding", "right binding"),
            );
            // The pole pivots freely in the fist and is pulled out of it by `GRIP_FORCE`.
            join(
                s(B::HandL, B::HandR),
                s(B::PoleL, B::PoleR),
                j(J::HandL, J::HandR),
                None,
                "",
            );
        }
        // Jointed segments overlap at the joint by design; so do pelvis and thorax across the
        // short abdomen. Every other pair collides (a boot and its ski once the binding releases).
        solver.collide_unjointed(&[(B::Pelvis as usize, B::Thorax as usize)]);
        Self { solver, points }
    }

    fn pose(&self) -> SkierPose {
        let p = std::array::from_fn(|i| {
            let (k, local) = self.points[i];
            self.solver.bodies[k].point(local)
        });
        let mut pose = SkierPose::from_points(p);
        pose.ski_up = [B::SkiL, B::SkiR].map(|b| self.solver.bodies[b as usize].q * Vec3::Y);
        pose.gaze = self.solver.bodies[B::Head as usize].q * Vec3::NEG_Z;
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::super::pose::{
        ANKLE_ABOVE_SKI, BONES, CHEST_TO_NECK, FOREARM, GEAR, HAND, HEEL_BACK, HIP_HALF_WIDTH,
        NECK_TO_HEAD, PELVIS_TO_WAIST, POLE_GRIP, POLE_LENGTH, SHIN, SHOULDER_HALF_WIDTH, SKI_BACK,
        SKI_FRONT, SKI_TIP_RISE, STANCE_WIDTH, THIGH, TOE_FORWARD, UPPER_ARM, WAIST_TO_CHEST,
    };
    use super::*;
    use crate::bike::{HILL_X, terrain_height};

    const DT: f32 = 1.0 / 120.0;

    /// Two-bone solution from `a` to `c` with the middle joint on the `hint` side.
    fn bend(a: Vec3, c: Vec3, l1: f32, l2: f32, hint: Vec3) -> Vec3 {
        let d = c - a;
        let dist = d.length();
        assert!(dist < l1 + l2, "unreachable test limb");
        let axis = d / dist;
        let x = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist);
        let h = (l1 * l1 - x * x).max(0.0).sqrt();
        a + axis * x + (hint - axis * hint.dot(axis)).normalize() * h
    }

    /// Plausible upright skier facing local -Z, ski bases on y = 0, placed by `rotation`/`origin`.
    fn standing(origin: Vec3, rotation: Quat) -> SkierPose {
        let mut p = [Vec3::ZERO; JOINTS];
        let fwd = Vec3::NEG_Z;
        let pelvis_y = SKI_THICKNESS + ANKLE_ABOVE_SKI + 0.8;
        let chest_y = pelvis_y + PELVIS_TO_WAIST + WAIST_TO_CHEST;
        p[J::Pelvis as usize] = Vec3::new(0.0, pelvis_y, 0.0);
        p[J::Waist as usize] = Vec3::new(0.0, pelvis_y + PELVIS_TO_WAIST, 0.0);
        p[J::Chest as usize] = Vec3::new(0.0, chest_y, 0.0);
        p[J::Neck as usize] = Vec3::new(0.0, chest_y + CHEST_TO_NECK, 0.0);
        p[J::Head as usize] = Vec3::new(0.0, chest_y + CHEST_TO_NECK + NECK_TO_HEAD, 0.0);
        for (sx, left) in [(-1.0, true), (1.0, false)] {
            let pick = |l: J, r: J| (if left { l } else { r }) as usize;
            let (hip, knee, ankle) = (
                pick(J::HipL, J::HipR),
                pick(J::KneeL, J::KneeR),
                pick(J::AnkleL, J::AnkleR),
            );
            let (shoulder, elbow, wrist) = (
                pick(J::ShoulderL, J::ShoulderR),
                pick(J::ElbowL, J::ElbowR),
                pick(J::WristL, J::WristR),
            );
            let hand = pick(J::HandL, J::HandR);
            let bind = Vec3::new(sx * STANCE_WIDTH / 2.0, SKI_THICKNESS, 0.0);
            p[pick(J::BindL, J::BindR)] = bind;
            p[pick(J::TipL, J::TipR)] = bind + fwd * SKI_FRONT + Vec3::Y * SKI_TIP_RISE;
            p[pick(J::TailL, J::TailR)] = bind - fwd * SKI_BACK;
            p[ankle] = bind + Vec3::Y * ANKLE_ABOVE_SKI;
            p[pick(J::HeelL, J::HeelR)] = bind - fwd * HEEL_BACK;
            p[pick(J::ToeL, J::ToeR)] = bind + fwd * TOE_FORWARD;
            p[hip] = Vec3::new(sx * HIP_HALF_WIDTH, pelvis_y, 0.0);
            p[knee] = bend(p[hip], p[ankle], THIGH, SHIN, fwd);
            p[shoulder] = Vec3::new(sx * SHOULDER_HALF_WIDTH, chest_y, 0.0);
            // Hands forward and out at belly height, poles trailing back and out.
            p[wrist] = Vec3::new(sx * 0.30, chest_y - 0.30, -0.30);
            p[elbow] = bend(
                p[shoulder],
                p[wrist],
                UPPER_ARM,
                FOREARM,
                Vec3::new(sx * 0.5, -1.0, 0.8),
            );
            p[hand] = p[wrist] + Vec3::new(0.0, -0.2, -1.0).normalize() * HAND;
            let pole_dir = Vec3::new(-sx * 0.3, 0.8, -0.5).normalize();
            p[pick(J::PoleTopL, J::PoleTopR)] = p[hand] + pole_dir * POLE_GRIP;
            p[pick(J::PoleTipL, J::PoleTipR)] = p[hand] - pole_dir * (POLE_LENGTH - POLE_GRIP);
        }
        p.iter_mut().for_each(|q| *q = origin + rotation * *q);
        SkierPose::from_points(p)
    }

    /// Skier standing on the 32 m hill at `z`, skis along the fall line (`yaw` 0) or rotated.
    fn on_hill(z: f32, yaw: f32) -> SkierPose {
        let ground = Vec3::new(HILL_X, terrain_height(HILL_X, z), z);
        let n = terrain_contact(ground, 0.0).1;
        standing(
            ground + n * 0.03,
            Quat::from_rotation_arc(Vec3::Y, n) * Quat::from_rotation_y(yaw),
        )
    }

    /// Ragdoll seeded at `pose` moving at `velocity` and spinning at `spin` (rad/s) about the pelvis.
    fn thrown(pose: &SkierPose, velocity: Vec3, spin: Vec3) -> SkiRagdoll {
        let centre = pose[J::Pelvis];
        let back = Quat::from_scaled_axis(-spin * DT);
        let mut earlier = *pose;
        earlier
            .p
            .iter_mut()
            .for_each(|p| *p = centre + back * (*p - centre) - velocity * DT);
        earlier.facing = back * earlier.facing;
        earlier.gaze = back * earlier.gaze;
        earlier.ski_up = earlier.ski_up.map(|u| back * u);
        let mut r = SkiRagdoll::default();
        r.activate(&earlier, pose, DT);
        r
    }

    fn body(r: &SkiRagdoll) -> &Solver {
        &r.body.as_ref().unwrap().solver
    }

    /// Steps `r` for `ticks`, asserting every tick that it stays finite and above the snow, that
    /// bones keep their length, that no two segments overlap and that joints never pass the hard
    /// stop past their (unbroken or broken) range. Returns the largest excess seen, rad.
    fn run_checked(r: &mut SkiRagdoll, ticks: usize, label: &str) -> f32 {
        let lengths = |p: &SkierPose| {
            BONES
                .iter()
                .chain(GEAR)
                .map(|&(a, b)| p[a].distance(p[b]))
                .collect::<Vec<_>>()
        };
        let rest = lengths(&r.pose().unwrap());
        let mut worst_excess = 0.0_f32;
        for tick in 0..ticks {
            r.step(DT);
            let p = r.pose().unwrap();
            for (i, q) in p.p.iter().enumerate() {
                assert!(
                    q.is_finite(),
                    "{label} tick {tick}: {:?} not finite",
                    ALL[i]
                );
                assert!(
                    q.y >= terrain_height(q.x, q.z) - 0.01,
                    "{label} tick {tick}: {:?} sank to {q}",
                    ALL[i]
                );
            }
            for (k, (now, len)) in lengths(&p).iter().zip(&rest).enumerate() {
                let (a, b) = BONES.iter().chain(GEAR).nth(k).unwrap();
                assert!(
                    (now - len).abs() < 0.02,
                    "{label} tick {tick}: {a:?}-{b:?} stretched {len} -> {now}"
                );
            }
            // Body segments: 5 mm. A thin pole sliding over a ski edge: 1 cm.
            let is_gear = |k: usize| k >= B::SkiL as usize;
            for (gear, limit) in [(false, -0.005), (true, -0.01)] {
                let (gap, (a, b)) =
                    body(r).worst_overlap(|a, b| (is_gear(a) && is_gear(b)) != gear);
                assert!(
                    gap > limit,
                    "{label} tick {tick}: segments {a}/{b} overlap {gap}"
                );
            }
            // Contacts are solved last, so a limb pinned between the snow and the body can be
            // forced a few degrees past its hard stop for a tick.
            for k in 0..body(r).joints.len() {
                let e = body(r).excess(k);
                worst_excess = worst_excess.max(e);
                assert!(
                    e < HARD_STOP + 6.0 * DEG,
                    "{label} tick {tick}: {} {:.1} deg out of range",
                    body(r).joints[k].name,
                    e / DEG
                );
            }
        }
        worst_excess
    }

    #[test]
    fn dropped_skier_comes_to_rest_intact() {
        let pose = standing(Vec3::new(-20.0, 2.0, 0.0), Quat::IDENTITY);
        let mut r = SkiRagdoll::default();
        r.activate(&pose, &pose, DT);
        run_checked(&mut r, 720, "drop");
        assert!(r.sleeping(), "not asleep after 6 s");
        assert!(r.centre().unwrap().0.y < 1.0, "pelvis must come down");
        assert!(
            r.injuries().is_empty(),
            "a 2 m drop broke {:?}",
            r.injuries()
        );
        assert_eq!(r.poles_off(), 0, "a 2 m drop lost a pole");
    }

    #[test]
    fn tumbles_keep_every_limb_out_of_the_body_and_inside_its_range() {
        let slope = terrain_contact(Vec3::new(HILL_X, 0.0, 0.0), 0.0).1;
        let down = Vec3::new(0.0, -slope.z.abs(), -1.0).normalize();
        for (label, velocity, spin) in [
            ("downhill", down * 15.0, Vec3::ZERO),
            ("front flip", down * 12.0, Vec3::X * -6.0),
            ("cartwheel", down * 12.0, Vec3::Z * 7.0),
            (
                "corkscrew",
                down * 14.0 + Vec3::X * 4.0,
                Vec3::new(3.0, 5.0, -4.0),
            ),
            ("backwards", -down * 6.0, Vec3::X * 4.0),
        ] {
            let mut r = thrown(&on_hill(0.0, 0.0), velocity, spin);
            let start = r.centre().unwrap().0;
            run_checked(&mut r, 480, label);
            let end = r.centre().unwrap().0;
            if label != "backwards" {
                assert!(
                    end.z < start.z - 5.0,
                    "{label}: pelvis only moved {start} -> {end}"
                );
            }
        }
    }

    #[test]
    fn a_hard_impact_breaks_joints_and_broken_limbs_still_collide() {
        // Head-first dive into flat snow at 22 m/s.
        let pose = standing(Vec3::new(-20.0, 1.5, 0.0), Quat::from_rotation_x(-1.9));
        let mut r = thrown(&pose, Vec3::new(0.0, -12.0, -18.0), Vec3::X * -4.0);
        run_checked(&mut r, 360, "dive");
        assert!(!r.injuries().is_empty(), "a 22 m/s dive broke nothing");
        assert!(r.poles_off() >= 1, "a 22 m/s dive kept both poles");
    }

    #[test]
    fn a_low_speed_fall_breaks_nothing() {
        let pose = standing(Vec3::new(-20.0, 0.0, 0.0), Quat::IDENTITY);
        let mut r = thrown(&pose, Vec3::X * 2.0, Vec3::Z * -2.0);
        run_checked(&mut r, 480, "topple");
        assert!(
            r.injuries().is_empty(),
            "toppling over broke {:?}",
            r.injuries()
        );
        assert_eq!(r.poles_off(), 0, "toppling over lost a pole");
    }

    /// Slide over 2 s of a ski released on the slope, along vs across the fall line (the skier is
    /// lifted clear so only the ski touches the snow).
    fn slide(yaw: f32) -> f32 {
        let pose = on_hill(-20.0, yaw);
        let mut r = SkiRagdoll::default();
        r.activate(&pose, &pose, DT);
        let b = &mut r.body.as_mut().unwrap().solver;
        for k in 0..b.joints.len() {
            if b.joints[k].detachable {
                b.fail(k);
            }
        }
        for (k, body) in b.bodies.iter_mut().enumerate() {
            if k != B::SkiL as usize && k != B::SkiR as usize {
                body.x.y += 100.0;
            }
        }
        let at = |r: &SkiRagdoll| r.pose().unwrap()[J::BindL];
        let start = at(&r);
        for _ in 0..240 {
            r.step(DT);
        }
        start.distance(at(&r))
    }

    #[test]
    fn skis_slide_along_their_axis_more_than_across() {
        let along = slide(0.0);
        let across = slide(std::f32::consts::FRAC_PI_2);
        assert!(
            along > across * 5.0 + 0.5,
            "along {along} vs across {across}"
        );
    }

    #[test]
    fn clamp_keeps_ranges_and_mirrors_left_joints() {
        let knee_back = Quat::from_rotation_x(20.0 * DEG);
        let c = clamp_rotation(knee_back, &KNEE, false, 0.0);
        assert!((c.to_scaled_axis().x - 3.0 * DEG).abs() < 1e-3, "{c:?}");
        let bent = Quat::from_rotation_x(-90.0 * DEG);
        assert!(clamp_rotation(bent, &KNEE, false, 0.0).angle_between(bent) < 1e-3);
        // Right hip abducts 45 deg (+Z swing); the left one the same amount the other way.
        let out_r = Quat::from_rotation_z(60.0 * DEG);
        let out_l = Quat::from_rotation_z(-60.0 * DEG);
        let r = clamp_rotation(out_r, &HIP, false, 0.0).to_scaled_axis().z / DEG;
        let l = clamp_rotation(out_l, &HIP, true, 0.0).to_scaled_axis().z / DEG;
        assert!((r - 45.0).abs() < 0.1 && (l + 45.0).abs() < 0.1, "{r} {l}");
        // A broken joint opens by the slack.
        let c = clamp_rotation(knee_back, &KNEE, false, BROKEN_SLACK);
        assert!(c.angle_between(knee_back) < 1e-3);
    }

    #[test]
    fn the_showcase_crash_tumbles_without_limbs_entering_the_body() {
        use super::super::anim::SkiAnimation;
        use super::super::physics::{SkiControls, Skier};
        use super::super::rig::solve;
        let mut demo = crate::ski::demo::SkiDemo::default();
        let (mut s, mut a) = (Skier::default(), SkiAnimation::default());
        let mut last = None;
        for i in 0..120 * 400 {
            let mut c = SkiControls::default();
            if i == 0 {
                demo.enabled = true;
                demo.begin_run(&mut s);
            }
            demo.drive(&mut s, &mut c, DT);
            a.update(&s, &c, DT);
            let pose = solve(&s, &a);
            s.step(&c, DT);
            if s.crash.is_some() {
                let mut r = SkiRagdoll::default();
                r.activate(&last.unwrap_or(pose), &pose, DT);
                run_checked(&mut r, 600, "showcase crash");
                return;
            }
            last = Some(pose);
        }
        panic!("the showcase never crashed");
    }

    #[test]
    fn a_perpendicular_landing_crashes_and_the_ragdoll_keeps_the_momentum() {
        use super::super::anim::SkiAnimation;
        use super::super::physics::{SkiControls, SkiCrashReason, Skier};
        use super::super::rig::solve;
        // Skis level, 8 m/s along the snow and 20 m/s into it (the speed after a 20 m drop).
        let mut s = Skier::default();
        s.reset_at(-40.0, 0.0, 0.0, 8.0);
        s.grounded = false;
        s.position.y = 2.0;
        s.velocity.y = -20.0;
        let (mut a, c) = (SkiAnimation::default(), SkiControls::default());
        let mut last = None;
        for _ in 0..120 {
            a.update(&s, &c, DT);
            let pose = solve(&s, &a);
            let before = s.velocity;
            s.step(&c, DT);
            if let Some(crash) = s.crash {
                assert_eq!(crash.reason, SkiCrashReason::HardImpact);
                let mut r = SkiRagdoll::default();
                r.activate(&last.unwrap_or(pose), &pose, DT);
                // The skier's own pre-impact velocity, not the one the landing corrected.
                let seed = r.centre().unwrap().1;
                assert!(
                    seed.distance(before) < 0.1 * before.length(),
                    "seeded {seed} for {before}"
                );
                run_checked(&mut r, 360, "perpendicular landing");
                assert!(r.skis_off() >= 1, "no binding released");
                assert!(r.poles_off() >= 1, "no pole dropped");
                return;
            }
            last = Some(pose);
        }
        panic!("the landing never crashed");
    }
}
