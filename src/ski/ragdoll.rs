//! Detached skier: an articulated rigid-body ragdoll solved with extended position-based dynamics
//! (Müller et al. 2020, one iteration per 1/2400 s substep).
//!
//! Eighteen segments carry anthropometric masses and inertias: pelvis, abdomen, thorax, head, upper
//! arms, forearms, hands, thighs, shins, boot-and-ski and poles. They are pinned together at the
//! skeleton's joints. Every joint keeps an anatomical swing/twist range, measured from the upright
//! stance, and breaks (its range opens by `BROKEN_SLACK`) once the torque holding it at the limit
//! exceeds its strength. Every pair of segments that is not jointed collides as rendered-size
//! capsules, so no limb passes through another limb or the body, broken or not. The snow gives
//! Coulomb friction (skis glide along their length and bite across it), and the body sleeps at rest.

use bevy::math::{Mat3, Quat, Vec3};
use bevy::prelude::Resource;

use super::pose::{J, JOINTS, SKI_THICKNESS, SKI_WIDTH, SkierPose};
use super::rig::closest_params;
use crate::bike::terrain_height;

const GRAVITY: f32 = 9.81;
/// Longest solver substep, s.
const SUBSTEP: f32 = 1.0 / 2400.0;
const MAX_SUBSTEPS: usize = 64;
/// Seed velocities beyond these (pose-difference glitches) are clamped: m/s, rad/s.
const MAX_SEED_SPEED: f32 = 40.0;
const MAX_SEED_SPIN: f32 = 25.0;
/// Spacing of terrain probes along a capsule, m.
const SAMPLE_SPACING: f32 = 0.08;
const AIR_DAMPING: f32 = 0.1;
/// Passive joint friction (relative spin decay rate, 1/s): muscle tone without driving a pose.
const JOINT_DAMPING: f32 = 12.0;
/// Spin decay of a segment lying in the snow, 1/s.
const ROLL_DAMPING: f32 = 5.0;
const RESTITUTION: f32 = 0.1;
/// Coulomb coefficients: body and boots on snow, limb on limb, skis along and across their axis.
const BODY_FRICTION: f32 = 0.6;
const LIMB_FRICTION: f32 = 0.4;
const SKI_ALONG: f32 = 0.05;
const SKI_ACROSS: f32 = 0.8;
const SLEEP_SPEED: f32 = 0.12;
const SLEEP_TIME: f32 = 0.6;
/// Averaging time of the torque loading a joint at its limit: shorter spikes are absorbed.
const LOAD_TIME: f32 = 0.02;
const DEG: f32 = std::f32::consts::PI / 180.0;
/// How far past its anatomical range a broken joint goes before soft tissue stops it.
const BROKEN_SLACK: f32 = 50.0 * DEG;
/// Position-only passes that remove seed overlaps before the first step.
const SETTLE_PASSES: usize = 40;
/// Rate at which a joint handed over outside its range is eased back into it, rad/s.
const SEED_RELAX: f32 = 4.0;
/// Past the end of its range a joint gives like a stiff spring (ligaments, the boot shell), so the
/// torque holding it is physical rather than a one-substep impulse; this much further on it stops
/// dead.
const HARD_STOP: f32 = 6.0 * DEG;

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

/// Anatomical range of a joint for the right side (left mirrors it), degrees, about the child
/// segment's frame relative to the parent's: `flex` swings about X, `side` about Z, `twist` turns
/// about the segment's own Y. A ball joint's swing stays inside the ellipse through the four
/// extremes; a `hinge` limits flex and side independently. `centre` (X, Z swing) shifts the middle
/// of the range off the upright stance. `strength`: torque about the flex, side and twist axes
/// that breaks the joint, N m (combined as an ellipsoid). `stiffness`: how hard the end of range
/// pushes back, N m per rad.
#[derive(Clone, Copy)]
struct Limit {
    centre: [f32; 2],
    flex: [f32; 2],
    side: [f32; 2],
    twist: [f32; 2],
    hinge: bool,
    strength: [f32; 3],
    stiffness: f32,
}

/// Ligament end-of-range stiffness, N m per rad.
const LIGAMENT: f32 = 2500.0;

const fn ball(flex: [f32; 2], side: [f32; 2], twist: [f32; 2], strength: [f32; 3]) -> Limit {
    Limit {
        centre: [0.0, 0.0],
        flex,
        side,
        twist,
        hinge: false,
        strength,
        stiffness: LIGAMENT,
    }
}

const fn hinged(flex: [f32; 2], side: [f32; 2], twist: [f32; 2], strength: [f32; 3]) -> Limit {
    Limit {
        hinge: true,
        ..ball(flex, side, twist, strength)
    }
}

/// Spine: forward bend is negative flex, a right side-bend negative side, a right turn negative twist.
/// The trunk's muscles and ribcage share the load, so it is far stronger than a single limb joint.
const LUMBAR: Limit = ball(
    [-50.0, 25.0],
    [-25.0, 25.0],
    [-12.0, 12.0],
    [600.0, 600.0, 300.0],
);
const THORACIC: Limit = ball(
    [-35.0, 15.0],
    [-20.0, 20.0],
    [-35.0, 35.0],
    [600.0, 600.0, 300.0],
);
const NECK: Limit = ball(
    [-50.0, 60.0],
    [-40.0, 40.0],
    [-75.0, 75.0],
    [225.0, 180.0, 120.0],
);
/// Shoulder (humerus on thorax), measured about an arm raised 45 deg sideways and 20 deg forward:
/// flexion to about 170 deg, extension 60, abduction 180, adduction across the chest; humeral
/// rotation about +-80.
const SHOULDER: Limit = Limit {
    centre: [20.0, 45.0],
    ..ball(
        [-90.0, 150.0],
        [-90.0, 140.0],
        [-80.0, 80.0],
        [120.0, 120.0, 70.0],
    )
};
/// Elbow: a hinge from straight to 150 deg; pronation/supination as twist.
const ELBOW: Limit = hinged(
    [-3.0, 150.0],
    [-5.0, 5.0],
    [-80.0, 80.0],
    [160.0, 90.0, 80.0],
);
const WRIST: Limit = ball(
    [-70.0, 70.0],
    [-25.0, 25.0],
    [-10.0, 10.0],
    [110.0, 90.0, 70.0],
);
/// Hip: flexion 120, extension 20, abduction 45, adduction 30, internal 35 / external 45 rotation.
const HIP: Limit = ball(
    [-20.0, 120.0],
    [-30.0, 45.0],
    [-45.0, 35.0],
    [420.0, 420.0, 280.0],
);
/// Knee: flexion (negative) to 150 deg, barely any hyperextension, side play or tibial rotation.
/// Varus/valgus and rotation tear it first.
const KNEE: Limit = hinged(
    [-150.0, 3.0],
    [-5.0, 5.0],
    [-20.0, 20.0],
    [250.0, 150.0, 100.0],
);
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

#[derive(Clone, Copy)]
struct Joint {
    parent: usize,
    child: usize,
    /// Anchor in each segment's local frame.
    pa: Vec3,
    pc: Vec3,
    /// Joint frame in the parent's local frame (the centre of the range).
    frame: Quat,
    limit: Option<Limit>,
    mirror: bool,
    name: &'static str,
    /// Averaged torque at the limit about the joint frame's flex, side and twist axes, N m.
    load: [f32; 3],
    /// A broken joint's range opens by `BROKEN_SLACK`; a released binding lets go of the ski.
    broken: bool,
    binding: bool,
    /// Extra range the joint starts with when the animation handed it over outside its anatomical
    /// range; it closes at `SEED_RELAX`, so the limb is eased in rather than snapped.
    allow: f32,
}

impl Joint {
    fn slack(&self) -> f32 {
        if self.broken { BROKEN_SLACK } else { 0.0 }.max(self.allow)
    }

    fn holds(&self) -> bool {
        !(self.binding && self.broken)
    }
}

#[derive(Clone, Copy)]
struct Rigid {
    x: Vec3,
    q: Quat,
    v: Vec3,
    w: Vec3,
    /// Constraint displacement and rotation this substep; velocities come from these rather than
    /// from position differences, which f32 rounds away far from the origin.
    dx: Vec3,
    dphi: Vec3,
    inv_mass: f32,
    /// Inverse inertia in the local frame.
    inv_inertia: Mat3,
    /// Radius of a sphere round `x` holding every collision capsule.
    bound: f32,
}

impl Rigid {
    fn inv_inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.q);
        r * self.inv_inertia * r.transpose()
    }

    fn point(&self, local: Vec3) -> Vec3 {
        self.x + self.q * local
    }

    fn velocity_at(&self, r: Vec3) -> Vec3 {
        self.v + self.w.cross(r)
    }

    /// Small rotation by the rotation vector `phi`.
    fn turn(&mut self, phi: Vec3) {
        self.q = (self.q + Quat::from_xyzw(phi.x, phi.y, phi.z, 0.0) * self.q * 0.5).normalize();
        self.dphi += phi;
    }

    fn shift_by(&mut self, d: Vec3) {
        self.x += d;
        self.dx += d;
    }

    /// Inverse mass felt along `n` at offset `r` from the centre of mass.
    fn weight(&self, r: Vec3, n: Vec3) -> f32 {
        let rn = r.cross(n);
        self.inv_mass + rn.dot(self.inv_inertia_world() * rn)
    }
}

/// Collision capsule fixed to a segment, endpoints in its local frame. `ignore`: a segment it may
/// touch (the grip section of a pole lies along its own wrist).
#[derive(Clone, Copy)]
struct Shape {
    body: usize,
    a: Vec3,
    b: Vec3,
    r: f32,
    ski: bool,
    ignore: Option<usize>,
}

impl Shape {
    fn meets(&self, other: &Shape) -> bool {
        self.ignore != Some(other.body) && other.ignore != Some(self.body)
    }
}

#[derive(Clone, Copy)]
struct Contact {
    a: usize,
    /// `None`: the snow.
    b: Option<usize>,
    ra: Vec3,
    rb: Vec3,
    /// Pushes `a` away from `b`.
    n: Vec3,
    /// Normal impulse of the position solve (kg m).
    lambda: f32,
    mu: f32,
    restitution: f32,
    /// World ski axis for anisotropic friction.
    ski: Option<Vec3>,
}

/// A segment's mass element or collision capsule in world space, from the pose.
struct Part {
    body: B,
    a: Vec3,
    b: Vec3,
    r: f32,
    mass: f32,
    collide: bool,
    ski: bool,
    ignore: Option<B>,
}

fn part(body: B, a: Vec3, b: Vec3, r: f32, mass: f32) -> Part {
    Part {
        body,
        a,
        b,
        r,
        mass,
        collide: true,
        ski: false,
        ignore: None,
    }
}

fn unit(v: Vec3) -> Vec3 {
    v.normalize_or_zero()
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
            ignore: Some(side_of(i, B::ForearmL, B::ForearmR)),
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

/// Rotation with X along `x` made perpendicular to `y`, Y along `y`, Z = X x Y.
fn basis(x: Vec3, y: Vec3) -> Quat {
    let y = y.normalize_or(Vec3::Y);
    let x = x
        .reject_from_normalized(y)
        .try_normalize()
        .unwrap_or_else(|| y.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y))).normalize()
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Flexion axis (X) of a two-bone limb bent at `mid`: `(end - mid) x (root - mid)`, blended towards
/// `fallback` as the limb straightens and its bend plane becomes undefined.
fn hinge(root: Vec3, mid: Vec3, end: Vec3, fallback: Vec3) -> Vec3 {
    let (a, b) = (root - mid, end - mid);
    let c = b.cross(a);
    let bend = c.length() / (a.length() * b.length()).max(1e-6);
    let w = smoothstep(0.05, 0.3, bend);
    (unit(c) * w + unit(fallback) * (1.0 - w)).normalize_or(fallback)
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

/// Centre of mass of every segment in pose `p`.
fn centres(parts: &[Part]) -> [Vec3; BODIES] {
    let mut sum = [(Vec3::ZERO, 0.0_f32); BODIES];
    for q in parts {
        let s = &mut sum[q.body as usize];
        s.0 += (q.a + q.b) * (0.5 * q.mass);
        s.1 += q.mass;
    }
    sum.map(|(m, w)| m / w)
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Inertia of a rod or ball `q` about the point `about`, world axes.
fn inertia(q: &Part, about: Vec3) -> Mat3 {
    let (m, r) = (q.mass, q.r);
    let d = q.b - q.a;
    let l = d.length();
    let own = if l < 1e-6 {
        Mat3::IDENTITY * (0.4 * m * r * r)
    } else {
        let u = d / l;
        let perp = m * (3.0 * r * r + l * l) / 12.0;
        Mat3::IDENTITY * perp + outer(u, u) * (0.5 * m * r * r - perp)
    };
    let o = (q.a + q.b) * 0.5 - about;
    own + (Mat3::IDENTITY * o.length_squared() - outer(o, o)) * m
}

/// Clamps the joint rotation `q` (child frame in the parent's joint frame) into `l`, widened by
/// `slack` rad. A ball joint's swing (about X and Z) is pulled radially into the ellipse through
/// the four range extremes, a hinge's flex and side are clamped separately; twist (about Y) is
/// clamped into its interval. `mirror` for the left side.
fn clamp_rotation(q: Quat, l: &Limit, mirror: bool, slack: f32) -> Quat {
    let q = if q.w < 0.0 { -q } else { q };
    let m = if mirror { -1.0 } else { 1.0 };
    let len = (q.y * q.y + q.w * q.w).sqrt();
    let twist = if len < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_xyzw(0.0, q.y / len, 0.0, q.w / len)
    };
    let swing = (q * twist.inverse()).to_scaled_axis();
    let (fx, sz, tw) = (swing.x, swing.z * m, 2.0 * twist.y.atan2(twist.w) * m);
    let open = |r: [f32; 2], cap: f32| {
        (
            (r[0] * DEG - slack).max(-cap),
            (r[1] * DEG + slack).min(cap),
        )
    };
    let swing_cap = 160.0 * DEG;
    let (f0, f1) = open(l.flex, swing_cap);
    let (s0, s1) = open(l.side, swing_cap);
    let (t0, t1) = open(l.twist, 170.0 * DEG);
    let (fx, sz) = if l.hinge {
        (fx.clamp(f0, f1), sz.clamp(s0, s1))
    } else {
        let ax = (if fx >= 0.0 { f1 } else { -f0 }).max(DEG);
        let az = (if sz >= 0.0 { s1 } else { -s0 }).max(DEG);
        let e = (fx / ax).powi(2) + (sz / az).powi(2);
        let k = if e > 1.0 { e.sqrt().recip() } else { 1.0 };
        (fx * k, sz * k)
    };
    Quat::from_scaled_axis(Vec3::new(fx, 0.0, sz * m)) * Quat::from_rotation_y(tw.clamp(t0, t1) * m)
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
        let (x0, q0) = (centres(&parts(previous)), frames(previous));
        for (k, b) in body.bodies.iter_mut().enumerate() {
            b.v = ((b.x - x0[k]) / dt).clamp_length_max(MAX_SEED_SPEED);
            let dq = b.q * q0[k].inverse();
            let dq = if dq.w < 0.0 { -dq } else { dq };
            b.w = (dq.to_scaled_axis() / dt).clamp_length_max(MAX_SEED_SPIN);
        }
        for k in 0..body.joints.len() {
            body.joints[k].allow = body.excess(k) * 1.2 + DEG;
        }
        body.settle();
        self.body = Some(body);
    }

    pub(crate) fn step(&mut self, dt: f32) {
        if let Some(body) = &mut self.body {
            body.step(dt);
        }
    }

    pub(crate) fn pose(&self) -> Option<SkierPose> {
        self.body.as_ref().map(Body::pose)
    }

    /// Pelvis position and velocity (camera target).
    pub(crate) fn centre(&self) -> Option<(Vec3, Vec3)> {
        self.body.as_ref().map(|b| {
            let (k, local) = b.points[J::Pelvis as usize];
            (b.bodies[k].point(local), b.bodies[k].v)
        })
    }

    pub(crate) fn sleeping(&self) -> bool {
        self.body.as_ref().is_some_and(|b| b.sleeping)
    }

    /// Joints broken so far, in the order they gave way.
    pub(crate) fn injuries(&self) -> &[&'static str] {
        self.body.as_ref().map_or(&[], |b| &b.injuries)
    }

    /// Skis whose binding has released.
    pub(crate) fn skis_off(&self) -> usize {
        self.body.as_ref().map_or(0, |b| {
            b.joints.iter().filter(|j| j.binding && j.broken).count()
        })
    }
}

struct Body {
    bodies: Vec<Rigid>,
    shapes: Vec<Shape>,
    /// Shape index range of each segment.
    range: [(usize, usize); BODIES],
    /// Segment pairs that collide (not jointed).
    pairs: Vec<(usize, usize)>,
    joints: Vec<Joint>,
    /// Owner segment and local position of every pose point.
    points: [(usize, Vec3); JOINTS],
    contacts: Vec<Contact>,
    injuries: Vec<&'static str>,
    sleeping: bool,
    quiet_time: f32,
}

impl Body {
    fn new(p: &SkierPose) -> Self {
        let parts = parts(p);
        let f = frames(p);
        let com = centres(&parts);
        let mut bodies = Vec::with_capacity(BODIES);
        for k in 0..BODIES {
            let own = parts.iter().filter(|q| q.body as usize == k);
            let mass: f32 = own.clone().map(|q| q.mass).sum();
            let world = own.fold(Mat3::ZERO, |sum, q| sum + inertia(q, com[k]));
            let r = Mat3::from_quat(f[k]);
            // Floor keeps thin poles and skis from spinning arbitrarily fast about their axis.
            let local = r.transpose() * world * r + Mat3::IDENTITY * (mass * 4e-4);
            bodies.push(Rigid {
                x: com[k],
                q: f[k],
                v: Vec3::ZERO,
                w: Vec3::ZERO,
                dx: Vec3::ZERO,
                dphi: Vec3::ZERO,
                inv_mass: 1.0 / mass,
                inv_inertia: local.inverse(),
                bound: 0.0,
            });
        }
        let mut shapes = Vec::new();
        let mut range = [(0, 0); BODIES];
        for (k, rk) in range.iter_mut().enumerate() {
            let start = shapes.len();
            let inv = f[k].inverse();
            for q in parts.iter().filter(|q| q.body as usize == k && q.collide) {
                let (a, b) = (inv * (q.a - com[k]), inv * (q.b - com[k]));
                bodies[k].bound = bodies[k].bound.max(a.length() + q.r).max(b.length() + q.r);
                shapes.push(Shape {
                    body: k,
                    a,
                    b,
                    r: q.r,
                    ski: q.ski,
                    ignore: q.ignore.map(|b| b as usize),
                });
            }
            *rk = (start, shapes.len());
        }
        let points = std::array::from_fn(|i| {
            let k = owner(ALL[i]) as usize;
            (k, f[k].inverse() * (p.p[i] - com[k]))
        });
        let mut joints = Vec::new();
        let mut join = |parent: B, child: B, at: J, limit: Option<Limit>, name: &'static str| {
            let (pi, ci) = (parent as usize, child as usize);
            let mirror = matches!(crate::ski::pose::side(at), crate::ski::pose::Side::Left);
            let frame = limit.map_or(Quat::IDENTITY, |l| {
                let m = if mirror { -1.0 } else { 1.0 };
                Quat::from_scaled_axis(Vec3::new(l.centre[0], 0.0, l.centre[1] * m) * DEG)
            });
            joints.push(Joint {
                parent: pi,
                child: ci,
                pa: f[pi].inverse() * (p[at] - com[pi]),
                pc: f[ci].inverse() * (p[at] - com[ci]),
                frame,
                limit,
                mirror,
                name,
                load: [0.0; 3],
                broken: false,
                binding: matches!(child, B::SkiL | B::SkiR),
                allow: 0.0,
            });
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
            // The pole pivots freely in the fist (strap).
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
        let joined = |a: usize, b: usize| {
            joints
                .iter()
                .any(|j| (j.parent, j.child) == (a, b) || (j.parent, j.child) == (b, a))
                || (a, b) == (B::Pelvis as usize, B::Thorax as usize)
        };
        let pairs = (0..BODIES)
            .flat_map(|a| (a + 1..BODIES).map(move |b| (a, b)))
            .filter(|&(a, b)| !joined(a, b))
            .collect();
        Self {
            bodies,
            shapes,
            range,
            pairs,
            joints,
            points,
            contacts: Vec::new(),
            injuries: Vec::new(),
            sleeping: false,
            quiet_time: 0.0,
        }
    }

    fn pose(&self) -> SkierPose {
        let p = std::array::from_fn(|i| {
            let (k, local) = self.points[i];
            self.bodies[k].point(local)
        });
        let mut pose = SkierPose::from_points(p);
        pose.ski_up = [B::SkiL, B::SkiR].map(|b| self.bodies[b as usize].q * Vec3::Y);
        pose.gaze = self.bodies[B::Head as usize].q * Vec3::NEG_Z;
        pose
    }

    /// Position-only passes over the seed pose: overlaps and out-of-range joints left by the
    /// animation are removed without turning the correction into speed.
    fn settle(&mut self) {
        for _ in 0..SETTLE_PASSES {
            self.solve_positions(1.0, false);
        }
        self.contacts.clear();
    }

    fn step(&mut self, dt: f32) {
        if self.sleeping || dt <= 0.0 {
            return;
        }
        let count = ((dt / SUBSTEP - 1e-3).ceil() as usize).clamp(1, MAX_SUBSTEPS);
        let h = dt / count as f32;
        for _ in 0..count {
            self.substep(h);
            if self.sleeping {
                break;
            }
        }
    }

    fn substep(&mut self, h: f32) {
        let before: Vec<(Vec3, Vec3)> = self.bodies.iter().map(|b| (b.v, b.w)).collect();
        let damping = (-AIR_DAMPING * h).exp();
        for b in &mut self.bodies {
            b.v.y -= GRAVITY * h;
            b.v *= damping;
            b.w *= damping;
            b.x += b.v * h;
            let w = b.w;
            b.turn(w * h);
            b.dx = Vec3::ZERO;
            b.dphi = Vec3::ZERO;
        }
        for j in &mut self.joints {
            j.allow = (j.allow - SEED_RELAX * h).max(0.0);
        }
        self.contacts.clear();
        self.solve_positions(h, true);
        for b in &mut self.bodies {
            b.v += b.dx / h;
            b.w += b.dphi / h;
        }
        self.solve_velocities(&before, h);

        let supported = self.contacts.iter().any(|c| c.b.is_none());
        // At rest when every capsule end is slow: a thin pole may keep spinning about its own
        // axis, which moves nothing.
        let quiet = self.shapes.iter().all(|s| {
            let b = &self.bodies[s.body];
            [s.a, s.b]
                .iter()
                .all(|&p| b.velocity_at(b.q * p).length() < SLEEP_SPEED)
        });
        self.quiet_time = if supported && quiet {
            self.quiet_time + h
        } else {
            0.0
        };
        if self.quiet_time > SLEEP_TIME {
            self.sleeping = true;
            for b in &mut self.bodies {
                b.v = Vec3::ZERO;
                b.w = Vec3::ZERO;
            }
        }
    }

    /// One pass over every constraint. `h` scales joint loads; `live` false (settling) neither
    /// loads nor breaks joints. Segment and snow contacts go last: whatever the joints leave,
    /// nothing ends a substep inside anything else.
    fn solve_positions(&mut self, h: f32, live: bool) {
        for k in 0..self.joints.len() {
            self.attach(k);
        }
        for k in 0..self.joints.len() {
            self.limit(k, h, live);
        }
        for k in 0..self.joints.len() {
            self.attach(k);
        }
        for k in 0..self.joints.len() {
            let joint = self.joints[k];
            if let Some(l) = joint.limit.filter(|_| joint.holds()) {
                let (_, stop) = self.overshoot(&joint, &l, HARD_STOP);
                self.twist_apart(joint.child, joint.parent, stop, 0.0, h);
            }
        }
        self.collide_segments();
        self.collide_snow();
    }

    /// Moves the point at offset `ra` of segment `a` by `delta` relative to the point at `rb` of
    /// `b` (`None`: the snow), split by generalized inverse mass. Returns the impulse, kg m.
    fn shift(&mut self, a: usize, b: Option<usize>, ra: Vec3, rb: Vec3, delta: Vec3) -> f32 {
        let c = delta.length();
        if c < 1e-9 {
            return 0.0;
        }
        let n = delta / c;
        let wa = self.bodies[a].weight(ra, n);
        let wb = b.map_or(0.0, |b| self.bodies[b].weight(rb, n));
        let p = n * (c / (wa + wb));
        let body = &mut self.bodies[a];
        body.shift_by(p * body.inv_mass);
        let turn = body.inv_inertia_world() * ra.cross(p);
        body.turn(turn);
        if let Some(b) = b {
            let body = &mut self.bodies[b];
            body.shift_by(-p * body.inv_mass);
            let turn = body.inv_inertia_world() * rb.cross(p);
            body.turn(-turn);
        }
        c / (wa + wb)
    }

    /// Turns segment `a` by the rotation vector `phi` relative to `b`, split by inertia, against a
    /// spring of `compliance` (rad per N m) over the substep `h`. Returns the angular impulse,
    /// kg m^2.
    fn twist_apart(&mut self, a: usize, b: usize, phi: Vec3, compliance: f32, h: f32) -> f32 {
        let angle = phi.length();
        if angle < 1e-7 {
            return 0.0;
        }
        let n = phi / angle;
        let (ia, ib) = (
            self.bodies[a].inv_inertia_world(),
            self.bodies[b].inv_inertia_world(),
        );
        let lambda = angle / (n.dot(ia * n) + n.dot(ib * n) + compliance / (h * h));
        self.bodies[a].turn(ia * n * lambda);
        self.bodies[b].turn(-(ib * n * lambda));
        lambda
    }

    fn attach(&mut self, k: usize) {
        let j = self.joints[k];
        if !j.holds() {
            return;
        }
        let (bp, bc) = (self.bodies[j.parent], self.bodies[j.child]);
        let (wp, wc) = (bp.point(j.pa), bc.point(j.pc));
        self.shift(j.child, Some(j.parent), wc - bc.x, wp - bp.x, wp - wc);
    }

    /// Joint rotation: child joint frame in the parent's.
    fn relative(&self, j: &Joint) -> (Quat, Quat) {
        let qp = self.bodies[j.parent].q * j.frame;
        (qp, qp.inverse() * self.bodies[j.child].q)
    }

    /// Rotation vector turning the child of `j` back to its range widened by `extra` rad.
    fn overshoot(&self, j: &Joint, l: &Limit, extra: f32) -> (Quat, Vec3) {
        let (qp, rel) = self.relative(j);
        let target = clamp_rotation(rel, l, j.mirror, j.slack() + extra);
        let fix = qp * target * rel.inverse() * qp.inverse();
        (qp, if fix.w < 0.0 { -fix } else { fix }.to_scaled_axis())
    }

    /// Range limit: a spring of the joint's stiffness past the end of range, and a hard stop
    /// `HARD_STOP` further on.
    fn limit(&mut self, k: usize, h: f32, live: bool) {
        let j = self.joints[k];
        let Some(l) = j.limit.filter(|_| j.holds()) else {
            return;
        };
        let (qp, phi) = self.overshoot(&j, &l, 0.0);
        let soft = self.twist_apart(j.child, j.parent, phi, 1.0 / l.stiffness, h);
        let (_, stop) = self.overshoot(&j, &l, HARD_STOP);
        let hard = self.twist_apart(j.child, j.parent, stop, 0.0, h);
        if !live {
            return;
        }
        // Torque holding the limit, about the joint frame's flex (X), side (Z) and twist (Y) axes.
        let torque = qp.inverse()
            * (phi.normalize_or_zero() * soft + stop.normalize_or_zero() * hard)
            / (h * h);
        let j = &mut self.joints[k];
        let mut over = 0.0;
        // A binding's heel releases only when the boot levers forward off the ski (-X).
        let flex = if j.binding {
            (-torque.x).max(0.0)
        } else {
            torque.x.abs()
        };
        for (a, t) in [flex, torque.z.abs(), torque.y.abs()]
            .into_iter()
            .enumerate()
        {
            j.load[a] += (t - j.load[a]) * (h / LOAD_TIME).min(1.0);
            over += (j.load[a] / l.strength[a]).powi(2);
        }
        if !j.broken && over > 1.0 {
            j.broken = true;
            if j.binding {
                self.pairs.push((j.parent, j.child));
            } else {
                self.injuries.push(j.name);
            }
        }
    }

    fn collide_segments(&mut self) {
        for k in 0..self.pairs.len() {
            let (a, b) = self.pairs[k];
            if self.bodies[a].x.distance(self.bodies[b].x)
                > self.bodies[a].bound + self.bodies[b].bound
            {
                continue;
            }
            for i in self.range[a].0..self.range[a].1 {
                for j in self.range[b].0..self.range[b].1 {
                    let (sa, sb) = (self.shapes[i], self.shapes[j]);
                    if !sa.meets(&sb) {
                        continue;
                    }
                    let (ba, bb) = (self.bodies[a], self.bodies[b]);
                    let (a0, a1, b0, b1) = (
                        ba.point(sa.a),
                        ba.point(sa.b),
                        bb.point(sb.a),
                        bb.point(sb.b),
                    );
                    let (s, t) = closest_params(a0, a1, b0, b1);
                    let (pa, pb) = (a0.lerp(a1, s), b0.lerp(b1, t));
                    let d = pa - pb;
                    let gap = d.length() - sa.r - sb.r;
                    if gap >= 0.0 {
                        continue;
                    }
                    let n = d.try_normalize().unwrap_or_else(|| unit(ba.x - bb.x));
                    let (ra, rb) = (pa - n * sa.r - ba.x, pb + n * sb.r - bb.x);
                    let lambda = self.shift(a, Some(b), ra, rb, n * -gap);
                    self.contacts.push(Contact {
                        a,
                        b: Some(b),
                        ra,
                        rb,
                        n,
                        lambda,
                        mu: LIMB_FRICTION,
                        restitution: 0.0,
                        ski: None,
                    });
                }
            }
        }
    }

    fn collide_snow(&mut self) {
        for i in 0..self.shapes.len() {
            let s = self.shapes[i];
            let len = (s.b - s.a).length();
            let samples = (len / SAMPLE_SPACING).ceil() as usize;
            for k in 0..=samples {
                let t = if samples == 0 {
                    0.0
                } else {
                    k as f32 / samples as f32
                };
                let body = self.bodies[s.body];
                let p = body.point(s.a.lerp(s.b, t));
                if p.y - terrain_height(p.x, p.z) > 2.0 * s.r + 0.1 {
                    continue;
                }
                let (depth, n) = terrain_contact(p, s.r);
                if depth <= 0.0 {
                    continue;
                }
                let r = p - n * s.r - body.x;
                let lambda = self.shift(s.body, None, r, Vec3::ZERO, n * depth);
                let q = self.bodies[s.body].q;
                self.contacts.push(Contact {
                    a: s.body,
                    b: None,
                    ra: r,
                    rb: Vec3::ZERO,
                    n,
                    lambda,
                    mu: BODY_FRICTION,
                    restitution: RESTITUTION,
                    ski: s.ski.then(|| unit(q * (s.b - s.a))),
                });
            }
        }
    }

    fn relative_velocity(&self, c: &Contact) -> Vec3 {
        let vb = c.b.map_or(Vec3::ZERO, |b| self.bodies[b].velocity_at(c.rb));
        self.bodies[c.a].velocity_at(c.ra) - vb
    }

    /// Applies impulse `p` at the contact: `+p` to `a`, `-p` to `b`.
    fn impulse(&mut self, c: &Contact, p: Vec3) {
        let a = &mut self.bodies[c.a];
        a.v += p * a.inv_mass;
        a.w += a.inv_inertia_world() * c.ra.cross(p);
        if let Some(b) = c.b {
            let b = &mut self.bodies[b];
            b.v -= p * b.inv_mass;
            b.w -= b.inv_inertia_world() * c.rb.cross(p);
        }
    }

    fn contact_weight(&self, c: &Contact, n: Vec3) -> f32 {
        self.bodies[c.a].weight(c.ra, n) + c.b.map_or(0.0, |b| self.bodies[b].weight(c.rb, n))
    }

    /// Coulomb friction along unit `dir` against relative speed `speed`, at most `max` impulse.
    fn friction(&mut self, c: &Contact, dir: Vec3, speed: f32, max: f32) {
        if dir == Vec3::ZERO {
            return;
        }
        let j = (speed / self.contact_weight(c, dir)).clamp(-max, max);
        self.impulse(c, -dir * j);
    }

    /// Restitution (from the pre-step approach speed) and friction from each contact's normal
    /// impulse, then passive joint damping. Position repair never becomes launch speed: the
    /// normal speed is reset to the bounce alone.
    fn solve_velocities(&mut self, before: &[(Vec3, Vec3)], h: f32) {
        for i in 0..self.contacts.len() {
            let c = self.contacts[i];
            let at = |k: usize, r: Vec3| before[k].0 + before[k].1.cross(r);
            let approach = (at(c.a, c.ra) - c.b.map_or(Vec3::ZERO, |b| at(b, c.rb))).dot(c.n);
            let bounce = if approach < -1.0 {
                -c.restitution * approach
            } else {
                0.0
            };
            let vn = self.relative_velocity(&c).dot(c.n);
            let p = (bounce - vn) / self.contact_weight(&c, c.n);
            self.impulse(&c, c.n * p);
            let v = self.relative_velocity(&c);
            let vt = v - c.n * v.dot(c.n);
            let max = c.lambda / h;
            match c.ski {
                Some(axis) => {
                    let along = unit(axis.reject_from(c.n));
                    let slide = vt.dot(along);
                    self.friction(&c, along, slide, SKI_ALONG * max);
                    let across = vt - along * slide;
                    self.friction(&c, unit(across), across.length(), SKI_ACROSS * max);
                }
                None => self.friction(&c, unit(vt), vt.length(), c.mu * max),
            }
        }
        // Rolling resistance: whatever lies in the snow digs in, so it cannot roll on forever.
        let roll = (-ROLL_DAMPING * h).exp();
        let mut rolled = [false; BODIES];
        for c in self.contacts.iter().filter(|c| c.b.is_none()) {
            if !std::mem::replace(&mut rolled[c.a], true) {
                self.bodies[c.a].w *= roll;
            }
        }
        let keep = 1.0 - (-JOINT_DAMPING * h).exp();
        for k in 0..self.joints.len() {
            let j = self.joints[k];
            let (ip, ic) = (
                self.bodies[j.parent].inv_inertia_world(),
                self.bodies[j.child].inv_inertia_world(),
            );
            let spin = self.bodies[j.child].w - self.bodies[j.parent].w;
            let rate = spin.length();
            if rate < 1e-6 {
                continue;
            }
            let n = spin / rate;
            let p = n * (rate * keep / (n.dot(ip * n) + n.dot(ic * n)));
            self.bodies[j.parent].w += ip * p;
            self.bodies[j.child].w -= ic * p;
        }
    }

    /// Angle (rad) by which joint `k` is outside its (possibly broken) range.
    fn excess(&self, k: usize) -> f32 {
        let j = &self.joints[k];
        let Some(l) = j.limit.filter(|_| j.holds()) else {
            return 0.0;
        };
        let rel = self.relative(j).1;
        rel.angle_between(clamp_rotation(rel, &l, j.mirror, j.slack()))
    }

    /// Deepest overlap (m, negative) between two segments that should not touch, and which: among
    /// pairs of skis and poles only (`gear`), or among pairs involving the body.
    #[cfg(test)]
    fn worst_overlap(&self, gear: bool) -> (f32, String) {
        const NAMES: [B; BODIES] = {
            use B::*;
            [
                Pelvis, Abdomen, Thorax, Head, UpperArmL, UpperArmR, ForearmL, ForearmR, HandL,
                HandR, ThighL, ThighR, ShinL, ShinR, BootL, BootR, SkiL, SkiR, PoleL, PoleR,
            ]
        };
        let is_gear = |k: usize| k >= B::SkiL as usize;
        let mut worst = (f32::MAX, String::new());
        for &(a, b) in &self.pairs {
            if (is_gear(a) && is_gear(b)) != gear {
                continue;
            }
            for i in self.range[a].0..self.range[a].1 {
                for j in self.range[b].0..self.range[b].1 {
                    let (sa, sb) = (self.shapes[i], self.shapes[j]);
                    if !sa.meets(&sb) {
                        continue;
                    }
                    let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                    let (a0, a1, b0, b1) = (
                        ba.point(sa.a),
                        ba.point(sa.b),
                        bb.point(sb.a),
                        bb.point(sb.b),
                    );
                    let (s, t) = closest_params(a0, a1, b0, b1);
                    let gap = a0.lerp(a1, s).distance(b0.lerp(b1, t)) - sa.r - sb.r;
                    if gap < worst.0 {
                        worst = (gap, format!("{:?}/{:?}", NAMES[a], NAMES[b]));
                    }
                }
            }
        }
        worst
    }
}

/// Penetration depth of a sphere at `p` into the terrain, and the surface normal.
fn terrain_contact(p: Vec3, radius: f32) -> (f32, Vec3) {
    let eps = 0.05;
    let dx = (terrain_height(p.x + eps, p.z) - terrain_height(p.x - eps, p.z)) / (2.0 * eps);
    let dz = (terrain_height(p.x, p.z + eps) - terrain_height(p.x, p.z - eps)) / (2.0 * eps);
    let n = Vec3::new(-dx, 1.0, -dz).normalize();
    (radius - (p.y - terrain_height(p.x, p.z)) * n.y, n)
}

#[cfg(test)]
mod tests {
    use super::super::pose::{
        ANKLE_ABOVE_SKI, BONES, CHEST_TO_NECK, FOREARM, GEAR, HAND, HEEL_BACK, HIP_HALF_WIDTH,
        NECK_TO_HEAD, PELVIS_TO_WAIST, POLE_GRIP, POLE_LENGTH, SHIN, SHOULDER_HALF_WIDTH, SKI_BACK,
        SKI_FRONT, SKI_TIP_RISE, STANCE_WIDTH, THIGH, TOE_FORWARD, UPPER_ARM, WAIST_TO_CHEST,
    };
    use super::*;
    use crate::bike::HILL_X;

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

    fn body(r: &SkiRagdoll) -> &Body {
        r.body.as_ref().unwrap()
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
            for (gear, limit) in [(false, -0.005), (true, -0.01)] {
                let (gap, which) = body(r).worst_overlap(gear);
                assert!(gap > limit, "{label} tick {tick}: {which} overlap {gap}");
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
    }

    /// Slide over 2 s of a ski released on the slope, along vs across the fall line (the skier is
    /// lifted clear so only the ski touches the snow).
    fn slide(yaw: f32) -> f32 {
        let pose = on_hill(-20.0, yaw);
        let mut r = SkiRagdoll::default();
        r.activate(&pose, &pose, DT);
        let b = r.body.as_mut().unwrap();
        for k in 0..b.joints.len() {
            if b.joints[k].binding {
                b.joints[k].broken = true;
                b.pairs.push((b.joints[k].parent, b.joints[k].child));
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
}
