//! Procedural showcase arena (flat white floor, three jumps, one big hill), bike, rider and skeleton overlay.
//!
//! Everything is generated at startup from primitives (no asset files). Per frame only
//! `Transform`s change: [`animate_bike`] solves one rig (named points in bike-local space)
//! and every moving part reads its end points from it.
//!
//! Bike-local frame: forward -Z, right +X, up +Y. Bike root = `Bike::position` with
//! yaw/pitch/roll; hubs sit at the shared contract positions, the rest is derived from them.

use bevy::asset::RenderAssetUsages;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::ops::{Index, IndexMut};

pub(crate) use crate::animation::AnimationState;
use crate::animation::{
    H_BAR, H_NO, H_ONE, H_SEAT, H_TIRE, H_TOBO, H_TUCK, HAND_KINDS, L_CAN, L_COUNTER, L_DANGLE,
    L_INDIAN, L_LIFT, L_NAC, L_NO, L_NOCAN, L_ONE, L_SUPER, L_TAIL, LEG_KINDS, Superman,
};
use crate::bike::{
    Bike, CollisionPose, CollisionSphere, Discipline, SUSPENSION_REST, WHEEL_RADIUS, WHEELBASE,
    terrain_height,
};
use crate::ragdoll::{COUNT, Ragdoll};

// ---------------------------------------------------------------------------------------------
// Bike geometry (meters, bike-local). Hubs come from the Bike contract; the rest is fixed here.
// ---------------------------------------------------------------------------------------------

const BB: Vec3 = Vec3::new(0.0, -0.32, 0.17); // bottom bracket
const SEAT_DIR: Vec3 = Vec3::new(0.0, 0.956, 0.2924); // 73 degree seat tube, leaning back
const STEER_AXIS: Vec3 = Vec3::new(0.0, 0.9135, 0.4067); // 66 degree head angle
const CROWN: Vec3 = Vec3::new(0.0, 0.0885, -0.4056); // fork crown, on the steering axis
const BAR_C: Vec3 = Vec3::new(0.0, 0.333, -0.363); // bar centre at zero steer
const FORK_X: f32 = 0.075; // fork legs clear the 5.5 cm half-width tire
const DROP_X: f32 = 0.08; // rear dropouts
const CRANK: f32 = 0.17;
const HIP_REST: Vec3 = Vec3::new(0.0, 0.392, 0.365);
// Rider proportions: our own constants, set near the decoded retail rider's bone lengths.
const HIP_HALF: f32 = 0.1016;
const SHOULDER_HALF: f32 = 0.178;
const THIGH: f32 = 0.459;
const SHIN: f32 = 0.412;
const UPPER_ARM: f32 = 0.31;
const FOREARM: f32 = 0.262;
const TORSO: f32 = 0.49;
/// Peak roll of the drawn bike under a standing pedalling rider, rad.
const ROCK: f32 = 0.05;
/// Rest hub height; rear shock / rocker positions are measured from it.
const HUB_REST_Y: f32 = -SUSPENSION_REST + 0.05;

const WRIST_OFF: Vec3 = Vec3::new(0.0, 0.03, 0.035); // wrist relative to the grip
const ANKLE_OFF: Vec3 = Vec3::new(0.0, 0.105, 0.08); // ankle relative to the pedal
const HEEL_OFF: Vec3 = Vec3::new(0.0, 0.057, 0.115);
const TOE_OFF: Vec3 = Vec3::new(0.0, 0.057, -0.13);
/// Free-limb targets stay a little inside the real reach.
const ARM_REACH: f32 = (UPPER_ARM + FOREARM) * 0.97;
const LEG_REACH: f32 = (THIGH + SHIN) * 0.97;

/// Per-discipline look that `BikeProfile` does not carry.
struct Look {
    frame: [f32; 3],
    jersey: [f32; 3],
    /// Seat post length from the bottom bracket (saddle height).
    post: f32,
    /// Rigid frame and fork: no rear shock, no stanchions, frame carried rigidly on the hubs.
    rigid: bool,
    shock: bool,
    knobs: bool,
}

impl Look {
    fn of(d: Discipline) -> Self {
        match d {
            Discipline::Downhill => Self {
                frame: [1.0, 0.34, 0.28],
                jersey: [0.10, 0.16, 0.28],
                post: 0.56,
                rigid: false,
                shock: true,
                knobs: true,
            },
            Discipline::Road => Self {
                frame: [0.16, 0.55, 0.95],
                jersey: [0.85, 0.12, 0.14],
                post: 0.70,
                rigid: true,
                shock: false,
                knobs: false,
            },
            Discipline::Slopestyle => Self {
                frame: [0.62, 0.27, 0.95],
                jersey: [0.92, 0.55, 0.10],
                post: 0.60,
                rigid: false,
                shock: true,
                knobs: true,
            },
            Discipline::Freeride => Self {
                frame: [0.20, 0.78, 0.36],
                jersey: [0.10, 0.40, 0.38],
                post: 0.66,
                rigid: false,
                shock: true,
                knobs: true,
            },
        }
    }
}

/// Named rig points. `N` must stay last.
#[derive(Clone, Copy, Debug)]
pub(crate) enum P {
    Bb,
    SeatLink,
    SeatTop,
    PostTop,
    HeadBot,
    HeadTop,
    Crown,
    SteerTop,
    RearHub,
    DropL,
    DropR,
    CsFrontL,
    CsFrontR,
    CsMidL,
    CsMidR,
    SsTopL,
    SsTopR,
    SsBowL,
    SsBowR,
    Rocker,
    ShockTop,
    ShockMid,
    FrontHub,
    CrownL,
    CrownR,
    DropFL,
    DropFR,
    LegTopL,
    LegTopR,
    StanBotL,
    StanBotR,
    BarC,
    BarInL,
    BarInR,
    BarOutL,
    BarOutR,
    GripInL,
    GripInR,
    GripL,
    GripR,
    LeverRootL,
    LeverRootR,
    LeverTipL,
    LeverTipR,
    CrankHubL,
    CrankHubR,
    CrankEndL,
    CrankEndR,
    PedalL,
    PedalR,
    ChainFrontTop,
    ChainFrontBot,
    ChainRearTop,
    ChainRearBot,
    DerailA,
    DerailB,
    Hip,
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
    Shoulder,
    ShoulderL,
    ShoulderR,
    ElbowL,
    ElbowR,
    WristL,
    WristR,
    Neck,
    Head,
    HandL,
    HandR,
    N,
}

const SIDES: [f32; 2] = [-1.0, 1.0];

fn side(l: P, r: P, s: f32) -> P {
    if s < 0.0 { l } else { r }
}

#[derive(Clone, Copy)]
struct Rig {
    p: [Vec3; P::N as usize],
    /// Front assembly rotation about the steering axis: steering, bar spin, table/x-up turn.
    steer: Quat,
    head_rot: Quat,
    /// Bike assembly (everything but the rider) -> rider space: rigid road carry and whip/table/...
    frame: Transform,
    /// Rear triangle about the steering axis, in frame space (tailwhip).
    rear: Transform,
    /// Crank angle including the crankflip spin.
    crank: f32,
}

impl Index<P> for Rig {
    type Output = Vec3;
    fn index(&self, i: P) -> &Vec3 {
        &self.p[i as usize]
    }
}

impl IndexMut<P> for Rig {
    fn index_mut(&mut self, i: P) -> &mut Vec3 {
        &mut self.p[i as usize]
    }
}

impl Rig {
    /// A front-assembly point (bike points are stored in frame space) in rider space.
    fn front_pt(&self, v: Vec3) -> Vec3 {
        self.frame.transform_point(v)
    }

    /// A rear-triangle point (pedals, saddle) in rider space.
    fn rear_pt(&self, v: Vec3) -> Vec3 {
        self.frame.transform_point(self.rear.transform_point(v))
    }
}

/// Two-bone IK: returns the middle joint for limb lengths `a`,`b`, bending towards `pole`.
fn ik(root: Vec3, target: Vec3, a: f32, b: f32, pole: Vec3) -> Vec3 {
    let d = target - root;
    let len = d.length().clamp((a - b).abs() + 1e-3, a + b - 1e-3);
    let dir = d.try_normalize().unwrap_or(Vec3::NEG_Y);
    let along = (a * a - b * b + len * len) / (2.0 * len);
    let h = (a * a - along * along).max(0.0).sqrt();
    let bend = (pole - dir * pole.dot(dir))
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    root + dir * along + bend * h
}

fn unit(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).normalize()
}

fn smooth01(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Two-bone limb: `(mid, end)` with exact bone lengths; an unreachable target is pulled onto the
/// reachable shell instead of stretching a bone.
fn limb(root: Vec3, target: Vec3, a: f32, b: f32, pole: Vec3) -> (Vec3, Vec3) {
    let d = target - root;
    let len = d.length().clamp((a - b).abs() + 1e-3, a + b - 1e-3);
    let end = root + d.try_normalize().unwrap_or(Vec3::NEG_Y) * len;
    (ik(root, end, a, b, pole), end)
}

/// Wrist target of a released arm for pose kind `k` (mirrored by `s`, in rider space).
fn arm_free(k: usize, s: f32, sh: Vec3, hover: Vec3, tire: Vec3, seat: Vec3) -> Vec3 {
    match k {
        H_ONE => sh + unit(0.85 * s, 0.45, -0.25) * ARM_REACH,
        H_NO => sh + unit(0.95 * s, 0.35, -0.15) * ARM_REACH,
        H_TUCK => sh + Vec3::new(-0.10 * s, -0.18, -0.30),
        H_BAR => hover,
        H_TIRE => tire,
        H_SEAT => seat,
        H_TOBO => sh + unit(0.45 * s, 0.55, -0.70) * ARM_REACH,
        _ => hover,
    }
}

/// Ankle target of a released leg for pose kind `k`; `sigma` is the trick side.
fn leg_free(k: usize, s: f32, sigma: f32, hip: Vec3, hj: Vec3, sup: &Superman) -> Vec3 {
    let near = s * sigma; // +1 on the trick side
    match k {
        L_ONE => hj + unit(0.85 * s, -0.35, -0.15) * (LEG_REACH * 0.95),
        L_NO => hj + unit(0.55 * s, -0.62, -0.50) * (LEG_REACH * 0.95),
        L_CAN => hip + Vec3::new(-0.26 * s, -0.40, -0.33),
        L_NOCAN => {
            hip + Vec3::new(
                -sigma * (0.30 + 0.04 * near),
                -0.42 + 0.04 * near,
                -0.28 + 0.10 * near,
            )
        }
        // Straight legs swing at the hip from just ahead of straight down to straight back; the
        // knees fold on the way back to the pedals.
        L_SUPER => {
            let phi = -0.23 + (1.89 + sup.level) * sup.legs;
            hj + unit(0.08 * s, -phi.cos(), phi.sin())
                * ((THIGH + SHIN) * 0.99 * (1.0 - 0.15 * sup.fold))
        }
        // Both feet go to the trick side: the near one up and back over the passing rear wheel,
        // the other across and forward, nearly straight.
        L_TAIL if near > 0.0 => hip + Vec3::new(0.62 * sigma, -0.05, 0.14),
        L_TAIL => hip + Vec3::new(0.58 * sigma, -0.15, -0.50),
        L_NAC => hj + Vec3::new(-0.20 * s, -0.02, 0.62),
        L_INDIAN => hj + unit(0.60 * s, -0.15, 0.55) * (LEG_REACH * 0.9),
        L_DANGLE => hj + Vec3::new(0.05 * s, -0.80, 0.05),
        L_COUNTER => hj + Vec3::new(-0.55 * sigma, -0.55, -0.10),
        L_LIFT => hj + Vec3::new(0.20 * s, -0.25, -0.45),
        _ => hj + Vec3::new(0.05 * s, -0.80, 0.05),
    }
}

/// Foot pitch about the pedal axle, rad (+ toe up): toe down at the bottom of the stroke while
/// pedalling, heel dropped otherwise. `s` is the side (-1 left); the left crank is half a turn on.
fn ankling(crank: f32, s: f32, pedal: f32) -> f32 {
    let phase = crank + if s < 0.0 { PI } else { 0.0 };
    -0.2 * pedal * phase.cos() + 0.15 * (1.0 - pedal)
}

fn rig(b: &Bike, a: &AnimationState) -> Rig {
    solve(b, a, true)
}

fn solve(b: &Bike, a: &AnimationState, rocking: bool) -> Rig {
    let look = Look::of(b.discipline);
    let prof = b.discipline.profile();
    let lay = a.bike_layer();
    let body = a.body();
    // `calm` = everything that turns the front assembly except bar spin; the bar line follows it.
    let calm = body.steer.unwrap_or(b.steering) + lay.bar_turn; // +steering = left = +yaw
    let q = Quat::from_axis_angle(STEER_AXIS, calm + a.bars.angle);
    let qc = Quat::from_axis_angle(STEER_AXIS, calm);
    let mut r = Rig {
        p: [Vec3::ZERO; P::N as usize],
        steer: q,
        head_rot: Quat::IDENTITY,
        frame: Transform::IDENTITY,
        rear: Transform::IDENTITY,
        crank: body.crank.unwrap_or(b.crank_phase) + a.crank.angle,
    };
    let st = |v: Vec3| CROWN + q * (v - CROWN); // steer about the head-tube axis
    let stc = |v: Vec3| CROWN + qc * (v - CROWN);
    let x = Vec3::X;
    // A rigid bike keeps its hubs at the design position; `r.frame` carries it on the contract hubs.
    let (sf, sr) = if look.rigid {
        (0.05, 0.05)
    } else {
        (b.suspension[0], b.suspension[1])
    };

    // Frame (unsteered).
    r[P::Bb] = BB;
    r[P::SeatLink] = BB + SEAT_DIR * 0.38;
    r[P::SeatTop] = BB + SEAT_DIR * 0.50;
    r[P::PostTop] = BB + SEAT_DIR * look.post;
    r[P::Crown] = CROWN;
    r[P::HeadBot] = CROWN + STEER_AXIS * 0.02;
    r[P::HeadTop] = CROWN + STEER_AXIS * 0.14;
    r[P::SteerTop] = CROWN + STEER_AXIS * 0.24;

    // Rear suspension: hub follows the contract, stays are bowed around the tire.
    let hub_r = Vec3::new(0.0, -SUSPENSION_REST + sr, WHEELBASE * 0.5);
    r[P::RearHub] = hub_r;
    let bow = r[P::SeatLink].lerp(hub_r, 0.45);
    let cs0 = BB + Vec3::new(0.0, 0.0, 0.03);
    for s in SIDES {
        r[side(P::DropL, P::DropR, s)] = hub_r + x * (DROP_X * s);
        r[side(P::CsFrontL, P::CsFrontR, s)] = cs0 + x * (0.04 * s);
        r[side(P::CsMidL, P::CsMidR, s)] = cs0.lerp(hub_r, 0.29) + x * (0.072 * s);
        r[side(P::SsTopL, P::SsTopR, s)] = r[P::SeatLink] + x * (0.025 * s);
        r[side(P::SsBowL, P::SsBowR, s)] = bow + x * (0.085 * s);
    }
    r[P::Rocker] = Vec3::new(0.0, -0.08 + 0.45 * (hub_r.y - HUB_REST_Y), 0.215);
    r[P::ShockTop] = r[P::HeadTop].lerp(r[P::SeatTop], 0.72);
    r[P::ShockMid] = r[P::ShockTop].lerp(r[P::Rocker], 0.5);

    // Front end: everything below/above the crown turns with the bars.
    let hub_f = Vec3::new(0.0, -SUSPENSION_REST + sf, -WHEELBASE * 0.5);
    r[P::FrontHub] = st(hub_f);
    let bar_ratio = prof.bar_width / 0.38;
    let grip_drop = prof.grip_drop;
    let mut grip_local = [Vec3::ZERO; 2]; // grips at zero steer, on the unsteered bar
    for (i, s) in SIDES.into_iter().enumerate() {
        let crown = st(CROWN + x * (FORK_X * s));
        let fork_drop = st(hub_f + x * (FORK_X * s));
        let dir = (crown - fork_drop).normalize();
        // Suspension fork: slider length 0.30, stanchion slides inside it. Rigid fork: one leg.
        let (leg_top, stan_bot) = if look.rigid {
            (crown, crown)
        } else {
            let top = fork_drop + dir * 0.30;
            (top, top - dir * 0.05)
        };
        r[side(P::CrownL, P::CrownR, s)] = crown;
        r[side(P::DropFL, P::DropFR, s)] = fork_drop;
        r[side(P::LegTopL, P::LegTopR, s)] = leg_top;
        r[side(P::StanBotL, P::StanBotR, s)] = stan_bot;
        // Flat bars, or drop bars: the grip end sits `grip_drop` lower and forward.
        let bar_in_l = BAR_C + Vec3::new(0.12 * bar_ratio * s, 0.0, 0.0);
        let bar_out_l = BAR_C
            + Vec3::new(
                prof.bar_width * s,
                0.015 - grip_drop,
                0.06 - grip_drop * 0.8,
            );
        let (bar_in, bar_out) = (st(bar_in_l), st(bar_out_l));
        grip_local[i] = bar_in_l.lerp(bar_out_l, lay.grip_slide);
        r[side(P::BarInL, P::BarInR, s)] = bar_in;
        r[side(P::BarOutL, P::BarOutR, s)] = bar_out;
        r[side(P::GripInL, P::GripInR, s)] = bar_in.lerp(bar_out, 0.54);
        r[side(P::GripL, P::GripR, s)] = st(grip_local[i]);
        r[side(P::LeverRootL, P::LeverRootR, s)] =
            st(BAR_C + Vec3::new(0.27 * bar_ratio * s, 0.0, 0.034));
        r[side(P::LeverTipL, P::LeverTipR, s)] = st(BAR_C
            + Vec3::new(
                0.255 * bar_ratio * s,
                -0.03 - grip_drop,
                -0.05 - grip_drop * 0.8,
            ));
    }
    r[P::BarC] = st(BAR_C);

    // Drivetrain.
    for (s, phase) in [(-1.0f32, r.crank + PI), (1.0, r.crank)] {
        let hub = BB + x * (0.075 * s);
        let end = hub + Quat::from_rotation_x(-phase) * Vec3::new(0.0, -CRANK, 0.0);
        r[side(P::CrankHubL, P::CrankHubR, s)] = hub;
        r[side(P::CrankEndL, P::CrankEndR, s)] = end;
        r[side(P::PedalL, P::PedalR, s)] = end + x * (0.05 * s);
    }
    r[P::ChainFrontTop] = BB + Vec3::new(0.07, 0.095, 0.0);
    r[P::ChainFrontBot] = BB + Vec3::new(0.07, -0.095, 0.0);
    r[P::ChainRearTop] = hub_r + Vec3::new(0.056, 0.052, 0.0);
    r[P::ChainRearBot] = hub_r + Vec3::new(0.056, -0.052, 0.0);
    r[P::DerailA] = hub_r + Vec3::new(0.062, -0.07, -0.005);
    r[P::DerailB] = hub_r + Vec3::new(0.062, -0.14, 0.03);

    // Bike assembly relative to the rider. Whip/table/turndown/invert pivot on the grip centre;
    // the roll axis follows the (turned) grip line for table-likes and the pitch axis is the grip
    // line, so the hands stay on the grips. Tailwhip spins the rear triangle about the steering axis.
    let grip_pts = [stc(grip_local[0]), stc(grip_local[1])];
    let pivot = (grip_pts[0] + grip_pts[1]) * 0.5;
    let bar_line = (grip_pts[1] - grip_pts[0])
        .try_normalize()
        .unwrap_or(qc * Vec3::X);
    let roll_axis = Vec3::NEG_Z
        .lerp(bar_line * a.side, lay.table_t)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    let pitch_axis = if bar_line.x < 0.0 {
        -bar_line
    } else {
        bar_line
    };
    let trick_q = Quat::from_rotation_y(lay.yaw)
        * Quat::from_axis_angle(roll_axis, lay.roll)
        * Quat::from_axis_angle(pitch_axis, lay.pitch);
    let trick = Transform {
        translation: pivot - trick_q * pivot,
        rotation: trick_q,
        scale: Vec3::ONE,
    };
    let carry = if look.rigid {
        Transform {
            translation: Vec3::new(0.0, (b.suspension[0] + b.suspension[1]) * 0.5 - 0.05, 0.0),
            rotation: Quat::from_rotation_x((b.suspension[0] - b.suspension[1]) / WHEELBASE),
            scale: Vec3::ONE,
        }
    } else {
        Transform::IDENTITY
    };
    // Standing strokes rock the drawn bike under the rider about the tyre contact line, against
    // the pelvis sway. Display only: `collision_pose` and `rider_points` solve without it.
    let rock = if rocking {
        Quat::from_rotation_z(-ROCK * body.pedal * body.stand * r.crank.sin())
    } else {
        Quat::IDENTITY
    };
    let ground = Vec3::new(
        0.0,
        -SUSPENSION_REST + (b.suspension[0] + b.suspension[1]) * 0.5 - WHEEL_RADIUS,
        0.0,
    );
    let rocked = Transform {
        translation: ground - rock * ground,
        rotation: rock,
        scale: Vec3::ONE,
    };
    r.frame = rocked.mul_transform(trick).mul_transform(carry);
    let tail_q = Quat::from_axis_angle(STEER_AXIS, a.tail.angle);
    r.rear = Transform {
        translation: CROWN - tail_q * CROWN,
        rotation: tail_q,
        scale: Vec3::ONE,
    };

    // Attachment targets in rider space: hands on the grips, feet on the pedals. The attached
    // foot pitches with the stroke (toe down at the bottom, heel dropped coasting).
    let fq = r.frame.rotation;
    let rq = fq * r.rear.rotation;
    let mut wrist_att = [Vec3::ZERO; 2];
    let mut ankle_att = [Vec3::ZERO; 2];
    let mut foot_att = [Quat::IDENTITY; 2];
    let mut grip_calm = [Vec3::ZERO; 2];
    for (i, s) in SIDES.into_iter().enumerate() {
        foot_att[i] = rq * Quat::from_rotation_x(ankling(r.crank, s, body.pedal));
        wrist_att[i] = r.front_pt(r[side(P::GripL, P::GripR, s)]) + fq * WRIST_OFF;
        ankle_att[i] = r.rear_pt(r[side(P::PedalL, P::PedalR, s)]) + foot_att[i] * ANKLE_OFF;
        grip_calm[i] = r.front_pt(grip_pts[i]);
    }

    // Hips: the desired seat/stance point, pulled into reach of the pedals that are still
    // attached. The reach is a soft minimum so the hip glides instead of snapping between the
    // two pedals, and a stroke never fully straightens the leg.
    let leg_max = (THIGH + SHIN) * 0.99;
    let sag = (body.sag * 0.5).clamp(-0.025, 0.06);
    // Standing strokes rock the pelvis over the pushing pedal; a lean carries it outboard of the
    // bike (the bike leans more than the rider).
    let sway = -body.pedal * (0.012 + 0.03 * body.stand) * r.crank.sin();
    let lean_x = 0.2 * body.turn;
    let [throw, twist, drop, tuck] = body.english;
    // Barrel roll: the hips counter the dropped shoulder; a tucked rotation sinks them.
    let mut hip = HIP_REST
        + SEAT_DIR * (look.post - 0.60)
        + Vec3::new(
            body.lateral + lean_x + sway + 0.04 * drop,
            0.22 * body.stand - 0.11 * body.crouch - sag - 0.07 * tuck,
            0.22 * body.back - 0.03 * body.stand + 0.04 * throw,
        );
    {
        const K: f32 = 0.05;
        let caps = [0, 1].map(|i| {
            let d = hip + x * (HIP_HALF * SIDES[i]) - ankle_att[i];
            let up = (leg_max * leg_max - d.x * d.x - d.z * d.z).max(1e-4).sqrt();
            ankle_att[i].y + up + a.foot_rel[i] * 10.0
        });
        let low = hip.y.min(caps[0]).min(caps[1]);
        let sum: f32 = [hip.y, caps[0], caps[1]]
            .iter()
            .map(|v| (-(v - low) / K).exp())
            .sum();
        hip.y = low - K * sum.ln();
    }
    let leg_max = (THIGH + SHIN) * 0.995;
    for _ in 0..6 {
        for (i, s) in SIDES.into_iter().enumerate() {
            let d = hip + x * (HIP_HALF * s) - ankle_att[i];
            let l = d.length();
            if l > leg_max {
                hip -= d * ((l - leg_max) / l * (1.0 - a.foot_rel[i]));
            }
        }
    }

    // Torso: shoulders sit where the torso circle (radius TORSO about the hips) meets the reach
    // circle about the attached grips, so arms keep a constant elbow bend and torso length is
    // exact. With hands released the torso direction is authored; a tire grab folds it forward.
    let wa = [1.0 - a.hand_rel[0], 1.0 - a.hand_rel[1]];
    let wsum = wa[0] + wa[1];
    let grips = if wsum > 1e-3 {
        (grip_calm[0] * wa[0] + grip_calm[1] * wa[1]) / wsum
    } else {
        (grip_calm[0] + grip_calm[1]) * 0.5
    };
    let reach =
        (0.44 + 0.09 * body.back.max(0.0) + 0.02 * body.stand - 0.14 * body.land).clamp(0.3, 0.53);
    let d = grips - hip;
    let raw = Vec2::new(d.y, d.z).length().max(1e-4);
    let dist = raw.clamp((TORSO - reach).abs() + 0.02, TORSO + reach - 0.02);
    let (dy, dz) = (d.y / raw, d.z / raw);
    let along = (TORSO * TORSO - reach * reach + dist * dist) / (2.0 * dist);
    let h = (TORSO * TORSO - along * along).max(0.0).sqrt();
    let (py, pz) = if dz <= 0.0 { (-dz, dy) } else { (dz, -dy) }; // perpendicular, pointing up
    let on_grips = Vec3::new(
        hip.x,
        hip.y + dy * along + py * h,
        hip.z + dz * along + pz * h,
    ) - hip;
    let released = (a.hand_rel[0] + a.hand_rel[1]) * 0.5;
    let free_dir = Vec3::new(0.0, 1.0, -(0.45 - 0.55 * body.back.clamp(-0.5, 1.0))).normalize();
    let fold_dir = unit(0.0, 0.2, -0.98);
    let dir = on_grips
        .normalize_or_zero()
        .lerp(free_dir, released)
        .lerp(fold_dir, 0.6 * body.fold)
        .try_normalize()
        .unwrap_or(Vec3::Y);
    let mut shoulder = hip + dir * (on_grips.length() * (1.0 - released) + TORSO * released);
    // Shoulders swing against the pelvis on a standing stroke; in a lean the torso stays more
    // upright than the bike (shoulders outboard of the pelvis).
    shoulder.x += 0.13 * body.turn - 1.6 * sway;
    // Air body English: the shoulders drop into a barrel roll and lead a backflip (thrown back
    // and up) or a front flip (over the bars); the arms bend to keep the grips.
    shoulder += Vec3::new(
        -0.12 * drop,
        -0.04 * drop.abs() + 0.03 * throw,
        0.07 * throw,
    );

    // Shoulders follow the grip line (bars turned, whipped or tabled); modulo 180 degrees so an
    // x-up does not flip them, and fading out when the grips sit together.
    let mut g = grip_calm[1] - grip_calm[0];
    if g.x < -0.12 {
        g = -g;
    }
    let sq = match g.try_normalize() {
        Some(gn) => {
            let line = Quat::from_rotation_arc(Vec3::X, gn);
            let f = (0.45 + 0.45 * smooth01(0.5, 1.2, line.angle_between(Quat::IDENTITY)))
                * smooth01(0.08, 0.25, g.length());
            Quat::IDENTITY.slerp(line, f)
        }
        None => Quat::IDENTITY,
    };
    // A spin is led by the shoulders, turned ahead of the hips and the bike.
    let sq = Quat::from_rotation_y(twist) * sq;
    // Released feet leave the pelvis free to follow the shoulders. Keeping it pinned to the
    // saddle during Superman/table combinations would stretch the spine.
    let free_feet = (a.foot_rel[0] + a.foot_rel[1]) * 0.5;
    // Superman: the body lies flat behind the gripped bars (trunk 75–81 degrees from vertical,
    // shoulders just above the wrists), arms straightening, while the bike swings under the bars.
    let sup = a.superman();
    let flat = sup.body * (1.0 - released);
    if flat > 0.0 {
        let w = (wrist_att[0] + wrist_att[1]) * 0.5;
        let lat = ((wrist_att[1].x - wrist_att[0].x).abs() * 0.5 - SHOULDER_HALF).max(0.0);
        let arm = 0.47 + 0.08 * sup.arms;
        let s_t = w + Vec3::new(0.0, 0.10, (arm * arm - 0.01 - lat * lat).max(0.0).sqrt());
        let tr = 1.31 + 0.10 * sup.legs + sup.level;
        let h_t = s_t + Vec3::new(0.0, -tr.cos(), tr.sin()) * TORSO;
        shoulder = shoulder.lerp(s_t, flat);
        hip = hip.lerp(h_t, flat);
    }
    // Tailwhip: the rider sinks behind the bars with the trunk nearly upright while the frame
    // spins under him.
    let whip = a.leg_w[L_TAIL] * free_feet;
    if whip > 0.0 {
        shoulder.y -= 0.08 * whip;
        let up = (shoulder - hip)
            .try_normalize()
            .unwrap_or(Vec3::Y)
            .lerp(Vec3::Y, 0.6)
            .normalize();
        hip = hip.lerp(shoulder - up * TORSO, whip);
    }
    // At full steer lock or a whip one grip swings away: pull the shoulders into reach of the
    // attached hands (torso stretches a few cm) rather than detaching a hand.
    let arm_max = (UPPER_ARM + FOREARM) * 0.985;
    for _ in 0..6 {
        for (i, s) in SIDES.into_iter().enumerate() {
            let d = shoulder + sq * (x * (SHOULDER_HALF * s)) - wrist_att[i];
            let l = d.length();
            if l > arm_max {
                shoulder -= d * ((l - arm_max) / l * wa[i]);
            }
        }
    }
    let torso_dir = (shoulder - hip).try_normalize().unwrap_or(Vec3::Y);
    hip = hip.lerp(shoulder - torso_dir * TORSO, free_feet);
    r[P::Hip] = hip;
    r[P::Shoulder] = shoulder;
    r[P::Waist] = hip.lerp(shoulder, 0.5);
    let fwd = (-(shoulder - hip).normalize().z).clamp(-0.3, 1.0);
    // Head stays level and looking ahead: it keeps little of the torso's roll and lean, and
    // nods against the torso's acceleration lag.
    r[P::Neck] = shoulder + Vec3::new(0.0, 0.07, -0.02).lerp(Vec3::new(0.0, 0.03, -0.08), flat);
    r[P::Neck].x *= 0.7;
    r[P::Head] = shoulder
        + Vec3::new(0.0, 0.145, -0.06 - 0.05 * fwd).lerp(Vec3::new(0.0, 0.05, -0.18), flat);
    r[P::Head].x *= 0.4;
    // The head leads every rotation: it turns into a spin ahead of the shoulders, tilts into a
    // barrel roll and looks back for the landing in a backflip (down past the bars in a front
    // flip).
    r.head_rot = Quat::from_rotation_y(0.7 * twist)
        * Quat::from_rotation_z(-0.8 * body.turn + 0.5 * drop)
        * Quat::from_rotation_x(0.55 * fwd - 0.05 + 0.8 * body.surge + 0.6 * throw);

    // Limbs: attached target blended with the authored free pose by the release weight.
    let hover_lift = Vec3::new(0.0, 0.15, 0.0);
    let post = r.rear_pt(r[P::PostTop]);
    for (i, s) in SIDES.into_iter().enumerate() {
        let hj = hip + x * (HIP_HALF * s);
        let free_ankle = {
            let (mut acc, mut sum) = (Vec3::ZERO, 0.0);
            for k in 0..LEG_KINDS {
                let w = a.leg_w[k];
                if w > 0.0 {
                    acc += leg_free(k, s, a.side, hip, hj, &sup) * w;
                    sum += w;
                }
            }
            if sum > 1e-3 {
                acc / sum
            } else {
                leg_free(L_DANGLE, s, a.side, hip, hj, &sup)
            }
        };
        let ankle_t = ankle_att[i].lerp(free_ankle, a.foot_rel[i]);
        let inside = (-s * body.turn / 0.4).clamp(0.0, 1.0);
        // Superman legs fold with the knees down, under the hip-ankle line.
        let knee_pole = Vec3::new(
            s * (0.3 + 0.9 * inside + 0.3 * body.crouch.max(0.0)),
            0.2,
            -1.0,
        )
        .lerp(
            Vec3::new(0.2 * s, -1.0, -0.3),
            a.leg_w[L_SUPER] * a.foot_rel[i],
        );
        let (knee, ankle) = limb(hj, ankle_t, THIGH, SHIN, knee_pole);
        let shin = (ankle - knee).try_normalize().unwrap_or(Vec3::NEG_Y);
        let foot_q = foot_att[i].slerp(Quat::from_rotation_arc(Vec3::NEG_Y, shin), a.foot_rel[i]);
        r[side(P::HipL, P::HipR, s)] = hj;
        r[side(P::KneeL, P::KneeR, s)] = knee;
        r[side(P::AnkleL, P::AnkleR, s)] = ankle;
        r[side(P::HeelL, P::HeelR, s)] = ankle + foot_q * (HEEL_OFF - ANKLE_OFF);
        r[side(P::ToeL, P::ToeR, s)] = ankle + foot_q * (TOE_OFF - ANKLE_OFF);

        let sh = shoulder + sq * (x * (SHOULDER_HALF * s));
        r[side(P::ShoulderL, P::ShoulderR, s)] = sh;
        let hover = grip_calm[i] + hover_lift + fq * WRIST_OFF;
        let tire = r.front_pt(st(hub_f + Vec3::new(0.075 * s, 0.287, 0.20))) + fq * WRIST_OFF;
        let seat = post + rq * Vec3::new(0.09 * s, 0.04, 0.08) + fq * WRIST_OFF;
        let free_wrist = {
            let (mut acc, mut sum) = (Vec3::ZERO, 0.0);
            for k in 0..HAND_KINDS {
                let w = a.hand_w[k];
                if w > 0.0 {
                    acc += arm_free(k, s, sh, hover, tire, seat) * w;
                    sum += w;
                }
            }
            if sum > 1e-3 {
                acc / sum
            } else {
                arm_free(H_NO, s, sh, hover, tire, seat)
            }
        };
        let wrist_t = wrist_att[i].lerp(free_wrist, a.hand_rel[i]);
        let inside = (-s * body.turn / 0.4).clamp(0.0, 1.0);
        let elbow_pole = Vec3::new(
            s * (0.55
                + 0.6 * body.crouch.clamp(0.0, 1.0)
                + 0.6 * inside
                + 0.5 * throw.max(0.0)
                + 0.3 * tuck),
            0.25,
            0.55,
        );
        let (elbow, wrist) = limb(sh, wrist_t, UPPER_ARM, FOREARM, elbow_pole);
        r[side(P::ElbowL, P::ElbowR, s)] = elbow;
        r[side(P::WristL, P::WristR, s)] = wrist;
        let hand_q = fq.slerp(Quat::IDENTITY, a.hand_rel[i]);
        r[side(P::HandL, P::HandR, s)] = wrist - hand_q * WRIST_OFF;
    }
    r
}

pub(crate) fn rider_points(b: &Bike, a: &AnimationState) -> [Vec3; COUNT] {
    let r = solve(b, a, false);
    std::array::from_fn(|i| r.p[P::Hip as usize + i])
}

/// Collision proxies in bike-local space as `(centre, radius, rider)`, in this order: pelvis, waist,
/// shoulders, head, left/right knee, left/right hand (all rider), saddle, bottom bracket, head
/// tube, bars (bike). Rider centres are the solved joints; bike centres go through the same
/// `frame`/`rear` transforms as the drawn tubes, so a tailwhip or table moves them exactly like
/// the meshes.
fn body_proxies(r: &Rig) -> [(Vec3, f32, bool); 12] {
    [
        (r[P::Hip], 0.13, true),
        (r[P::Waist], 0.14, true),
        (r[P::Shoulder], 0.14, true),
        (r[P::Head], 0.12, true),
        (r[P::KneeL], 0.09, true),
        (r[P::KneeR], 0.09, true),
        (r[P::HandL], 0.07, true),
        (r[P::HandR], 0.07, true),
        (r.rear_pt(r[P::PostTop]), 0.08, false),
        (r.rear_pt(r[P::Bb]), 0.10, false),
        (r.front_pt(r[P::HeadTop]), 0.08, false),
        (r.front_pt(r[P::BarC]), 0.08, false),
    ]
}

/// The pose the physics collides with: the solved rig, not a separate model. Wheel hubs are the
/// rest (zero strut compression) positions of the actual front/rear assembly, so tables, whips,
/// inverts and tailwhips move them; each axle is the wheel's local X pushed through the same
/// transforms the wheel mesh uses (frame, steering/bar spin for the front, rear triangle for the rear).
pub(crate) fn collision_pose(b: &Bike, a: &AnimationState) -> CollisionPose {
    let r = solve(b, a, false);
    let mut rest = b.clone();
    rest.suspension = [0.0; 2];
    let z = solve(&rest, a, false);
    CollisionPose {
        wheel_rest: [z.front_pt(z[P::FrontHub]), z.rear_pt(z[P::RearHub])],
        wheel_axes: [
            (r.frame.rotation * r.steer * Vec3::X).normalize(),
            (r.frame.rotation * r.rear.rotation * Vec3::X).normalize(),
        ],
        bodies: body_proxies(&r).map(|(offset, radius, rider)| CollisionSphere {
            offset,
            radius,
            rider,
        }),
        hand_release: a.hand_rel,
        foot_release: a.foot_rel,
    }
}

// ---------------------------------------------------------------------------------------------
// Per-frame animation
// ---------------------------------------------------------------------------------------------

/// Debug view switches (F1 / F2 in `game`).
#[derive(Resource)]
pub(crate) struct SkeletonDebug {
    /// Draw the solved rig as colored bones and joints, over all geometry.
    pub enabled: bool,
    /// Show the rider's body meshes (the bike is always shown).
    pub rider_mesh: bool,
}

impl Default for SkeletonDebug {
    fn default() -> Self {
        Self {
            // Web showcase (phones have no F1): show the rider, not the debug overlay.
            enabled: !cfg!(target_arch = "wasm32"),
            rider_mesh: true,
        }
    }
}

/// Root of every rider mesh; its `Visibility` hides the whole rider (helmet, goggles, shoes included).
#[derive(Component)]
pub(crate) struct RiderMesh;

type Bones = &'static [(P, P)];

const SPINE_BONES: Bones = &[
    (P::Hip, P::Waist),
    (P::Waist, P::Shoulder),
    (P::Shoulder, P::Neck),
    (P::Neck, P::Head),
];
const LEFT_BONES: Bones = &[
    (P::Hip, P::HipL),
    (P::HipL, P::KneeL),
    (P::KneeL, P::AnkleL),
    (P::AnkleL, P::HeelL),
    (P::AnkleL, P::ToeL),
    (P::Shoulder, P::ShoulderL),
    (P::ShoulderL, P::ElbowL),
    (P::ElbowL, P::WristL),
    (P::WristL, P::HandL),
];
const RIGHT_BONES: Bones = &[
    (P::Hip, P::HipR),
    (P::HipR, P::KneeR),
    (P::KneeR, P::AnkleR),
    (P::AnkleR, P::HeelR),
    (P::AnkleR, P::ToeR),
    (P::Shoulder, P::ShoulderR),
    (P::ShoulderR, P::ElbowR),
    (P::ElbowR, P::WristR),
    (P::WristR, P::HandR),
];
/// Bone sets and their sRGB colors: spine/head lime, left cyan, right magenta.
pub(crate) const SKELETON: [(Bones, [f32; 3]); 3] = [
    (SPINE_BONES, [0.55, 1.0, 0.1]),
    (LEFT_BONES, [0.1, 0.85, 1.0]),
    (RIGHT_BONES, [1.0, 0.2, 0.75]),
];
// Darker hues keep the isolated rig legible against the white floor.
const SKELETON_ISOLATED: [[f32; 3]; 3] =
    [[0.06, 0.34, 0.12], [0.05, 0.27, 0.70], [0.65, 0.06, 0.34]];
const JOINT_RADIUS: f32 = 0.03;
const HEAD_RADIUS: f32 = 0.09;

/// Crashed rider: jersey and bones turn red.
const CRASH_JERSEY: [f32; 3] = [0.85, 0.04, 0.04];
const CRASH_BONE: [f32; 3] = [1.0, 0.1, 0.1];
const CRASH_BONE_ISOLATED: [f32; 3] = [0.75, 0.0, 0.0];

/// Bones and joints from the same `Rig` that drives the meshes, in world space.
fn draw_skeleton(g: &mut Gizmos, origin: Vec3, q: Quat, r: &Rig, over_mesh: bool, crashed: bool) {
    let world = |p: P| origin + q * r[p];
    let tint = |overlay: [f32; 3], isolated: [f32; 3]| {
        let [cr, cg, cb] = match (crashed, over_mesh) {
            (true, true) => CRASH_BONE,
            (true, false) => CRASH_BONE_ISOLATED,
            (false, true) => overlay,
            (false, false) => isolated,
        };
        Color::srgb(cr, cg, cb)
    };
    g.sphere(
        world(P::Hip),
        JOINT_RADIUS * 1.5,
        tint(SKELETON[0].1, SKELETON_ISOLATED[0]),
    )
    .resolution(12);
    for ((bones, overlay), isolated) in SKELETON.iter().zip(SKELETON_ISOLATED) {
        let color = tint(*overlay, isolated);
        for &(a, b) in *bones {
            g.line(world(a), world(b), color);
            let radius = if matches!(b, P::Head) {
                HEAD_RADIUS
            } else {
                JOINT_RADIUS
            };
            g.sphere(world(b), radius, color).resolution(12);
        }
    }
}

/// Every animated entity carries one of these; `animate_bike` updates all in a single query.
#[derive(Component, Clone, Copy)]
pub(crate) enum Part {
    Root,
    /// Unit mesh (extent 1, long axis Y) stretched from `a` to `b`; `dx`,`dz` are cross-section sizes.
    Link {
        a: P,
        b: P,
        dx: f32,
        dz: f32,
    },
    /// Translation-only follower.
    At(P),
    Head,
    Wheel {
        front: bool,
    },
    Crank,
    /// Bike assembly node: whip/table/turndown/invert and the rigid road carry.
    Frame,
    /// Rear triangle node under `Frame`: spins about the steering axis for a tailwhip.
    Rear,
    /// Tire mesh: its axle-direction scale is the discipline tire width.
    Tire,
}

/// Materials recoloured per discipline.
#[derive(Resource)]
pub(crate) struct BikePaint {
    frame: Handle<StandardMaterial>,
    jersey: Handle<StandardMaterial>,
}

/// Parts that only some disciplines show.
#[derive(Component, Clone, Copy)]
pub(crate) enum Gear {
    Shock,
    Knobs,
    Stanchion,
}

fn tube_tf(a: Vec3, b: Vec3, dx: f32, dz: f32) -> Transform {
    let d = b - a;
    let len = d.length().max(1e-5);
    Transform {
        translation: (a + b) * 0.5,
        rotation: Quat::from_rotation_arc(Vec3::Y, d.try_normalize().unwrap_or(Vec3::Y)),
        scale: Vec3::new(dx, len, dz),
    }
}

/// What the display draws for one fixed tick: the solved rig (the ragdoll once crashed) plus the
/// bike root and wheel angle.
#[derive(Clone, Copy)]
struct Frame {
    rig: Rig,
    position: Vec3,
    root: Quat,
    yaw: f32,
    wheel: f32,
    discipline: Discipline,
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    a + ((b - a + PI).rem_euclid(TAU) - PI) * t
}

fn lerp_transform(a: &Transform, b: &Transform, t: f32) -> Transform {
    Transform {
        translation: a.translation.lerp(b.translation, t),
        rotation: a.rotation.slerp(b.rotation, t),
        scale: Vec3::ONE,
    }
}

impl Frame {
    fn new(bike: &Bike, anim: &AnimationState, ragdoll: &Ragdoll) -> Self {
        let mut rig = rig(bike, anim);
        let root = bike.orientation();
        if let Some(body) = &ragdoll.body {
            for i in 0..COUNT {
                rig.p[P::Hip as usize + i] = root.inverse() * (body.positions[i] - bike.position);
            }
            let up = (rig[P::Head] - rig[P::Neck]).normalize_or_zero();
            let right = (rig[P::ShoulderR] - rig[P::ShoulderL]).normalize_or_zero();
            let back = right.cross(up).normalize_or_zero();
            let up = back.cross(right).normalize_or_zero();
            if right.length_squared() > 0.9 && up.length_squared() > 0.9 {
                rig.head_rot = Quat::from_mat3(&Mat3::from_cols(right, up, back));
            }
        }
        Self {
            rig,
            position: bike.position,
            root,
            yaw: bike.yaw,
            wheel: bike.wheel_phase,
            discipline: bike.discipline,
        }
    }

    /// Joints and positions lerp, rotations slerp, angle accumulators take the short way round.
    fn lerp(&self, next: &Self, t: f32) -> Self {
        let (a, b) = (&self.rig, &next.rig);
        let mut rig = *b;
        for i in 0..rig.p.len() {
            rig.p[i] = a.p[i].lerp(b.p[i], t);
        }
        rig.steer = a.steer.slerp(b.steer, t);
        rig.head_rot = a.head_rot.slerp(b.head_rot, t);
        rig.frame = lerp_transform(&a.frame, &b.frame, t);
        rig.rear = lerp_transform(&a.rear, &b.rear, t);
        rig.crank = lerp_angle(a.crank, b.crank, t);
        Self {
            rig,
            position: self.position.lerp(next.position, t),
            root: self.root.slerp(next.root, t),
            yaw: lerp_angle(self.yaw, next.yaw, t),
            wheel: lerp_angle(self.wheel, next.wheel, t),
            discipline: next.discipline,
        }
    }

    fn world_hip(&self) -> Vec3 {
        self.position + self.root * self.rig[P::Hip]
    }
}

/// The drawn bike for the last two fixed ticks; display frames blend between them (the bike
/// is drawn up to one tick late but never at a raw 120 Hz state). Also what the camera follows.
#[derive(Resource, Default)]
pub(crate) struct BikeFrames {
    previous: Option<Frame>,
    current: Option<Frame>,
}

/// Further than this in one tick is a reset or teleport: nothing blends across it.
const TELEPORT: f32 = 2.0;

impl BikeFrames {
    fn push(&mut self, frame: Frame) {
        let continuous = self.current.as_ref().is_some_and(|c| {
            c.discipline == frame.discipline && c.position.distance(frame.position) < TELEPORT
        });
        self.previous = if continuous { self.current } else { None };
        self.current = Some(frame);
    }

    fn pair(&self) -> Option<(&Frame, &Frame)> {
        let c = self.current.as_ref()?;
        Some((self.previous.as_ref().unwrap_or(c), c))
    }

    /// The frame to draw at `alpha` (`Time<Fixed>::overstep_fraction`); `fallback` before any tick.
    fn blend(&self, alpha: f32, fallback: impl FnOnce() -> Frame) -> Frame {
        self.pair().map_or_else(fallback, |(p, c)| p.lerp(c, alpha))
    }

    /// Interpolated bike position for the camera.
    pub(crate) fn root(&self, alpha: f32) -> Option<Vec3> {
        self.pair().map(|(p, c)| p.position.lerp(c.position, alpha))
    }

    /// Interpolated heading for the camera (radians, wrapped).
    pub(crate) fn yaw(&self, alpha: f32) -> Option<f32> {
        self.pair().map(|(p, c)| lerp_angle(p.yaw, c.yaw, alpha))
    }

    /// Interpolated world pelvis of the drawn rider (the ragdoll's once crashed).
    pub(crate) fn hip(&self, alpha: f32) -> Option<Vec3> {
        self.pair()
            .map(|(p, c)| p.world_hip().lerp(c.world_hip(), alpha))
    }
}

/// Fixed-step system: store the drawn frame of this tick (runs after `game::simulate`).
pub fn record_frame(
    bike: Res<Bike>,
    anim: Res<AnimationState>,
    ragdoll: Res<Ragdoll>,
    mut frames: ResMut<BikeFrames>,
) {
    frames.push(Frame::new(&bike, &anim, &ragdoll));
}

pub fn animate_bike(
    bike: Res<Bike>,
    anim: Res<AnimationState>,
    ragdoll: Res<Ragdoll>,
    frames: Res<BikeFrames>,
    fixed: Res<Time<Fixed>>,
    debug: Res<SkeletonDebug>,
    paint: Res<BikePaint>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut dressed: Local<Option<(Discipline, bool)>>,
    mut parts: Query<(&Part, &mut Transform)>,
    mut rider: Query<&mut Visibility, With<RiderMesh>>,
    mut gear: Query<(&Gear, &mut Visibility), Without<RiderMesh>>,
    mut gizmos: Gizmos,
) {
    // The animation is advanced by the fixed physics step (and frozen by the game when paused or
    // crashed); this only draws it, with the physical root orientation.
    let look = Look::of(bike.discipline);
    let crashed = bike.crash.is_some();
    if *dressed != Some((bike.discipline, crashed)) {
        *dressed = Some((bike.discipline, crashed));
        let jersey = if crashed { CRASH_JERSEY } else { look.jersey };
        for (handle, [r, g, b]) in [(&paint.frame, look.frame), (&paint.jersey, jersey)] {
            if let Some(m) = materials.get_mut(handle) {
                m.base_color = Color::srgb(r, g, b);
            }
        }
        for (g, mut v) in &mut gear {
            let on = match *g {
                Gear::Shock => look.shock,
                Gear::Knobs => look.knobs,
                Gear::Stanchion => !look.rigid,
            };
            *v = if on {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
    }
    let frame = frames.blend(fixed.overstep_fraction(), || {
        Frame::new(&bike, &anim, &ragdoll)
    });
    let rig = frame.rig;
    let tire_width = bike.discipline.profile().tire_width;
    for (part, mut tf) in &mut parts {
        match *part {
            Part::Root => {
                tf.translation = frame.position;
                tf.rotation = frame.root;
            }
            Part::Frame => *tf = rig.frame,
            Part::Rear => *tf = rig.rear,
            Part::Tire => tf.scale = Vec3::new(1.0, tire_width, 1.0),
            Part::Link { a, b, dx, dz } => *tf = tube_tf(rig[a], rig[b], dx, dz),
            Part::At(p) => tf.translation = rig[p],
            Part::Head => {
                tf.translation = rig[P::Head];
                tf.rotation = rig.head_rot;
            }
            Part::Wheel { front } => {
                // Wheels roll about local X; -phase makes the top move forward (-Z).
                let spin = Quat::from_rotation_x(-frame.wheel);
                if front {
                    tf.translation = rig[P::FrontHub];
                    tf.rotation = rig.steer * spin;
                } else {
                    tf.translation = rig[P::RearHub];
                    tf.rotation = spin;
                }
            }
            Part::Crank => {
                tf.translation = BB;
                tf.rotation = Quat::from_rotation_x(-rig.crank);
            }
        }
    }
    let shown = if debug.rider_mesh {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut rider {
        v.set_if_neq(shown);
    }
    if debug.enabled {
        draw_skeleton(
            &mut gizmos,
            frame.position,
            frame.root,
            &rig,
            debug.rider_mesh,
            crashed,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------------------------

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gizmo_config: ResMut<GizmoConfigStore>,
) {
    // Skeleton lines always draw over the (much thicker) body geometry.
    let (cfg, _) = gizmo_config.config_mut::<DefaultGizmoConfigGroup>();
    cfg.depth_bias = -1.0;
    cfg.line.width = 3.0;

    // Neutral white studio: pale clear color, white ambient, one shadow-casting sun.
    let sun = Vec3::new(-0.55, 0.72, 0.30).normalize(); // direction towards the sun
    commands.insert_resource(ClearColor(Color::srgb(0.93, 0.93, 0.94)));
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 350.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            minimum_distance: 0.1,
            maximum_distance: 140.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
        .build(),
        Transform::default().looking_to(-sun, Vec3::Y),
    ));

    let floor = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.9),
        perceptual_roughness: 1.0,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(floor_mesh())),
        MeshMaterial3d(floor),
        Transform::default(),
    ));
    spawn_bike(&mut commands, &mut meshes, &mut materials);
}

fn mat(
    m: &mut Assets<StandardMaterial>,
    c: Color,
    rough: f32,
    metal: f32,
) -> Handle<StandardMaterial> {
    m.add(StandardMaterial {
        base_color: c,
        perceptual_roughness: rough,
        metallic: metal,
        ..default()
    })
}

/// `terrain_height` is exactly zero outside these bounds: the jump lane around x = 0 and the hill.
const FLOOR_X: (f32, f32) = (-4.0, 108.0);
const FLOOR_Z: (f32, f32) = (-150.0, 65.0);
const FLOOR_STEP: f32 = 0.5;
/// Fine sampling for the narrow jump lane and for z up to the end of the three jumps (and the
/// hill's kicker); the broad hill is smooth enough for `FLOOR_STEP`.
const FLOOR_FINE: f32 = 0.125;
const JUMP_LANE_X: f32 = 4.0;
const FINE_Z_END: f32 = -20.0;
const FLOOR_FAR: f32 = 1000.0;

/// Grid lines: a far cell, the samples `lo..=hi` (`FLOOR_FINE` apart inside `fine`, `FLOOR_STEP`
/// elsewhere; every bound is a multiple of the fine step so the walk is exact), a far cell.
fn floor_axis(lo: f32, hi: f32, fine: (f32, f32)) -> Vec<f32> {
    let mut v = vec![-FLOOR_FAR];
    let mut t = lo;
    while t < hi - 1e-4 {
        v.push(t);
        t += if t >= fine.0 - 1e-4 && t < fine.1 - 1e-4 {
            FLOOR_FINE
        } else {
            FLOOR_STEP
        };
    }
    v.push(hi);
    v.push(FLOOR_FAR);
    v
}

fn floor_axes() -> (Vec<f32>, Vec<f32>) {
    (
        floor_axis(FLOOR_X.0, FLOOR_X.1, (-JUMP_LANE_X, JUMP_LANE_X)),
        floor_axis(FLOOR_Z.0, FLOOR_Z.1, (FLOOR_Z.0, FINE_Z_END)),
    )
}

/// White floor: `terrain_height` sampled on the grid above, with huge outer cells (flat there) so
/// the plane reaches the horizon.
fn floor_mesh() -> Mesh {
    let (xs, zs) = floor_axes();
    let (nx, nz) = (xs.len(), zs.len());
    let h = |i: usize, j: usize| terrain_height(xs[i], zs[j]);
    let mut pos = Vec::with_capacity(nx * nz);
    let mut nor = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            let (i0, i1) = (i.saturating_sub(1), (i + 1).min(nx - 1));
            let (j0, j1) = (j.saturating_sub(1), (j + 1).min(nz - 1));
            let dx = (h(i1, j) - h(i0, j)) / (xs[i1] - xs[i0]);
            let dz = (h(i, j1) - h(i, j0)) / (zs[j1] - zs[j0]);
            pos.push([xs[i], h(i, j), zs[j]]);
            nor.push(Vec3::new(-dx, 1.0, -dz).normalize().to_array());
        }
    }
    let mut idx = Vec::with_capacity((nx - 1) * (nz - 1) * 6);
    for j in 0..nz - 1 {
        for i in 0..nx - 1 {
            let a = (j * nx + i) as u32;
            let (b, c) = (a + 1, a + nx as u32);
            idx.extend([a, c, b, b, c, c + 1]); // CCW seen from +Y
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
    .with_inserted_indices(Indices::U32(idx))
}

fn cyl(dia: f32, h: f32, res: u32) -> Mesh {
    Cylinder::new(dia * 0.5, h).mesh().resolution(res).build()
}

fn cube(x: f32, y: f32, z: f32) -> Mesh {
    Cuboid::new(x, y, z).into()
}

// ---------------------------------------------------------------------------------------------
// Bike + rider construction
// ---------------------------------------------------------------------------------------------

struct Kit {
    cube: Handle<Mesh>,
    cyl: Handle<Mesh>,
    ball: Handle<Mesh>,
    coral: Handle<StandardMaterial>,
    black: Handle<StandardMaterial>,
    steel: Handle<StandardMaterial>,
    silver: Handle<StandardMaterial>,
    rubber: Handle<StandardMaterial>,
    knob: Handle<StandardMaterial>,
    white: Handle<StandardMaterial>,
    amber: Handle<StandardMaterial>,
    jersey: Handle<StandardMaterial>,
    shorts: Handle<StandardMaterial>,
    skin: Handle<StandardMaterial>,
    helmet: Handle<StandardMaterial>,
    shoe: Handle<StandardMaterial>,
    lens: Handle<StandardMaterial>,
}

struct Build<'a, 'w, 's> {
    c: &'a mut Commands<'w, 's>,
    parent: Entity,
}

impl Build<'_, '_, '_> {
    fn spawn(
        &mut self,
        mesh: &Handle<Mesh>,
        m: &Handle<StandardMaterial>,
        tf: Transform,
        part: Option<Part>,
    ) -> Entity {
        let mut e = self.c.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(m.clone()),
            tf,
            ChildOf(self.parent),
        ));
        if let Some(p) = part {
            e.insert(p);
        }
        e.id()
    }

    /// Fixed child.
    fn solid(
        &mut self,
        mesh: &Handle<Mesh>,
        m: &Handle<StandardMaterial>,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
    ) -> Entity {
        self.spawn(
            mesh,
            m,
            Transform {
                translation: pos,
                rotation: rot,
                scale,
            },
            None,
        )
    }

    /// Rig-driven tube between two points.
    fn link(
        &mut self,
        mesh: &Handle<Mesh>,
        m: &Handle<StandardMaterial>,
        a: P,
        b: P,
        dx: f32,
        dz: f32,
    ) -> Entity {
        self.spawn(
            mesh,
            m,
            Transform::default(),
            Some(Part::Link { a, b, dx, dz }),
        )
    }

    /// Rig-driven blob at a point.
    fn at(
        &mut self,
        mesh: &Handle<Mesh>,
        m: &Handle<StandardMaterial>,
        p: P,
        scale: Vec3,
    ) -> Entity {
        self.spawn(mesh, m, Transform::from_scale(scale), Some(Part::At(p)))
    }

    /// Mark a part as shown by some disciplines only.
    fn gear(&mut self, e: Entity, g: Gear) {
        self.c.entity(e).insert(g);
    }

    /// Empty parent entity (wheel, crank, head) whose children share its animated transform.
    fn node(&mut self, part: Part) -> Entity {
        self.c
            .spawn((
                part,
                Transform::default(),
                Visibility::default(),
                ChildOf(self.parent),
            ))
            .id()
    }
}

fn spawn_bike(c: &mut Commands, meshes: &mut Assets<Mesh>, mats: &mut Assets<StandardMaterial>) {
    let k = Kit {
        cube: meshes.add(cube(1.0, 1.0, 1.0)),
        cyl: meshes.add(cyl(1.0, 1.0, 14)),
        ball: meshes.add(Sphere::new(0.5).mesh().uv(18, 12)),
        coral: mat(mats, Color::srgb(1.0, 0.34, 0.28), 0.35, 0.1),
        black: mat(mats, Color::srgb(0.04, 0.04, 0.05), 0.5, 0.3),
        steel: mat(mats, Color::srgb(0.30, 0.31, 0.33), 0.4, 0.5),
        silver: mat(mats, Color::srgb(0.78, 0.80, 0.84), 0.3, 0.6),
        rubber: mat(mats, Color::srgb(0.03, 0.03, 0.035), 0.95, 0.0),
        knob: mat(mats, Color::srgb(0.08, 0.08, 0.085), 0.95, 0.0),
        white: mat(mats, Color::srgb(0.92, 0.92, 0.90), 0.6, 0.0),
        amber: mat(mats, Color::srgb(1.0, 0.72, 0.15), 0.4, 0.3),
        jersey: mat(mats, Color::srgb(0.10, 0.16, 0.28), 0.8, 0.0),
        shorts: mat(mats, Color::srgb(0.04, 0.045, 0.06), 0.85, 0.0),
        skin: mat(mats, Color::srgb(0.76, 0.57, 0.44), 0.7, 0.0),
        helmet: mat(mats, Color::srgb(0.07, 0.08, 0.10), 0.35, 0.2),
        shoe: mat(mats, Color::srgb(0.82, 0.84, 0.88), 0.7, 0.0),
        lens: mat(mats, Color::srgb(0.02, 0.03, 0.05), 0.1, 0.6),
    };
    c.insert_resource(BikePaint {
        frame: k.coral.clone(),
        jersey: k.jersey.clone(),
    });
    // Tire: 5.5 cm tube whose outer radius is the contract WHEEL_RADIUS; rim sits inside it.
    let tire = meshes.add(
        Torus::new(WHEEL_RADIUS - 0.11, WHEEL_RADIUS)
            .mesh()
            .major_resolution(48)
            .minor_resolution(14)
            .build(),
    );
    let rim = meshes.add(
        Torus::new(0.234, 0.29)
            .mesh()
            .major_resolution(48)
            .minor_resolution(10)
            .build(),
    );
    let ring = meshes.add(
        Torus::new(0.09, 0.105)
            .mesh()
            .major_resolution(32)
            .minor_resolution(6)
            .build(),
    );

    // Hierarchy: root -> frame -> (front group, rear node -> rear group); the rider hangs off root.
    // Front group = fork, bars, front wheel; rear group = frame triangle, swingarm, drivetrain,
    // saddle, rear wheel. A tailwhip rotates only the rear node about the steering axis.
    let root = c
        .spawn((Part::Root, Transform::default(), Visibility::default()))
        .id();
    let mut b = Build { c, parent: root };
    let frame = b.node(Part::Frame);
    b.parent = frame;
    let rear = b.node(Part::Rear);
    let (cyl, cube, ball) = (&k.cyl, &k.cube, &k.ball);
    let rz = Quat::from_rotation_z(FRAC_PI_2);

    // Front group: head tube, steerer and stem.
    b.link(cyl, &k.coral, P::HeadBot, P::HeadTop, 0.058, 0.058);
    b.link(cyl, &k.black, P::Crown, P::SteerTop, 0.030, 0.030);
    b.link(cyl, &k.steel, P::SteerTop, P::BarC, 0.040, 0.040);

    // Suspension fork: stanchions slide into the lower legs as the front compresses.
    b.link(cyl, &k.black, P::CrownL, P::CrownR, 0.036, 0.036);
    for (crown, drop, leg_top, stan_bot) in [
        (P::CrownL, P::DropFL, P::LegTopL, P::StanBotL),
        (P::CrownR, P::DropFR, P::LegTopR, P::StanBotR),
    ] {
        b.link(cyl, &k.black, drop, leg_top, 0.044, 0.044);
        let stan = b.link(cyl, &k.silver, crown, stan_bot, 0.030, 0.030);
        b.gear(stan, Gear::Stanchion);
        b.at(ball, &k.steel, drop, Vec3::splat(0.038));
    }

    // Handlebar, grips and brake levers.
    b.link(cyl, &k.silver, P::BarInL, P::BarInR, 0.022, 0.022);
    for (bar_in, bar_out, grip_in, lever_root, lever_tip) in [
        (
            P::BarInL,
            P::BarOutL,
            P::GripInL,
            P::LeverRootL,
            P::LeverTipL,
        ),
        (
            P::BarInR,
            P::BarOutR,
            P::GripInR,
            P::LeverRootR,
            P::LeverTipR,
        ),
    ] {
        b.link(cyl, &k.silver, bar_in, bar_out, 0.022, 0.022);
        b.link(cyl, &k.black, grip_in, bar_out, 0.036, 0.036);
        b.link(cyl, &k.silver, lever_root, lever_tip, 0.011, 0.011);
    }
    spawn_wheel(&mut b, &k, &tire, &rim, true);

    // Rear group. Main triangle, seat post.
    b.parent = rear;
    b.link(cyl, &k.coral, P::HeadTop, P::SeatTop, 0.044, 0.044);
    b.link(cyl, &k.coral, P::HeadBot, P::Bb, 0.054, 0.054);
    b.link(cyl, &k.coral, P::Bb, P::SeatTop, 0.040, 0.040);
    b.link(cyl, &k.silver, P::SeatTop, P::PostTop, 0.027, 0.027);
    b.solid(cyl, &k.steel, BB, rz, Vec3::new(0.056, 0.09, 0.056));

    // Rear swingarm: bowed chain/seat stays, rocker and coil shock (hidden on the rigid road bike).
    b.link(cyl, &k.coral, P::CsFrontL, P::CsFrontR, 0.028, 0.028);
    for (cs_f, cs_m, drop, ss_top, ss_bow) in [
        (P::CsFrontL, P::CsMidL, P::DropL, P::SsTopL, P::SsBowL),
        (P::CsFrontR, P::CsMidR, P::DropR, P::SsTopR, P::SsBowR),
    ] {
        b.link(cyl, &k.coral, cs_f, cs_m, 0.028, 0.028);
        b.link(cyl, &k.coral, cs_m, drop, 0.026, 0.026);
        b.link(cyl, &k.coral, ss_top, ss_bow, 0.022, 0.022);
        b.link(cyl, &k.coral, ss_bow, drop, 0.022, 0.022);
        let arm = b.link(cyl, &k.black, P::Rocker, ss_bow, 0.020, 0.020);
        b.gear(arm, Gear::Shock);
        b.at(ball, &k.steel, drop, Vec3::splat(0.036));
    }
    let shock_top = b.link(cyl, &k.black, P::ShockTop, P::ShockMid, 0.034, 0.034);
    b.gear(shock_top, Gear::Shock);
    let shock_coil = b.link(cyl, &k.amber, P::ShockMid, P::Rocker, 0.044, 0.044);
    b.gear(shock_coil, Gear::Shock);
    let rocker = b.at(ball, &k.steel, P::Rocker, Vec3::splat(0.04));
    b.gear(rocker, Gear::Shock);

    // Drivetrain: crank arms, pedals, chain and derailleur (chainring itself spins with the crank).
    for (hub, end, pedal) in [
        (P::CrankHubL, P::CrankEndL, P::PedalL),
        (P::CrankHubR, P::CrankEndR, P::PedalR),
    ] {
        b.link(cyl, &k.silver, hub, end, 0.030, 0.030);
        b.link(cyl, &k.steel, end, pedal, 0.014, 0.014);
        b.at(cube, &k.black, pedal, Vec3::new(0.10, 0.022, 0.11));
    }
    b.link(
        cyl,
        &k.steel,
        P::ChainFrontTop,
        P::ChainRearTop,
        0.012,
        0.012,
    );
    b.link(
        cyl,
        &k.steel,
        P::ChainFrontBot,
        P::ChainRearBot,
        0.012,
        0.012,
    );
    b.link(cyl, &k.black, P::DerailA, P::DerailB, 0.018, 0.018);
    b.at(ball, &k.steel, P::DerailA, Vec3::splat(0.036));
    b.at(ball, &k.steel, P::DerailB, Vec3::splat(0.036));

    // Saddle rides the post top, whose height depends on the discipline.
    let saddle = b.node(Part::At(P::PostTop));
    let mut sb = Build {
        c: &mut *b.c,
        parent: saddle,
    };
    sb.solid(
        cube,
        &k.black,
        Vec3::new(0.0, 0.038, 0.045),
        Quat::IDENTITY,
        Vec3::new(0.13, 0.04, 0.14),
    );
    sb.solid(
        cube,
        &k.black,
        Vec3::new(0.0, 0.036, -0.075),
        Quat::IDENTITY,
        Vec3::new(0.06, 0.034, 0.14),
    );

    // Chainring on the crank axle.
    let crank = b.node(Part::Crank);
    let mut cb = Build {
        c: &mut *b.c,
        parent: crank,
    };
    cb.solid(&ring, &k.steel, Vec3::new(0.07, 0.0, 0.0), rz, Vec3::ONE);
    cb.solid(
        cyl,
        &k.black,
        Vec3::new(0.066, 0.0, 0.0),
        rz,
        Vec3::new(0.05, 0.02, 0.05),
    );
    for i in 0..5 {
        let a = Quat::from_rotation_x(i as f32 * TAU / 5.0);
        cb.solid(
            cube,
            &k.steel,
            a * Vec3::new(0.066, 0.05, 0.0),
            a,
            Vec3::new(0.006, 0.05, 0.016),
        );
    }

    spawn_wheel(&mut b, &k, &tire, &rim, false);
    b.parent = root;
    spawn_rider(&mut b, &k);
}

/// Tire, rim, hub, spokes, rotor (and cassette on the rear). Everything spins with the `Wheel` node.
fn spawn_wheel(b: &mut Build, k: &Kit, tire: &Handle<Mesh>, rim: &Handle<Mesh>, front: bool) {
    let node = b.node(Part::Wheel { front });
    let mut w = Build {
        c: &mut *b.c,
        parent: node,
    };
    let rz = Quat::from_rotation_z(FRAC_PI_2);
    w.spawn(
        tire,
        &k.rubber,
        Transform::from_rotation(rz),
        Some(Part::Tire),
    );
    w.solid(rim, &k.silver, Vec3::ZERO, rz, Vec3::ONE);
    w.solid(
        &k.cyl,
        &k.steel,
        Vec3::ZERO,
        rz,
        Vec3::new(0.045, 0.12, 0.045),
    );
    for s in SIDES {
        w.solid(
            &k.cyl,
            &k.steel,
            Vec3::X * (0.04 * s),
            rz,
            Vec3::new(0.08, 0.008, 0.08),
        );
        for i in 0..16 {
            // Tangentially laced: each side's spokes lean the opposite way, offset half a pitch.
            let a = (i as f32 + if s > 0.0 { 0.5 } else { 0.0 }) * TAU / 16.0;
            let ra = a + 0.5 * s;
            let hub_end = Vec3::new(0.04 * s, 0.035 * a.cos(), 0.035 * a.sin());
            let rim_end = Vec3::new(0.0, 0.25 * ra.cos(), 0.25 * ra.sin());
            w.spawn(
                &k.cyl,
                &k.silver,
                tube_tf(hub_end, rim_end, 0.004, 0.004),
                None,
            );
        }
        // Sidewall label: makes wheel rotation readable.
        w.solid(
            &k.cube,
            &k.white,
            Vec3::new(0.057 * s, 0.30, 0.0),
            Quat::IDENTITY,
            Vec3::new(0.006, 0.07, 0.022),
        );
    }
    w.solid(
        &k.cyl,
        &k.steel,
        Vec3::new(-0.036, 0.0, 0.0),
        rz,
        Vec3::new(0.16, 0.004, 0.16),
    );
    if !front {
        for (x, dia) in [(0.045, 0.11), (0.056, 0.085), (0.067, 0.06)] {
            w.solid(
                &k.cyl,
                &k.silver,
                Vec3::new(x, 0.0, 0.0),
                rz,
                Vec3::new(dia, 0.01, dia),
            );
        }
    }
    for i in 0..32 {
        let a = Quat::from_rotation_x(i as f32 * TAU / 32.0);
        let x = if i % 2 == 0 { 0.018 } else { -0.018 };
        let knob = w.solid(
            &k.cube,
            &k.knob,
            a * Vec3::new(x, WHEEL_RADIUS - 0.002, 0.0),
            a,
            Vec3::new(0.045, 0.014, 0.035),
        );
        w.gear(knob, Gear::Knobs);
    }
}

fn spawn_rider(root: &mut Build, k: &Kit) {
    let rider = root
        .c
        .spawn((
            RiderMesh,
            Transform::default(),
            Visibility::default(),
            ChildOf(root.parent),
        ))
        .id();
    let mut b = Build {
        c: &mut *root.c,
        parent: rider,
    };
    let (cyl, ball, cube) = (&k.cyl, &k.ball, &k.cube);
    b.at(ball, &k.shorts, P::Hip, Vec3::new(0.27, 0.17, 0.23));
    b.link(cyl, &k.jersey, P::Hip, P::Waist, 0.30, 0.20);
    b.link(cyl, &k.jersey, P::Waist, P::Shoulder, 0.34, 0.21);
    b.link(cyl, &k.jersey, P::ShoulderL, P::ShoulderR, 0.09, 0.09);
    b.link(cyl, &k.skin, P::Shoulder, P::Head, 0.075, 0.075);

    let limbs = [
        // hip, knee, ankle, heel, toe, shoulder, elbow, wrist, grip
        (
            P::HipL,
            P::KneeL,
            P::AnkleL,
            P::HeelL,
            P::ToeL,
            P::ShoulderL,
            P::ElbowL,
            P::WristL,
            P::HandL,
        ),
        (
            P::HipR,
            P::KneeR,
            P::AnkleR,
            P::HeelR,
            P::ToeR,
            P::ShoulderR,
            P::ElbowR,
            P::WristR,
            P::HandR,
        ),
    ];
    for (hip, knee, ankle, heel, toe, shoulder, elbow, wrist, grip) in limbs {
        b.at(ball, &k.shorts, hip, Vec3::splat(0.13));
        b.link(cyl, &k.shorts, hip, knee, 0.115, 0.115);
        b.at(ball, &k.shorts, knee, Vec3::splat(0.115));
        b.link(cyl, &k.shorts, knee, ankle, 0.088, 0.088);
        b.link(cube, &k.shoe, heel, toe, 0.09, 0.075);

        b.at(ball, &k.jersey, shoulder, Vec3::splat(0.105));
        b.link(cyl, &k.jersey, shoulder, elbow, 0.085, 0.085);
        b.at(ball, &k.jersey, elbow, Vec3::splat(0.08));
        b.link(cyl, &k.jersey, elbow, wrist, 0.068, 0.068);
        b.at(ball, &k.shorts, grip, Vec3::new(0.085, 0.06, 0.075)); // glove wrapped on the grip
    }

    // Head: skull, goggles, helmet with coral stripe and visor; tilts with the torso.
    let head = b.node(Part::Head);
    let mut h = Build {
        c: &mut *b.c,
        parent: head,
    };
    let id = Quat::IDENTITY;
    h.solid(ball, &k.skin, Vec3::ZERO, id, Vec3::new(0.165, 0.21, 0.20));
    h.solid(
        ball,
        &k.helmet,
        Vec3::new(0.0, 0.05, 0.02),
        id,
        Vec3::new(0.205, 0.17, 0.27),
    );
    h.solid(
        cube,
        &k.coral,
        Vec3::new(0.0, 0.134, 0.02),
        id,
        Vec3::new(0.03, 0.012, 0.27),
    );
    h.solid(
        cube,
        &k.helmet,
        Vec3::new(0.0, 0.072, -0.135),
        Quat::from_rotation_x(-0.2),
        Vec3::new(0.17, 0.012, 0.09),
    );
    h.solid(
        cube,
        &k.lens,
        Vec3::new(0.0, 0.018, -0.097),
        id,
        Vec3::new(0.165, 0.055, 0.04),
    );
    h.solid(
        cube,
        &k.black,
        Vec3::new(0.0, 0.018, 0.0),
        id,
        Vec3::new(0.19, 0.03, 0.2),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{B_INVERT, B_TABLE, Input};
    use crate::bike::{BikeTrick, Controls, HandTrick, LegTrick};

    const DT: f32 = 1.0 / 30.0;
    /// Long enough for the slowest return (Superman) to start after the mid-flight sample.
    const FLIGHT: f32 = 1.5;

    /// One solved frame: finite joints, exact bone lengths, limbs fully on the bike reach their
    /// grips/pedals (hands may cost the torso a few cm of stretch, hence the looser tolerance).
    fn check(b: &Bike, a: &AnimationState, tol_hand: f32, tol_foot: f32, tag: &str) {
        let r = rig(b, a);
        assert!(r.p.iter().all(|v| v.is_finite()), "{tag}: non-finite joint");
        assert!(
            r.frame.translation.is_finite()
                && r.frame.rotation.is_finite()
                && r.rear.rotation.is_finite()
                && r.steer.is_finite()
                && r.head_rot.is_finite(),
            "{tag}: non-finite transform"
        );
        let fq = r.frame.rotation;
        let rq = fq * r.rear.rotation;
        let len = |p: P, q: P| r[p].distance(r[q]);
        for (i, s) in SIDES.into_iter().enumerate() {
            let l = |l: P, rt: P| side(l, rt, s);
            let (hip, knee, ankle) = (
                l(P::HipL, P::HipR),
                l(P::KneeL, P::KneeR),
                l(P::AnkleL, P::AnkleR),
            );
            let (sh, el, wr) = (
                l(P::ShoulderL, P::ShoulderR),
                l(P::ElbowL, P::ElbowR),
                l(P::WristL, P::WristR),
            );
            assert!((len(hip, knee) - THIGH).abs() < 2e-3, "{tag}: thigh");
            assert!((len(knee, ankle) - SHIN).abs() < 2e-3, "{tag}: shin");
            assert!((len(sh, el) - UPPER_ARM).abs() < 2e-3, "{tag}: upper arm");
            assert!((len(el, wr) - FOREARM).abs() < 2e-3, "{tag}: forearm");
            if a.foot_rel[i] == 0.0 {
                let ank = ankling(r.crank, s, a.pedal);
                let want = r.rear_pt(r[l(P::PedalL, P::PedalR)])
                    + rq * Quat::from_rotation_x(ank) * ANKLE_OFF;
                let gap = r[ankle].distance(want);
                assert!(gap < tol_foot, "{tag}: foot off pedal by {gap}");
            }
            if a.hand_rel[i] == 0.0 {
                let want = r.front_pt(r[l(P::GripL, P::GripR)]) + fq * WRIST_OFF;
                let gap = r[wr].distance(want);
                assert!(gap < tol_hand, "{tag}: hand off grip by {gap}");
            }
        }
        assert!(
            (len(P::Hip, P::Shoulder) - TORSO).abs() < 0.15,
            "{tag}: torso length"
        );
    }

    /// Every limb on the bike, for every posture, steer and crank angle.
    #[test]
    fn rider_stays_attached() {
        for steering in [-0.5, 0.0, 0.5] {
            for crank in [0.0, 1.3, 3.1, 5.0] {
                for (stand, crouch, back) in [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (0.55, 0.55, -0.5),
                ] {
                    for &d in Discipline::ALL {
                        let mut bike = Bike::default();
                        bike.select_discipline(d);
                        bike.steering = steering;
                        bike.crank_phase = crank;
                        bike.suspension = [0.2, 0.1];
                        let mut a = AnimationState::default();
                        (a.stand, a.crouch, a.back, a.lateral) = (stand, crouch, back, 0.03);
                        check(&bike, &a, 5e-3, 5e-3, &format!("{d:?} {steering} {crank}"));
                    }
                }
            }
        }
    }

    /// Hips, torso and head must be drawn as one connected tree from the pelvis, hands included.
    #[test]
    fn skeleton_is_one_connected_tree() {
        let bones: Vec<(P, P)> = SKELETON
            .iter()
            .flat_map(|(b, _)| b.iter().copied())
            .collect();
        let mut seen = vec![P::Hip as usize];
        loop {
            let before = seen.len();
            for &(a, b) in &bones {
                if seen.contains(&(a as usize)) && !seen.contains(&(b as usize)) {
                    seen.push(b as usize);
                }
            }
            if seen.len() == before {
                break;
            }
        }
        for p in [
            P::Waist,
            P::Shoulder,
            P::Neck,
            P::Head,
            P::HipL,
            P::HipR,
            P::KneeL,
            P::KneeR,
            P::AnkleL,
            P::AnkleR,
            P::HeelL,
            P::HeelR,
            P::ToeL,
            P::ToeR,
            P::ShoulderL,
            P::ShoulderR,
            P::ElbowL,
            P::ElbowR,
            P::WristL,
            P::WristR,
            P::HandL,
            P::HandR,
        ] {
            assert!(
                seen.contains(&(p as usize)),
                "{p:?} not reached from the pelvis"
            );
        }
        // No bone runs to a bike grip: a released hand must not draw a wrist-to-grip line.
        assert!(
            bones
                .iter()
                .all(|&(a, b)| !matches!(a, P::GripL | P::GripR)
                    && !matches!(b, P::GripL | P::GripR))
        );
    }

    /// The four disciplines differ in bars, saddle height, suspension parts and colour.
    #[test]
    fn disciplines_have_distinct_configurations() {
        let mut seen: Vec<[f32; 3]> = Vec::new();
        for &d in Discipline::ALL {
            let mut bike = Bike::default();
            bike.select_discipline(d);
            let r = rig(&bike, &AnimationState::default());
            let cfg = [
                r[P::GripR].x - r[P::GripL].x,
                r[P::GripR].y,
                r[P::PostTop].y,
            ];
            assert!(
                seen.iter()
                    .all(|c| c.iter().zip(cfg).any(|(a, b)| (a - b).abs() > 1e-3)),
                "{d:?} duplicates another configuration"
            );
            seen.push(cfg);
            let look = Look::of(d);
            assert_eq!(look.rigid, d == Discipline::Road);
            assert_eq!(look.shock, d != Discipline::Road);
        }
        let road = Look::of(Discipline::Road);
        assert!(
            Discipline::ALL
                .iter()
                .filter(|&&d| d != Discipline::Road)
                .all(|&d| Look::of(d).frame != road.frame)
        );
    }

    fn both(one_sided: bool, si: usize) -> [f32; 2] {
        if one_sided {
            let mut t = [0.0; 2];
            t[si] = 1.0;
            t
        } else {
            [1.0; 2]
        }
    }

    /// One scripted ride: ground (tricks requested and refused), takeoff, a flight with the trick
    /// layers held, a landing, then recovery with the tricks still requested.
    fn run(d: Discipline, h: HandTrick, l: LegTrick, bt: BikeTrick, side: f32) -> AnimationState {
        let mut bike = Bike::default();
        bike.select_discipline(d);
        let mut a = AnimationState::default();
        let tag = format!("{d:?}/{h:?}/{l:?}/{bt:?}/side {side}");
        let ctl = Controls {
            pedal: 1.0,
            hand_trick: h,
            leg_trick: l,
            bike_trick: bt,
            trick_side: side,
            ..Controls::default()
        };
        let mut frame = 0u32;
        let mut step = |a: &mut AnimationState, grounded: bool, air_time: f32, landing_in: f32| {
            frame += 1;
            let f = frame as f32;
            bike.steering = 0.5 * (f * 0.37).sin();
            bike.crank_phase = (f * 0.9).rem_euclid(TAU);
            bike.suspension = [
                0.02 + 0.1 * (f * 0.21).sin().abs(),
                0.08 * (f * 0.13).cos().abs(),
            ];
            bike.grounded = [grounded; 2];
            bike.air_time = air_time;
            a.update(
                &Input {
                    c: ctl,
                    roll: 0.1 * (frame as f32 * 0.05).sin(),
                    suspension: bike.suspension,
                    crank_phase: bike.crank_phase,
                    ride_stand: d.profile().ride_stand,
                    speed: 8.0,
                    vy: 3.0 - 9.81 * air_time,
                    pitch: 0.0,
                    steering: bike.steering,
                    grounded: [grounded; 2],
                    air_time,
                    landing_in,
                    impact: 6.0,
                    pitch_rate: 0.0,
                    yaw_rate: 0.0,
                    roll_rate: 0.0,
                },
                DT,
            );
            check(&bike, a, 0.1, 0.05, &tag);
        };
        let spins = |a: &AnimationState| [&a.bars, &a.tail, &a.crank].iter().all(|s| s.settled());

        // Grounded: every air-only layer is refused.
        for _ in 0..12 {
            step(&mut a, true, 0.0, 0.0);
            assert!(a.layers_idle(), "{tag}: trick accepted on the ground");
            assert_eq!(
                (a.hand, a.leg, a.bike),
                (HandTrick::None, LegTrick::None, BikeTrick::None)
            );
        }

        // Flight; sampled mid-way while the tricks are still held.
        let si = (side > 0.0) as usize;
        let mut t = 0.0;
        let mut sampled = false;
        while t < FLIGHT - 1e-4 {
            t += DT;
            step(&mut a, false, t, (FLIGHT - t).max(0.0));
            if !sampled && t >= 0.55 {
                sampled = true;
                let suppressed = h == HandTrick::Barspin
                    && matches!(bt, BikeTrick::Table | BikeTrick::XUp | BikeTrick::EuroTable);
                let forced = !suppressed
                    && matches!(
                        bt,
                        BikeTrick::Invert | BikeTrick::EuroTable | BikeTrick::Crankflip
                    );
                let want_h = match h {
                    HandTrick::None => [0.0; 2],
                    HandTrick::OneHand | HandTrick::TireGrab | HandTrick::SeatGrab => {
                        both(true, si)
                    }
                    _ => [1.0; 2],
                };
                let want_f = match l {
                    _ if forced => [1.0; 2],
                    LegTrick::None => [0.0; 2],
                    LegTrick::OneFoot | LegTrick::CanCan | LegTrick::NacNac => both(true, si),
                    _ => [1.0; 2],
                };
                for i in 0..2 {
                    assert!(
                        (a.hand_rel[i] - want_h[i]).abs() < 0.05,
                        "{tag}: hand layer {i}"
                    );
                    assert!(
                        (a.foot_rel[i] - want_f[i]).abs() < 0.05,
                        "{tag}: foot layer {i}"
                    );
                }
                assert_eq!((a.hand, a.leg), (h, l), "{tag}: reported tricks");
                assert_eq!(
                    a.bike,
                    if suppressed { BikeTrick::None } else { bt },
                    "{tag}"
                );
                assert_eq!(a.side, side, "{tag}: side latch");
                assert_eq!(a.mode, "air");
            }
        }
        assert!(sampled);
        // Contact is imminent: limbs are back on the bike and every spin has finished its turn.
        assert!(
            a.hand_rel.iter().chain(&a.foot_rel).all(|&w| w < 0.02),
            "{tag}: not regrabbed"
        );
        // (The crankflip report weight is informational; it trails the finishing crank spin.)
        assert!(
            a.bike_w
                .iter()
                .enumerate()
                .all(|(k, &w)| k == crate::animation::B_CRANK || w < 0.02),
            "{tag}: bike layer still on"
        );
        assert!(spins(&a), "{tag}: spin unfinished at contact");

        // Landed with the tricks still requested: compression, recovery, no re-entry.
        for n in 0..40 {
            step(&mut a, true, 0.0, 0.0);
            if n == 0 {
                assert_eq!(a.phase, "landing", "{tag}");
            }
            if n >= 20 {
                assert!(a.layers_idle(), "{tag}: layer left after landing");
            }
        }
        assert_eq!(a.phase, "ground", "{tag}");
        assert!(a.land.abs() < 0.05, "{tag}: landing spring");

        a
    }

    /// Every hand x leg x bike trick on every discipline and both sides. The animation only
    /// blends limbs on and off the bike; it never rights or corrects the root.
    #[test]
    fn trick_matrix_is_finite_attached_independent_and_recovers() {
        for &d in Discipline::ALL {
            for &h in HandTrick::ALL {
                for &l in LegTrick::ALL {
                    for &bt in BikeTrick::ALL {
                        for side in [-1.0, 1.0] {
                            run(d, h, l, bt, side);
                        }
                    }
                }
            }
        }
    }

    fn posed(d: Discipline, f: impl Fn(&mut Bike, &mut AnimationState)) -> (Bike, AnimationState) {
        let mut bike = Bike::default();
        bike.select_discipline(d);
        bike.suspension = [0.1, 0.06];
        let mut a = AnimationState::default();
        f(&mut bike, &mut a);
        (bike, a)
    }

    fn tailwhip(_: &mut Bike, a: &mut AnimationState) {
        a.tail.angle = PI;
        a.leg_w[L_TAIL] = 1.0;
        a.foot_rel = [1.0; 2];
    }

    fn table(_: &mut Bike, a: &mut AnimationState) {
        a.bike_w[B_TABLE] = 1.0;
    }

    #[test]
    fn a_tabled_assembly_crashes_where_an_upright_bike_can_land() {
        for tabled in [false, true] {
            let (mut bike, anim) = posed(Discipline::Freeride, |b, a| {
                if tabled {
                    table(b, a);
                }
            });
            bike.position.y += 1.5;
            bike.velocity = Vec3::new(0.0, -2.0, -5.0);
            for _ in 0..240 {
                bike.collision_pose = collision_pose(&bike, &anim);
                bike.step(&Controls::default(), 1.0 / 120.0);
                if bike.crash.is_some() {
                    for sphere in bike.collision_pose.bodies.iter().filter(|s| !s.rider) {
                        let p = bike.position + bike.orientation() * sphere.offset;
                        assert!(p.y - sphere.radius >= terrain_height(p.x, p.z) - 1e-3);
                    }
                }
            }
            assert_eq!(
                bike.crash.is_some(),
                tabled,
                "posed assembly contact: {:?}",
                bike.crash
            );
        }
    }

    /// Hubs are the rest positions of the actual assembly and the axles are unit and really turn
    /// with steering, tailwhip, table and invert.
    #[test]
    fn wheel_hubs_and_axles_follow_the_assembly() {
        for &d in Discipline::ALL {
            let (bike, a) = posed(d, |_, _| {});
            let calm = collision_pose(&bike, &a);
            let hubs = [
                Vec3::new(0.0, -SUSPENSION_REST, -WHEELBASE / 2.0),
                Vec3::new(0.0, -SUSPENSION_REST, WHEELBASE / 2.0),
            ];
            for w in 0..2 {
                assert!(
                    calm.wheel_rest[w].distance(hubs[w]) < 1e-3,
                    "{d:?}: hub {w}"
                );
                assert!(calm.wheel_axes[w].dot(Vec3::X) > 0.9999, "{d:?}: axle {w}");
            }

            let (bike, a) = posed(d, |b, _| b.steering = 0.5);
            let p = collision_pose(&bike, &a);
            assert!(
                (p.wheel_axes[0].dot(Vec3::X) - 0.5_f32.cos()).abs() < 1e-3,
                "{d:?}: steer"
            );
            assert!(p.wheel_axes[1].dot(Vec3::X) > 0.9999, "{d:?}: rear steers");

            let (bike, a) = posed(d, tailwhip);
            let p = collision_pose(&bike, &a);
            assert!(p.wheel_axes[1].dot(Vec3::X) < -0.999, "{d:?}: whipped axle");
            assert!(
                p.wheel_axes[0].dot(Vec3::X) > 0.9999,
                "{d:?}: front axle whipped"
            );
            assert!(
                p.wheel_rest[1].distance(calm.wheel_rest[1]) > 0.5,
                "{d:?}: whipped hub"
            );

            let (bike, a) = posed(d, table);
            let p = collision_pose(&bike, &a);
            assert!(
                p.wheel_axes.iter().all(|x| x.dot(Vec3::X) < 0.95),
                "{d:?}: tabled axles"
            );

            let (bike, a) = posed(d, |_, a| a.bike_w[B_INVERT] = 1.0);
            let p = collision_pose(&bike, &a);
            assert!(
                p.wheel_rest[1].y > 0.5,
                "{d:?}: inverted rear hub {:?}",
                p.wheel_rest[1]
            );

            for p in [calm, p] {
                assert!(p.wheel_axes.iter().all(|x| (x.length() - 1.0).abs() < 1e-4));
            }
        }
    }

    /// Terrain is flat outside the rendered bounds (also just past every edge), and the hill is
    /// inside them.
    #[test]
    fn floor_covers_all_terrain_relief() {
        let inside = |x: f32, z: f32| {
            (FLOOR_X.0..=FLOOR_X.1).contains(&x) && (FLOOR_Z.0..=FLOOR_Z.1).contains(&z)
        };
        for i in -60..=260 {
            for j in -80..=60 {
                let (x, z) = (i as f32 * 0.9 - 50.0 + 0.013, j as f32 * 5.3 + 0.011);
                if !inside(x, z) {
                    assert!(
                        terrain_height(x, z).abs() < 1e-4,
                        "relief outside floor at {x},{z}"
                    );
                }
            }
        }
        for t in 0..=200 {
            let u = t as f32 / 200.0;
            let x = FLOOR_X.0 + (FLOOR_X.1 - FLOOR_X.0) * u;
            let z = FLOOR_Z.0 + (FLOOR_Z.1 - FLOOR_Z.0) * u;
            for (px, pz) in [
                (x, FLOOR_Z.0 - 0.01),
                (x, FLOOR_Z.1 + 0.01),
                (FLOOR_X.0 - 0.01, z),
                (FLOOR_X.1 + 0.01, z),
            ] {
                assert!(
                    terrain_height(px, pz).abs() < 1e-4,
                    "relief leaks at {px},{pz}"
                );
            }
        }
        let peak = (0..=430)
            .map(|j| terrain_height(60.0, FLOOR_Z.0 + j as f32 * 0.5))
            .fold(0.0_f32, f32::max);
        assert!(peak > 5.0, "no hill in the rendered floor: peak {peak}");
    }

    /// At every cell centre (where the two triangles meet) the mesh stays close to the terrain the
    /// physics collides with, so the wheels do not visibly sink or float.
    #[test]
    fn floor_grid_follows_the_terrain() {
        let (xs, zs) = floor_axes();
        let mut worst = 0.0_f32;
        for i in 1..xs.len() - 2 {
            for j in 1..zs.len() - 2 {
                let mid = terrain_height((xs[i] + xs[i + 1]) * 0.5, (zs[j] + zs[j + 1]) * 0.5);
                let mesh =
                    (terrain_height(xs[i + 1], zs[j]) + terrain_height(xs[i], zs[j + 1])) * 0.5;
                worst = worst.max((mid - mesh).abs());
            }
        }
        assert!(
            worst < 0.05,
            "floor mesh deviates from terrain by {worst} m"
        );
    }

    const JOINTS: [&str; COUNT] = [
        "Hip",
        "HipL",
        "HipR",
        "KneeL",
        "KneeR",
        "AnkleL",
        "AnkleR",
        "HeelL",
        "HeelR",
        "ToeL",
        "ToeR",
        "Waist",
        "Shoulder",
        "ShoulderL",
        "ShoulderR",
        "ElbowL",
        "ElbowR",
        "WristL",
        "WristR",
        "Neck",
        "Head",
        "HandL",
        "HandR",
    ];

    /// Scripted ride at the fixed 120 Hz step, exactly as `game::simulate` runs it: `(label,
    /// seconds, controls)`. Returns the rider pose (bike space) after every tick with its label.
    fn ride() -> Vec<(&'static str, [Vec3; COUNT])> {
        let go = Controls {
            pedal: 1.0,
            ..Controls::default()
        };
        let script = [
            ("idle", 1.0, Controls::default()),
            ("pedal", 3.0, go),
            ("sprint", 3.0, Controls { sprint: true, ..go }),
            ("coast", 1.2, Controls::default()),
            (
                "brake",
                1.2,
                Controls {
                    brake: 1.0,
                    ..Controls::default()
                },
            ),
            ("pedal", 2.0, go),
            (
                "turn",
                1.5,
                Controls {
                    steering: 1.0,
                    ..go
                },
            ),
            (
                "turn",
                1.5,
                Controls {
                    steering: -1.0,
                    ..go
                },
            ),
            ("pedal", 1.0, go),
            ("hop", 0.3, Controls { hop: true, ..go }),
            ("pedal", 1.5, go),
            ("sprint", 6.0, Controls { sprint: true, ..go }),
            ("coast", 3.0, Controls::default()),
        ];
        let dt = 1.0 / 120.0;
        let mut bike = Bike::default();
        bike.select_discipline(Discipline::Freeride);
        let mut anim = AnimationState::default();
        let mut out = Vec::new();
        for (label, secs, ctl) in script {
            for _ in 0..(secs / dt) as usize {
                anim.update(&Input::from_bike(&bike, &ctl), dt);
                bike.collision_pose = collision_pose(&bike, &anim);
                bike.step(&ctl, dt);
                let phase = if anim.phase == "ground" {
                    label
                } else {
                    anim.phase
                };
                out.push((phase, rider_points(&bike, &anim)));
            }
        }
        out
    }

    /// Per-tick joint travel and its change (millimetres per tick, relative to the bike): a pose
    /// that snaps shows up as a spike in the second difference.
    #[test]
    fn poses_never_snap_between_ticks() {
        let frames = ride();
        // label -> (steps, snaps with joint and tick) per tick, worst joint of the tick.
        let mut by_label: Vec<(&str, Vec<(f32, f32, usize, usize)>)> = Vec::new();
        for (n, w) in frames.windows(3).enumerate() {
            let label = w[2].0;
            let (mut step, mut snap, mut at) = (0.0_f32, 0.0_f32, 0);
            for j in 0..COUNT {
                let v0 = w[1].1[j] - w[0].1[j];
                let v1 = w[2].1[j] - w[1].1[j];
                step = step.max(v1.length());
                let a = (v1 - v0).length();
                if a > snap {
                    (snap, at) = (a, j);
                }
            }
            let row = (step * 1e3, snap * 1e3, at, n);
            match by_label.iter_mut().find(|e| e.0 == label) {
                Some(e) => e.1.push(row),
                None => by_label.push((label, vec![row])),
            }
        }
        for (label, rows) in &mut by_label {
            let step = rows.iter().map(|r| r.0).fold(0.0, f32::max);
            rows.sort_by(|a, b| a.1.total_cmp(&b.1));
            let p99 = rows[rows.len() * 99 / 100].1;
            let top = rows[rows.len() - 1];
            println!(
                "PROBE {label:>11}: step max {step:.2} mm/tick | snap p99 {p99:.2} max {:.2} mm/tick^2 ({} @tick {})",
                top.1, JOINTS[top.2], top.3
            );
            // The pre-spring pose measured up to 38.6 mm/tick^2 (sprint hip kinks, steering and
            // coast steps); a landing kick is the hardest honest case.
            assert!(
                top.1 < 9.0,
                "{label}: pose snaps ({} {:.1})",
                JOINTS[top.2],
                top.1
            );
        }
        assert!(
            by_label.iter().any(|e| e.0 == "landing"),
            "script never landed"
        );
    }

    /// Display frames blend the last two ticks: endpoints are exact, a quarter step is a quarter
    /// of the way, a reset or teleport never blends across.
    #[test]
    fn frames_blend_between_ticks_and_never_across_a_reset() {
        let (mut bike, anim) = (Bike::default(), AnimationState::default());
        let ragdoll = Ragdoll::default();
        let mut frames = BikeFrames::default();
        assert!(frames.root(0.5).is_none());
        frames.push(Frame::new(&bike, &anim, &ragdoll));
        bike.position += Vec3::new(0.0, 0.0, -0.1);
        bike.steering = 0.1;
        frames.push(Frame::new(&bike, &anim, &ragdoll));
        let (a, b) = (frames.previous.unwrap(), frames.current.unwrap());
        assert!(frames.root(0.0).unwrap().distance(a.position) < 1e-6);
        assert!(frames.root(1.0).unwrap().distance(b.position) < 1e-6);
        let q = frames.blend(0.25, || unreachable!());
        assert!((q.position - (a.position + 0.25 * (b.position - a.position))).length() < 1e-6);
        for i in 0..a.rig.p.len() {
            let want = a.rig.p[i].lerp(b.rig.p[i], 0.25);
            assert!(q.rig.p[i].distance(want) < 1e-6);
        }
        assert!(
            (lerp_angle(3.1, -3.1, 0.5) - PI).abs() < 0.05,
            "angles blend the short way"
        );
        bike.position += Vec3::new(30.0, 0.0, 0.0);
        frames.push(Frame::new(&bike, &anim, &ragdoll));
        assert!(frames.root(0.5).unwrap().distance(bike.position) < 1e-6);
    }

    /// Freewheeling in a straight line the drawn cranks settle level, whatever the last stroke
    /// left.
    #[test]
    fn coasting_cranks_settle_level() {
        let dt = 1.0 / 120.0;
        for stop in [0.0, 0.7, 1.9, 3.3, 4.6] {
            let mut bike = Bike::default();
            bike.reset_at(-20.0, 8.0, 0.0);
            let mut anim = AnimationState::default();
            let go = Controls {
                pedal: 1.0,
                ..Controls::default()
            };
            let mut t = 0.0;
            while t < 3.0 + stop + 3.0 {
                let ctl = if t < 3.0 + stop {
                    go
                } else {
                    Controls::default()
                };
                anim.update(&Input::from_bike(&bike, &ctl), dt);
                bike.step(&ctl, dt);
                t += dt;
            }
            let crank = anim.body().crank.unwrap();
            assert!(
                crank.cos().abs() < 0.05,
                "stop {stop}: crank {crank} not level"
            );
        }
    }

    /// The drawn bike rocks under a standing pedalling rider against the pelvis sway; the
    /// collision pose never sees it and the hands stay on the drawn grips.
    #[test]
    fn standing_stroke_rocks_the_drawn_bike_only() {
        for &d in Discipline::ALL {
            for crank in [0.3, 1.2, 2.5, 3.9, 5.5_f32] {
                let mut bike = Bike::default();
                bike.select_discipline(d);
                bike.crank_phase = crank;
                let mut a = AnimationState::default();
                (a.stand, a.pedal) = (1.0, 1.0);
                check(&bike, &a, 5e-3, 5e-3, &format!("{d:?} crank {crank}"));
                let mut still = a.clone();
                still.pedal = 0.0;
                let (c0, c1) = (collision_pose(&bike, &a), collision_pose(&bike, &still));
                assert_eq!(c0.wheel_axes, c1.wheel_axes, "{d:?}: physics saw the rock");
                assert_eq!(c0.wheel_rest, c1.wheel_rest);
                let roll = rig(&bike, &a).frame.rotation.to_euler(EulerRot::ZXY).0;
                let want = -ROCK * crank.sin();
                assert!((roll - want).abs() < 1e-3, "{d:?}: rock {roll} vs {want}");
            }
        }
        // Right foot driving down: pelvis to the right, bike under it rolled to the left.
        let mut bike = Bike::default();
        bike.crank_phase = 1.75 * PI;
        let mut a = AnimationState::default();
        (a.stand, a.pedal) = (1.0, 1.0);
        let r = rig(&bike, &a);
        let level = rig(&bike, &AnimationState::default());
        assert!(
            r[P::Hip].x > level[P::Hip].x + 0.02,
            "pelvis should go over the pushing pedal"
        );
        assert!(
            (r.frame.rotation * Vec3::Y).x < -0.02,
            "the bike should roll left under the pushing rider"
        );
    }

    fn interior(a: Vec3, b: Vec3, c: Vec3) -> f32 {
        (a - b).angle_between(c - b).to_degrees()
    }

    /// Landing metrics at the moment of contact, the deepest compression and after recovery,
    /// from a real jump on the first ramp (elbow/knee flexion in degrees, upper-arm flare away
    /// from the sagittal plane in degrees, heights in bike space in cm).
    #[test]
    fn landing_compresses_fast_recovers_without_overshoot() {
        use crate::ragdoll::index as ix;
        let dt = 1.0 / 120.0;
        let ctl = Controls {
            pedal: 1.0,
            sprint: true,
            ..Controls::default()
        };
        let mut bike = Bike::default();
        bike.select_discipline(Discipline::Freeride);
        bike.reset_at(0.0, -8.0, 9.0);
        let mut anim = AnimationState::default();
        let mut ticks: Vec<(bool, [Vec3; COUNT])> = Vec::new();
        for _ in 0..(10.0 / dt) as usize {
            anim.update(&Input::from_bike(&bike, &ctl), dt);
            bike.collision_pose = collision_pose(&bike, &anim);
            bike.step(&ctl, dt);
            ticks.push((
                bike.grounded[0] || bike.grounded[1],
                rider_points(&bike, &anim),
            ));
        }
        assert!(bike.crash.is_none(), "the scripted jump crashed");
        let start = ticks
            .iter()
            .position(|t| !t.0)
            .expect("never left the ground");
        let contact = start
            + ticks[start..]
                .iter()
                .position(|t| t.0)
                .expect("never landed");
        assert!(contact - start > 30, "flight too short to judge a landing");
        let metrics = |p: &[Vec3; COUNT]| {
            let mut m = [0.0_f32; 5];
            for (s, (sh, el, wr, hp, kn, an)) in [
                (
                    P::ShoulderL,
                    P::ElbowL,
                    P::WristL,
                    P::HipL,
                    P::KneeL,
                    P::AnkleL,
                ),
                (
                    P::ShoulderR,
                    P::ElbowR,
                    P::WristR,
                    P::HipR,
                    P::KneeR,
                    P::AnkleR,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let _ = s;
                let (sh, el, wr) = (p[ix(sh)], p[ix(el)], p[ix(wr)]);
                let up = el - sh;
                m[0] += 0.5 * (180.0 - interior(sh, el, wr));
                m[1] += 0.5 * up.x.abs().atan2(up.y.hypot(up.z)).to_degrees();
                m[2] += 0.5 * (180.0 - interior(p[ix(hp)], p[ix(kn)], p[ix(an)]));
            }
            m[3] = p[ix(P::Shoulder)].y;
            m[4] = p[ix(P::Hip)].y;
            m
        };
        let pre = metrics(&ticks[contact - 12].1);
        let deepest = (contact..contact + 72)
            .min_by(|&a, &b| metrics(&ticks[a].1)[3].total_cmp(&metrics(&ticks[b].1)[3]))
            .unwrap();
        let deep = metrics(&ticks[deepest].1);
        let settled = metrics(&ticks[(contact + 150).min(ticks.len() - 1)].1);
        let t_deep = (deepest - contact) as f32 * dt;
        let (mut t_back, mut over) = (f32::NAN, 0.0_f32);
        for k in deepest..(contact + 150).min(ticks.len()) {
            let y = metrics(&ticks[k].1)[3];
            if t_back.is_nan() && y >= deep[3] + 0.9 * (settled[3] - deep[3]) {
                t_back = (k - contact) as f32 * dt;
            }
            over = over.max(y - settled[3]);
        }
        println!(
            "PROBE LAND air {:.2}s | pre-contact elbow {:.0} flare {:.0} knee {:.0} | deepest at {:.2}s: elbow {:.0} flare {:.0} knee {:.0} chest drop {:.1} cm pelvis drop {:.1} cm | 90% recovered at {:.2}s, overshoot {:.1} cm | settled elbow {:.0} knee {:.0}",
            (contact - start) as f32 * dt,
            pre[0],
            pre[1],
            pre[2],
            t_deep,
            deep[0],
            deep[1],
            deep[2],
            (pre[3] - deep[3]) * 100.0,
            (pre[4] - deep[4]) * 100.0,
            t_back,
            over * 100.0,
            settled[0],
            settled[2]
        );
        assert!(
            (0.12..0.35).contains(&t_deep),
            "deepest compression at {t_deep}"
        );
        assert!(over < 0.03, "recovery overshoots by {over} m");
    }

    /// Joint ranges over a steady stroke, in the drawn bike's frame, to compare against decoded
    /// retail pedal loops (knee flexion deg, pelvis height above the bottom bracket, torso lean
    /// from vertical, elbow flexion, pelvis lateral sway peak-to-peak, head height). Prints only.
    #[test]
    fn pedal_stroke_metrics() {
        let dt = 1.0 / 120.0;
        for (label, sprint, warm) in [("pedal", false, 8.0), ("sprint", true, 8.0)] {
            let ctl = Controls {
                pedal: 1.0,
                sprint,
                ..Controls::default()
            };
            let mut bike = Bike::default();
            bike.select_discipline(Discipline::Downhill);
            bike.reset_at(-20.0, 8.0, 0.0);
            let mut anim = AnimationState::default();
            let (mut lo, mut hi) = ([f32::MAX; 8], [f32::MIN; 8]);
            let (mut crank, mut last) = (0.0_f32, bike.crank_phase);
            let record = (2.0 / dt) as usize;
            for k in 0..((warm / dt) as usize + record) {
                anim.update(&Input::from_bike(&bike, &ctl), dt);
                bike.collision_pose = collision_pose(&bike, &anim);
                bike.step(&ctl, dt);
                if k < (warm / dt) as usize {
                    last = bike.crank_phase;
                    continue;
                }
                crank += (bike.crank_phase - last + PI).rem_euclid(TAU) - PI;
                last = bike.crank_phase;
                let r = rig(&bike, &anim);
                let f = r.frame;
                let rel = |p: P| f.rotation.inverse() * (r[p] - f.translation);
                let flex = |a: P, b: P, c: P| 180.0 - interior(rel(a), rel(b), rel(c));
                let (hip, sh) = (rel(P::Hip), rel(P::Shoulder));
                let v = [
                    flex(P::HipL, P::KneeL, P::AnkleL),
                    flex(P::HipR, P::KneeR, P::AnkleR),
                    hip.y - BB.y,
                    (sh - hip).z.abs().atan2((sh - hip).y).to_degrees(),
                    0.5 * (flex(P::ShoulderL, P::ElbowL, P::WristL)
                        + flex(P::ShoulderR, P::ElbowR, P::WristR)),
                    hip.x * 100.0,
                    rel(P::Head).y - BB.y,
                    f.rotation.to_euler(EulerRot::ZXY).0.to_degrees(),
                ];
                for i in 0..8 {
                    lo[i] = lo[i].min(v[i]);
                    hi[i] = hi[i].max(v[i]);
                }
            }
            println!(
                "PROBE PEDAL {label}: {:.0} rpm | knee flex L {:.0}-{:.0} R {:.0}-{:.0} | pelvis {:.2}-{:.2} m (p2p {:.1} cm) | torso lean {:.0}-{:.0} | elbow {:.0}-{:.0} | pelvis sway p2p {:.1} cm | head {:.2}-{:.2} m | drawn bike roll amplitude {:.1} deg",
                crank / TAU / 2.0 * 60.0,
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                lo[2],
                hi[2],
                (hi[2] - lo[2]) * 100.0,
                lo[3],
                hi[3],
                lo[4],
                hi[4],
                hi[5] - lo[5],
                lo[6],
                hi[6],
                (hi[7] - lo[7]) * 0.5
            );
        }
    }

    /// Superman and tailwhip returns play out instead of snapping back: the Superman feet take
    /// about 0.6 s to reach the pedals, a released tailwhip frame keeps turning the way it was
    /// kicked, and its trick-side foot is caught before the other. Both are done by contact.
    #[test]
    fn superman_and_tailwhip_returns_take_their_time() {
        let (dt, flight, hold) = (1.0 / 120.0, 2.2, 1.2);
        let wrap = |x: f32| (x + PI).rem_euclid(TAU) - PI;
        for l in [LegTrick::Superman, LegTrick::Tailwhip] {
            let mut a = AnimationState::default();
            let (mut t, mut start, mut caught) = (0.0_f32, None, [None; 2]);
            let mut prev = 0.0;
            while t < flight {
                let c = Controls {
                    leg_trick: if t < hold { l } else { LegTrick::None },
                    trick_side: -1.0,
                    ..Controls::default()
                };
                a.update(
                    &Input {
                        c,
                        ride_stand: 0.5,
                        speed: 8.0,
                        vy: 0.0,
                        pitch: 0.0,
                        steering: 0.0,
                        roll: 0.0,
                        suspension: [0.05; 2],
                        crank_phase: 0.0,
                        grounded: [false; 2],
                        air_time: t,
                        landing_in: flight - t,
                        impact: 3.0,
                        pitch_rate: 0.0,
                        yaw_rate: 0.0,
                        roll_rate: 0.0,
                    },
                    dt,
                );
                t += dt;
                let turn = wrap(a.tail.angle - prev);
                prev = a.tail.angle;
                if t > hold {
                    assert!(turn <= 1e-4, "{l:?}: the frame reversed at {t:.3}");
                    let returning = a.foot_rel.iter().any(|&f| f < 0.98);
                    start = start.or(returning.then_some(t));
                    for s in 0..2 {
                        if a.foot_rel[s] < 0.02 {
                            caught[s] = caught[s].or(Some(t));
                        }
                    }
                }
            }
            let (start, near, far) = (start.unwrap(), caught[0].unwrap(), caught[1].unwrap());
            if l == LegTrick::Superman {
                assert!(
                    (0.5..0.7).contains(&(near - start)),
                    "Superman return {}",
                    near - start
                );
            } else {
                assert!(far - near > 0.1, "tailwhip catch {near:.3} / {far:.3}");
            }
            assert!(
                a.tail.settled() && a.sup_t == 0.0,
                "{l:?}: unfinished at contact"
            );
        }
    }
}
