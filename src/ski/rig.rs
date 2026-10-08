//! Skier rig: turns the physics `Skier` and the animation weights into a world-space `SkierPose`.
//!
//! Everything hangs off the boots: each ski is a rigid line (Tail - Bind - Tip) locked to its boot,
//! the legs are two-bone IK from the hips to the ankles, the spine and arms follow, and every pole is
//! rigid in a fist. All fixed body lengths are exact in every pose.

use super::anim::SkiAnimation;
use super::physics::{Grab, STROKE_PUSH, Skier, skate_stroke, stroke_time};
use super::pose::*;
use crate::bike::terrain_height;
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

const SIDE: [f32; 2] = [-1.0, 1.0];
/// Poles keep their basket tips at least this far above the snow.
const POLE_CLEARANCE: f32 = 0.03;
/// Skate geometry. Both skis point `SKATE_V_YAW` (at full speed) away from the travel line, a V of
/// about 25 degrees, and the skier travels along the ski it glides on, so the glide ski never slides
/// sideways and the body weaves a little from side to side. Feet stand `SKATE_STANCE` m from the
/// centre line, the landing ski sets down `SKATE_LAND_IN` m inside its place, the recovering ski
/// lifts `SKATE_LIFT` m, the pushing foot trails `SKATE_BACK` m with its edge at `SKATE_EDGE` rad
/// (glide ski `SKATE_GLIDE_EDGE`), and the body leans `SKATE_LEAN` rad over the glide ski.
const SKATE_V_YAW: f32 = 0.22;
const SKATE_STANCE: f32 = 0.18;
const SKATE_LAND_IN: f32 = 0.03;
const SKATE_LIFT: f32 = 0.09;
const SKATE_BACK: f32 = 0.18;
const SKATE_EDGE: f32 = 0.5;
const SKATE_GLIDE_EDGE: f32 = 0.06;
const SKATE_LEAN: f32 = 0.14;
/// How much of a full double-pole push the poles add to each skate stroke (V2).
const V2_POLE: f32 = 0.25;
/// Snowplow ski angle, rad.
const PLOW_ANGLE: f32 = 0.40;
/// Visual landing skid: the skis slew this far across the line of travel at full effect, rad (~45 deg).
const LAND_SKID_YAW: f32 = 0.8;
/// Each foot steps this far outwards at full landing compression, m.
const LAND_STANCE: f32 = 0.06;

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

/// Two-bone limb: `(mid, end)` with exact bone lengths; an unreachable target is pulled onto the
/// reachable shell instead of stretching a bone.
fn limb(root: Vec3, target: Vec3, a: f32, b: f32, pole: Vec3) -> (Vec3, Vec3) {
    let d = target - root;
    let len = d.length().clamp((a - b).abs() + 1e-3, a + b - 1e-3);
    let end = root + d.try_normalize().unwrap_or(Vec3::NEG_Y) * len;
    (ik(root, end, a, b, pole), end)
}

/// `target` eased in towards the reach limit of a limb rooted at `root`: the last 15 % of the reach
/// is approached asymptotically, so a hand that goes out of reach never stops with a kink.
fn soft_reach(root: Vec3, target: Vec3, reach: f32) -> Vec3 {
    let d = target - root;
    let (len, knee) = (d.length(), 0.85 * reach);
    if len <= knee {
        return target;
    }
    root + d / len * (knee + (reach - knee) * (1.0 - (-(len - knee) / (reach - knee)).exp()))
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn unit(v: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(Vec3::Y)
}

/// One ski of a skate stroke at progress `p` (`skate_stroke`): the pushing ski edges, runs out and
/// back while the hips cross over to the other ski, then lifts, swings in and lands in the V as the
/// next stroke begins; the other ski lands, takes the weight and glides. The two weights always sum
/// to one. `reach` is how far the pushing foot runs out sideways, m.
struct SkateSki {
    weight: f32,
    out: f32,
    back: f32,
    lift: f32,
    /// Yaw of the ski away from the travel line, rad.
    yaw: f32,
    edge: f32,
}

fn skate_ski(pushing: bool, p: f32, reach: f32, yaw: f32) -> SkateSki {
    let cross = smooth(p / STROKE_PUSH);
    if !pushing {
        // Lands inside its place and moves out as the weight arrives, then glides.
        return SkateSki {
            weight: cross,
            out: -SKATE_LAND_IN * (1.0 - cross),
            back: 0.0,
            lift: 0.0,
            yaw,
            edge: SKATE_GLIDE_EDGE * cross,
        };
    }
    let recover = smooth((p - STROKE_PUSH) / 0.40);
    let spread = smooth(p / 0.30) * (1.0 - recover);
    let q = ((p - 0.40) / 0.52).clamp(0.0, 1.0);
    SkateSki {
        weight: 1.0 - cross,
        out: reach * spread - SKATE_LAND_IN * smooth((p - 0.8) / 0.2),
        back: SKATE_BACK * spread,
        lift: SKATE_LIFT * (PI * q).sin(),
        yaw,
        edge: lerp(SKATE_GLIDE_EDGE, SKATE_EDGE, smooth(p / 0.15))
            * (1.0 - smooth((p - 0.38) / 0.14)),
    }
}

/// Sideways drift (m, towards the glide side) of the skier at stroke progress `p`, for `sweep` =
/// speed * stroke time * tan(ski yaw): the lateral speed follows the weighted skis' directions
/// (the glide ski's while it carries the weight), averaged to zero over two strokes.
fn skate_drift(p: f32, sweep: f32) -> f32 {
    let f = if p < STROKE_PUSH {
        let q = p / STROKE_PUSH;
        2.0 * STROKE_PUSH * (q.powi(3) - 0.5 * q.powi(4)) - p
    } else {
        p - STROKE_PUSH
    };
    sweep * (f - 0.5 * (1.0 - STROKE_PUSH))
}

/// Periodic cubic Hermite through `keys` (time 0..1, ascending, first at 0): continuous in value and
/// velocity at every key and across the wrap.
fn loop_spline(keys: &[(f32, Vec3)], u: f32) -> Vec3 {
    let n = keys.len() as isize;
    let u = u.rem_euclid(1.0);
    let i = keys.iter().rposition(|k| k.0 <= u).unwrap_or(0) as isize;
    let at = |j: isize| {
        let k = keys[j.rem_euclid(n) as usize];
        (k.0 + j.div_euclid(n) as f32, k.1)
    };
    let ((t0, p0), (t1, p1), (tm, pm), (t2, p2)) = (at(i), at(i + 1), at(i - 1), at(i + 2));
    let d = t1 - t0;
    let (m0, m1) = ((p1 - pm) / (t1 - tm) * d, (p2 - p0) / (t2 - t0) * d);
    let t = (u - t0) / d;
    let (t2_, t3) = (t * t, t * t * t);
    p0 * (2.0 * t3 - 3.0 * t2_ + 1.0)
        + m0 * (t3 - 2.0 * t2_ + t)
        + p1 * (3.0 * t2_ - 2.0 * t3)
        + m1 * (t3 - t2_)
}

/// Fist of a double-pole push relative to its shoulder (x outward, y up, -z forward, heading frame)
/// over the cycle `u` (0 = poles planted): up and forward at the plant, pulled down past the hips
/// with the elbows opening, released behind, then swung back forward with the elbows bent.
const POLE_HAND: [(f32, Vec3); 7] = [
    (0.00, Vec3::new(0.04, 0.05, -0.42)),
    (0.12, Vec3::new(0.03, -0.18, -0.38)),
    (0.25, Vec3::new(0.03, -0.42, -0.14)),
    (0.38, Vec3::new(0.06, -0.50, 0.34)),
    (0.50, Vec3::new(0.08, -0.40, 0.40)),
    (0.65, Vec3::new(0.12, -0.36, 0.10)),
    (0.82, Vec3::new(0.08, -0.10, -0.40)),
];

/// Trunk and knee flexion of a double-pole push (x): -0.3 tall and up on the toes at the plant,
/// 1 at the end of the crunch, then rising back.
const POLE_CRUNCH: [(f32, Vec3); 6] = [
    (0.00, Vec3::new(-0.20, 0.0, 0.0)),
    (0.18, Vec3::new(0.45, 0.0, 0.0)),
    (0.38, Vec3::new(1.00, 0.0, 0.0)),
    (0.60, Vec3::new(0.55, 0.0, 0.0)),
    (0.80, Vec3::new(0.00, 0.0, 0.0)),
    (0.93, Vec3::new(-0.30, 0.0, 0.0)),
];

fn pole_crunch(u: f32) -> f32 {
    loop_spline(&POLE_CRUNCH, u).x
}

/// Pole basket track over a double-pole cycle `u` in 0..1 (0 = planted): (metres ahead of the
/// binding, metres lifted above the snow). The basket is planted ahead of the binding and sweeps
/// back `sweep` m beyond 0.45 m behind it, leaves the snow, trails, then comes down again over the
/// last 15 % of the cycle so the plant is a descent, not a drop.
fn pole_tip_track(u: f32, sweep: f32) -> (f32, f32) {
    let end = -(0.45 + sweep);
    if u < 0.38 {
        (lerp(0.30, end, smooth(u / 0.38)), 0.0)
    } else if u < 0.56 {
        let t = smooth((u - 0.38) / 0.18);
        (lerp(end, end - 0.15, t), 0.30 * t)
    } else if u < 0.78 {
        (lerp(end - 0.15, 0.40, smooth((u - 0.56) / 0.22)), 0.30)
    } else {
        let t = smooth((u - 0.78) / 0.22);
        (lerp(0.40, 0.30, t), 0.30 * (1.0 - t))
    }
}

/// How firmly the pole points at its basket: while planted or about to be, not in the swing.
fn pole_aim(u: f32) -> f32 {
    if u < 0.38 {
        1.0
    } else if u < 0.60 {
        1.0 - smooth((u - 0.38) / 0.22)
    } else {
        smooth((u - 0.76) / 0.24)
    }
}

/// Pole axis (basket to grip) closest to `wish` that keeps the basket above the snow, for a pole held
/// with the wrist at `wrist`. Fallback for a hand the arm cannot lift high enough; near vertical
/// this rotation is abrupt, so `solve` lifts the hand first (`basket_lift`).
fn fit_pole(wrist: Vec3, wish: Vec3) -> Vec3 {
    let reach = POLE_LENGTH - HAND - POLE_GRIP;
    let wish = unit(wish);
    let dir = Vec3::new(wish.x, 0.0, wish.z)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    let mut floor = f32::MIN;
    let mut d = wish;
    for _ in 0..6 {
        let tip = wrist - d * reach;
        floor = floor.max(terrain_height(tip.x, tip.z) + POLE_CLEARANCE);
        let max_y = ((wrist.y - floor) / reach).clamp(-1.0, 1.0);
        d = if wish.y <= max_y {
            wish
        } else {
            dir * (1.0 - max_y * max_y).sqrt() + Vec3::Y * max_y
        };
    }
    d
}

/// Height the fist at `fist` must rise so a pole along `d` keeps its basket above the snow. A
/// quadratic blend over `BASKET_SOFT` either side of contact keeps the hand's velocity continuous
/// when the basket touches or leaves the snow.
fn basket_lift(fist: Vec3, d: Vec3) -> f32 {
    const BASKET_SOFT: f32 = 0.10;
    let tip = fist - d * (POLE_LENGTH - POLE_GRIP);
    let x = terrain_height(tip.x, tip.z) + POLE_CLEARANCE - tip.y;
    if x <= -BASKET_SOFT {
        0.0
    } else if x < BASKET_SOFT {
        (x + BASKET_SOFT).powi(2) / (4.0 * BASKET_SOFT)
    } else {
        x
    }
}

/// A leg's pose when it is part of a grab. `ankle` is relative to the pelvis in the pelvis frame
/// (x outward from the leg's own side, y up, z back); angles are relative to the pelvis frame
/// (yaw: tip outward, pitch: tip up, roll: outer edge down).
#[derive(Clone, Copy)]
struct Leg {
    ankle: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
}

const fn leg(x: f32, y: f32, z: f32, yaw: f32, pitch: f32, roll: f32) -> Leg {
    Leg {
        ankle: Vec3::new(x, y, z),
        yaw,
        pitch,
        roll,
    }
}

/// A hand closing on a ski: `other` = the non-grabbed ski, `cross` = the hand of the opposite side
/// of that ski. The point lies `along` the ski from the binding (+ towards the tip), `lat` outside
/// its centre line and `up` above the top surface.
#[derive(Clone, Copy)]
struct Hold {
    other: bool,
    cross: bool,
    along: f32,
    lat: f32,
    up: f32,
}

const fn hold(other: bool, cross: bool, along: f32, lat: f32, up: f32) -> Hold {
    Hold {
        other,
        cross,
        along,
        lat,
        up,
    }
}

struct Spec {
    /// `[grabbed ski, other ski]`.
    legs: [Leg; 2],
    holds: [Option<Hold>; 2],
    /// Free hand relative to the chest in the chest frame (x outward).
    free: Vec3,
    /// Extra forward spine flex.
    flex: f32,
}

const FREE_ARM: Vec3 = Vec3::new(0.55, 0.05, -0.15);
const RELAXED: Leg = leg(0.12, -0.62, 0.05, 0.0, -0.1, 0.0);

fn grab_spec(g: Grab) -> Spec {
    match g {
        Grab::Mute | Grab::None => Spec {
            legs: [leg(0.05, -0.25, -0.20, 0.0, 0.4, 0.2), RELAXED],
            holds: [Some(hold(false, true, 0.15, 0.04, 0.045)), None],
            free: FREE_ARM,
            flex: 1.0,
        },
        Grab::Safety => Spec {
            legs: [leg(0.20, -0.12, -0.25, 0.0, 0.3, 0.35), RELAXED],
            holds: [Some(hold(false, false, -0.02, 0.07, 0.02)), None],
            free: FREE_ARM,
            flex: 1.2,
        },
        Grab::Japan => Spec {
            legs: [leg(0.05, -0.10, -0.10, -0.3, -0.5, 0.2), RELAXED],
            holds: [Some(hold(false, true, 0.05, 0.04, 0.045)), None],
            free: FREE_ARM,
            flex: 1.0,
        },
        Grab::Tail => Spec {
            legs: [leg(0.15, -0.10, -0.20, 0.0, -1.0, 0.0), RELAXED],
            holds: [
                Some(hold(false, false, -(SKI_BACK - 0.08), 0.0, 0.045)),
                None,
            ],
            free: FREE_ARM,
            flex: 0.3,
        },
        Grab::Tip => Spec {
            legs: [leg(0.15, -0.25, 0.0, 0.0, 0.5, 0.0), RELAXED],
            holds: [Some(hold(false, false, SKI_FRONT - 0.10, 0.0, 0.045)), None],
            free: FREE_ARM,
            flex: 0.9,
        },
        Grab::TruckDriver => Spec {
            legs: [
                leg(0.10, -0.30, 0.25, 0.0, 0.7, 0.0),
                leg(0.10, -0.30, 0.25, 0.0, 0.7, 0.0),
            ],
            holds: [
                Some(hold(false, false, SKI_FRONT - 0.10, 0.0, 0.045)),
                Some(hold(true, false, SKI_FRONT - 0.10, 0.0, 0.045)),
            ],
            free: FREE_ARM,
            flex: 0.8,
        },
        Grab::Daffy => Spec {
            legs: [
                leg(0.10, -0.60, -0.40, 0.0, 0.25, 0.0),
                leg(0.10, -0.55, 0.40, 0.0, -0.2, 0.0),
            ],
            holds: [None, None],
            free: Vec3::new(0.60, 0.15, -0.20),
            flex: 0.35,
        },
        Grab::SpreadEagle => Spec {
            legs: [leg(0.45, -0.55, 0.0, 0.45, 0.1, 0.4); 2],
            holds: [None, None],
            free: Vec3::new(0.62, 0.30, -0.10),
            flex: 0.0,
        },
        Grab::IronCross => Spec {
            legs: [
                leg(-0.10, -0.60, -0.05, -0.28, 0.15, 0.0),
                leg(-0.10, -0.55, -0.05, -0.28, 0.15, 0.0),
            ],
            holds: [None, None],
            free: Vec3::new(0.60, 0.02, -0.05),
            flex: 0.3,
        },
    }
}

/// Point on a ski `along` its binding (+ towards the tip), following the upturned shovel.
fn ski_point(bind: Vec3, f: Vec3, u: Vec3, side: f32, along: f32, lat: f32, up: f32) -> Vec3 {
    let rise = if along > 0.0 {
        SKI_TIP_RISE * along / SKI_FRONT
    } else {
        0.0
    };
    bind + f * along + u * (up + rise) + f.cross(u) * (side * lat)
}

/// Solves the whole skier in world space.
pub(crate) fn solve(s: &Skier, a: &SkiAnimation) -> SkierPose {
    let w = &a.w;
    let r = s.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let spec = (a.grab != Grab::None).then(|| grab_spec(a.grab));
    let gw = if spec.is_some() { w.grab } else { 0.0 };
    let gi = usize::from(w.grab_side > 0.0);

    // --- Skis, neutral placement in the platform frame. --------------------------------------
    // Skating: one ski pushes per stroke while the hips cross over to the other (`skate_ski`). The
    // strokes grow with speed, and from a standing start they are short and upright.
    let speed = s.velocity.length();
    let amp = smooth(speed / 3.0);
    let sk = w.skate;
    let (pusher, stroke_p) = skate_stroke(s.skate_phase);
    let v_yaw = SKATE_V_YAW * lerp(0.6, 1.0, amp);
    let sweep = speed * stroke_time(speed) * v_yaw.tan();
    let reach = (0.5 * sweep).clamp(0.08, 0.18);
    let skis = [0, 1].map(|i| skate_ski(i == pusher, stroke_p, reach, v_yaw));
    // The skier drifts and faces towards the glide side, then swings over to the other.
    let g_glide = SIDE[1 - pusher];
    let drift = sk * g_glide * skate_drift(stroke_p, sweep);
    let facing = sk * g_glide * v_yaw * (2.0 * skis[1 - pusher].weight - 1.0);
    let rb = r * Quat::from_rotation_y(-facing);
    let (right_b, fwd_b) = (rb * Vec3::X, rb * Vec3::NEG_Z);
    let grounded_w = 1.0 - w.air;
    let mut q_n = [Quat::IDENTITY; 2];
    let mut ankle_n = [Vec3::ZERO; 2];
    for i in 0..2 {
        let g = SIDE[i];
        let ks = &skis[i];
        // Plow: tips together (inward). Skate: skis in a V.
        let vis_yaw = w.skid_side * LAND_SKID_YAW * w.land_skid;
        let yaw = g * (PLOW_ANGLE * w.plow - sk * ks.yaw) + vis_yaw;
        // Right edge down is positive; plow and skate push edge the inside edges.
        let roll = s.edge - g * (0.38 * w.plow + sk * ks.edge) + 0.35 * vis_yaw;
        let pitch = -(0.12 * w.air + 0.25 * w.extend);
        let base_q = r * Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
        let roll_q = Quat::from_axis_angle(Vec3::NEG_Z, roll);
        // The tilted ski rolls about its lower edge, which stays on the snow.
        let edge = Vec3::new(roll.signum() * SKI_WIDTH * 0.5, 0.0, 0.0);
        let shift = base_q * (edge - roll_q * edge) * grounded_w;
        // The pushing skate leg runs out to the side and back, the recovering ski lifts.
        let base = s.position
            + right
                * (drift
                    + g * (lerp(STANCE_WIDTH * 0.5, SKATE_STANCE, sk)
                        + 0.03 * w.plow
                        + sk * ks.out
                        + LAND_STANCE * w.land))
            - fwd * (sk * ks.back)
            + up * (sk * ks.lift)
            + shift;
        let q = base_q * roll_q;
        q_n[i] = q;
        ankle_n[i] = base + q * Vec3::Y * (SKI_THICKNESS + ANKLE_ABOVE_SKI);
    }

    // --- Pelvis. -----------------------------------------------------------------------------
    // Skating: the hips sit over the weighted ski and the body leans over it.
    let over_right = skis[1].weight - skis[0].weight;
    let lean = s.lean + SKATE_LEAN * sk * amp * over_right;
    let roll_p = rb * Quat::from_axis_angle(Vec3::NEG_Z, lean);
    let col = roll_p * Vec3::Y;
    let over = ankle_n[0] * skis[0].weight + ankle_n[1] * skis[1].weight
        - right_b * (HIP_HALF_WIDTH * over_right);
    let m0 = ((ankle_n[0] + ankle_n[1]) * 0.5).lerp(over, sk);
    let pole_u = s.pole_phase.rem_euclid(TAU) / TAU;
    // Double pole (and the poling of each V2 stroke): up on the toes at the plant, a crunch of
    // trunk, hips and knees through the push, rising again in the recovery.
    let tall = w.pole + V2_POLE * w.v2;
    let crunch = w.pole * pole_crunch(pole_u) + V2_POLE * w.v2 * pole_crunch(stroke_p);
    let rise = if stroke_p < STROKE_PUSH {
        (PI * stroke_p / STROKE_PUSH).sin()
    } else {
        0.0
    };
    let carve = w.carve.abs();
    // Idle life: slow weight shifts between the feet and breathing, off the same clock, fading in
    // tucks and the air. Incommensurate periods never repeat visibly.
    let calm = (1.0 - w.air) * (1.0 - 0.8 * w.tuck);
    let sway = calm * (0.010 * (0.7 * w.clock).sin() + 0.006 * (1.9 * w.clock + 1.0).sin());
    let breathe = calm * (1.7 * w.clock).sin();
    // Athletic stance: ankles and knees flexed at rest, deeper with speed, turns and impacts.
    let dist = (0.76
        - 0.12 * w.crouch
        - 0.07 * carve
        - 0.36 * w.preload * (1.0 - w.extend)
        - 0.18 * w.tuck * (1.0 - 0.5 * w.air)
        - 0.12 * w.air_tuck
        - 0.08 * w.air
        - 0.20 * w.land
        - 0.10 * w.plow
        + w.skate * (0.08 - 0.05 * amp + 0.02 * amp * rise)
        + 0.07 * tall
        - 0.11 * crunch
        + 0.14 * w.extend)
        .clamp(0.40, 0.85);
    // Hips sit back over the heels as the knees flex.
    let k = ((0.80 - dist) / 0.40).clamp(0.0, 1.0);
    let lateral = right * (0.05 * w.carve + sway);
    let pelvis = m0 + (col * dist - fwd_b * (0.05 + 0.20 * k)).normalize() * dist + lateral;

    // --- Torso frames. -----------------------------------------------------------------------
    // Slip angle of the velocity relative to the skis (positive = towards the left), faded in with
    // speed so it never cuts out at a threshold.
    let vel_flat = s.velocity - up * s.velocity.dot(up);
    let slip = match vel_flat.try_normalize() {
        Some(n) => {
            let f_flat = unit(fwd - up * fwd.dot(up));
            let angle = f_flat
                .cross(n)
                .dot(up)
                .atan2(f_flat.dot(n))
                .clamp(-1.2, 1.2);
            angle * smooth((vel_flat.length() - 0.3) / 1.2)
        }
        None => 0.0,
    };
    // Upper body counter-rotates towards the fall line (lagging the skis' turn); in switch it
    // twists to look the way it travels.
    let turn_twist = 0.45 * w.carve + w.counter + 0.6 * slip * w.skid
        - 0.85 * w.skid_side * LAND_SKID_YAW * w.land_skid;
    let chest_twist = turn_twist + 1.6 * w.switch * w.switch_side;
    let qp = roll_p * Quat::from_rotation_y(0.25 * chest_twist);
    // Angulation: the torso is inclined less than the hips; sway rolls it a little against them.
    let qc = rb
        * Quat::from_axis_angle(Vec3::NEG_Z, lean * 0.45 - 2.0 * sway)
        * Quat::from_rotation_y(chest_twist);
    // The head stays level and keeps looking down the hill while the chest twists and leans.
    let qh = qc.slerp(r * Quat::from_rotation_y(chest_twist), 0.6);
    let gaze = unit(
        r * Quat::from_rotation_y(chest_twist - 0.6 * turn_twist - 0.2 * w.carve) * Vec3::NEG_Z,
    );
    let flex = (0.28
        + 0.30 * w.crouch
        + 0.10 * carve
        + 0.80 * w.preload * (1.0 - w.extend)
        + 1.7 * w.fold
        + 0.02 * breathe
        + 1.05 * w.tuck * (1.0 - 0.3 * w.air)
        + 0.35 * w.air_tuck
        + 0.18 * tall
        + 0.78 * crunch
        + 0.10 * w.plow
        + w.skate * (0.15 + 0.25 * amp)
        - 0.25 * w.extend
        + spec.as_ref().map_or(0.0, |sp| sp.flex) * gw)
        .clamp(0.0, 1.55);
    let tilt = |q: Quat, f: f32| q * Quat::from_rotation_x(-f) * Vec3::Y;
    let waist = pelvis + tilt(qp.slerp(qc, 0.5), 0.45 * flex) * PELVIS_TO_WAIST;
    let chest = waist + tilt(qc, 0.85 * flex) * WAIST_TO_CHEST;
    let neck = chest + tilt(qc, 0.95 * flex) * CHEST_TO_NECK;
    let head = neck + tilt(qh, 0.35 * flex) * NECK_TO_HEAD;
    let qchest = qc * Quat::from_rotation_x(-0.85 * flex);

    // --- Legs and skis (skis follow the feet). ---------------------------------------------
    let hip_x = qp * Vec3::X;
    let body_fwd = qp * Vec3::NEG_Z;
    let mut bind = [Vec3::ZERO; 2];
    let mut f = [Vec3::ZERO; 2];
    let mut u = [Vec3::ZERO; 2];
    let mut knee = [Vec3::ZERO; 2];
    let mut ankle = [Vec3::ZERO; 2];
    let mut hip = [Vec3::ZERO; 2];
    for i in 0..2 {
        let g = SIDE[i];
        let (q, target) = match &spec {
            Some(sp) => {
                let l = if i == gi { sp.legs[0] } else { sp.legs[1] };
                let q_spec = qp
                    * Quat::from_rotation_y(-g * l.yaw)
                    * Quat::from_rotation_x(l.pitch)
                    * Quat::from_axis_angle(Vec3::NEG_Z, g * l.roll);
                let t_spec = pelvis + qp * Vec3::new(g * l.ankle.x, l.ankle.y, l.ankle.z);
                (q_n[i].slerp(q_spec, gw), ankle_n[i].lerp(t_spec, gw))
            }
            None => (q_n[i], ankle_n[i]),
        };
        hip[i] = pelvis + hip_x * (g * HIP_HALF_WIDTH);
        // Knees track forward over the skis, pinch inward in a plow and drive into a carve.
        let pole = (q * Vec3::NEG_Z + body_fwd) * 0.5 - hip_x * (g * (0.12 + 0.45 * w.plow))
            + hip_x * (0.35 * w.carve);
        let (k_i, a_i) = limb(hip[i], target, THIGH, SHIN, pole);
        knee[i] = k_i;
        ankle[i] = a_i;
        f[i] = q * Vec3::NEG_Z;
        u[i] = q * Vec3::Y;
        bind[i] = a_i - u[i] * ANKLE_ABOVE_SKI;
    }

    // --- Arms: (fist target, pole axis basket->grip) per hand. -----------------------------
    let shoulder_x = qchest * Vec3::X;
    let mut held: [Option<Vec3>; 2] = [None, None];
    if let Some(sp) = &spec {
        for h in sp.holds.iter().flatten() {
            let ski = if h.other { 1 - gi } else { gi };
            let hand = if h.cross { 1 - ski } else { ski };
            held[hand] = Some(ski_point(
                bind[ski], f[ski], u[ski], SIDE[ski], h.along, h.lat, h.up,
            ));
        }
    }
    let mut wrist = [Vec3::ZERO; 2];
    let mut shoulder = [Vec3::ZERO; 2];
    let mut elbow = [Vec3::ZERO; 2];
    let mut pole_dir = [Vec3::Y; 2];
    let mut fist = [Vec3::ZERO; 2];
    for h in 0..2 {
        let g = SIDE[h];
        shoulder[h] = chest + shoulder_x * (g * SHOULDER_HALF_WIDTH);
        // Neutral: hands forward at belly height, elbows bent, poles angled back.
        let mut ht = chest + qc * Vec3::new(g * 0.27, -0.30, -0.40);
        // The trailing pole hangs from the heading frame, not the leaned torso, so an inclined
        // body does not drive the downhill basket into the snow.
        let mut dw =
            unit(rb * Quat::from_rotation_y(chest_twist) * Vec3::new(g * 0.12, 0.75, -0.65));
        // Skating without poles: the arm opposite the pushing leg swings forward and across.
        let swing = w.skate * (1.0 - w.v2) * g * s.skate_phase.sin();
        ht += qc * Vec3::new(-0.04 * g * swing, 0.10 * swing, -0.30 * swing);
        // A double-pole push at cycle position `u`: the fist and the pole axis it holds. The fist is
        // pulled onto the pole length around the basket while the basket is planted.
        let hq = rb * Quat::from_rotation_y(chest_twist);
        let sweep = (0.20 + 0.04 * speed).min(0.55);
        let pole_arm = |u: f32| {
            let key = loop_spline(&POLE_HAND, u);
            let wish = shoulder[h] + hq * Vec3::new(g * key.x, key.y, key.z);
            let (ahead, lift) = pole_tip_track(u, sweep);
            let tip_xz = s.position + right * (g * 0.26) + fwd * ahead;
            let tip = Vec3::new(
                tip_xz.x,
                terrain_height(tip_xz.x, tip_xz.z) + POLE_CLEARANCE + 0.005 + lift,
                tip_xz.z,
            );
            let toward = unit(wish - tip);
            let aim = pole_aim(u);
            (
                wish.lerp(tip + toward * (POLE_LENGTH - POLE_GRIP), aim),
                toward,
                aim,
            )
        };
        if w.pole > 1e-3 {
            let (hand, toward, aim) = pole_arm(pole_u);
            ht = ht.lerp(hand, w.pole);
            dw = dw.lerp(toward, aim * w.pole);
        }
        if w.v2 > 1e-3 {
            let (hand, toward, aim) = pole_arm(stroke_p);
            ht = ht.lerp(hand, w.v2);
            dw = dw.lerp(toward, aim * w.v2);
        }
        // Turn-change plant: the new inside pole reaches ahead, touches and the skier passes it.
        if w.plant_w[h] > 1e-3 {
            let p = w.plant[h];
            let reach = lerp(0.95, -0.25, smooth((p - 0.3) / 0.4));
            let tip_xz = s.position + right * (g * 0.42) + fwd * reach;
            let tip = Vec3::new(
                tip_xz.x,
                terrain_height(tip_xz.x, tip_xz.z)
                    + POLE_CLEARANCE
                    + 0.005
                    + 0.5 * smooth((p - 0.55) / 0.35),
                tip_xz.z,
            );
            let toward = unit(ht - tip);
            ht = ht.lerp(tip + toward * (POLE_LENGTH - POLE_GRIP), w.plant_w[h]);
            dw = dw.lerp(toward, w.plant_w[h]);
        }
        // Pop: the arms swing back with the crouch, then drive up and forward with the legs.
        ht += qc * Vec3::new(0.0, -0.05, 0.22) * (w.preload * (1.0 - w.extend));
        ht += qc * Vec3::new(0.0, 0.12, -0.16) * w.extend;
        // Landing: fists drop towards the knees while the legs absorb.
        ht += qc * Vec3::new(0.0, -0.22, -0.18) * w.land;
        if w.tuck > 1e-3 {
            // Tuck: fists forward at knee height, poles along the back.
            ht = ht.lerp(chest + qc * Vec3::new(g * 0.14, -0.36, -0.42), w.tuck);
            dw = dw.lerp(unit(qchest * Vec3::new(g * 0.05, 1.0, 0.0)), w.tuck);
        }
        if w.air > 1e-3 {
            // Airborne: arms out for balance.
            ht = ht.lerp(chest + qc * Vec3::new(g * 0.55, 0.05, -0.18), w.air);
            dw = dw.lerp(unit(qc * Vec3::new(g * 0.25, 0.5, -0.7)), w.air);
        }
        // Free hands trail the body's acceleration.
        ht += w.hand[h] * ((1.0 - w.pole) * (1.0 - 0.6 * w.tuck));
        if let Some(sp) = &spec {
            let (gt, gd) = match held[h] {
                Some(p) => (p, unit(body_fwd * 0.9 + shoulder_x * (g * 0.3) + up * 0.2)),
                None => (
                    chest + qc * Vec3::new(g * sp.free.x, sp.free.y, sp.free.z),
                    unit(qc * Vec3::new(g * 0.25, 0.5, -0.7)),
                ),
            };
            ht = ht.lerp(gt, gw);
            dw = dw.lerp(gd, gw);
        }
        let dw = unit(dw);
        // Keep the wished pole direction and raise the fist over the snow; the arm IK follows.
        let ht = ht + Vec3::Y * basket_lift(ht, dw);
        let d = dw;
        let poling = w.pole.max(w.v2);
        let elbow_pole =
            qc * Vec3::new(g * 0.5, -0.7, 0.3).lerp(Vec3::new(g * 0.6, -0.2, 0.8), poling);
        let wrist_target = soft_reach(shoulder[h], ht - d * HAND, UPPER_ARM + FOREARM);
        let (e, wr) = limb(shoulder[h], wrist_target, UPPER_ARM, FOREARM, elbow_pole);
        let d = fit_pole(wr, d);
        elbow[h] = e;
        wrist[h] = wr;
        pole_dir[h] = d;
        fist[h] = wr + d * HAND;
    }

    // --- Assemble. -----------------------------------------------------------------------
    let mut pose = SkierPose {
        p: [Vec3::ZERO; JOINTS],
        ski_up: u,
        facing: unit(qc * Vec3::NEG_Z),
        gaze,
    };
    pose[J::Pelvis] = pelvis;
    pose[J::Waist] = waist;
    pose[J::Chest] = chest;
    pose[J::Neck] = neck;
    pose[J::Head] = head;
    const HIP: [J; 2] = [J::HipL, J::HipR];
    const KNEE: [J; 2] = [J::KneeL, J::KneeR];
    const ANKLE: [J; 2] = [J::AnkleL, J::AnkleR];
    const HEEL: [J; 2] = [J::HeelL, J::HeelR];
    const TOE: [J; 2] = [J::ToeL, J::ToeR];
    const BIND: [J; 2] = [J::BindL, J::BindR];
    const TIP: [J; 2] = [J::TipL, J::TipR];
    const TAIL: [J; 2] = [J::TailL, J::TailR];
    const SHOULDER: [J; 2] = [J::ShoulderL, J::ShoulderR];
    const ELBOW: [J; 2] = [J::ElbowL, J::ElbowR];
    const WRIST: [J; 2] = [J::WristL, J::WristR];
    const HANDS: [J; 2] = [J::HandL, J::HandR];
    const POLE_TOP: [J; 2] = [J::PoleTopL, J::PoleTopR];
    const POLE_TIP: [J; 2] = [J::PoleTipL, J::PoleTipR];
    for i in 0..2 {
        pose[HIP[i]] = hip[i];
        pose[KNEE[i]] = knee[i];
        pose[ANKLE[i]] = ankle[i];
        pose[HEEL[i]] = bind[i] - f[i] * HEEL_BACK;
        pose[TOE[i]] = bind[i] + f[i] * TOE_FORWARD;
        pose[BIND[i]] = bind[i];
        pose[TAIL[i]] = bind[i] - f[i] * SKI_BACK;
        pose[TIP[i]] = bind[i] + f[i] * SKI_FRONT + u[i] * SKI_TIP_RISE;
        pose[SHOULDER[i]] = shoulder[i];
        pose[ELBOW[i]] = elbow[i];
        pose[WRIST[i]] = wrist[i];
        pose[HANDS[i]] = fist[i];
        let top = fist[i] + pole_dir[i] * POLE_GRIP;
        pose[POLE_TOP[i]] = top;
        pose[POLE_TIP[i]] = top - pole_dir[i] * POLE_LENGTH;
    }
    // Touchdown settle: rigidly carry the whole pose back towards its pre-snap attitude/height.
    if s.land_tilt != Vec3::ZERO || s.land_shift != Vec3::ZERO {
        let q = Quat::from_scaled_axis(s.land_tilt);
        for p in &mut pose.p {
            *p = s.position + s.land_shift + q * (*p - s.position);
        }
        pose.ski_up = pose.ski_up.map(|v| q * v);
        pose.facing = q * pose.facing;
        pose.gaze = q * pose.gaze;
    }
    pose
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ski::physics::SkiControls;

    const DT: f32 = 1.0 / 120.0;
    const SCENES: &[&str] = &[
        "slow", "carve_l", "carve_r", "tuck", "plow", "hockey", "skate", "pole", "air0", "air1",
        "switch", "landing",
    ];

    fn bone_len(a: J, b: J) -> f32 {
        use J::*;
        match (a, b) {
            (Pelvis, Waist) => PELVIS_TO_WAIST,
            (Waist, Chest) => WAIST_TO_CHEST,
            (Chest, Neck) => CHEST_TO_NECK,
            (Neck, Head) => NECK_TO_HEAD,
            (Pelvis, HipL | HipR) => HIP_HALF_WIDTH,
            (HipL, KneeL) | (HipR, KneeR) => THIGH,
            (KneeL, AnkleL) | (KneeR, AnkleR) => SHIN,
            (AnkleL, HeelL) | (AnkleR, HeelR) => ANKLE_ABOVE_SKI.hypot(HEEL_BACK),
            (AnkleL, ToeL) | (AnkleR, ToeR) => ANKLE_ABOVE_SKI.hypot(TOE_FORWARD),
            (HeelL, ToeL) | (HeelR, ToeR) => HEEL_BACK + TOE_FORWARD,
            (Chest, ShoulderL | ShoulderR) => SHOULDER_HALF_WIDTH,
            (ShoulderL, ElbowL) | (ShoulderR, ElbowR) => UPPER_ARM,
            (ElbowL, WristL) | (ElbowR, WristR) => FOREARM,
            (WristL, HandL) | (WristR, HandR) => HAND,
            _ => panic!("unlisted bone {a:?}-{b:?}"),
        }
    }

    fn base() -> Skier {
        let mut s = Skier::default();
        s.position = Vec3::new(20.0, 0.0, -250.0);
        s.rotation = Quat::IDENTITY;
        s.velocity = Vec3::ZERO;
        s.grounded = true;
        s
    }

    /// Drives `s`/`c` for scenario `name` at time `t`.
    fn scene(name: &str, t: f32, s: &mut Skier, c: &mut SkiControls) {
        *s = base();
        s.rotation = Quat::IDENTITY;
        *c = SkiControls {
            grab: c.grab,
            trick_side: c.trick_side,
            ..Default::default()
        };
        match name {
            "slow" => s.velocity = Vec3::new(0.0, 0.0, -2.0),
            "carve_l" | "carve_r" => {
                let k = if name == "carve_l" { -1.0 } else { 1.0 };
                s.velocity = Vec3::new(k * 3.0, 0.0, -12.0);
                s.edge = 0.6 * k;
                s.lean = 0.7 * k;
                c.steer = k;
            }
            "tuck" => {
                s.velocity = Vec3::new(0.0, 0.0, -25.0);
                s.tuck = 1.0;
                c.tuck = true;
            }
            "plow" => {
                s.velocity = Vec3::new(0.0, 0.0, -4.0);
                s.plow = 1.0;
            }
            "hockey" => {
                s.rotation = Quat::from_rotation_y(1.2);
                s.velocity = Vec3::new(0.0, 0.0, -14.0);
                s.skid = 1.0;
                s.edge = 0.8;
                s.lean = 0.5;
            }
            "skate" => {
                s.velocity = Vec3::new(0.0, 0.0, -3.0);
                s.skate = 1.0;
                s.skate_phase = t * 9.0;
            }
            "pole" => {
                s.velocity = Vec3::new(0.0, 0.0, -7.0);
                s.pole = 1.0;
                s.pole_phase = t * 7.0;
            }
            "air0" | "air1" => {
                s.position.y = 100.0;
                s.grounded = false;
                s.air_time = 0.5 + t;
                s.preload = if name == "air1" { 1.0 } else { 0.0 };
                s.rotation = Quat::from_euler(EulerRot::YXZ, t * 2.0, t * 1.3, t * 0.7);
            }
            "switch" => {
                s.velocity = Vec3::new(0.0, 0.0, 6.0);
                s.switch = true;
            }
            "landing" => {
                s.compression = 1.0;
                s.impact = 8.0;
            }
            _ => unreachable!(),
        }
    }

    fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        (p - (a + ab * t)).length()
    }

    fn ski_dist(p: &SkierPose, hand: J, ski: usize) -> f32 {
        let (tail, bind, tip) = if ski == 0 {
            (J::TailL, J::BindL, J::TipL)
        } else {
            (J::TailR, J::BindR, J::TipR)
        };
        seg_dist(p[hand], p[tail], p[bind]).min(seg_dist(p[hand], p[bind], p[tip]))
    }

    fn check(p: &SkierPose, what: &str) {
        for (i, v) in p.p.iter().enumerate() {
            assert!(v.is_finite(), "{what}: joint {i} not finite");
        }
        for &(a, b) in BONES {
            let l = (p[a] - p[b]).length();
            assert!(
                (l - bone_len(a, b)).abs() < 1e-3,
                "{what}: bone {a:?}-{b:?} is {l}"
            );
        }
        let boots = [
            (J::AnkleL, J::HeelL, J::ToeL, J::BindL, J::TipL, J::TailL),
            (J::AnkleR, J::HeelR, J::ToeR, J::BindR, J::TipR, J::TailR),
        ];
        for (i, &(ankle, heel, toe, bind, tip, tail)) in boots.iter().enumerate() {
            let up = p.ski_up[i];
            assert!((up.length() - 1.0).abs() < 1e-3, "{what}: ski_up length");
            let f = (p[toe] - p[heel]).normalize();
            assert!(
                f.dot(up).abs() < 1e-3,
                "{what}: ski axis not perpendicular to ski_up"
            );
            let eps = 1e-3;
            assert!(
                (p[ankle] - p[bind] - up * ANKLE_ABOVE_SKI).length() < eps,
                "{what}: ankle"
            );
            assert!(
                (p[heel] - (p[bind] - f * HEEL_BACK)).length() < eps,
                "{what}: heel"
            );
            assert!(
                (p[toe] - (p[bind] + f * TOE_FORWARD)).length() < eps,
                "{what}: toe"
            );
            assert!(
                (p[tail] - (p[bind] - f * SKI_BACK)).length() < eps,
                "{what}: tail"
            );
            let tip_ref = p[bind] + f * SKI_FRONT + up * SKI_TIP_RISE;
            assert!((p[tip] - tip_ref).length() < eps, "{what}: tip");
        }
        for (hand, top, tip) in [
            (J::HandL, J::PoleTopL, J::PoleTipL),
            (J::HandR, J::PoleTopR, J::PoleTipR),
        ] {
            let shaft = (p[tip] - p[top]).normalize();
            assert!(
                ((p[tip] - p[top]).length() - POLE_LENGTH).abs() < 1e-3,
                "{what}: pole length"
            );
            assert!(
                ((p[hand] - p[top]).length() - POLE_GRIP).abs() < 1e-3,
                "{what}: grip"
            );
            assert!(
                (p[hand] - (p[top] + shaft * POLE_GRIP)).length() < 1e-3,
                "{what}: hand off shaft"
            );
            let ground = terrain_height(p[tip].x, p[tip].z);
            assert!(
                p[tip].y >= ground - 0.01,
                "{what}: pole tip {} below snow {ground}",
                p[tip].y
            );
        }
    }

    #[test]
    fn every_grab_and_scene_keeps_exact_rig() {
        for &name in SCENES {
            for &g in Grab::ALL {
                for side in [-1.0, 1.0] {
                    let mut a = SkiAnimation::default();
                    let (mut s, mut c) = (
                        base(),
                        SkiControls {
                            grab: g,
                            trick_side: side,
                            ..Default::default()
                        },
                    );
                    for step in 0..300 {
                        let t = step as f32 * DT;
                        scene(name, t, &mut s, &mut c);
                        a.update(&s, &c, DT);
                        if step % 20 == 19 {
                            let p = solve(&s, &a);
                            let what = format!("{name} {g:?} side {side} t={t:.2}");
                            check(&p, &what);
                            if !name.starts_with("air") && name != "skate" {
                                for bind in [J::BindL, J::BindR] {
                                    let y = p[bind].y;
                                    assert!(
                                        (SKI_THICKNESS - 0.005..SKI_THICKNESS + 0.06).contains(&y),
                                        "{what}: ski bind {y} off the snow"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn neutral_stance_stands_on_the_snow() {
        let a = SkiAnimation::default();
        let mut s = base();
        s.velocity = Vec3::new(0.0, 0.0, -1.0);
        let p = solve(&s, &a);
        check(&p, "neutral");
        for (bind, tip, tail) in [(J::BindL, J::TipL, J::TailL), (J::BindR, J::TipR, J::TailR)] {
            assert!(
                (p[bind].y - SKI_THICKNESS).abs() < 0.03,
                "bind {}",
                p[bind].y
            );
            assert!(
                (p[tail].y - SKI_THICKNESS).abs() < 0.03,
                "tail {}",
                p[tail].y
            );
            assert!(
                (p[tip].y - SKI_THICKNESS - SKI_TIP_RISE).abs() < 0.03,
                "tip {}",
                p[tip].y
            );
        }
        let h = p[J::Head].y;
        assert!((1.55..=1.8).contains(&h), "standing head height {h}");
        // Skis sit STANCE_WIDTH apart and point downhill.
        assert!(((p[J::BindL] - p[J::BindR]).length() - STANCE_WIDTH).abs() < 0.01);
        assert!((p[J::TipL] - p[J::TailL]).dot(Vec3::NEG_Z) > 1.4);
    }

    #[test]
    fn tuck_lowers_the_head() {
        let mut a = SkiAnimation::default();
        let (mut s, mut c) = (base(), SkiControls::default());
        for step in 0..360 {
            scene("tuck", step as f32 * DT, &mut s, &mut c);
            a.update(&s, &c, DT);
        }
        let p = solve(&s, &a);
        let h = p[J::Head].y;
        assert!((0.8..=1.3).contains(&h), "tuck head height {h}");
        // Poles lie along the back: nearly horizontal.
        let shaft = (p[J::PoleTopL] - p[J::PoleTipL]).normalize();
        assert!(shaft.y.abs() < 0.6, "tucked pole slope {}", shaft.y);
    }

    #[test]
    fn pole_plant_sweeps_back_over_the_snow() {
        let mut a = SkiAnimation::default();
        let (mut s, mut c) = (base(), SkiControls::default());
        let mut tip_z = Vec::new();
        for step in 0..360 {
            scene("pole", step as f32 * DT, &mut s, &mut c);
            s.pole_phase = 0.0;
            a.update(&s, &c, DT);
        }
        let planted = solve(&s, &a);
        assert!(
            planted[J::PoleTipL].y < 0.1,
            "planted tip {}",
            planted[J::PoleTipL].y
        );
        assert!(
            planted[J::PoleTipL].z < s.position.z - 0.2,
            "planted tip not ahead"
        );
        for k in 0..6 {
            s.pole_phase = 0.38 * TAU * k as f32 / 5.0 * 0.999;
            for _ in 0..120 {
                a.update(&s, &c, DT);
            }
            tip_z.push(solve(&s, &a)[J::PoleTipL].z);
        }
        assert!(
            tip_z.windows(2).all(|w| w[1] > w[0] - 0.02),
            "tips sweep back: {tip_z:?}"
        );
        assert!(
            tip_z[5] > s.position.z,
            "tip ends behind the skier: {tip_z:?}"
        );
    }

    #[test]
    fn held_grab_brings_hands_to_the_skis() {
        let mut fails = Vec::new();
        for &g in &[
            Grab::Mute,
            Grab::Safety,
            Grab::Japan,
            Grab::Tail,
            Grab::Tip,
            Grab::TruckDriver,
        ] {
            for side in [-1.0, 1.0] {
                let mut a = SkiAnimation::default();
                let (mut s, mut c) = (
                    base(),
                    SkiControls {
                        grab: g,
                        trick_side: side,
                        ..Default::default()
                    },
                );
                let mut p = solve(&s, &a);
                for step in 0..180 {
                    scene("air0", step as f32 * DT, &mut s, &mut c);
                    a.update(&s, &c, DT);
                    p = solve(&s, &a);
                }
                assert_eq!(a.grab, g);
                check(&p, &format!("held {g:?}"));
                let d = |hand: J| ski_dist(&p, hand, 0).min(ski_dist(&p, hand, 1));
                let (dl, dr) = (d(J::HandL), d(J::HandR));
                let ok = if g == Grab::TruckDriver {
                    dl < 0.1 && dr < 0.1
                } else {
                    dl.min(dr) < 0.1
                };
                if !ok {
                    fails.push(format!(
                        "{g:?} side {side}: hands {dl:.3} {dr:.3} m from skis"
                    ));
                }
            }
        }
        assert!(fails.is_empty(), "{fails:#?}");
    }

    /// Largest per-tick second difference of any joint (metres per tick^2) while `drive` steers
    /// the real physics -> animation -> rig chain at 120 Hz. `drive` returns true when it teleports.
    fn worst_snap(
        ticks: usize,
        mut drive: impl FnMut(usize, &mut Skier, &mut SkiControls) -> bool,
    ) -> f32 {
        let (mut s, mut a) = (Skier::default(), SkiAnimation::default());
        let mut hist: Vec<SkierPose> = Vec::new();
        let mut worst = 0.0_f32;
        for i in 0..ticks {
            let mut c = SkiControls::default();
            if drive(i, &mut s, &mut c) {
                hist.clear();
            }
            a.update(&s, &c, DT);
            s.step(&c, DT);
            if s.crash.is_some() {
                break;
            }
            hist.push(solve(&s, &a));
            if let [.., p0, p1, p2] = hist.as_slice() {
                for j in 0..JOINTS {
                    worst = worst.max((p2.p[j] - 2.0 * p1.p[j] + p0.p[j]).length());
                }
            }
        }
        worst
    }

    #[test]
    fn poses_never_snap_between_ticks() {
        let flat = |i: usize, s: &mut Skier, v: f32| {
            if i == 0 {
                s.reset_at(-40.0, 0.0, 0.0, v);
            }
        };
        // Before smoothing these were 0.04-0.37 m/tick^2 (edge changes, hockey-stop end, pole plant,
        // jump pop). Gravity alone is 0.7 mm/tick^2; a pop's launch impulse is ~0.03.
        let cases: [(&str, f32, f32); 5] = [
            (
                "carve edge changes",
                worst_snap(600, |i, s, c| {
                    flat(i, s, 14.0);
                    c.steer = if (i / 120) % 2 == 0 { 1.0 } else { -1.0 };
                    false
                }),
                0.02,
            ),
            (
                "hockey stop",
                worst_snap(600, |i, s, c| {
                    flat(i, s, 15.0);
                    c.brake = f32::from(u8::from(i > 60));
                    false
                }),
                0.03,
            ),
            (
                "skate into double pole",
                worst_snap(1200, |i, s, c| {
                    flat(i, s, 0.0);
                    c.push = 1.0;
                    false
                }),
                0.02,
            ),
            (
                "push, glide, push again",
                worst_snap(2880, |i, s, c| {
                    flat(i, s, 0.0);
                    c.push = f32::from(u8::from((i / 720) % 2 == 0));
                    false
                }),
                0.02,
            ),
            (
                "jump pops and landings",
                worst_snap(900, |i, s, c| {
                    flat(i, s, 10.0);
                    c.jump = (i % 150) < 50 && i > 20;
                    false
                }),
                0.06,
            ),
        ];
        for (name, worst, limit) in cases {
            assert!(
                worst < limit,
                "{name}: {:.0} mm/tick^2 >= {:.0}",
                worst * 1e3,
                limit * 1e3
            );
        }
    }

    #[test]
    fn kicker_landings_settle_instead_of_teleporting() {
        // The F6 loop lands pitched 17 deg tail-first; the old snap moved the tips 0.6 m in a tick.
        let mut demo = crate::ski::demo::SkiDemo::default();
        let worst = worst_snap(120 * 70, |i, s, c| {
            if i == 0 {
                demo.enabled = true;
                demo.begin_run(s);
            }
            demo.drive(s, c, DT)
        });
        // What remains is the normal velocity stopping at the snow (~13 m/s * 1/120 s).
        assert!(worst < 0.16, "{:.0} mm/tick^2", worst * 1e3);
    }

    /// Poses of a 12 m/s flat run that steers right from tick 60, one per tick.
    fn right_turn_entry(ticks: usize) -> Vec<(Skier, SkierPose)> {
        let (mut s, mut a) = (Skier::default(), SkiAnimation::default());
        let mut out = Vec::new();
        for i in 0..ticks {
            if i == 0 {
                s.reset_at(-40.0, 0.0, 0.0, 12.0);
            }
            let c = SkiControls {
                steer: f32::from(u8::from(i >= 60)),
                ..Default::default()
            };
            a.update(&s, &c, DT);
            s.step(&c, DT);
            out.push((s.clone(), solve(&s, &a)));
        }
        out
    }

    #[test]
    fn turn_entry_plants_the_inside_pole_and_turns_the_chest_against_the_skis() {
        let run = right_turn_entry(240);
        let tip_y = run.iter().map(|(_, p)| p[J::PoleTipR].y);
        let low = tip_y.clone().fold(f32::MAX, f32::min);
        let high = tip_y.skip(150).fold(0.0_f32, f32::max);
        assert!(low < 0.08, "inside pole never touches the snow ({low})");
        assert!(high > 0.2, "inside pole stays on the snow ({high})");
        // Skis turn right (negative yaw rate); the chest faces further left than the skis.
        let (s, p) = &run[120];
        let fwd = s.rotation * Vec3::NEG_Z;
        assert!(fwd.cross(p.facing).y > 0.1, "chest not counter-rotated");
    }

    #[test]
    fn head_keeps_its_gaze_while_the_chest_twists() {
        let run = right_turn_entry(240);
        let (s, p) = &run[120];
        let fwd = s.rotation * Vec3::NEG_Z;
        assert!(
            p.gaze.dot(p.facing) < 1.0 - 1e-3 && p.gaze.is_normalized(),
            "gaze follows the chest"
        );
        let off = |v: Vec3| fwd.cross(v).y.abs();
        assert!(
            off(p.gaze) < off(p.facing),
            "head turned further than the chest"
        );
    }

    #[test]
    fn touchdown_compresses_in_about_a_fifth_of_a_second_and_recovers_in_half_a_second() {
        let (mut s, mut a) = (Skier::default(), SkiAnimation::default());
        let (mut touch, mut land, mut gap) = (None, Vec::new(), Vec::new());
        let mut was_air = false;
        for i in 0..600 {
            if i == 0 {
                s.reset_at(-40.0, 0.0, 0.0, 10.0);
            }
            let c = SkiControls {
                jump: i > 20 && i < 70,
                ..Default::default()
            };
            a.update(&s, &c, DT);
            s.step(&c, DT);
            if was_air && s.grounded && touch.is_none() {
                touch = Some(i);
            }
            was_air |= !s.grounded;
            if touch.is_some() {
                land.push(a.w.land);
                let p = solve(&s, &a);
                gap.push((p[J::BindL] - p[J::BindR]).length());
                if land.len() > 150 {
                    break;
                }
            }
        }
        let peak = land.iter().copied().fold(0.0_f32, f32::max);
        let at = land.iter().position(|&l| l == peak).unwrap() as f32 * DT;
        assert!(
            (0.12..0.30).contains(&at),
            "max compression after {at:.2} s"
        );
        assert!(peak > 0.2, "no compression on touchdown ({peak})");
        let widest = gap.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            widest > STANCE_WIDTH + 0.02,
            "feet stay together ({widest})"
        );
        let after = land[(0.6 / DT) as usize];
        assert!(after < 0.35 * peak, "still {after} of {peak} after 0.6 s");
        assert!(land.iter().all(|&l| l > -0.1 * peak), "rebound overshoots");
    }

    /// Skates from rest on flat snow for `secs`, handing the per-tick skier, weights and pose to `f`.
    fn skate_run(secs: f32, mut f: impl FnMut(&Skier, &SkiAnimation, &SkierPose)) {
        let (mut s, mut a) = (Skier::default(), SkiAnimation::default());
        s.reset_at(-40.0, 0.0, 0.0, 0.0);
        let c = SkiControls {
            push: 1.0,
            ..Default::default()
        };
        for _ in 0..(secs / DT) as usize {
            a.update(&s, &c, DT);
            s.step(&c, DT);
            f(&s, &a, &solve(&s, &a));
        }
    }

    #[test]
    fn skate_ski_hands_over_cleanly_between_strokes() {
        for p in (0..=20).map(|k| k as f32 / 20.0) {
            let (push, glide) = (skate_ski(true, p, 0.1, 0.2), skate_ski(false, p, 0.1, 0.2));
            assert!((push.weight + glide.weight - 1.0).abs() < 1e-5, "p={p}");
            assert!(glide.lift == 0.0 && glide.out <= 0.0);
            assert!(
                push.lift == 0.0 || push.weight < 0.05,
                "p={p} lifts a loaded ski"
            );
        }
        // The end of one stroke is the start of the next with the roles swapped.
        let (end, start) = (
            skate_ski(true, 1.0, 0.1, 0.2),
            skate_ski(false, 0.0, 0.1, 0.2),
        );
        assert!(end.weight < 1e-5 && start.weight < 1e-5);
        assert!((end.out - start.out).abs() < 1e-5 && end.lift < 1e-5 && end.edge < 1e-5);
        assert!((end.yaw - start.yaw).abs() < 1e-5);
        let (end, start) = (
            skate_ski(false, 1.0, 0.1, 0.2),
            skate_ski(true, 0.0, 0.1, 0.2),
        );
        assert!((end.weight - 1.0).abs() < 1e-5 && (start.weight - 1.0).abs() < 1e-5);
        assert!((end.yaw - start.yaw).abs() < 1e-5 && (end.out - start.out).abs() < 1e-5);
    }

    #[test]
    fn skating_glides_on_a_flat_ski_lifts_the_recovering_one_and_shifts_the_hips() {
        let (mut lifted, mut lat) = (0.0_f32, (0.0_f32, 0.0_f32));
        let mut prev: Option<(SkierPose, Vec3)> = None;
        skate_run(5.0, |s, _, p| {
            let (pusher, stroke) = skate_stroke(s.skate_phase);
            let (glider, bind) = (1 - pusher, [J::BindL, J::BindR]);
            if s.speed() > 1.0 {
                // The ski that carries the weight lies flat on the snow ...
                let gy = p[bind[glider]].y;
                if stroke > 0.6 {
                    assert!(
                        (gy - SKI_THICKNESS).abs() < 0.004,
                        "glide ski {gy} off the snow at {stroke}"
                    );
                }
                // ... and runs along its own axis: it has no sideways speed over the snow.
                if let (Some((q, _)), true) = (&prev, stroke > 0.5) {
                    let v = (p[bind[glider]] - q[bind[glider]]) / DT;
                    let axis = (p[[J::TipL, J::TipR][glider]] - p[[J::TailL, J::TailR][glider]])
                        .normalize();
                    let side = axis.cross(Vec3::Y).normalize();
                    assert!(v.dot(side).abs() < 0.2, "glide ski slides {}", v.dot(side));
                }
                lifted = lifted.max(p[bind[pusher]].y - SKI_THICKNESS);
                let l = (p[J::Pelvis] - s.position).dot(s.rotation * Vec3::X);
                lat = (lat.0.min(l), lat.1.max(l));
            }
            prev = Some((*p, s.position));
        });
        assert!(lifted > 0.06, "recovering ski never lifts ({lifted})");
        assert!(lat.1 - lat.0 > 0.3, "hips stay put: {lat:?}");
    }

    #[test]
    fn the_skis_form_a_wide_v_with_tails_close_at_the_end_of_each_push() {
        let mut seen = 0;
        skate_run(5.0, |s, _, p| {
            let (_, stroke) = skate_stroke(s.skate_phase);
            if !(2.5..3.8).contains(&s.speed()) || !(0.38..0.44).contains(&stroke) {
                return;
            }
            seen += 1;
            let across = |j: J, k: J| (p[j] - p[k]).dot(s.rotation * Vec3::X).abs();
            let (bind, tip, tail) = (
                across(J::BindL, J::BindR),
                across(J::TipL, J::TipR),
                across(J::TailL, J::TailR),
            );
            assert!(
                (0.42..0.65).contains(&bind),
                "push foot {bind} m from the glide foot"
            );
            assert!(
                (0.7..1.0).contains(&tip) && tail < 0.4,
                "tips {tip} tails {tail}"
            );
            // Each ski points out of the travel line by about 12 degrees.
            for (tip, tail) in [(J::TipL, J::TailL), (J::TipR, J::TailR)] {
                let a = p[tip] - p[tail];
                let out = a
                    .dot(s.rotation * Vec3::X)
                    .abs()
                    .atan2(-a.dot(s.rotation * Vec3::Z));
                assert!((0.14..0.30).contains(&out), "ski yaw {out}");
            }
            // The glide knee is bent about 50-70 degrees, the push leg nearly straight.
            let flex = |h: J, k: J, a: J| (p[k] - p[h]).angle_between(p[a] - p[k]).to_degrees();
            let (kl, kr) = (
                flex(J::HipL, J::KneeL, J::AnkleL),
                flex(J::HipR, J::KneeR, J::AnkleR),
            );
            let (glide, push) = if skate_stroke(s.skate_phase).0 == 0 {
                (kr, kl)
            } else {
                (kl, kr)
            };
            assert!(
                (48.0..72.0).contains(&glide) && push < 40.0,
                "knees glide {glide} push {push}"
            );
        });
        assert!(seen >= 6, "only {seen} samples");
    }

    #[test]
    fn first_strokes_are_upright_then_the_skier_leans_into_the_rhythm() {
        let (mut first, mut late) = ((90.0_f32, 0.0_f32), (90.0_f32, 0.0_f32));
        skate_run(3.5, |s, a, p| {
            let t = (p[J::Neck] - p[J::Pelvis])
                .angle_between(s.rotation * Vec3::Y)
                .to_degrees();
            let slot = if a.w.clock < 1.0 {
                &mut first
            } else if a.w.clock > 2.5 {
                &mut late
            } else {
                return;
            };
            *slot = (slot.0.min(t), slot.1.max(t));
        });
        assert!(first.1 < 28.0, "start leans {first:?}");
        assert!(
            late.1 > first.1 + 8.0 && late.1 < 55.0,
            "{first:?} -> {late:?}"
        );
    }
}
