//! Authored, composable rider/bike animation layers. The solved pose feeds the physics through
//! `scene::collision_pose`; root pitch/roll/yaw (full flips and barrels included) is physical and
//! lives in `Bike`, never here. None of it is claimed to be the retail game's tracks or timing.
//!
//! Layers (all smoothed, all independent unless a priority rule below says otherwise):
//! * body posture: seated/standing/brake/turn/manual/nose-manual/hop preload+takeoff, air tuck then
//!   extension, landing compression (impact captured before contact) and recovery;
//! * hand release (`HandTrick`), foot release (`LegTrick`) and bike motion (`BikeTrick`), each with
//!   per-kind blend weights, per-side release weights and bar/tail/crank spin accumulators.
//!
//! Name-derived families: OneHand, NoHand, TuckNoHand, Barspin, TireGrab, SeatGrab, Toboggan;
//! OneFoot, NoFoot, CanCan, NoFootCan, Superman, Tailwhip, NacNac, Indian; Whip, Table, XUp,
//! Turndown, EuroTable, Invert, Crankflip. Poses are authored from the trick names. Superman and
//! tailwhip timing (release order, swing, return and catch) is tuned against measurements of the
//! decoded retail clips; the poses are ours, so parity is unverified. Not implemented: BIKEFLIP,
//! BRIFLIP, HDFLIP, GRIZZAIR, CANONBALL, TSUNAMI and any PS/discipline binding.
//!
//! Priority rules (encoded in [`AnimationState::update`], reported through `note`):
//! 1. Air-only: tricks are refused on the ground and during the first `AIR_MIN` seconds of flight.
//! 2. Landing: new holds stop `regrab_lead` s before the predicted contact (longer for Superman
//!    and tailwhip, whose returns take longer); hands/feet regrab and bar/tail/crank spins finish
//!    their revolution (or unwind) before contact. Nothing else is corrected for the landing: body
//!    rotation is whatever the physics produced.
//! 3. Barspin owns the front assembly: Table, X-up and Euro table are suppressed while it runs.
//! 4. Invert, Euro table and Crankflip force both feet off the pedals; an explicit `LegTrick` keeps
//!    authority over the leg pose (the forced leg pose is used only when the leg trick is `None`).
//! 5. Body offsets of hand and leg tricks add; the free limb poses stay independent.
//! 6. A bar/tail/crank spin that is still unfinished on the ground is finished quickly (`spin-recover`).

use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use crate::bike::{
    Bike, BikeTrick, Controls, HandTrick, LegTrick, SUSPENSION_REST, WHEEL_RADIUS, WHEELBASE,
    terrain_height,
};
use crate::scene::smooth01;

const GRAVITY: f32 = 9.81;
const MAX_DT: f32 = 1.0 / 15.0;
/// Seconds of flight before air-only tricks may start.
const AIR_MIN: f32 = 0.05;
/// Tricks stop being held, and limbs regrab, this long before the predicted contact.
const REGRAB_LEAD: f32 = 0.45;
/// Hands/feet always head back to the bike once contact is this close.
const LAST_REGRAB: f32 = 0.18;
/// A finishing spin keeps hands/feet free until this many radians remain.
const BUSY_OWED: f32 = 0.8;
/// A finishing tailwhip starts the foot catch this many radians before the frame comes round.
const TAIL_CATCH: f32 = 1.75;
/// Spins aim to be complete this long before contact.
const LAND_MARGIN: f32 = 0.15;
const RELEASE_OMEGA: f32 = 19.0;
/// Hands/feet back onto grips/pedals in about 0.3 s; faster when contact is closer than that.
const REGRAB_OMEGA: f32 = 16.0;
/// Tailwhip catch: the trick-side foot is back in about 0.2 s, the other in about 0.4 s.
const CATCH_NEAR_OMEGA: f32 = 22.0;
const CATCH_FAR_OMEGA: f32 = 12.0;
const WEIGHT_OMEGA: f32 = 19.0;
/// Superman: stretch-out clock limit and return duration, s.
const SUP_IN: f32 = 1.1;
const SUP_OUT: f32 = 0.63;
/// The bike swings nose-up about the gripped bars while the rider lies flat, rad.
const SUP_PITCH: f32 = 1.55;
const SPIN_ACCEL: f32 = 14.0;
const SETTLE: f32 = 0.01;
const MIN_FINISH_RATE: f32 = 3.0;
const RECOVER_RATE: f32 = 12.0;
const MAX_FINISH_RATE: f32 = 32.0;
const BAR_RATE: f32 = 16.0;
const TAIL_RATE: f32 = 12.6;
const CRANK_RATE: f32 = 13.0;
const LAND_IN: f32 = 20.0;
const LAND_OUT: f32 = 12.0;
const LAND_HOLD: f32 = 0.22;
const LAND_CROUCH: f32 = 4.2;
const SURGE_OMEGA: f32 = 10.0;
const SURGE_ZETA: f32 = 0.45;
const SURGE_GAIN: f32 = 0.03;
/// Suspension compression of the settled bike, m.
const SAG_REST: f32 = 0.05;
const TAKEOFF_T: f32 = 0.16;
/// Time constant of the low-passed ground acceleration that tells climbing speed from top speed, s.
const ACCEL_LP: f32 = 0.4;

/// Deterministic per-landing jitter in -1..1, from a hash of the landing count and `salt`, so no
/// two landings are absorbed alike but replays and tests stay exact.
pub(crate) fn jitter(count: u32, salt: u32) -> f32 {
    let mut x = count.wrapping_mul(0x9E37_79B9) ^ salt.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32 * 2.0 - 1.0
}

// Free-pose kinds. Public tricks first, then poses forced by a bike trick.
pub(crate) const HAND_KINDS: usize = 7;
pub(crate) const H_ONE: usize = 0;
pub(crate) const H_NO: usize = 1;
pub(crate) const H_TUCK: usize = 2;
pub(crate) const H_BAR: usize = 3;
pub(crate) const H_TIRE: usize = 4;
pub(crate) const H_SEAT: usize = 5;
pub(crate) const H_TOBO: usize = 6;
const HAND_REPORT: [HandTrick; HAND_KINDS] = [
    HandTrick::OneHand,
    HandTrick::NoHand,
    HandTrick::TuckNoHand,
    HandTrick::Barspin,
    HandTrick::TireGrab,
    HandTrick::SeatGrab,
    HandTrick::Toboggan,
];
/// Kinds that release only the hand on the trick side.
const HAND_ONE_SIDED: [bool; HAND_KINDS] = [true, false, false, false, true, true, false];
/// Body offsets (crouch, back) while the kind is fully released.
const HAND_BODY: [(f32, f32); HAND_KINDS] = [
    (0.0, 0.0),
    (0.0, 0.0),
    (0.5, -0.3),
    (0.0, 0.0),
    (0.6, -0.5),
    (0.0, 0.7),
    (0.2, 0.9),
];

pub(crate) const LEG_KINDS: usize = 11;
pub(crate) const L_ONE: usize = 0;
pub(crate) const L_NO: usize = 1;
pub(crate) const L_CAN: usize = 2;
pub(crate) const L_NOCAN: usize = 3;
pub(crate) const L_SUPER: usize = 4;
pub(crate) const L_TAIL: usize = 5;
pub(crate) const L_NAC: usize = 6;
pub(crate) const L_INDIAN: usize = 7;
/// Invert: legs hang free.
pub(crate) const L_DANGLE: usize = 8;
/// Euro table: legs thrown opposite the table side.
pub(crate) const L_COUNTER: usize = 9;
/// Crankflip: knees lifted clear of the spinning cranks.
pub(crate) const L_LIFT: usize = 10;
const LEG_PUBLIC: usize = 8;
const LEG_REPORT: [LegTrick; LEG_PUBLIC] = [
    LegTrick::OneFoot,
    LegTrick::NoFoot,
    LegTrick::CanCan,
    LegTrick::NoFootCan,
    LegTrick::Superman,
    LegTrick::Tailwhip,
    LegTrick::NacNac,
    LegTrick::Indian,
];
const LEG_ONE_SIDED: [bool; LEG_KINDS] = [
    true, false, true, false, false, false, true, false, false, false, false,
];
const LEG_BODY: [(f32, f32); LEG_KINDS] = [
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (2.0, 0.6),
    (0.0, 0.2),
    (0.1, 0.3),
    (-0.05, 0.5),
    (0.0, 0.0),
    (0.2, 0.0),
];

pub(crate) const BIKE_KINDS: usize = 7;
pub(crate) const B_WHIP: usize = 0;
pub(crate) const B_TABLE: usize = 1;
pub(crate) const B_XUP: usize = 2;
pub(crate) const B_TURN: usize = 3;
pub(crate) const B_EURO: usize = 4;
pub(crate) const B_INVERT: usize = 5;
pub(crate) const B_CRANK: usize = 6;
const BIKE_REPORT: [BikeTrick; BIKE_KINDS] = [
    BikeTrick::Whip,
    BikeTrick::Table,
    BikeTrick::XUp,
    BikeTrick::Turndown,
    BikeTrick::EuroTable,
    BikeTrick::Invert,
    BikeTrick::Crankflip,
];

fn hand_index(t: HandTrick) -> Option<usize> {
    HAND_REPORT.iter().position(|&k| k == t)
}

fn leg_index(t: LegTrick) -> Option<usize> {
    LEG_REPORT.iter().position(|&k| k == t)
}

fn bike_index(t: BikeTrick) -> Option<usize> {
    BIKE_REPORT.iter().position(|&k| k == t)
}

/// Holds stop this long before contact so the return fits: Superman's return takes `SUP_OUT`,
/// and a tailwhip's second foot is caught about 0.4 s after the frame comes round.
fn regrab_lead(leg: LegTrick) -> f32 {
    match leg {
        LegTrick::Superman => SUP_OUT + 0.12,
        LegTrick::Tailwhip => 0.65,
        _ => REGRAB_LEAD,
    }
}

/// Critically damped spring (exact step): `x` reaches `target` without overshoot and with
/// continuous velocity `v`, so a changing target never snaps the pose. `omega` is rad/s.
fn spring(x: &mut f32, v: &mut f32, target: f32, omega: f32, dt: f32) {
    let e = (-omega * dt).exp();
    let d = *x - target;
    let t = *v + omega * d;
    *x = target + (d + t * dt) * e;
    *v = (*v - omega * t * dt) * e;
    if (target - *x).abs() < 1e-3 && v.abs() < 1e-2 {
        *x = target;
        *v = 0.0;
    }
}

/// A 0..1 blend weight on a spring; a retargeted spring may carry momentum past the ends.
fn weight(x: &mut f32, v: &mut f32, target: f32, omega: f32, dt: f32) {
    spring(x, v, target, omega, dt);
    if !(0.0..=1.0).contains(x) {
        *x = x.clamp(0.0, 1.0);
        *v = 0.0;
    }
}

/// Angle accumulator in `[0, TAU)` that always ends on a whole turn.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Spin {
    pub angle: f32,
    pub rate: f32,
}

impl Spin {
    pub fn settled(&self) -> bool {
        self.angle == 0.0 && self.rate == 0.0
    }

    /// Direction in which the spin will finish: on with the rotation while it is moving (a kicked
    /// frame or thrown bar never reverses), otherwise back the short way.
    fn dir(&self) -> f32 {
        if self.rate.abs() > 0.5 {
            self.rate.signum()
        } else if self.angle < PI {
            -1.0
        } else {
            1.0
        }
    }

    /// Radians still to turn to land on a whole revolution.
    pub fn owed(&self) -> f32 {
        if self.angle == 0.0 {
            0.0
        } else if self.dir() > 0.0 {
            TAU - self.angle
        } else {
            self.angle
        }
    }

    /// `hold` keeps accelerating towards `target` rate (airborne only). Otherwise finish the turn:
    /// momentum-limited and eased at the end, but fast enough to be done `LAND_MARGIN` before
    /// the contact `lead` seconds away, or at `RECOVER_RATE` once grounded.
    pub fn step(&mut self, target: f32, hold: bool, grounded: bool, lead: f32, dt: f32) {
        if hold && !grounded {
            self.rate += (target - self.rate) * (1.0 - (-SPIN_ACCEL * dt).exp());
            self.angle = self.angle + self.rate * dt;
            self.angle = self.angle.rem_euclid(TAU);
            if self.angle >= TAU {
                self.angle = 0.0;
            }
            return;
        }
        let owed = self.owed();
        if owed <= SETTLE {
            self.angle = 0.0;
            self.rate = 0.0;
            return;
        }
        let dir = self.dir();
        // Ease out: each 1/30 s removes about 40% of what is left (retail tailwhip catch).
        let mut r = self.rate.abs().max(MIN_FINISH_RATE).min(owed * 15.0 + 1.0);
        if grounded {
            r = r.max(RECOVER_RATE);
        } else {
            r = r.max(owed / ((lead - LAND_MARGIN) * 0.9).max(0.04));
        }
        r = r.min(MAX_FINISH_RATE);
        let step = r * dt;
        if owed - step <= SETTLE {
            self.angle = 0.0;
            self.rate = 0.0;
        } else {
            self.angle = (self.angle + dir * step).rem_euclid(TAU);
            self.rate = dir * r;
        }
    }
}

/// Everything the animation reads each frame; plain data so it can be driven without a `Bike`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Input {
    pub c: Controls,
    pub ride_stand: f32,
    /// Horizontal speed, m/s.
    pub speed: f32,
    pub vy: f32,
    pub pitch: f32,
    pub steering: f32,
    /// Bike lean, + left.
    pub roll: f32,
    pub suspension: [f32; 2],
    pub crank_phase: f32,
    pub grounded: [bool; 2],
    pub air_time: f32,
    /// Predicted seconds to the next wheel contact (infinite when none within the horizon).
    pub landing_in: f32,
    /// Predicted downward speed at that contact, m/s.
    pub impact: f32,
    /// Body-frame angular velocity, rad/s: pitch (+ nose up), yaw (+ left), roll (+ left).
    pub pitch_rate: f32,
    pub yaw_rate: f32,
    pub roll_rate: f32,
    /// Pitch relative to the ground under the bike, rad (+ nose up).
    pub ground_pitch: f32,
}

impl Input {
    pub fn from_bike(b: &Bike, c: &Controls) -> Self {
        let (landing_in, impact) = predict_landing(b);
        Self {
            c: *c,
            ride_stand: b.discipline.profile().ride_stand,
            speed: Vec2::new(b.velocity.x, b.velocity.z).length(),
            vy: b.velocity.y,
            pitch: b.pitch,
            steering: b.steering,
            roll: b.roll,
            suspension: b.suspension,
            crank_phase: b.crank_phase,
            grounded: b.grounded,
            air_time: b.air_time,
            landing_in,
            impact,
            pitch_rate: b.pitch_rate,
            yaw_rate: b.yaw_rate,
            roll_rate: b.roll_rate,
            ground_pitch: {
                let (s, c) = b.yaw.sin_cos();
                let (x, z, d) = (b.position.x, b.position.z, 0.5);
                let rise =
                    terrain_height(x - s * d, z - c * d) - terrain_height(x + s * d, z + c * d);
                b.pitch - (rise / (2.0 * d)).atan()
            },
        }
    }
}

/// Ballistic prediction of the first wheel contact: `(seconds, downward speed)`.
pub(crate) fn predict_landing(b: &Bike) -> (f32, f32) {
    let q = b.orientation();
    let hubs = [-WHEELBASE / 2.0, WHEELBASE / 2.0]
        .map(|z| b.position + q * Vec3::new(0.0, -SUSPENSION_REST, z));
    for step in 1..=60 {
        let t = step as f32 * 0.02;
        for hub in hubs {
            let p = hub + b.velocity * t - Vec3::Y * (0.5 * GRAVITY * t * t);
            if p.y - terrain_height(p.x, p.z) - WHEEL_RADIUS <= 0.0 {
                return (t, (GRAVITY * t - b.velocity.y).max(0.0));
            }
        }
    }
    (f32::INFINITY, 0.0)
}

/// Rider posture, smoothed bases plus the trick offsets.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Body {
    pub stand: f32,
    pub crouch: f32,
    pub back: f32,
    pub lateral: f32,
    /// 0..1 torso folded over the front wheel (tire grab).
    pub fold: f32,
    /// Bike lean into the corner, rad (+ left).
    pub turn: f32,
    /// 0..1 pedalling with the cranks driven.
    pub pedal: f32,
    /// Torso lag behind the bike's acceleration (adds to `back`).
    pub surge: f32,
    /// Smoothed mean suspension compression above the settled sag, m.
    pub sag: f32,
    /// Drawn crank angle (spring-followed, includes the coasting reposition); `None` = the bike's.
    pub crank: Option<f32>,
    /// Drawn bar angle; `None` = the bike's steering.
    pub steer: Option<f32>,
    /// 0..1 landing compression (arms and legs absorb).
    pub land: f32,
    /// Air body English `[throw, twist, drop, tuck]`, see [`AnimationState::english`].
    pub english: [f32; 4],
    /// 0..1 how hard the rider is driving the pedals; see [`AnimationState::effort`].
    pub effort: f32,
    /// 0..1 spun out at top speed in a sprint: low and forward.
    pub top: f32,
    /// Landing jolt `[side, fore, twist]`, see [`AnimationState::jolt`].
    pub jolt: [f32; 3],
}

/// Bike-assembly motion relative to the rider, as angles (see `scene::rig` for the geometry).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BikeLayer {
    /// About the roll axis (forward, blended into the bar line for table-likes); + rolls to `side`.
    pub roll: f32,
    /// About the bar line; negative lifts the rear.
    pub pitch: f32,
    /// About the vertical through the bars; + kicks the rear to the right.
    pub yaw: f32,
    /// Extra front assembly turn about the steering axis, + left.
    pub bar_turn: f32,
    /// 0..1 how much the roll axis follows the (turned) bars.
    pub table_t: f32,
    /// Where the hands sit along the bar (0 inner clamp .. 1 outer end); slides inboard for X-up.
    pub grip_slide: f32,
}

/// Superman channels, each 0..1 (see [`AnimationState::superman`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Superman {
    /// Straight legs swung from just ahead of straight down to straight back.
    pub legs: f32,
    /// Bike swung nose-up about the gripped bars.
    pub bike: f32,
    /// Arms straightened to full reach.
    pub arms: f32,
    /// Body laid flat behind the bars.
    pub body: f32,
    /// Knees folded on the way back to the pedals.
    pub fold: f32,
    /// Bike nose-up pitch (rad) the pose is levelled against, so the body lies flat in the world.
    pub level: f32,
}

/// Spring velocities for the smoothed weights and posture values of [`AnimationState`].
#[derive(Clone, Copy, Debug, Default)]
struct Rates {
    stand: f32,
    crouch: f32,
    back: f32,
    lateral: f32,
    hand_w: [f32; HAND_KINDS],
    leg_w: [f32; LEG_KINDS],
    bike_w: [f32; BIKE_KINDS],
    hand_rel: [f32; 2],
    foot_rel: [f32; 2],
    turn: f32,
    pedal: f32,
    attack: f32,
    pump: f32,
    surge: f32,
    crank: f32,
    steer: f32,
    english: [f32; 4],
    effort: f32,
    top: f32,
    jolt: [f32; 3],
}

#[derive(Resource, Clone, Debug)]
pub(crate) struct AnimationState {
    /// Body state: idle, pedal, sprint, coast, brake, turn, manual, nose-manual, hop, air.
    pub mode: &'static str,
    /// Timing: ground, preload, takeoff, air-tuck, air-extend, landing, recovery, spin-recover.
    pub phase: &'static str,
    /// Tricks currently being shown (still shown while releasing/regrabbing), else `None`.
    pub hand: HandTrick,
    pub leg: LegTrick,
    pub bike: BikeTrick,
    /// Why a request was changed (priority rules), else "".
    pub note: &'static str,
    /// Set by the game while paused, unfocused or crashed: nothing advances.
    pub frozen: bool,
    pub stand: f32,
    pub crouch: f32,
    pub back: f32,
    pub lateral: f32,
    pub land: f32,
    land_v: f32,
    land_hit: f32,
    /// Smoothed bike lean (+ left), pedalling weight, descent attack weight, suspension-driven
    /// absorption (m) and the longitudinal torso lag.
    pub turn: f32,
    pub pedal: f32,
    pub attack: f32,
    pub pump: f32,
    pub surge: f32,
    /// Body English in the air: `[throw, twist, drop, tuck]`. Throw: head and shoulders thrown
    /// back (+, backflip) or over the bars (-). Twist: shoulders and head leading a spin (+ left).
    /// Drop: shoulder dropped into a barrel roll (+ left). Tuck: 0..1 bike pulled in to rotate.
    pub english: [f32; 4],
    /// 0..1 how hard the rider drives the pedals: sprinting, or pedalling while speed climbs.
    pub effort: f32,
    /// 0..1 spun out at top speed in a sprint.
    pub top: f32,
    accel_lp: f32,
    /// Landing jolt, -1..1 each, scaled by the impact: `side` throws the body to the low side of
    /// a leaned touchdown (+ left), `fore` over the bars (+) or back as the rear wheel lands first,
    /// `twist` turns the shoulders (+ left). Set from the landing's direction plus a per-landing
    /// jitter, held while the landing compresses, then released on a spring.
    pub jolt: [f32; 3],
    jolt_hit: [f32; 3],
    landings: u32,
    /// Drawn (spring-followed, unwrapped) crank angle and bar angle; `None` before the first
    /// update, when the bike's own are used.
    crank_un: f32,
    crank_vis: f32,
    prev_crank: Option<f32>,
    steer_vis: Option<f32>,
    prev_speed: Option<f32>,
    impact: f32,
    since_land: f32,
    aloft: bool,
    /// Trick side latched at trick start: -1 left, +1 right.
    pub side: f32,
    pub hand_w: [f32; HAND_KINDS],
    pub leg_w: [f32; LEG_KINDS],
    pub bike_w: [f32; BIKE_KINDS],
    /// Per side (left, right) 0 on the grip/pedal .. 1 fully released.
    pub hand_rel: [f32; 2],
    pub foot_rel: [f32; 2],
    pub bars: Spin,
    pub tail: Spin,
    pub crank: Spin,
    /// Superman clocks: seconds into the stretch (held during the return) and 0..1 return progress.
    pub sup_t: f32,
    pub sup_u: f32,
    sup_level: f32,
    /// Seconds the tailwhip has been requested.
    tail_t: f32,
    /// Spring velocities of the smoothed values above.
    rate: Rates,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            mode: "idle",
            phase: "ground",
            hand: HandTrick::None,
            leg: LegTrick::None,
            bike: BikeTrick::None,
            note: "",
            frozen: false,
            stand: 0.0,
            crouch: 0.0,
            back: 0.0,
            lateral: 0.0,
            land: 0.0,
            land_v: 0.0,
            land_hit: 0.0,
            turn: 0.0,
            pedal: 0.0,
            attack: 0.0,
            pump: 0.0,
            surge: 0.0,
            english: [0.0; 4],
            effort: 0.0,
            top: 0.0,
            accel_lp: 0.0,
            jolt: [0.0; 3],
            jolt_hit: [0.0; 3],
            landings: 0,
            crank_un: 0.0,
            crank_vis: 0.0,
            prev_crank: None,
            steer_vis: None,
            prev_speed: None,
            impact: 0.0,
            since_land: 10.0,
            aloft: false,
            side: 1.0,
            hand_w: [0.0; HAND_KINDS],
            leg_w: [0.0; LEG_KINDS],
            bike_w: [0.0; BIKE_KINDS],
            hand_rel: [0.0; 2],
            foot_rel: [0.0; 2],
            bars: Spin::default(),
            tail: Spin::default(),
            crank: Spin::default(),
            sup_t: 0.0,
            sup_u: 0.0,
            sup_level: 0.0,
            tail_t: 0.0,
            rate: Rates::default(),
        }
    }
}

impl AnimationState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn spins(&self) -> [&Spin; 3] {
        [&self.bars, &self.tail, &self.crank]
    }

    /// No trick layer is shown or finishing (the trick side may be re-latched).
    pub(crate) fn layers_idle(&self) -> bool {
        self.hand_w
            .iter()
            .chain(&self.leg_w)
            .chain(&self.bike_w)
            .all(|&w| w == 0.0)
            && self.hand_rel == [0.0; 2]
            && self.foot_rel == [0.0; 2]
            && self.spins().iter().all(|s| s.settled())
            && self.sup_t == 0.0
    }

    /// Superman pose channels, from the retail clip timings: feet off at once, the straight legs
    /// swing back over 0.1–0.45 s, arms straighten from 0.2 s and the bike keeps swinging nose-up
    /// (easing out) until `SUP_IN`. The return (`SUP_OUT`) swings the bike and body back first,
    /// folds the knees early and puts the feet down last.
    pub fn superman(&self) -> Superman {
        let t = self.sup_t;
        if t == 0.0 {
            return Superman::default();
        }
        let ease = |a: f32, b: f32| {
            let x = ((t - a) / (b - a)).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let u = self.sup_u;
        let stay = (1.0 - u) * (1.0 - u);
        Superman {
            legs: ease(0.10, 0.45),
            bike: ((t / SUP_IN).min(1.0) * FRAC_PI_2).sin() * stay,
            arms: ((t - 0.2) / 0.47).clamp(0.0, 1.0) * stay,
            body: ease(0.0, 0.2) * stay,
            fold: (PI * u.sqrt()).sin(),
            level: self.sup_level,
        }
    }

    pub fn body(&self) -> Body {
        let (mut crouch, mut back) = (self.crouch, self.back);
        let hr = self.hand_rel[0].max(self.hand_rel[1]);
        let fr = self.foot_rel[0].max(self.foot_rel[1]);
        for (w, (c, b)) in self.hand_w.iter().zip(HAND_BODY) {
            crouch += w * hr * c;
            back += w * hr * b;
        }
        for (w, (c, b)) in self.leg_w.iter().zip(LEG_BODY) {
            crouch += w * fr * c;
            back += w * fr * b;
        }
        let table_t = (self.bike_w[B_TABLE] + self.bike_w[B_EURO]).min(1.0);
        Body {
            stand: self.stand,
            crouch: crouch.clamp(-0.4, 2.4),
            back: (back + self.surge).clamp(-0.7, 1.0),
            lateral: self.lateral - self.side * 0.08 * table_t,
            fold: self.hand_w[H_TIRE] * hr,
            turn: self.turn,
            pedal: self.pedal,
            surge: self.surge,
            sag: self.pump,
            english: self.english,
            effort: self.effort,
            top: self.top,
            jolt: self.jolt,
            crank: self.prev_crank.map(|_| self.crank_vis),
            steer: self.steer_vis,
            land: self.land.clamp(0.0, 1.0),
        }
    }

    pub fn bike_layer(&self) -> BikeLayer {
        let (w, s) = (&self.bike_w, self.side);
        let table_t = (w[B_TABLE] + w[B_EURO]).min(1.0);
        let sup = self.superman();
        BikeLayer {
            roll: s * (0.30 * w[B_WHIP] + 1.35 * w[B_TABLE] + 1.45 * w[B_EURO]),
            pitch: -(0.9 * w[B_TURN] + 2.35 * w[B_INVERT])
                + (SUP_PITCH - sup.level)
                    * sup.bike
                    * (1.0 - 0.5 * (self.hand_rel[0] + self.hand_rel[1])),
            yaw: s * (0.30 * w[B_WHIP] + 0.35 * w[B_TURN] + 0.30 * w[B_EURO]),
            bar_turn: s * (0.94 * FRAC_PI_2 * table_t + 0.92 * PI * w[B_XUP]),
            table_t,
            grip_slide: 0.77 + (-0.35 - 0.77) * w[B_XUP],
        }
    }

    pub fn update(&mut self, i: &Input, dt: f32) {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, MAX_DT)
        } else {
            0.0
        };
        if dt == 0.0 {
            return;
        }
        let c = &i.c;
        let grounded = i.grounded[0] || i.grounded[1];
        let air = !grounded && i.air_time >= AIR_MIN;
        let lead = if air { i.landing_in } else { 0.0 };
        let free = air && lead > regrab_lead(c.leg_trick);

        // Landing impact is captured from the prediction while still airborne, then kicks the
        // compression spring on the first contact frame.
        if !grounded {
            if !self.aloft {
                self.aloft = true;
                self.impact = 0.0;
            }
            if air && i.landing_in.is_finite() {
                self.impact = (i.impact / 8.0).clamp(0.0, 1.2);
            }
        } else if self.aloft {
            self.aloft = false;
            self.landings = self.landings.wrapping_add(1);
            let n = self.landings;
            let hit = self.impact.max((-i.vy / 8.0).clamp(0.0, 1.2)) * (1.0 + 0.15 * jitter(n, 1));
            self.land_hit = hit;
            self.land_v = 0.0;
            self.since_land = 0.0;
            self.impact = 0.0;
            // The body keeps going the way the landing throws it: to the low side of a leaned
            // touchdown, back as the rear wheel lands first, over the bars on a nose-first one
            // (or when still rotating forward), never twice the same.
            let fore = -i.ground_pitch / 0.35 - i.pitch_rate / 4.0;
            self.jolt_hit = [
                (i.roll / 0.35 + 0.35 * jitter(n, 2)).clamp(-1.0, 1.0) * hit,
                (fore + 0.3 * jitter(n, 3)).clamp(-1.0, 1.0) * hit,
                (0.1 * i.yaw_rate + 0.5 * jitter(n, 4)).clamp(-1.0, 1.0) * hit,
            ];
        }
        self.since_land = (self.since_land + dt).min(10.0);
        // Contact -> deepest compression over LAND_HOLD, then a slower recovery.
        let (target, omega) = if self.since_land < LAND_HOLD {
            (self.land_hit, LAND_IN)
        } else {
            (0.0, LAND_OUT)
        };
        spring(&mut self.land, &mut self.land_v, target, omega, dt);
        let held = (self.since_land < LAND_HOLD) as u8 as f32;
        for k in 0..3 {
            spring(
                &mut self.jolt[k],
                &mut self.rate.jolt[k],
                self.jolt_hit[k] * held,
                omega,
                dt,
            );
        }

        if self.layers_idle() {
            self.side = if c.trick_side < 0.0 { -1.0 } else { 1.0 };
        }
        let si = (self.side > 0.0) as usize;

        // Requests: air only, and only while a landing is not imminent.
        let (hand_req, leg_req, mut bike_req) = if free {
            (c.hand_trick, c.leg_trick, c.bike_trick)
        } else {
            (HandTrick::None, LegTrick::None, BikeTrick::None)
        };
        self.note = "";
        let bars_busy = !self.bars.settled() && self.bars.owed() > BUSY_OWED && lead > LAST_REGRAB;
        if (hand_req == HandTrick::Barspin || bars_busy)
            && matches!(
                bike_req,
                BikeTrick::Table | BikeTrick::XUp | BikeTrick::EuroTable
            )
        {
            bike_req = BikeTrick::None;
            self.note = "barspin owns the bars: table/x-up/euro table suppressed";
        }

        // Hands.
        let mut hand_want = [false; HAND_KINDS];
        let mut hand_t = [0.0_f32; 2];
        if let Some(k) = hand_index(hand_req) {
            hand_want[k] = true;
            hand_t = if HAND_ONE_SIDED[k] {
                let mut t = [0.0; 2];
                t[si] = 1.0;
                t
            } else {
                [1.0; 2]
            };
        }
        if bars_busy {
            hand_want[H_BAR] = true;
            hand_t = [1.0; 2];
        }

        // Feet.
        let tail_busy = !self.tail.settled() && self.tail.owed() > TAIL_CATCH && lead > LAST_REGRAB;
        let crank_busy =
            !self.crank.settled() && self.crank.owed() > BUSY_OWED && lead > LAST_REGRAB;
        let mut leg_want = [false; LEG_KINDS];
        let mut leg_t = [0.0_f32; 2];
        if let Some(k) = leg_index(leg_req) {
            leg_want[k] = true;
            leg_t = if LEG_ONE_SIDED[k] {
                let mut t = [0.0; 2];
                t[si] = 1.0;
                t
            } else {
                [1.0; 2]
            };
        }
        // Tailwhip: the trick-side foot leaves first, the other once the rear is swinging round.
        self.tail_t = if leg_req == LegTrick::Tailwhip {
            self.tail_t + dt
        } else {
            0.0
        };
        if leg_req == LegTrick::Tailwhip && self.tail_t < 0.1 {
            leg_t[1 - si] = 0.0;
        }
        let forced = match bike_req {
            BikeTrick::Invert => Some(L_DANGLE),
            BikeTrick::EuroTable => Some(L_COUNTER),
            BikeTrick::Crankflip => Some(L_LIFT),
            _ => None,
        };
        if let Some(k) = forced {
            leg_t = [1.0; 2];
            if leg_req == LegTrick::None {
                leg_want[k] = true;
            }
        }
        if tail_busy && leg_req != LegTrick::Tailwhip {
            leg_t = [1.0; 2];
            leg_want[L_TAIL] = true;
        }
        if crank_busy {
            leg_t = [1.0; 2];
            if leg_req == LegTrick::None {
                leg_want[L_LIFT] = true;
            }
        }
        // Superman runs on its clip clocks; the return always plays out, quicker if contact is near.
        self.sup_level = i.pitch.clamp(-0.7, 0.7);
        if leg_req == LegTrick::Superman && self.sup_u == 0.0 {
            self.sup_t = (self.sup_t + dt).min(SUP_IN);
        } else if self.sup_t > 0.0 {
            let left = if air {
                (lead - LAND_MARGIN).max(0.05)
            } else {
                0.2
            };
            self.sup_u += dt * (1.0 / SUP_OUT).max((1.0 - self.sup_u) / left);
            if self.sup_u >= 1.0 {
                (self.sup_t, self.sup_u) = (0.0, 0.0);
            }
        }
        if self.sup_t > 0.0 {
            leg_t = [1.0; 2];
            leg_want[L_SUPER] = true;
        }

        // Bike.
        let mut bike_want = [false; BIKE_KINDS];
        if let Some(k) = bike_index(bike_req) {
            bike_want[k] = true;
        }
        if crank_busy {
            bike_want[B_CRANK] = true;
        }

        // Spins.
        self.bars.step(
            self.side * BAR_RATE,
            hand_req == HandTrick::Barspin,
            grounded,
            lead,
            dt,
        );
        self.tail.step(
            self.side * TAIL_RATE,
            leg_req == LegTrick::Tailwhip,
            grounded,
            lead,
            dt,
        );
        self.crank.step(
            self.side * CRANK_RATE,
            bike_req == BikeTrick::Crankflip,
            grounded,
            lead,
            dt,
        );

        // Blend weights and release.
        for k in 0..HAND_KINDS {
            weight(
                &mut self.hand_w[k],
                &mut self.rate.hand_w[k],
                hand_want[k] as u8 as f32,
                WEIGHT_OMEGA,
                dt,
            );
        }
        for k in 0..LEG_KINDS {
            weight(
                &mut self.leg_w[k],
                &mut self.rate.leg_w[k],
                leg_want[k] as u8 as f32,
                WEIGHT_OMEGA,
                dt,
            );
        }
        for k in 0..BIKE_KINDS {
            let t = bike_want[k];
            let omega = if t && self.bike_w[k] < 1.0 {
                16.0
            } else {
                22.0
            };
            weight(
                &mut self.bike_w[k],
                &mut self.rate.bike_w[k],
                t as u8 as f32,
                omega,
                dt,
            );
        }
        // Regrabs speed up when contact is closer than they take.
        let late = if air { (5.0 / lead).min(60.0) } else { 38.0 };
        let tail_catch = self.leg_w[L_TAIL] > 0.02;
        for s in 0..2 {
            let omega = if hand_t[s] > self.hand_rel[s] {
                RELEASE_OMEGA
            } else {
                REGRAB_OMEGA.max(late)
            };
            let (x, v) = (&mut self.hand_rel[s], &mut self.rate.hand_rel[s]);
            weight(x, v, hand_t[s], omega, dt);
            let regrab = match (tail_catch, s == si) {
                (true, true) => CATCH_NEAR_OMEGA,
                (true, false) => CATCH_FAR_OMEGA,
                _ => REGRAB_OMEGA,
            };
            let omega = if leg_t[s] > self.foot_rel[s] {
                // A tailwhip kick takes the feet off fast.
                if leg_want[L_TAIL] {
                    30.0
                } else {
                    RELEASE_OMEGA
                }
            } else {
                regrab.max(late)
            };
            let (x, v) = (&mut self.foot_rel[s], &mut self.rate.foot_rel[s]);
            weight(x, v, leg_t[s], omega, dt);
        }
        // Superman return: the feet come down onto the pedals last, easing out.
        if self.sup_u > 0.0 {
            let off = 1.0 - (self.sup_u * FRAC_PI_2).sin();
            for s in 0..2 {
                if self.foot_rel[s] > off {
                    self.foot_rel[s] = off;
                    self.rate.foot_rel[s] = 0.0;
                }
            }
        }

        // Body posture targets (additive terms), then smoothing.
        let ride = i.ride_stand.clamp(0.0, 1.0);
        let moving = i.speed > 0.6;
        let pedaling = c.pedal > 0.1;
        let sprint = c.sprint && pedaling;
        let manual = c.manual || c.wheelie;
        let nose = c.nose_manual && !manual;
        let ext = if air && lead.is_finite() {
            1.0 - ((lead - 0.10) / 0.40).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let takeoff = if !grounded && i.vy > 0.3 {
            (1.0 - i.air_time / TAKEOFF_T).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut stand;
        let mut crouch;
        let mut back;
        let lateral;
        let mut attack = 0.0_f32;
        if !grounded {
            stand = 0.45 + 0.55 * ext + 0.5 * takeoff;
            crouch = 0.7 - 0.95 * ext - 0.5 * takeoff;
            back = c.air_pitch * 0.5;
            lateral = -c.air_roll * 0.06;
            // Tailwhip catch: lunge forward over the bars as the frame comes round.
            let catch = (!self.tail.settled()
                && leg_req != LegTrick::Tailwhip
                && self.tail.owed() < TAIL_CATCH) as u8 as f32;
            back -= 0.9 * catch;
            stand += 0.3 * catch;
        } else {
            stand = if sprint {
                1.0
            } else if pedaling {
                0.25 + ride * 0.7
            } else if moving {
                ride
            } else {
                0.0
            };
            crouch = 0.15 * (i.steering.abs() / 0.5).min(1.0) * moving as u8 as f32 + c.brake * 0.2;
            back = c.brake * 0.7 - i.pitch * 0.5 - 0.5 * (pedaling && !sprint) as u8 as f32;
            attack = ((i.speed - 4.0) / 6.0).clamp(0.0, 1.0) * ride * (!pedaling) as u8 as f32;
            crouch += 0.3 * attack;
            back += 0.2 * attack;
            // Driving hard the rider gets low and forward over the bars; spun out at top speed
            // they sit lower still, chest down, out of the wind.
            crouch += 0.3 * self.effort + 0.5 * self.top;
            back -= 0.25 * self.effort + 0.3 * self.top;
            stand -= 0.45 * self.top;
            lateral = 0.0;
            if c.hop {
                stand = 0.0;
                crouch = 1.0;
            }
            if manual {
                stand = 0.5;
                crouch = -0.1;
                back = 1.0;
            } else if nose {
                stand = 0.7;
                crouch = 0.25;
                back = -0.7;
            }
        }
        stand = (stand - 0.4 * self.land).clamp(0.0, 1.0);
        // Rider extends into compressions and gives over crests.
        let absorb = ((i.suspension[0] + i.suspension[1]) * 0.5 - SAG_REST).clamp(-0.05, 0.08);
        crouch -= 3.0 * absorb;
        crouch += LAND_CROUCH * self.land;
        back = back.clamp(-0.7, 1.0);
        let omega = if c.hop { 22.0 } else { 14.0 };
        spring(&mut self.stand, &mut self.rate.stand, stand, omega, dt);
        spring(&mut self.crouch, &mut self.rate.crouch, crouch, omega, dt);
        spring(&mut self.back, &mut self.rate.back, back, omega, dt);
        spring(
            &mut self.lateral,
            &mut self.rate.lateral,
            lateral,
            omega,
            dt,
        );
        let lean = if grounded {
            i.roll.clamp(-0.6, 0.6)
        } else {
            0.0
        };
        spring(&mut self.turn, &mut self.rate.turn, lean, 9.0, dt);
        let pedal_on = (grounded && pedaling && moving) as u8 as f32;
        spring(&mut self.pedal, &mut self.rate.pedal, pedal_on, 8.0, dt);
        spring(&mut self.attack, &mut self.rate.attack, attack, 6.0, dt);
        spring(&mut self.pump, &mut self.rate.pump, absorb, 24.0, dt);
        // Body English: the rider throws head and shoulders the way the bike is asked to turn and
        // keeps leading the rotation while it runs (shoulders and head first, the hips and bike
        // follow), pulls the bike in to rotate, and opens up again to spot the landing.
        let english = if grounded {
            [0.0; 4]
        } else {
            let open = 1.0 - ext;
            [
                (0.6 * c.air_pitch + 0.08 * i.pitch_rate).clamp(-1.0, 1.0) * open,
                (-0.35 * c.air_yaw + 0.06 * i.yaw_rate).clamp(-0.6, 0.6),
                (-0.5 * c.air_roll + 0.07 * i.roll_rate).clamp(-0.8, 0.8) * open,
                ((i.pitch_rate.abs() / 6.0).max(i.roll_rate.abs() / 8.0)).min(1.0) * open,
            ]
        };
        for k in 0..4 {
            let omega = if grounded { 8.0 } else { 11.0 };
            spring(
                &mut self.english[k],
                &mut self.rate.english[k],
                english[k],
                omega,
                dt,
            );
        }
        // Secondary motion: the torso lags the bike's longitudinal acceleration on a lightly
        // damped spring (accelerating pushes it back, braking throws it forward).
        let accel = match self.prev_speed {
            Some(p) if grounded => ((i.speed - p) / dt).clamp(-12.0, 12.0),
            _ => 0.0,
        };
        self.prev_speed = Some(i.speed);
        let (x, v) = (&mut self.surge, &mut self.rate.surge);
        *v += (SURGE_OMEGA * SURGE_OMEGA * (SURGE_GAIN * accel - *x)
            - 2.0 * SURGE_ZETA * SURGE_OMEGA * *v)
            * dt;
        *x += *v * dt;
        // Effort: a rider driving the bike hard (sprinting, or pedalling while the speed still
        // climbs) throws it from side to side under them and pulls on the bars. Once the speed
        // stops climbing in a sprint the rider is spun out at top speed and drops low and
        // forward, spinning rather than stomping.
        self.accel_lp += (accel - self.accel_lp) * (dt / ACCEL_LP).min(1.0);
        let driving = (grounded && pedaling && moving) as u8 as f32;
        let top = driving
            * sprint as u8 as f32
            * smooth01(6.0, 9.0, i.speed)
            * (1.0 - smooth01(0.3, 1.2, self.accel_lp));
        let climbing = smooth01(0.2, 1.5, self.accel_lp);
        let effort = driving
            * if sprint {
                1.0 - 0.6 * top
            } else {
                0.4 * climbing
            };
        spring(&mut self.effort, &mut self.rate.effort, effort, 5.0, dt);
        spring(&mut self.top, &mut self.rate.top, top, 3.0, dt);
        // The drawn crank follows the physical one on a spring (the bike's crank stops dead the
        // tick pedalling stops). Coasting, it settles level, or outside pedal down in a corner,
        // instead of wherever the last stroke left it.
        let wrap = |a: f32| (a + PI).rem_euclid(TAU) - PI;
        let level = [FRAC_PI_2, 3.0 * FRAC_PI_2]
            .map(|t| wrap(t - i.crank_phase))
            .into_iter()
            .min_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(0.0);
        let outside = wrap(if self.turn > 0.0 { 0.0 } else { PI } - i.crank_phase);
        let corner = (self.turn.abs() / 0.25).min(1.0);
        let coast = (grounded && moving && !pedaling) || !grounded;
        let reposition = if coast {
            level + (outside - level) * corner
        } else {
            0.0
        };
        match self.prev_crank {
            Some(p) => self.crank_un += wrap(i.crank_phase - p),
            None => (self.crank_un, self.crank_vis) = (i.crank_phase, i.crank_phase),
        }
        self.prev_crank = Some(i.crank_phase);
        // Stiff while it tracks the physical crank closely, soft while it travels far (so a
        // reposition or the return to pedalling never whips the cranks round).
        let err = (self.crank_un + reposition - self.crank_vis).abs();
        let omega = if coast {
            7.0
        } else {
            10.0 + 20.0 * (1.0 - err / 0.5).clamp(0.0, 1.0)
        };
        spring(
            &mut self.crank_vis,
            &mut self.rate.crank,
            self.crank_un + reposition,
            omega,
            dt,
        );
        // Bars follow the steering input on a spring; the bike's own steering is first order.
        let bars = self.steer_vis.get_or_insert(i.steering);
        spring(bars, &mut self.rate.steer, i.steering, 25.0, dt);

        // Reporting.
        let unsettled = self.spins().iter().any(|s| !s.settled());
        self.mode = if !grounded {
            "air"
        } else if c.hop {
            "hop"
        } else if manual {
            "manual"
        } else if nose {
            "nose-manual"
        } else if c.brake > 0.3 {
            "brake"
        } else if moving && i.steering.abs() > 0.15 {
            "turn"
        } else if sprint {
            "sprint"
        } else if pedaling {
            "pedal"
        } else if moving {
            "coast"
        } else {
            "idle"
        };
        self.phase = if !grounded {
            if i.air_time < TAKEOFF_T && i.vy > 0.3 {
                "takeoff"
            } else if ext > 0.5 {
                "air-extend"
            } else {
                "air-tuck"
            }
        } else if unsettled {
            "spin-recover"
        } else if self.since_land < 0.18 {
            "landing"
        } else if self.since_land < 0.7 && self.land.abs() > 0.04 {
            "recovery"
        } else if c.hop {
            "preload"
        } else {
            "ground"
        };
        let dominant = |w: &[f32]| {
            w.iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.02)
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, _)| k)
        };
        let hands = self.hand_rel[0].max(self.hand_rel[1]) > 0.02 || !self.bars.settled();
        self.hand = dominant(&self.hand_w[..])
            .filter(|_| hands)
            .map_or(HandTrick::None, |k| HAND_REPORT[k]);
        let feet = self.foot_rel[0].max(self.foot_rel[1]) > 0.02
            || !self.tail.settled()
            || !self.crank.settled();
        self.leg = dominant(&self.leg_w[..LEG_PUBLIC])
            .filter(|_| feet)
            .map_or(LegTrick::None, |k| LEG_REPORT[k]);
        self.bike = dominant(&self.bike_w[..]).map_or(BikeTrick::None, |k| BIKE_REPORT[k]);
    }
}
