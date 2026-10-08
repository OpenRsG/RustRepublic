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
//! Turndown, EuroTable, Invert, Crankflip. Poses are authored from the trick names; the retail
//! animation tracks are not decoded, so parity is unverified. Not implemented: BIKEFLIP, BRIFLIP,
//! HDFLIP, GRIZZAIR, CANONBALL, TSUNAMI and any PS/discipline binding.
//!
//! Priority rules (encoded in [`AnimationState::update`], reported through `note`):
//! 1. Air-only: tricks are refused on the ground and during the first `AIR_MIN` seconds of flight.
//! 2. Landing: new holds stop `REGRAB_LEAD` s before the predicted contact; hands/feet regrab and
//!    bar/tail/crank spins finish their revolution (or unwind) before contact. Nothing else is
//!    corrected for the landing: body rotation is whatever the physics produced.
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

const GRAVITY: f32 = 9.81;
const MAX_DT: f32 = 1.0 / 15.0;
/// Seconds of flight before air-only tricks may start.
const AIR_MIN: f32 = 0.05;
/// Tricks stop being held, and limbs regrab, this long before the predicted contact.
pub(crate) const REGRAB_LEAD: f32 = 0.45;
/// Hands/feet always head back to the bike once contact is this close.
const LAST_REGRAB: f32 = 0.18;
/// A finishing spin keeps hands/feet free until this many radians remain.
const BUSY_OWED: f32 = 0.8;
/// Spins aim to be complete this long before contact.
const LAND_MARGIN: f32 = 0.15;
const RELEASE_OMEGA: f32 = 19.0;
const REGRAB_OMEGA: f32 = 38.0;
const WEIGHT_OMEGA: f32 = 19.0;
const SPIN_ACCEL: f32 = 14.0;
const SETTLE: f32 = 0.01;
const MIN_FINISH_RATE: f32 = 3.0;
const RECOVER_RATE: f32 = 12.0;
const MAX_FINISH_RATE: f32 = 32.0;
const BAR_RATE: f32 = 16.0;
const TAIL_RATE: f32 = 11.0;
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
    (0.1, 0.75),
    (0.2, 0.0),
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

    /// Direction in which the spin will finish: on with the current rotation when it is already
    /// more than ~30% of a turn along, otherwise back the short way.
    fn dir(&self) -> f32 {
        let fwd = if self.rate >= 0.0 {
            TAU - self.angle
        } else {
            self.angle
        };
        if self.rate.abs() > 0.5 && fwd <= 1.4 * PI {
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
        let mut r = self.rate.abs().max(MIN_FINISH_RATE).min(owed * 10.0 + 1.0);
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
            crank: self.prev_crank.map(|_| self.crank_vis),
            steer: self.steer_vis,
            land: self.land.clamp(0.0, 1.0),
        }
    }

    pub fn bike_layer(&self) -> BikeLayer {
        let (w, s) = (&self.bike_w, self.side);
        let table_t = (w[B_TABLE] + w[B_EURO]).min(1.0);
        BikeLayer {
            roll: s * (0.30 * w[B_WHIP] + 1.35 * w[B_TABLE] + 1.45 * w[B_EURO]),
            pitch: -(0.9 * w[B_TURN] + 2.35 * w[B_INVERT]),
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
        let free = air && lead > REGRAB_LEAD;

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
            let hit = self.impact.max((-i.vy / 8.0).clamp(0.0, 1.2));
            self.land_hit = hit;
            self.land_v = 0.0;
            self.since_land = 0.0;
            self.impact = 0.0;
        }
        self.since_land = (self.since_land + dt).min(10.0);
        // Contact -> deepest compression over LAND_HOLD, then a slower recovery.
        let (target, omega) = if self.since_land < LAND_HOLD {
            (self.land_hit, LAND_IN)
        } else {
            (0.0, LAND_OUT)
        };
        spring(&mut self.land, &mut self.land_v, target, omega, dt);

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
        let tail_busy = !self.tail.settled() && self.tail.owed() > BUSY_OWED && lead > LAST_REGRAB;
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
        if tail_busy {
            leg_t = [1.0; 2];
            leg_want[L_TAIL] = true;
        }
        if crank_busy {
            leg_t = [1.0; 2];
            if leg_req == LegTrick::None {
                leg_want[L_LIFT] = true;
            }
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
        for s in 0..2 {
            let omega = if hand_t[s] > self.hand_rel[s] {
                RELEASE_OMEGA
            } else {
                REGRAB_OMEGA
            };
            let (x, v) = (&mut self.hand_rel[s], &mut self.rate.hand_rel[s]);
            weight(x, v, hand_t[s], omega, dt);
            let omega = if leg_t[s] > self.foot_rel[s] {
                RELEASE_OMEGA
            } else {
                REGRAB_OMEGA
            };
            let (x, v) = (&mut self.foot_rel[s], &mut self.rate.foot_rel[s]);
            weight(x, v, leg_t[s], omega, dt);
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
