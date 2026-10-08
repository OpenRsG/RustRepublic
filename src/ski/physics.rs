//! Arcade ski dynamics, fixed-step, no physics engine.
//!
//! Frame: `rotation` columns are +X right, +Y ski-base normal, +Z back (tips at -Z). World +Y is up
//! and forward at yaw 0 is -Z; `heading` is the yaw readout (+ turns left, so a right turn lowers
//! it). `edge` is the physical ski tilt about the ski axis (+ onto the right edges); `lean` is the
//! whole-body inclination towards the turn centre. Everything is per unit mass.
//!
//! Grounded: the skier rides the snow. Gravity is projected on the slope plane, snow friction and
//! quadratic drag act along the velocity, and the edges remove the sideways velocity of the skis up
//! to a grip limit (sidecut carving). Exceeding the limit, or swinging the skis across the travel
//! (hockey stop), makes the skis skid. Pushing (skate, double-pole), braking (snowplow, hockey
//! stop), jump preload and the leg spring are arcade models, not retail parity.
//!
//! Airborne: the centre of mass is ballistic and `rotation` is a free quaternion driven by bounded
//! body-axis rate servos (pitch, yaw, roll). A touchdown is judged at first contact against the
//! surface (attitude, skis across travel, angular speed, closing speed) before any penetration is
//! corrected; a failed landing or head/pelvis striking the snow is a crash, after which `step` only
//! advances `crash.elapsed` and leaves the state for the ragdoll seed.

use bevy::math::{Mat3, Quat, Vec3};
use bevy::prelude::Resource;
use std::f32::consts::{PI, TAU};

use super::pose::{SKI_BACK, SKI_FRONT, SKI_TIP_RISE};
use crate::bike::{HILL_START_Z, HILL_X, terrain_height};

/// Declares a `Copy` selector enum (first variant is the default) with `ALL` and `label()`.
macro_rules! choice {
    ($(#[$m:meta])* $name:ident { $first:ident = $fl:literal $(, $v:ident = $l:literal)* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub enum $name {
            #[default]
            $first,
            $($v),*
        }

        impl $name {
            pub const ALL: &[Self] = &[Self::$first $(, Self::$v)*];

            pub fn label(self) -> &'static str {
                match self {
                    Self::$first => $fl,
                    $(Self::$v => $l),*
                }
            }
        }
    };
}

choice!(
    /// Grab requested while airborne (animation input; no physics effect).
    Grab {
        None = "None",
        Mute = "Mute",
        Safety = "Safety",
        Japan = "Japan",
        Tail = "Tail",
        Tip = "Tip",
        TruckDriver = "Truck driver",
        Daffy = "Daffy",
        SpreadEagle = "Spread eagle",
        IronCross = "Iron cross",
    }
);

/// Why the skier crashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkiCrashReason {
    BadLanding,
    BodyImpact,
    HardImpact,
    CaughtEdge,
}

impl SkiCrashReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::BadLanding => "Bad landing",
            Self::BodyImpact => "Body impact",
            Self::HardImpact => "Hard impact",
            Self::CaughtEdge => "Caught edge",
        }
    }
}

/// A crash in progress; the skier is frozen for the ragdoll until reset.
#[derive(Clone, Copy, Debug)]
pub struct SkiCrash {
    pub reason: SkiCrashReason,
    /// Closing speed (landings) or speed (other crashes), m/s.
    pub impact: f32,
    pub elapsed: f32,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct SkiControls {
    /// W 0..1: skate (slow) / double-pole (medium) propulsion.
    pub push: f32,
    /// S 0..1: snowplow when slow, hockey-stop skid when fast.
    pub brake: f32,
    /// A/D -1..1, +1 = turn right (carve).
    pub steer: f32,
    /// Shift: aero tuck.
    pub tuck: bool,
    /// Space HELD: crouch preload; release (held -> released edge) pops while grounded.
    pub jump: bool,
    /// -1..1, +1 nose (ski tips) up / backflip direction.
    pub air_pitch: f32,
    /// -1..1, +1 roll right.
    pub air_roll: f32,
    /// -1..1, +1 spin right.
    pub air_yaw: f32,
    /// Ctrl held: higher air authority (full flips).
    pub flip: bool,
    /// Requested grab (animation-only for physics).
    pub grab: Grab,
    /// -1 left / +1 right.
    pub trick_side: f32,
}

const G: f32 = 9.81;
const STEP_HZ: f32 = 120.0;
/// Larger `dt` is clamped, then split into <= 1/120 s substeps (so <= 8 substeps).
const MAX_FRAME_DT: f32 = 1.0 / 15.0;
/// A substep moves the root or the rotating head at most this far, metres.
const MAX_STEP_TRAVEL: f32 = 0.08;
const MAX_SUBSTEPS: u32 = 32;
const TIP_RADIUS: f32 = 0.9;
const SLOPE_EPS: f32 = 0.05;

// Snow and air resistance.
const SNOW_FRICTION: f32 = 0.05;
/// 0.5 * rho * CdA / m for an upright skier, 1/m.
const DRAG: f32 = 0.0035;
const TUCK_DRAG_CUT: f32 = 0.45;

// Carving.
const SIDECUT_RADIUS: f32 = 14.0;
const EDGE_RESPONSE: f32 = 7.0;
/// Fastest edge change, rad/s: an edge-to-edge transition takes about 0.4 s.
const EDGE_SLEW: f32 = 4.5;
/// Natural frequency of the critically damped body inclination, rad/s.
const LEAN_OMEGA: f32 = 7.0;
/// Natural frequency at which the drawn pose settles from its touchdown attitude and height onto
/// the snapped skis, rad/s (about 0.15 s; the skis visibly slap flat instead of teleporting).
const LAND_SETTLE_OMEGA: f32 = 18.0;
/// Edge and inclination relax towards zero in the air at this rate, 1/s.
const AIR_SETTLE: f32 = 3.0;
/// The heading rate follows its carve / pivot / hockey-stop target with this rate, 1/s, so the
/// skis never start swinging at full speed within one tick.
const TURN_RESPONSE: f32 = 20.0;
const GRIP_BASE: f32 = 0.5;
const GRIP_EDGE: f32 = 0.9;
const SKID_GRIP_LOSS: f32 = 0.45;
const PIVOT_RATE: f32 = 2.2;
const SWITCH_SPEED: f32 = 1.0;
const MAX_LEAN: f32 = 1.2;

// Propulsion. Skate below ~4 m/s, double-pole up to ~8.5 m/s, nothing beyond ~9 m/s.
const SKATE_ACCEL: f32 = 1.5;
const POLE_ACCEL: f32 = 1.4;
const V2_ACCEL: f32 = 0.6;
/// Skate stroke time from standstill and at 4 m/s, s.
const STROKE_SLOW: f32 = 0.95;
const STROKE_FAST: f32 = 0.60;
/// Fraction of a stroke over which the pushing leg works while the hips cross to the other ski.
pub const STROKE_PUSH: f32 = 0.44;
/// Double-pole cycle time below 3.5 m/s and above 7 m/s, s.
const POLE_CYCLE_SLOW: f32 = 1.2;
const POLE_CYCLE_FAST: f32 = 0.9;
/// Fraction of a pole cycle (or of a V2 stroke) spent pushing.
pub const POLE_PUSH_SPAN: f32 = 0.35;
/// Poles join the skating from this speed range (m/s): free skate below, V2 above.
const V2_SPEED: (f32, f32) = (1.8, 2.8);

// Braking.
const PLOW_DECEL: f32 = 2.5;
const HOCKEY_SPEED: f32 = 6.0;
const HOCKEY_END: f32 = 1.0;
const HOCKEY_ANGLE: f32 = 1.4;
const HOCKEY_RATE: f32 = 8.0;
const HOCKEY_EDGE: f32 = 0.9;

// Edge catching (needs a fast slide on the leading edge, never part of a hockey stop).
const CATCH_SPEED: f32 = 8.0;
const CATCH_EDGE: f32 = 0.6;
const CATCH_TIME: f32 = 0.25;

// Legs and jumping.
const LEG_K: f32 = 150.0;
const LEG_D: f32 = 19.6;
const LEG_TRAVEL: f32 = 0.35;
const PRELOAD_TIME: f32 = 0.35;
const POP_BASE: f32 = 2.0;
const POP_GAIN: f32 = 2.2;
/// Ground gap under the free-flight path that counts as having left the snow, metres.
const SEPARATION_TOL: f32 = 1e-3;

// Air control: body-axis rate servos.
/// Maximum spin rate about the skier's up axis, rad/s.
pub const AIR_YAW_RATE: f32 = 7.0;
/// Pitch rate authority without / with `flip` held, rad/s.
pub const AIR_PITCH_RATE: f32 = 2.5;
pub const AIR_FLIP_RATE: f32 = 6.0;
/// Roll rate authority without / with `flip` held, rad/s.
pub const AIR_ROLL_RATE: f32 = 3.0;
const FLIP_ROLL_RATE: f32 = 5.0;
const YAW_ACCEL: f32 = 18.0;
const PITCH_ACCEL: f32 = 8.0;
const FLIP_ACCEL: f32 = 16.0;
const ROLL_ACCEL: f32 = 12.0;
const AIR_DAMP: f32 = 1.5;
const FLIP_DAMP: f32 = 0.3;
const YAW_DAMP: f32 = 0.8;
const WINDUP_YAW: f32 = 3.0;
/// Centre of mass above the base point; rotation in the air is about it.
const COM_HEIGHT: f32 = 0.9;

// Landing judgement against the surface normal under the skis.
const LAND_TILT_LIMIT: f32 = 0.96;
const LAND_CROSS_LIMIT: f32 = 0.61;
const LAND_CROSS_MIN_SPEED: f32 = 2.0;
const LAND_SPIN_LIMIT: f32 = 6.0;
const HARD_IMPACT_SPEED: f32 = 17.0;
const LAND_ABSORB_SPEED: f32 = 12.0;
/// Leg compression rate per unit `impact / LAND_ABSORB_SPEED` at touchdown, 1/s; the leg spring
/// then peaks near full compression about 0.09 s later for a 12 m/s closing speed.
const LAND_LEG_KICK: f32 = 29.0;

// Body proxies against the snow while airborne.
const HEAD_HEIGHT: f32 = 1.55;
const HEAD_RADIUS: f32 = 0.12;
const PELVIS_HEIGHT: f32 = 0.9;
const PELVIS_RADIUS: f32 = 0.15;
const TUCK_HEAD_DROP: f32 = 0.22;

/// Clamps a caller-provided number; non-finite becomes zero.
fn sane(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() { x.clamp(lo, hi) } else { 0.0 }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The skate stroke of a `Skier::skate_phase`: the pushing leg (0 left, 1 right) and progress 0..1.
pub(crate) fn skate_stroke(phase: f32) -> (usize, f32) {
    let x = phase.rem_euclid(TAU) / PI;
    (usize::from(x >= 1.0), x.fract())
}

/// Seconds one skate stroke takes at `speed` m/s along the skis.
pub(crate) fn stroke_time(speed: f32) -> f32 {
    lerp(STROKE_SLOW, STROKE_FAST, (speed / 4.0).clamp(0.0, 1.0))
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn wrap(a: f32) -> f32 {
    (a + PI).rem_euclid(TAU) - PI
}

/// First-order smoothing factor for a substep.
fn ease(rate: f32, h: f32) -> f32 {
    1.0 - (-rate * h).exp()
}

/// Critically damped second-order step of `x` (rate `v`) towards `target`: velocity stays
/// continuous when the target jumps, so poses driven by it never snap.
fn spring(x: &mut f32, v: &mut f32, target: f32, omega: f32, h: f32) {
    *v += (omega * omega * (target - *x) - 2.0 * omega * *v) * h;
    *x += *v * h;
}

fn yaw_of(forward: Vec3) -> f32 {
    (-forward.x).atan2(-forward.z)
}

/// Unit snow normal at `(x, z)` from a finite difference of the terrain height.
pub fn surface_normal(x: f32, z: f32) -> Vec3 {
    let dx =
        (terrain_height(x + SLOPE_EPS, z) - terrain_height(x - SLOPE_EPS, z)) / (2.0 * SLOPE_EPS);
    let dz =
        (terrain_height(x, z + SLOPE_EPS) - terrain_height(x, z - SLOPE_EPS)) / (2.0 * SLOPE_EPS);
    Vec3::new(-dx, 1.0, -dz).normalize()
}

/// Ski forward and right axes on the plane with normal `n` for a heading.
fn frame(n: Vec3, heading: f32) -> (Vec3, Vec3) {
    let flat = Quat::from_rotation_y(heading) * Vec3::NEG_Z;
    let f = (flat - n * flat.dot(n)).try_normalize().unwrap_or(flat);
    (f, f.cross(n))
}

/// Platform orientation resting on the snow: tips along the heading, no edge tilt.
fn orient(n: Vec3, heading: f32) -> Quat {
    let (f, r) = frame(n, heading);
    Quat::from_mat3(&Mat3::from_cols(r, n, -f)).normalize()
}

/// Rate servo about one axis: slew to the commanded rate while `active`, otherwise damp.
fn axis_rate(w: f32, target: f32, accel: f32, damp: f32, active: bool, h: f32) -> f32 {
    if active {
        w + (target - w).clamp(-accel * h, accel * h)
    } else {
        w * (-damp * h).exp()
    }
}

#[derive(Resource, Clone, Debug)]
pub struct Skier {
    /// Snow contact point under the midpoint of the bindings (ski base level).
    pub position: Vec3,
    pub velocity: Vec3,
    /// Platform frame: local -Z = ski tips, +Y = ski base normal, +X right. No edge tilt.
    pub rotation: Quat,
    /// World rad/s.
    pub angular_velocity: Vec3,
    /// Yaw readout of `rotation`.
    pub heading: f32,
    /// Signed ski edge angle about the ski axis, rad; + = tilted onto right edges (right turn).
    pub edge: f32,
    /// Signed whole-body inclination into the turn (centripetal balance), rad; + right.
    pub lean: f32,
    /// Visual touchdown settle, world space: the drawn pose is rotated by `land_tilt` (scaled
    /// axis) about `position` and shifted by `land_shift`, which reproduces the pre-snap pose at
    /// touchdown; both settle to zero. Physics never reads them.
    pub land_tilt: Vec3,
    pub land_shift: Vec3,
    pub grounded: bool,
    /// Seconds since leaving the snow; 0 while grounded.
    pub air_time: f32,
    /// 0..1 dynamic leg absorption (landing impact spring / bumps).
    pub compression: f32,
    /// 0..1 jump crouch while Space held.
    pub preload: f32,
    /// 0..1 smoothed tuck.
    pub tuck: f32,
    /// 0..1 snowplow wedge amount.
    pub plow: f32,
    /// 0..1 lateral skid amount (hockey stop / sliding turn; snow spray).
    pub skid: f32,
    /// 0..1 skating-propulsion weight.
    pub skate: f32,
    /// Rad, advances while skating; sin > 0 = left leg pushing.
    pub skate_phase: f32,
    /// 0..1 double-pole weight.
    pub pole: f32,
    /// Rad, 0..TAU per double-pole cycle (0 = poles planted).
    pub pole_phase: f32,
    /// 0..1 weight of the poles joining the skating (V2: a double-pole push on every stroke).
    pub v2: f32,
    /// Travelling tails-first.
    pub switch: bool,
    pub distance: f32,
    /// Closing speed of the last touchdown, m/s.
    pub impact: f32,
    /// Accumulated signed yaw rotation this airtime, rad (+ right).
    pub air_spin: f32,
    /// Accumulated signed pitch rotation this airtime, rad (+ back flip).
    pub air_flip: f32,
    /// `air_spin` / `air_flip` of the last airtime, stored at touchdown.
    pub last_spin: f32,
    pub last_flip: f32,
    pub crash: Option<SkiCrash>,
    /// Leg spring velocity.
    leg_velocity: f32,
    lean_rate: f32,
    /// Heading rate while grounded, rad/s.
    turn_rate: f32,
    land_tilt_rate: Vec3,
    land_shift_rate: Vec3,
    jump_held: bool,
    hockey: bool,
    hockey_side: f32,
    catch_time: f32,
}

impl Default for Skier {
    fn default() -> Self {
        Self::placed(HILL_X, HILL_START_Z, 0.0, 0.0)
    }
}

impl Skier {
    fn placed(x: f32, z: f32, yaw: f32, speed: f32) -> Self {
        let n = surface_normal(x, z);
        let (f, _) = frame(n, yaw);
        Self {
            position: Vec3::new(x, terrain_height(x, z), z),
            velocity: f * speed,
            rotation: orient(n, yaw),
            angular_velocity: Vec3::ZERO,
            heading: wrap(yaw),
            edge: 0.0,
            lean: 0.0,
            land_tilt: Vec3::ZERO,
            land_shift: Vec3::ZERO,
            grounded: true,
            air_time: 0.0,
            compression: 0.0,
            preload: 0.0,
            tuck: 0.0,
            plow: 0.0,
            skid: 0.0,
            skate: 0.0,
            skate_phase: 0.0,
            pole: 0.0,
            pole_phase: 0.0,
            v2: 0.0,
            switch: false,
            distance: 0.0,
            impact: 0.0,
            air_spin: 0.0,
            air_flip: 0.0,
            last_spin: 0.0,
            last_flip: 0.0,
            crash: None,
            leg_velocity: 0.0,
            lean_rate: 0.0,
            turn_rate: 0.0,
            land_tilt_rate: Vec3::ZERO,
            land_shift_rate: Vec3::ZERO,
            jump_held: false,
            hockey: false,
            hockey_side: 1.0,
            catch_time: 0.0,
        }
    }

    /// Back to the summit start.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Places the skier on the snow at `(x, z)` facing `yaw`, moving `speed` m/s along the skis.
    /// Clears any crash. Test setup only; the game starts runs from the summit.
    #[cfg(test)]
    pub fn reset_at(&mut self, x: f32, z: f32, yaw: f32, speed: f32) {
        *self = Self::placed(x, z, yaw, speed);
    }

    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Angular velocity in the platform frame: x pitch (+ nose up), y yaw (- spins right), z roll
    /// (- rolls right), rad/s.
    pub fn body_rates(&self) -> Vec3 {
        self.rotation.inverse() * self.angular_velocity
    }

    /// Advance by `dt` seconds (clamped to 1/15 s and split into substeps of at most 1/120 s and
    /// `MAX_STEP_TRAVEL` metres of root or head travel). A crashed skier only ages the crash.
    pub fn step(&mut self, controls: &SkiControls, dt: f32) {
        let dt = sane(dt, 0.0, MAX_FRAME_DT);
        if dt == 0.0 {
            return;
        }
        if let Some(crash) = &mut self.crash {
            crash.elapsed += dt;
            return;
        }
        let c = SkiControls {
            push: sane(controls.push, 0.0, 1.0),
            brake: sane(controls.brake, 0.0, 1.0),
            steer: sane(controls.steer, -1.0, 1.0),
            air_pitch: sane(controls.air_pitch, -1.0, 1.0),
            air_roll: sane(controls.air_roll, -1.0, 1.0),
            air_yaw: sane(controls.air_yaw, -1.0, 1.0),
            trick_side: sane(controls.trick_side, -1.0, 1.0),
            ..*controls
        };
        // The release edge is consumed by the first substep so a frame never pops twice.
        let pop = self.jump_held && !c.jump;
        self.jump_held = c.jump;
        let travel = (self.speed() + self.angular_velocity.length() * TIP_RADIUS) * dt;
        let n = ((dt * STEP_HZ - 1e-3).ceil() as u32)
            .max((travel / MAX_STEP_TRAVEL).ceil() as u32)
            .clamp(1, MAX_SUBSTEPS);
        for i in 0..n {
            let h = dt / n as f32;
            self.settle(&c, h);
            if self.grounded {
                self.ground_step(&c, h, pop && i == 0);
            } else {
                self.air_step(&c, h);
            }
            if self.crash.is_some() {
                break;
            }
        }
    }

    fn fail(&mut self, reason: SkiCrashReason, impact: f32) {
        self.crash = Some(SkiCrash {
            reason,
            impact,
            elapsed: 0.0,
        });
    }

    /// Tuck smoothing and the leg spring, common to ground and air.
    fn settle(&mut self, c: &SkiControls, h: f32) {
        self.tuck += (f32::from(u8::from(c.tuck)) - self.tuck) * ease(6.0, h);
        let k = LAND_SETTLE_OMEGA;
        for (x, v) in [
            (&mut self.land_tilt, &mut self.land_tilt_rate),
            (&mut self.land_shift, &mut self.land_shift_rate),
        ] {
            *v += (-k * k * *x - 2.0 * k * *v) * h;
            *x += *v * h;
        }
        let target = self.preload * 0.7 + self.tuck * 0.1;
        self.leg_velocity += (LEG_K * (target - self.compression) - LEG_D * self.leg_velocity) * h;
        self.compression += self.leg_velocity * h;
        if self.compression < 0.0 {
            self.compression = 0.0;
            self.leg_velocity = self.leg_velocity.max(0.0);
        } else if self.compression > 1.0 {
            self.compression = 1.0;
            self.leg_velocity = self.leg_velocity.min(0.0);
        }
    }

    fn drag(&self) -> f32 {
        DRAG * (1.0 - TUCK_DRAG_CUT * self.tuck)
    }

    fn ground_step(&mut self, c: &SkiControls, h: f32, pop: bool) {
        let n = surface_normal(self.position.x, self.position.z);
        let gravity = Vec3::NEG_Y * G;
        let mut v = self.velocity;

        // Jump preload and pop along the surface normal.
        if c.jump {
            self.preload = (self.preload + h / PRELOAD_TIME).min(1.0);
        }
        if pop {
            v += n * (POP_BASE + POP_GAIN * self.preload);
        }
        if !c.jump {
            self.preload = 0.0;
        }

        v += (gravity - n * gravity.dot(n)) * h;
        let (f, _) = frame(n, self.heading);

        // Propulsion along the skis.
        let along = v.dot(f);
        let pushing = c.push > 0.0 && c.brake < 0.1 && !self.hockey && along > -0.5;
        let blend = smoothstep(3.5, 4.5, along);
        let skate_t = if pushing { c.push * (1.0 - blend) } else { 0.0 };
        let pole_t = if pushing {
            c.push * blend * (1.0 - smoothstep(8.0, 9.0, along))
        } else {
            0.0
        };
        self.skate += (skate_t - self.skate) * ease(8.0, h);
        self.pole += (pole_t - self.pole) * ease(8.0, h);
        // V2: once moving, the poles push on every stroke.
        let v2_t = skate_t * smoothstep(V2_SPEED.0, V2_SPEED.1, along);
        self.v2 += (v2_t - self.v2) * ease(8.0, h);
        // A skate stroke (one leg push, half of `skate_phase`) takes about 0.95 s from a standing
        // start and 0.6 s at 4 m/s, a double-pole cycle 1.2 s up to 3.5 m/s and 0.9 s from 7 m/s.
        // The phases run at a rate that follows the (eased) weights, so a stroke starts from rest
        // and slows to a stop smoothly instead of freezing; a fresh start begins at the first push.
        if skate_t > 0.01 || self.skate > 1e-4 {
            let stroke = stroke_time(along);
            let rate = (self.skate * 2.0).min(1.0);
            self.skate_phase = (self.skate_phase + rate * PI * h / stroke).rem_euclid(TAU);
        } else {
            self.skate_phase = 0.0;
        }
        if pole_t > 0.01 || self.pole > 1e-4 {
            let cycle = lerp(
                POLE_CYCLE_SLOW,
                POLE_CYCLE_FAST,
                ((along - 3.5) / 3.5).clamp(0.0, 1.0),
            );
            let rate = (self.pole * 2.5).min(1.0);
            self.pole_phase = (self.pole_phase + rate * TAU * h / cycle).rem_euclid(TAU);
        } else {
            self.pole_phase = 0.0;
        }
        // The leg push and the V2 pole push are half-sines inside their windows, so speed is gained
        // while the leg extends and the trunk crunches and the skier glides in between. Each mean
        // over a stroke / cycle is the constant named.
        let (_, p) = skate_stroke(self.skate_phase);
        let skate_acc = if p < STROKE_PUSH {
            SKATE_ACCEL * self.skate * (PI / 2.0 / STROKE_PUSH) * (PI * p / STROKE_PUSH).sin()
        } else {
            0.0
        };
        let v2_acc = if p < POLE_PUSH_SPAN {
            V2_ACCEL * self.v2 * (PI / 2.0 / POLE_PUSH_SPAN) * (PI * p / POLE_PUSH_SPAN).sin()
        } else {
            0.0
        };
        let x = self.pole_phase / TAU;
        let pole_acc = if x < POLE_PUSH_SPAN {
            POLE_ACCEL * self.pole * (PI / 2.0 / POLE_PUSH_SPAN) * (PI * x / POLE_PUSH_SPAN).sin()
        } else {
            0.0
        };
        v += f * ((skate_acc + v2_acc + pole_acc) * h);

        // Drag and snow friction (friction stops the skier outright at rest).
        v /= 1.0 + self.drag() * v.length() * h;
        let s = v.length();
        if s > 1e-6 {
            v *= (s - SNOW_FRICTION * G * n.y * h).max(0.0) / s;
        }

        // Braking.
        let s = v.length();
        let braking = c.brake > 0.05;
        if braking && !self.hockey && s >= HOCKEY_SPEED {
            self.hockey = true;
            self.hockey_side = if c.steer.abs() > 0.2 {
                c.steer.signum()
            } else if self.lean.abs() > 0.05 {
                self.lean.signum()
            } else {
                1.0
            };
        }
        if self.hockey && (!braking || s < HOCKEY_END) {
            self.hockey = false;
        }
        if self.hockey && c.steer.abs() > 0.2 {
            self.hockey_side = c.steer.signum();
        }
        let plow_t = if braking && !self.hockey {
            c.brake
        } else {
            0.0
        };
        self.plow += (plow_t - self.plow) * ease(10.0, h);
        if self.plow > 0.02 && s > 1e-6 {
            v *= (s - PLOW_DECEL * self.plow * h).max(0.0) / s;
        }

        // Edge and heading: carve, pivot at low speed, or swing across for a hockey stop.
        let s = v.length();
        let sigma = if self.switch { -1.0 } else { 1.0 };
        let edge_max = 0.4 + 0.6 * smoothstep(1.0, 8.0, s);
        let edge_target = if self.hockey {
            self.hockey_side * HOCKEY_EDGE
        } else {
            sigma * c.steer * edge_max
        };
        let step = (edge_target - self.edge) * ease(EDGE_RESPONSE, h);
        self.edge += step.clamp(-EDGE_SLEW * h, EDGE_SLEW * h);
        let turn_target = if self.hockey {
            let dir = if s > 0.5 { v / s } else { f };
            let axis = if self.switch { -dir } else { dir };
            let target = yaw_of(axis) - self.hockey_side * HOCKEY_ANGLE;
            (wrap(target - self.heading) * HOCKEY_RATE).clamp(-HOCKEY_RATE, HOCKEY_RATE)
        } else {
            let pivot = 1.0 - smoothstep(1.5, 6.0, s);
            -v.dot(f) * self.edge.sin() / SIDECUT_RADIUS - c.steer * PIVOT_RATE * pivot
        };
        self.turn_rate += (turn_target - self.turn_rate) * ease(TURN_RESPONSE, h);
        self.heading = wrap(self.heading + self.turn_rate * h);

        // Edge grip removes the sideways velocity of the skis up to its limit; the rest skids.
        let (f, r) = frame(n, self.heading);
        let speed_in = v.length();
        let side = v.dot(r);
        let cap = G
            * (GRIP_BASE + GRIP_EDGE * self.edge.sin().abs())
            * (1.0 - SKID_GRIP_LOSS * self.skid);
        // A hockey stop scrapes the skis across the snow: friction opposes the travel itself, so
        // the velocity slows along its own line and can never turn back.
        let (removed, left) = if self.hockey && speed_in > 1e-6 {
            let slow = (cap * h).min(speed_in);
            v *= 1.0 - slow / speed_in;
            (slow * side / speed_in, side.abs())
        } else {
            let removed = side.clamp(-cap * h, cap * h);
            v -= r * removed;
            (removed, (side - removed).abs())
        };
        self.skid += (((left - 0.4) / 2.0).clamp(0.0, 1.0) - self.skid) * ease(12.0, h);
        let lean_target = (-removed / h / G).atan().clamp(-MAX_LEAN, MAX_LEAN);
        spring(
            &mut self.lean,
            &mut self.lean_rate,
            lean_target,
            LEAN_OMEGA,
            h,
        );
        self.lean = self.lean.clamp(-MAX_LEAN, MAX_LEAN);

        let along = v.dot(f);
        if along < -SWITCH_SPEED {
            self.switch = true;
        } else if along > SWITCH_SPEED {
            self.switch = false;
        }

        // Catching an edge: sliding fast with the leading edge dug in.
        let s = v.length();
        let catching = !self.hockey
            && s > CATCH_SPEED
            && self.edge.abs() > CATCH_EDGE
            && self.skid > 0.5
            && self.edge * side > 0.0
            && side.abs() > 0.866 * speed_in;
        self.catch_time = if catching {
            self.catch_time + h
        } else {
            (self.catch_time - 2.0 * h).max(0.0)
        };
        if self.catch_time > CATCH_TIME {
            self.velocity = v;
            self.fail(SkiCrashReason::CaughtEdge, s);
            return;
        }

        // Move. Leave the snow when the free-flight path clears it (lips, crests, pop); only a
        // sub-millimetre gap is snapped.
        let x = self.position.x + v.x * h;
        let z = self.position.z + v.z * h;
        let ground = terrain_height(x, z);
        let free_y = self.position.y + v.y * h - 0.5 * G * h * h;
        self.distance += Vec3::new(v.x, 0.0, v.z).length() * h;
        if pop || free_y - ground > SEPARATION_TOL {
            v += n * (gravity.dot(n) * h);
            self.position = Vec3::new(x, free_y.max(ground), z);
            self.velocity = v;
            self.take_off(c);
            return;
        }
        self.position = Vec3::new(x, ground, z);
        let n2 = surface_normal(x, z);
        // Terrain curvature: absorb the closing speed with the legs, keep the speed.
        let closing = -v.dot(n2);
        if closing > 0.0 {
            self.compression = (self.compression + closing * h / LEG_TRAVEL).min(1.0);
        }
        let speed = v.length();
        v += n2 * closing;
        if let Some(dir) = v.try_normalize() {
            v = dir * speed;
        }
        self.velocity = v;
        let before = self.rotation;
        self.rotation = orient(n2, self.heading);
        let (axis, mut angle) = (self.rotation * before.inverse()).to_axis_angle();
        if angle > PI {
            angle -= TAU;
        }
        self.angular_velocity = axis * (angle / h);
        self.air_time = 0.0;
    }

    fn take_off(&mut self, c: &SkiControls) {
        self.grounded = false;
        self.air_time = 0.0;
        self.air_spin = 0.0;
        self.air_flip = 0.0;
        self.plow = 0.0;
        self.catch_time = 0.0;
        self.hockey = false;
        // Wind-up: holding spin at take-off starts the rotation immediately.
        self.angular_velocity = self.up() * (-c.air_yaw * WINDUP_YAW);
    }

    fn air_step(&mut self, c: &SkiControls, h: f32) {
        self.air_time += h;
        self.preload = (self.preload - 3.0 * h).max(0.0);
        for w in [
            &mut self.skate,
            &mut self.pole,
            &mut self.v2,
            &mut self.skid,
        ] {
            *w -= *w * ease(6.0, h);
        }
        self.edge -= self.edge * ease(AIR_SETTLE, h);
        spring(
            &mut self.lean,
            &mut self.lean_rate,
            0.0,
            LEAN_OMEGA * 0.5,
            h,
        );

        let mut w = self.body_rates();
        let (pitch_max, pitch_accel, pitch_damp, roll_max) = if c.flip {
            (AIR_FLIP_RATE, FLIP_ACCEL, FLIP_DAMP, FLIP_ROLL_RATE)
        } else {
            (AIR_PITCH_RATE, PITCH_ACCEL, AIR_DAMP, AIR_ROLL_RATE)
        };
        w.x = axis_rate(
            w.x,
            c.air_pitch * pitch_max,
            pitch_accel,
            pitch_damp,
            c.air_pitch.abs() > 0.02,
            h,
        );
        w.y = axis_rate(
            w.y,
            -c.air_yaw * AIR_YAW_RATE,
            YAW_ACCEL,
            YAW_DAMP,
            c.air_yaw.abs() > 0.02,
            h,
        );
        w.z = axis_rate(
            w.z,
            -c.air_roll * roll_max,
            ROLL_ACCEL,
            AIR_DAMP,
            c.air_roll.abs() > 0.02,
            h,
        );
        self.air_spin -= w.y * h;
        self.air_flip += w.x * h;

        // Ballistic centre of mass; the body turns about it.
        let com = Vec3::Y * COM_HEIGHT;
        let centre = self.position + self.rotation * com;
        let mut v = self.velocity;
        v.y -= G * h;
        v /= 1.0 + self.drag() * v.length() * h;
        let q = (self.rotation * Quat::from_scaled_axis(w * h)).normalize();
        self.position = centre + v * h - q * com;
        self.rotation = q;
        self.angular_velocity = q * w;
        self.velocity = v;
        let f = self.forward();
        if f.x.hypot(f.z) > 0.2 {
            self.heading = yaw_of(f);
        }

        // Head or pelvis in the snow.
        let head = HEAD_HEIGHT * (1.0 - TUCK_HEAD_DROP * self.tuck);
        for (height, radius) in [(head, HEAD_RADIUS), (PELVIS_HEIGHT, PELVIS_RADIUS)] {
            let p = self.position + q * (Vec3::Y * height);
            if terrain_height(p.x, p.z) > p.y - radius {
                self.fail(SkiCrashReason::BodyImpact, v.length());
                return;
            }
        }

        // Skis touching down: binding point, tip and tail.
        let pen = [
            Vec3::ZERO,
            Vec3::new(0.0, SKI_TIP_RISE, -SKI_FRONT),
            Vec3::new(0.0, 0.0, SKI_BACK),
        ]
        .iter()
        .map(|&o| {
            let p = self.position + q * o;
            terrain_height(p.x, p.z) - p.y
        })
        .fold(f32::MIN, f32::max);
        if pen > 0.0 {
            let n = surface_normal(self.position.x, self.position.z);
            if self.air_time > 0.1 || v.dot(n) < 0.0 {
                self.land(pen, n);
            }
        }
    }

    /// First contact: judged on the pre-correction state, then either snapped to the snow or a crash.
    fn land(&mut self, pen: f32, n: Vec3) {
        let v = self.velocity;
        let impact = (-v.dot(n)).max(0.0);
        let tilt = self.up().dot(n).clamp(-1.0, 1.0).acos();
        let spin = self.angular_velocity.length();
        let tangent = v - n * v.dot(n);
        let speed = tangent.length();
        let f = self.forward();
        let ski = f - n * f.dot(n);
        let cross = match (speed > LAND_CROSS_MIN_SPEED, ski.try_normalize()) {
            (true, Some(ski)) => ski.dot(tangent / speed).abs().clamp(0.0, 1.0).acos(),
            _ => 0.0,
        };
        self.impact = impact;
        self.last_spin = self.air_spin;
        self.last_flip = self.air_flip;
        let reason = if impact > HARD_IMPACT_SPEED {
            Some(SkiCrashReason::HardImpact)
        } else if tilt > LAND_TILT_LIMIT || cross > LAND_CROSS_LIMIT || spin > LAND_SPIN_LIMIT {
            Some(SkiCrashReason::BadLanding)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.position.y += pen;
            self.fail(reason, impact);
            return;
        }
        // Snap the skis to the surface keeping the heading; a backwards touchdown is a switch
        // landing. The drawn pose starts from the touchdown attitude and height and settles.
        if f.x.hypot(f.z) > 0.1 {
            self.heading = yaw_of(f);
        }
        self.switch = speed > SWITCH_SPEED && tangent.dot(ski) < 0.0;
        self.velocity = tangent;
        let before = (self.position, self.rotation);
        self.position.y = terrain_height(self.position.x, self.position.z);
        self.rotation = orient(n, self.heading);
        let mut offset = before.1 * self.rotation.inverse();
        if offset.w < 0.0 {
            offset = -offset;
        }
        self.land_tilt = offset.to_scaled_axis();
        self.land_shift = before.0 - self.position;
        self.land_tilt_rate = Vec3::ZERO;
        self.land_shift_rate = Vec3::ZERO;
        self.turn_rate = 0.0;
        self.angular_velocity = Vec3::ZERO;
        self.grounded = true;
        self.air_time = 0.0;
        self.air_spin = 0.0;
        self.air_flip = 0.0;
        self.leg_velocity = (impact / LAND_ABSORB_SPEED).min(1.0) * LAND_LEG_KICK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bike::HILL_LIP_Z;
    use std::f32::consts::FRAC_PI_2;

    const DT: f32 = 1.0 / 120.0;
    /// Flat arena, far from the jump lane and the hill.
    const FLAT_X: f32 = -40.0;

    fn run(s: &mut Skier, c: &SkiControls, seconds: f32) {
        for _ in 0..(seconds / DT).round() as usize {
            s.step(c, DT);
        }
    }

    fn flat(speed: f32) -> Skier {
        let mut s = Skier::default();
        s.reset_at(FLAT_X, 0.0, 0.0, speed);
        s
    }

    fn travel_yaw(s: &Skier) -> f32 {
        (-s.velocity.x).atan2(-s.velocity.z)
    }

    #[test]
    fn rest_on_flat_stays_on_snow() {
        let mut s = flat(0.0);
        run(&mut s, &SkiControls::default(), 5.0);
        assert!(s.grounded && s.crash.is_none());
        assert!(
            s.position.distance(Vec3::new(FLAT_X, 0.0, 0.0)) < 1e-3,
            "{:?}",
            s.position
        );
        assert!(s.speed() < 1e-3);
    }

    #[test]
    fn straight_hill_run_exceeds_14ms_before_the_kicker() {
        let mut s = Skier::default();
        s.reset_at(HILL_X, 15.0, 0.0, 0.0);
        let mut top = 0.0_f32;
        for _ in 0..(40.0 / DT) as usize {
            s.step(&SkiControls::default(), DT);
            assert!(s.crash.is_none(), "{:?}", s.crash);
            if s.position.z < -60.0 {
                break;
            }
            top = top.max(s.speed());
        }
        assert!(
            s.position.z < -60.0,
            "never reached the foot: {:?}",
            s.position
        );
        assert!(top > 14.0, "top speed {top}");
        assert!((s.position.x - HILL_X).abs() < 0.5);
    }

    #[test]
    fn carving_turns_heading_and_velocity_together() {
        let mut s = flat(10.0);
        let c = SkiControls {
            steer: 0.5,
            ..default_controls()
        };
        run(&mut s, &c, 2.0);
        assert!(s.heading < -0.4, "heading {}", s.heading);
        assert!(
            wrap(travel_yaw(&s) - s.heading).abs() < 0.15,
            "slip {}",
            wrap(travel_yaw(&s) - s.heading)
        );
        assert!(s.skid < 0.3 && s.lean > 0.05 && s.edge > 0.3);
        assert!(s.speed() > 8.0);
    }

    #[test]
    fn switch_carve_mirrors() {
        let mut s = flat(8.0);
        s.heading = PI;
        s.rotation = orient(Vec3::Y, PI);
        let c = SkiControls {
            steer: 1.0,
            ..default_controls()
        };
        run(&mut s, &c, 1.5);
        assert!(s.switch);
        assert!(wrap(travel_yaw(&s)) < -0.1, "travel yaw {}", travel_yaw(&s));
        assert!(s.edge < 0.0 && s.lean < 0.0);
    }

    #[test]
    fn hockey_stop_from_15ms_stops_without_reversing() {
        let mut s = flat(15.0);
        let c = SkiControls {
            brake: 1.0,
            ..default_controls()
        };
        let mut stopped = false;
        for _ in 0..(8.0 / DT) as usize {
            s.step(&c, DT);
            assert!(s.velocity.z <= 1e-3, "reversed {:?}", s.velocity);
            assert!(s.crash.is_none());
            if s.speed() < 0.05 {
                stopped = true;
                break;
            }
        }
        assert!(
            stopped && -s.position.z < 25.0,
            "stopped={stopped} distance {}",
            -s.position.z
        );
        assert!(s.skid > 0.0 || s.speed() < 0.05);
    }

    #[test]
    fn skate_then_double_pole_gain_speed() {
        let mut s = flat(0.0);
        let c = SkiControls {
            push: 1.0,
            ..default_controls()
        };
        run(&mut s, &c, 3.5);
        assert!((3.0..5.0).contains(&s.speed()), "skate speed {}", s.speed());
        run(&mut s, &c, 15.0);
        assert!(
            s.speed() > 6.0 && s.speed() < 9.5,
            "pole speed {}",
            s.speed()
        );
    }

    #[test]
    fn strokes_start_slow_speed_up_and_gain_speed_only_while_pushing() {
        let mut s = flat(0.0);
        let c = SkiControls {
            push: 1.0,
            ..default_controls()
        };
        let (mut strokes, mut cycles) = (Vec::new(), Vec::new());
        let (mut gain_push, mut gain_glide) = (0.0_f32, 0.0_f32);
        let (mut stroke_ticks, mut cycle_ticks) = (0usize, 0usize);
        let mut prev = (skate_stroke(0.0).0, 0.0_f32, 0.0_f32);
        for _ in 0..(20.0 / DT) as usize {
            s.step(&c, DT);
            let (idx, p) = skate_stroke(s.skate_phase);
            stroke_ticks += 1;
            cycle_ticks += 1;
            if s.skate > 0.9 && s.speed() < 3.5 && s.speed() > 1.0 {
                let dv = s.speed() - prev.1;
                if p < STROKE_PUSH {
                    gain_push += dv;
                } else {
                    gain_glide += dv;
                }
            }
            if idx != prev.0 {
                strokes.push((stroke_ticks as f32 * DT, s.speed()));
                stroke_ticks = 0;
            }
            if s.pole_phase < prev.2 && s.pole > 0.9 {
                cycles.push((cycle_ticks as f32 * DT, s.speed()));
                cycle_ticks = 0;
            } else if s.pole < 0.9 {
                cycle_ticks = 0;
            }
            prev = (idx, s.speed(), s.pole_phase);
        }
        // Strokes: slow from a standing start, about 0.6 s a leg by 4 m/s.
        assert!(strokes[0].0 > 0.85, "first stroke {:?}", strokes[0]);
        assert!(
            strokes.iter().any(|&(t, v)| v > 3.0 && t < 0.7),
            "{strokes:?}"
        );
        // Double-pole cycles run 1.2 s to 0.9 s and shorten with speed.
        cycles.remove(0); // the first one is the partial cycle the pole weight arrived in
        assert!(cycles.len() >= 3, "{cycles:?}");
        assert!(
            cycles.iter().all(|&(t, _)| (0.86..1.25).contains(&t)),
            "{cycles:?}"
        );
        assert!(
            cycles[0].0 > cycles[cycles.len() - 1].0 + 0.08,
            "{cycles:?}"
        );
        // Speed is gained in the push window; between pushes only drag and friction act.
        assert!(gain_push > 0.5, "gain {gain_push}");
        assert!(gain_glide < 0.0, "glide gain {gain_glide}");
    }

    #[test]
    fn jump_on_flat_lifts_off_and_lands_safely() {
        let mut s = flat(0.0);
        run(
            &mut s,
            &SkiControls {
                jump: true,
                ..default_controls()
            },
            0.4,
        );
        assert!(s.grounded && s.preload > 0.99 && s.compression > 0.3);
        let mut top = 0.0_f32;
        let mut left = false;
        for _ in 0..(3.0 / DT) as usize {
            s.step(&default_controls(), DT);
            left |= !s.grounded;
            top = top.max(s.position.y);
        }
        assert!(left && top > 0.5 && top < 1.5, "apex {top}");
        assert!(s.grounded && s.crash.is_none() && s.impact > 3.0);
    }

    #[test]
    fn kicker_straight_air_lands_safely() {
        let mut s = Skier::default();
        s.reset_at(HILL_X, 15.0, 0.0, 0.0);
        let mut airborne = 0.0_f32;
        let mut closing = 0.0_f32;
        for _ in 0..(60.0 / DT) as usize {
            s.step(&SkiControls::default(), DT);
            airborne = airborne.max(s.air_time);
            if s.position.z < HILL_LIP_Z - 10.0 && s.grounded {
                closing = s.impact;
                break;
            }
        }
        assert!(s.crash.is_none(), "{:?} at {:?}", s.crash, s.position);
        assert!(airborne > 1.0, "airtime {airborne}");
        assert!(
            s.grounded && (10.0..17.0).contains(&closing),
            "closing {closing}"
        );
    }

    #[test]
    fn sideways_landing_crashes() {
        let mut s = flat(10.0);
        s.grounded = false;
        s.position.y = 0.5;
        s.velocity = Vec3::new(0.0, -1.0, -10.0);
        s.rotation = Quat::from_rotation_y(FRAC_PI_2);
        for _ in 0..(1.0 / DT) as usize {
            s.step(&default_controls(), DT);
        }
        assert_eq!(s.crash.map(|c| c.reason), Some(SkiCrashReason::BadLanding));
    }

    #[test]
    fn inverted_drop_crashes_and_freezes() {
        let mut s = flat(0.0);
        s.grounded = false;
        s.position.y = 3.0;
        s.rotation = Quat::from_rotation_z(PI);
        for _ in 0..(2.0 / DT) as usize {
            s.step(&default_controls(), DT);
        }
        let crash = s.crash.expect("inverted landing must crash");
        let frozen = s.position;
        s.step(&default_controls(), DT);
        assert_eq!(s.position, frozen);
        assert!(s.crash.unwrap().elapsed > crash.elapsed);
    }

    #[test]
    fn air_servos_spin_and_flip() {
        let mut s = flat(0.0);
        s.grounded = false;
        s.position.y = 40.0;
        let spin = SkiControls {
            air_yaw: 1.0,
            ..default_controls()
        };
        run(&mut s, &spin, 1.0);
        assert!(s.air_spin > 5.5 && s.air_spin < 7.5, "spin {}", s.air_spin);
        let flip = SkiControls {
            air_pitch: 1.0,
            flip: true,
            ..default_controls()
        };
        run(&mut s, &flip, 1.0);
        assert!(s.air_flip > 4.5, "flip {}", s.air_flip);
    }

    fn default_controls() -> SkiControls {
        SkiControls::default()
    }
}
