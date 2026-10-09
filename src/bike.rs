//! Arcade bicycle dynamics, fixed-step, no physics engine.
//!
//! Frame: bike-local forward is -Z, right +X, up +Y. The orientation is a free quaternion, so
//! flips, barrel rolls and tumbles are physical. `yaw`, `pitch`, `roll` are read-only readouts of
//! it: `yaw` heading (+left), `pitch` nose elevation (+up), `roll` lean `atan2(right.y, up.y)` (+left).
//! `pitch_rate`, `yaw_rate`, `roll_rate` are the body-frame angular velocity about the bike's own
//! X (right), Y (up) and Z (back) axes, rad/s.
//! `Controls.steering` and `Controls.air_roll` are positive to the RIGHT, `air_pitch` positive nose-up.
//!
//! Model: the chassis is a point mass with an arcade rotational inertia, carried by two massless
//! wheels on spring/damper struts pushing along the terrain normal. Everything is per unit mass.
//! Every ground force (strut, drive, brake) acts at the tyre patch, so its pitch torque is the real
//! root-to-patch lever crossed with the force; drive and brake forces are capped by the wheel's
//! strut load (`GROUND_MU`), so a wheel that unloads under braking fades out continuously rather
//! than switching on a contact flag. Steering and lean are arcade-assisted. A landing is judged on
//! touchdown against the surface (attitude, slip, spin, closing speed, grip support); a failed
//! landing, or any rider/frame proxy striking the floor while riding, is a crash: the bike alone
//! (wheels and frame proxies) then tumbles under gravity and friction until reset, while the
//! detached rider is simulated separately (`ragdoll`).

use bevy::prelude::{Quat, Resource, Vec2, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub const WHEEL_RADIUS: f32 = 0.35;
pub const WHEELBASE: f32 = 1.30;
pub const SUSPENSION_REST: f32 = 0.40;
/// Hard upper bound of `Bike::suspension` (bump stop), metres.
pub const MAX_COMPRESSION: f32 = 0.22;
/// Bound of `Bike::steering`, radians.
pub const MAX_STEER_ANGLE: f32 = 0.5;

const GRAVITY: f32 = 9.81;
const STEP_HZ: f32 = 120.0;
/// Larger `dt` is clamped, then split into <= 1/120 s substeps (so <= 8 substeps).
const MAX_FRAME_DT: f32 = 1.0 / 15.0;

// Suspension strut limits (per wheel, per unit mass). Spring and damper come from `BikeProfile`;
// Downhill static sag = g / (2 * 98) = 5 cm.
const STOP_START: f32 = 0.14;
const STOP_K: f32 = 600.0;
const MAX_WHEEL_FORCE: f32 = 60.0;
const GYRATION_SQ: f32 = 0.2;
// Tyre friction coefficient: the longitudinal (drive/brake) force a wheel can transmit is at most
// this times its strut load, so a wheel that unloads fades out smoothly instead of switching.
const GROUND_MU: f32 = 1.2;
const SLOPE_EPS: f32 = 0.05;

// Drive. Pedal force is capped at low speed (scaled by profile power / Downhill power) and
// power-limited above it.
const PEDAL_POWER: f32 = 4.5;
const PEDAL_FORCE: f32 = 3.5;
const MIN_POWER_SPEED: f32 = 0.5;
const ROLLING_RESISTANCE: f32 = 0.3;
const BRAKE_DECEL: f32 = 9.0;
const BRAKE_FRONT: f32 = 0.65;
const BRAKE_REAR: f32 = 0.35;
const TIRE_GRIP: f32 = 20.0;

// Steering and lean.
const STEER_RESPONSE: f32 = 12.0;
const STEER_SPEED_REF: f32 = 5.0;
const MAX_LATERAL_ACCEL: f32 = 7.0;
const MAX_LEAN: f32 = 0.5;
const LEAN_GAIN: f32 = 0.9;
const LEAN_RESPONSE: f32 = 8.0;

// Tricks.
const HOP_PITCH_KICK: f32 = 1.5;
const WHEELIE_TORQUE: f32 = 50.0;
const WHEELIE_PITCH: f32 = 0.75;
const WHEELIE_SOFT: f32 = 0.35;
const WHEELIE_DAMP: f32 = 14.0;
const AIR_PITCH_ACCEL: f32 = 10.0;
const AIR_PITCH_DAMP: f32 = 3.0;
const AIR_ROLL_ACCEL: f32 = 8.0;
const AIR_ROLL_DAMP: f32 = 3.0;

// Balance assists (grounded manual / nose manual): a pitch servo about the ground slope that
// fades in with forward speed, so a standing bike still drops its wheel.
const MANUAL_PITCH: f32 = 0.5;
const NOSE_PITCH: f32 = 0.4;
const BALANCE_SPEED: (f32, f32) = (1.0, 3.0);
const BALANCE_K: f32 = 60.0;
const BALANCE_D: f32 = 14.0;
/// The assist holds fully within the first pitch error (rad) and lets go by the second.
const BALANCE_CAPTURE: (f32, f32) = (0.2, 0.4);
/// Pitch acceleration the rider's weight shift gives on one wheel, rad/s^2 (+ nose up).
const RIDER_BALANCE: f32 = 4.0;
/// Pitch to the ground under the one grounded wheel past which the rider falls off: looped out
/// backwards on the rear wheel, over the bars on the front, rad.
const TIP_OVER_PITCH: f32 = 1.25;
/// Strut force (in g) whose torque pitches a bike landing on one wheel; the rest is absorbed.
const ONE_WHEEL_TORQUE_G: f32 = 1.5;

// Air yaw: physical spin about the bike's own up axis.
const AIR_YAW_ACCEL: f32 = 12.0;
const AIR_YAW_DAMP: f32 = 1.5;
const MAX_AIR_YAW: f32 = 7.0;

// Drivetrain visuals.
const GEAR_RATIO: f32 = 0.45;
const MIN_CADENCE: f32 = 4.0;
const MAX_CADENCE: f32 = 15.0;
const AIR_WHEEL_DECAY: f32 = 0.2;

// Course: three jumps, crest forward distances along -Z, quadratic rise and fall faces.
const JUMP_CRESTS: [f32; 3] = [35.0, 85.0, 135.0];
const JUMP_HEIGHT: f32 = 1.8;
const JUMP_RISE: f32 = 7.0;
const JUMP_FALL: f32 = 9.0;
const JUMP_ROUND: f32 = 0.8;

// Tyre torus: the contact reach of a tilted wheel shrinks from WHEEL_RADIUS to this tube radius.
const TIRE_TUBE: f32 = 0.05;

// Full rotations. `Controls.flip` raises air pitch/roll authority and releases rate damping.
// Without it, damping fades out beyond `ASSIST_TILT`; there is no target-attitude torque.
const FLIP_AUTHORITY: f32 = 2.2;
const FLIP_MAX_RATE: f32 = 7.0;
const ASSIST_TILT: (f32, f32) = (0.7, 1.3);

// Landing judgement, all relative to the surface normal under the touching wheels.
/// Largest nose-to-surface angle, radians.
pub const LAND_PITCH_LIMIT: f32 = 1.05;
/// Largest sideways angle of the axle to the surface, radians.
pub const LAND_ROLL_LIMIT: f32 = 0.6;
/// Largest angle between the bike's up axis and the surface normal, radians.
pub const LAND_TILT_LIMIT: f32 = 1.15;
/// Largest angle between travel and heading along the surface, radians (above `LAND_SLIP_MIN_SPEED`).
pub const LAND_SLIP_LIMIT: f32 = 0.7;
const LAND_SLIP_MIN_SPEED: f32 = 3.0;
/// Largest angular speed at touchdown, rad/s.
pub const LAND_SPIN_LIMIT: f32 = 5.0;
/// Wheel closing speed along the surface normal that is always a crash, m/s. A flat landing from
/// an 8.6 m drop closes at sqrt(2 g h) = 13 m/s, about what a full-suspension MTB can absorb.
pub const HARD_IMPACT_SPEED: f32 = 13.0;
/// Fraction of `HARD_IMPACT_SPEED` that already crashes a landing using more than
/// `LAND_MARGINAL_USE` of any attitude limit: the suspension is spent and nothing is left to
/// correct the error.
const HARD_IMPACT_MARGINAL: f32 = 0.65;
/// Fraction of a pitch/roll/tilt/slip/spin limit above which a landing counts as marginal.
const LAND_MARGINAL_USE: f32 = 0.5;
/// Hands (or feet) count as holding when `2 - released[0] - released[1]` reaches this.
const SUPPORT_MIN: f32 = 0.5;
/// Rider/frame proxy penetration that counts as hitting the floor, metres.
const BODY_STRIKE_DEPTH: f32 = 0.04;

// Crash tumble: sphere proxies against the terrain, impulse resolved.
const CRASH_ITERATIONS: usize = 3;
const CRASH_RESTITUTION: f32 = 0.3;
const CRASH_BOUNCE_MIN: f32 = 1.0;
const CRASH_FRICTION: f32 = 0.6;
const CRASH_WHEEL_FRICTION: f32 = 0.15;
const CRASH_SPIN_DAMP: f32 = 0.8;
const CRASH_AIR_DRAG: f32 = 0.02;
const SLEEP_SPEED: f32 = 0.08;
const SLEEP_SPIN: f32 = 0.15;

// Swept stepping: a substep moves the root or any rotating point at most this far, metres.
const MAX_STEP_TRAVEL: f32 = 0.08;
const MAX_SUBSTEPS: u32 = 32;
const TIP_RADIUS: f32 = 0.9;

// Hill at x = HILL_X, separate from the jump lane at x = 0. Relief lies within x [-4, 108], z [-150, 65].
pub const HILL_X: f32 = 60.0;
/// Summit start point, on the flat top.
pub const HILL_START_Z: f32 = 24.0;
/// Kicker lip.
pub const HILL_LIP_Z: f32 = -84.0;
pub const HILL_HEIGHT: f32 = 32.0;
const HILL_SUMMIT_Z: (f32, f32) = (20.0, 45.0);
const HILL_FOOT_Z: f32 = -60.0;
const HILL_BACK_Z: f32 = 65.0;
const HILL_HALF_FLAT: f32 = 12.0;
const HILL_HALF_WIDTH: f32 = 48.0;
const KICKER_HEIGHT: f32 = 7.0;
const KICKER_LENGTH: f32 = 10.0;
const KICKER_FALL: f32 = 4.0;
const KICKER_ROUND: f32 = 1.0;
/// Landing below the kicker: a table rising from the kicker's foot to a crest, then a long
/// downslope that both the shorter ski flights and the longer bike flights land on. Crest, height
/// and the lengths of its faces, m.
const HILL_LANDING_Z: f32 = -102.0;
const HILL_LANDING_HEIGHT: f32 = 6.0;
const HILL_LANDING_RISE: f32 = 14.0;
const HILL_LANDING_FALL: f32 = 26.0;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// C1 polynomial minimum: rounds the crest where the rise and fall parabolas meet.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Height of a mound `s` metres past its crest (negative before it): quadratic rise and fall
/// faces, rounded where they meet.
fn mound(s: f32, height: f32, rise: f32, fall: f32, round: f32) -> f32 {
    let up = height * (1.0 + s / rise).max(0.0).powi(2);
    let down = height * (1.0 - s / fall).max(0.0).powi(2);
    smooth_min(up, down, round)
}

/// Height of one jump `s` metres past its crest (negative before it).
fn jump_profile(s: f32) -> f32 {
    mound(s, JUMP_HEIGHT, JUMP_RISE, JUMP_FALL, JUMP_ROUND)
}

/// Hill landing transition: its curved fall face meets the descent of a kicker flight at a few
/// degrees, so the bike closes on the surface at a few m/s instead of its full fall speed.
fn landing_profile(z: f32) -> f32 {
    mound(
        HILL_LANDING_Z - z,
        HILL_LANDING_HEIGHT,
        HILL_LANDING_RISE,
        HILL_LANDING_FALL,
        JUMP_ROUND,
    )
}

/// Hill height profile along z: flat summit, smooth descent to the foot, smooth back taper.
fn hill_profile(z: f32) -> f32 {
    HILL_HEIGHT
        * smoothstep(HILL_FOOT_Z, HILL_SUMMIT_Z.0, z)
        * (1.0 - smoothstep(HILL_SUMMIT_Z.1, HILL_BACK_Z, z))
}

/// Kicker height `HILL_LIP_Z - z` metres past its lip: eased rise, then a steep rounded backside.
fn kicker_profile(z: f32) -> f32 {
    let s = HILL_LIP_Z - z;
    let u = (1.0 + s / KICKER_LENGTH).clamp(0.0, 1.0);
    let rise = KICKER_HEIGHT * u * u * (2.0 - u) + KICKER_HEIGHT / KICKER_LENGTH * s.max(0.0);
    let fall = KICKER_HEIGHT * (1.0 - s / KICKER_FALL).max(0.0).powi(2);
    smooth_min(rise, fall, KICKER_ROUND)
}

/// Flat showcase floor with three smooth jump ramps on x = 0 and the big hill + kicker at `HILL_X`.
pub fn terrain_height(x: f32, z: f32) -> f32 {
    let jump_width = 1.0 - smoothstep(2.0, 4.0, x.abs());
    let jumps: f32 = JUMP_CRESTS.iter().map(|d| jump_profile(-z - d)).sum();
    let hill_width = 1.0 - smoothstep(HILL_HALF_FLAT, HILL_HALF_WIDTH, (x - HILL_X).abs());
    jump_width * jumps + hill_width * (hill_profile(z) + kicker_profile(z) + landing_profile(z))
}

/// Terrain gradient (dh/dx, dh/dz).
fn terrain_slope(x: f32, z: f32) -> Vec2 {
    Vec2::new(
        terrain_height(x + SLOPE_EPS, z) - terrain_height(x - SLOPE_EPS, z),
        terrain_height(x, z + SLOPE_EPS) - terrain_height(x, z - SLOPE_EPS),
    ) / (2.0 * SLOPE_EPS)
}

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
    /// Bike type; selects the authored `BikeProfile`.
    Discipline {
        Downhill = "Downhill",
        Road = "Road",
        Slopestyle = "Slopestyle",
        Freeride = "Freeride",
    }
);
choice!(
    /// Hand/handlebar pose requested while airborne (animation input; no physics effect).
    HandTrick {
        None = "None",
        OneHand = "One hand",
        NoHand = "No hands",
        TuckNoHand = "Tuck no-hand",
        Barspin = "Barspin",
        TireGrab = "Tire grab",
        SeatGrab = "Seat grab",
        Toboggan = "Toboggan",
    }
);
choice!(
    /// Leg/foot pose requested while airborne (animation input; no physics effect).
    LegTrick {
        None = "None",
        OneFoot = "One foot",
        NoFoot = "No feet",
        CanCan = "Can-can",
        NoFootCan = "No-foot can",
        Superman = "Superman",
        Tailwhip = "Tailwhip",
        NacNac = "Nac-nac",
        Indian = "Indian air",
    }
);
choice!(
    /// Bike-body motion requested while airborne (animation input; no physics effect).
    BikeTrick {
        None = "None",
        Whip = "Whip",
        Table = "Table",
        XUp = "X-up",
        Turndown = "Turndown",
        EuroTable = "Euro table",
        Invert = "Invert",
        Crankflip = "Crankflip",
    }
);

/// Why the bike crashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashReason {
    BadLanding,
    RiderImpact,
    FrameImpact,
    MissingSupport,
    HardImpact,
    LoopedOut,
    OverTheBars,
}

impl CrashReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::BadLanding => "Bad landing",
            Self::RiderImpact => "Rider impact",
            Self::FrameImpact => "Frame impact",
            Self::MissingSupport => "Missing support",
            Self::HardImpact => "Hard impact",
            Self::LoopedOut => "Looped out",
            Self::OverTheBars => "Over the bars",
        }
    }
}

/// A crash in progress; the bike tumbles passively until it is reset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crash {
    pub reason: CrashReason,
    /// Closing speed against the surface when the crash was declared, m/s.
    pub impact: f32,
    /// Seconds since the crash.
    pub elapsed: f32,
}

/// A collision proxy, bike-root local.
#[derive(Clone, Copy, Debug)]
pub struct CollisionSphere {
    pub offset: Vec3,
    pub radius: f32,
    /// Rider body (rider impact) as opposed to bike frame (frame impact).
    pub rider: bool,
}

/// Solved-rig collision data, supplied by the scene before every step.
#[derive(Clone, Copy, Debug)]
pub struct CollisionPose {
    /// Hub positions, bike-root local, at zero strut compression: `[front, rear]`.
    pub wheel_rest: [Vec3; 2],
    /// Unit wheel axle directions, bike-root local.
    pub wheel_axes: [Vec3; 2],
    pub bodies: [CollisionSphere; 12],
    /// 0 holding .. 1 released: `[left, right]` hand on the bar, foot on the pedal.
    pub hand_release: [f32; 2],
    pub foot_release: [f32; 2],
}

const fn proxy(x: f32, y: f32, z: f32, radius: f32, rider: bool) -> CollisionSphere {
    CollisionSphere {
        offset: Vec3::new(x, y, z),
        radius,
        rider,
    }
}

/// The design pose: used when no rig has been solved (pure simulation).
impl Default for CollisionPose {
    fn default() -> Self {
        Self {
            wheel_rest: [
                Vec3::new(0.0, -SUSPENSION_REST, -WHEELBASE / 2.0),
                Vec3::new(0.0, -SUSPENSION_REST, WHEELBASE / 2.0),
            ],
            wheel_axes: [Vec3::X; 2],
            bodies: [
                proxy(0.0, -0.18, -0.25, 0.07, false),
                proxy(0.0, -0.30, 0.06, 0.07, false),
                proxy(0.0, -0.12, 0.45, 0.07, false),
                proxy(0.0, 0.02, 0.38, 0.08, false),
                proxy(0.0, 0.10, -0.52, 0.07, false),
                proxy(0.0, 0.14, 0.40, 0.12, true),
                proxy(0.0, 0.58, 0.10, 0.15, true),
                proxy(0.0, 0.98, -0.18, 0.11, true),
                proxy(-0.14, 0.10, -0.08, 0.08, true),
                proxy(0.14, 0.10, -0.08, 0.08, true),
                proxy(-0.20, 0.16, -0.55, 0.06, true),
                proxy(0.20, 0.16, -0.55, 0.06, true),
            ],
            hand_release: [0.0; 2],
            foot_release: [0.0; 2],
        }
    }
}

/// Authored per-discipline tuning. Wheelbase, wheel radius, strut rest length and the
/// `MAX_COMPRESSION` contact bound are shared by every profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BikeProfile {
    /// Strut spring rate per wheel per unit mass; static sag is `g / (2 spring)`.
    pub spring: f32,
    /// Strut damper rate.
    pub damper: f32,
    /// Pedal power; also scales the low-speed pedal force cap.
    pub power: f32,
    pub sprint_power: f32,
    /// Quadratic air drag coefficient.
    pub drag: f32,
    /// Hop launch speed, m/s.
    pub hop_speed: f32,
    /// Multiplier on airborne pitch/roll/yaw/flip authority.
    pub air_control: f32,
    /// Handlebar outer half width, metres (scene).
    pub bar_width: f32,
    /// Grip drop below the stem, metres (scene; drop bars).
    pub grip_drop: f32,
    /// Tyre width factor vs Downhill; scales lateral grip and rolling resistance.
    pub tire_width: f32,
    /// Rider posture: 0 tucked/seated .. 1 standing attack stance (scene).
    pub ride_stand: f32,
}

impl Discipline {
    pub fn profile(self) -> BikeProfile {
        match self {
            // The original baseline tuning.
            Self::Downhill => BikeProfile {
                spring: 98.0,
                damper: 9.0,
                power: 4.5,
                sprint_power: 9.0,
                drag: 0.0035,
                hop_speed: 3.4,
                air_control: 1.0,
                bar_width: 0.40,
                grip_drop: 0.0,
                tire_width: 1.0,
                ride_stand: 0.85,
            },
            // Stiff, fast, narrow tyres, drop bars; sluggish in the air.
            Self::Road => BikeProfile {
                spring: 145.0,
                damper: 13.0,
                power: 6.0,
                sprint_power: 11.5,
                drag: 0.0018,
                hop_speed: 2.2,
                air_control: 0.6,
                bar_width: 0.21,
                grip_drop: 0.14,
                tire_width: 0.55,
                ride_stand: 0.2,
            },
            // Playful: strong hop, quick air control.
            Self::Slopestyle => BikeProfile {
                spring: 110.0,
                damper: 10.5,
                power: 4.8,
                sprint_power: 9.5,
                drag: 0.0032,
                hop_speed: 4.2,
                air_control: 1.35,
                bar_width: 0.36,
                grip_drop: 0.0,
                tire_width: 0.9,
                ride_stand: 0.7,
            },
            // Long-travel feel: softest spring, deepest sag, wide tyres.
            Self::Freeride => BikeProfile {
                spring: 82.0,
                damper: 8.0,
                power: 4.6,
                sprint_power: 9.2,
                drag: 0.0040,
                hop_speed: 3.6,
                air_control: 1.1,
                bar_width: 0.39,
                grip_drop: 0.0,
                tire_width: 1.15,
                ride_stand: 0.9,
            },
        }
    }
}

#[derive(Resource, Clone, Debug)]
pub struct Bike {
    pub position: Vec3,
    pub velocity: Vec3,
    /// Readouts of the orientation, see the module docs. Never write them.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Front wheel angle, radians, +left, within `MAX_STEER_ANGLE`.
    pub steering: f32,
    /// Wheel rolling angle, radians in [0, TAU); increases while rolling forward.
    pub wheel_phase: f32,
    /// Crank angle, radians in [0, TAU); 0 = right crank down; advances only while pedaling.
    pub crank_phase: f32,
    /// Strut compression in metres, `[front, rear]`, 0 when extended, <= `MAX_COMPRESSION`.
    pub suspension: [f32; 2],
    pub grounded: [bool; 2],
    /// Seconds since the last wheel contact; 0 while any wheel touches.
    pub air_time: f32,
    /// Horizontal distance travelled, metres.
    pub distance: f32,
    pub discipline: Discipline,
    /// Body-frame angular velocity about the right (X), up (Y) and back (Z) axes, rad/s.
    pub pitch_rate: f32,
    pub yaw_rate: f32,
    pub roll_rate: f32,
    /// Solved-rig collision data; the scene refreshes it before every step.
    pub collision_pose: CollisionPose,
    /// Set once the bike has crashed; controls are then ignored until a reset.
    pub crash: Option<Crash>,
    rot: Quat,
    wheel_omega: f32,
    hop_latch: bool,
}

#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct Controls {
    /// 0..1 pedal effort.
    pub pedal: f32,
    /// 0..1 brake lever.
    pub brake: f32,
    /// -1..1, +1 steers right.
    pub steering: f32,
    /// Raises pedal power while held.
    pub sprint: bool,
    /// Held state; a hop fires only on the released-to-held edge while on the ground.
    pub hop: bool,
    /// Held state; grounded wheelie assist.
    pub wheelie: bool,
    /// -1..1, +1 nose up (airborne only).
    pub air_pitch: f32,
    /// -1..1, +1 rolls right (airborne only).
    pub air_roll: f32,
    /// -1..1, +1 yaws right (airborne only; physical spin, no ground effect).
    pub air_yaw: f32,
    /// Held state; grounded manual balance assist (rear wheel, no pedaling needed).
    pub manual: bool,
    /// Held state; grounded nose manual balance assist (front wheel). Ignored while `manual`.
    pub nose_manual: bool,
    /// Held state; raises air pitch/roll authority and releases rate damping, so flips and
    /// barrel rolls carry physical momentum.
    pub flip: bool,
    /// Requested hand/leg/bike tricks and side (-1 left, +1 right); animation inputs.
    pub hand_trick: HandTrick,
    pub leg_trick: LegTrick,
    pub bike_trick: BikeTrick,
    pub trick_side: f32,
}

struct Contact {
    compression: f32,
    /// Closing rate of the strut (positive = compressing), m/s.
    rate: f32,
    ground_vy: f32,
    slope: Vec2,
    /// Unit terrain normal under the hub.
    normal: Vec3,
    /// Hub speed into the surface, m/s (positive = closing).
    closing: f32,
    /// World vector from the root to the tyre contact patch: where ground forces act.
    lever: Vec3,
}

/// Penetration of a sphere into the terrain (negative = clear) and the unit surface normal below it.
fn sphere_gap(centre: Vec3, radius: f32) -> (f32, Vec3) {
    let s = terrain_slope(centre.x, centre.z);
    let n = Vec3::new(-s.x, 1.0, -s.y).normalize();
    (
        radius - (centre.y - terrain_height(centre.x, centre.z)) * n.y,
        n,
    )
}

fn sane(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() { x.clamp(lo, hi) } else { 0.0 }
}

impl Default for Bike {
    fn default() -> Self {
        Self::placed(Discipline::default(), 0.0, 8.0, 0.0)
    }
}

impl Bike {
    /// Rolling at `speed` along the slope at (x, z), facing -Z, settled on the struts of
    /// `discipline`.
    fn placed(discipline: Discipline, x: f32, z: f32, speed: f32) -> Self {
        let sag = GRAVITY / (2.0 * discipline.profile().spring);
        let pose = CollisionPose::default();
        let (mut pitch, mut y) = (0.0_f32, 0.0_f32);
        for _ in 0..4 {
            let arms = pose.wheel_rest.map(|r| Quat::from_rotation_x(pitch) * r);
            let ground = arms.map(|a| {
                let s = terrain_slope(x + a.x, z + a.z);
                terrain_height(x + a.x, z + a.z) + WHEEL_RADIUS * (1.0 + s.length_squared()).sqrt()
            });
            y = (ground[0] - arms[0].y + ground[1] - arms[1].y) * 0.5 - sag;
            pitch = ((ground[0] - ground[1]) / WHEELBASE)
                .clamp(-1.0, 1.0)
                .asin();
        }
        let rot = Quat::from_rotation_x(pitch);
        let mut bike = Self {
            position: Vec3::new(x, y, z),
            velocity: rot * Vec3::NEG_Z * speed,
            yaw: 0.0,
            pitch,
            roll: 0.0,
            steering: 0.0,
            wheel_phase: 0.0,
            crank_phase: 0.0,
            suspension: [sag; 2],
            grounded: [true; 2],
            air_time: 0.0,
            distance: 0.0,
            discipline,
            pitch_rate: 0.0,
            yaw_rate: 0.0,
            roll_rate: 0.0,
            collision_pose: pose,
            crash: None,
            rot,
            wheel_omega: speed / WHEEL_RADIUS,
            hop_latch: false,
        };
        bike.sync_attitude();
        bike
    }

    /// Fresh bike of `discipline` at (x, z), keeping the current collision pose.
    fn restart(&mut self, discipline: Discipline, x: f32, z: f32, speed: f32) {
        let pose = self.collision_pose;
        *self = Self::placed(discipline, x, z, speed);
        self.collision_pose = pose;
    }

    /// Switches profile and resets the bike to the settled start pose.
    pub fn select_discipline(&mut self, discipline: Discipline) {
        self.restart(discipline, 0.0, 8.0, 0.0);
    }

    /// Resets to the start pose, keeping the selected discipline.
    pub fn reset(&mut self) {
        self.select_discipline(self.discipline);
    }

    /// Restarts rolling at `speed` m/s down the local slope at (x, z), facing -Z, settled on the
    /// struts (no airborne drop); clears any crash. Keeps the discipline.
    pub fn reset_at(&mut self, x: f32, z: f32, speed: f32) {
        self.restart(self.discipline, x, z, speed);
    }

    pub fn orientation(&self) -> Quat {
        self.rot
    }

    fn angular_velocity(&self) -> Vec3 {
        Vec3::new(self.pitch_rate, self.yaw_rate, self.roll_rate)
    }

    fn set_angular_velocity(&mut self, w: Vec3) {
        (self.pitch_rate, self.yaw_rate, self.roll_rate) = (w.x, w.y, w.z);
    }

    /// Refreshes the yaw/pitch/roll readouts. The heading comes from the nose, or the axle when the
    /// nose is vertical, and is kept continuous through half-flips.
    fn sync_attitude(&mut self) {
        let (fwd, right, up) = (
            self.rot * Vec3::NEG_Z,
            self.rot * Vec3::X,
            self.rot * Vec3::Y,
        );
        let wrap = |a: f32| (a + PI).rem_euclid(TAU) - PI;
        let heading = if fwd.x.hypot(fwd.z) > 0.2 {
            (-fwd.x).atan2(-fwd.z)
        } else {
            (-right.z).atan2(right.x)
        };
        let mut d = wrap(heading - self.yaw);
        if d.abs() > FRAC_PI_2 {
            d = wrap(d + PI);
        }
        self.yaw = wrap(self.yaw + d);
        self.pitch = fwd.y.clamp(-1.0, 1.0).asin();
        self.roll = right.y.atan2(up.y);
    }

    /// Advance by `dt` seconds (clamped to 1/15 s and split into substeps of at most 1/120 s and
    /// `MAX_STEP_TRAVEL` metres of root or tip travel).
    pub fn step(&mut self, controls: &Controls, dt: f32) {
        let dt = sane(dt, 0.0, MAX_FRAME_DT);
        if dt == 0.0 {
            return;
        }
        let c = Controls {
            pedal: sane(controls.pedal, 0.0, 1.0),
            brake: sane(controls.brake, 0.0, 1.0),
            steering: sane(controls.steering, -1.0, 1.0),
            air_pitch: sane(controls.air_pitch, -1.0, 1.0),
            air_roll: sane(controls.air_roll, -1.0, 1.0),
            air_yaw: sane(controls.air_yaw, -1.0, 1.0),
            trick_side: sane(controls.trick_side, -1.0, 1.0),
            ..*controls
        };
        let hop_edge = c.hop && !self.hop_latch;
        self.hop_latch = c.hop;
        let travel = (self.velocity.length() + self.angular_velocity().length() * TIP_RADIUS) * dt;
        let n = ((dt * STEP_HZ - 1e-3).ceil() as u32)
            .max((travel / MAX_STEP_TRAVEL).ceil() as u32)
            .clamp(1, MAX_SUBSTEPS);
        for i in 0..n {
            // The edge is consumed by the first substep so a held key cannot re-fire.
            self.substep(&c, dt / n as f32, hop_edge && i == 0);
        }
    }

    fn contact(&self, i: usize) -> Contact {
        let q = self.rot;
        let arm = q * self.collision_pose.wheel_rest[i];
        let hub = self.position + arm;
        let slope = terrain_slope(hub.x, hub.z);
        let up_len = (1.0 + slope.length_squared()).sqrt();
        let normal = Vec3::new(-slope.x, 1.0, -slope.y) / up_len;
        // A wheel is a torus: tilting its axle off the surface plane lowers its contact reach.
        let side = (q * self.collision_pose.wheel_axes[i])
            .dot(normal)
            .clamp(-1.0, 1.0);
        let reach = (WHEEL_RADIUS - TIRE_TUBE) * (1.0 - side * side).sqrt() + TIRE_TUBE;
        let hub_velocity = self.velocity + (q * self.angular_velocity()).cross(arm);
        let ground_vy = slope.dot(Vec2::new(hub_velocity.x, hub_velocity.z));
        let compression = terrain_height(hub.x, hub.z) + reach * up_len - hub.y;
        // The wheel centre is the rest hub lifted by the compression; the patch is one reach below
        // it along the surface normal.
        let patch = hub + Vec3::Y * compression - normal * reach;
        Contact {
            compression,
            rate: ground_vy - hub_velocity.y,
            ground_vy,
            slope,
            normal,
            closing: -hub_velocity.dot(normal),
            lever: patch - self.position,
        }
    }

    /// Judges a touchdown on the attitude and velocity at first contact, before struts, bump
    /// stops or the lean servo can alter them.
    fn assess_landing(&self, contacts: &[Contact; 2], grounded: [bool; 2]) -> Option<Crash> {
        let (mut n, mut impact) = (Vec3::ZERO, 0.0_f32);
        for i in 0..2 {
            if grounded[i] {
                n += contacts[i].normal;
                impact = impact.max(contacts[i].closing);
            }
        }
        let n = if n.length_squared() > 1e-6 {
            n.normalize()
        } else {
            Vec3::Y
        };
        let (up, fwd, right) = (
            self.rot * Vec3::Y,
            self.rot * Vec3::NEG_Z,
            self.rot * Vec3::X,
        );
        let along = self.velocity - n * self.velocity.dot(n);
        let heading = fwd - n * fwd.dot(n);
        let slip = if along.length() > LAND_SLIP_MIN_SPEED && heading.length_squared() > 1e-4 {
            along.angle_between(heading)
        } else {
            0.0
        };
        let use_of_limits = [
            up.dot(n).clamp(-1.0, 1.0).acos() / LAND_TILT_LIMIT,
            fwd.dot(n).abs() / LAND_PITCH_LIMIT.sin(),
            right.dot(n).abs() / LAND_ROLL_LIMIT.sin(),
            (0..2)
                .filter(|&i| grounded[i])
                .map(|i| {
                    (self.rot * self.collision_pose.wheel_axes[i])
                        .dot(contacts[i].normal)
                        .abs()
                        / LAND_ROLL_LIMIT.sin()
                })
                .fold(0.0, f32::max),
            slip / LAND_SLIP_LIMIT,
            self.angular_velocity().length() / LAND_SPIN_LIMIT,
        ]
        .into_iter()
        .fold(0.0, f32::max);
        let misaligned = use_of_limits > 1.0;
        let reason = if impact > HARD_IMPACT_SPEED
            || impact > HARD_IMPACT_SPEED * HARD_IMPACT_MARGINAL
                && use_of_limits > LAND_MARGINAL_USE
        {
            CrashReason::HardImpact
        } else if misaligned {
            CrashReason::BadLanding
        } else if Self::holding(self.collision_pose.foot_release) < SUPPORT_MIN {
            // Both feet off the pedals at touchdown (with or without hands) cannot be ridden out.
            CrashReason::MissingSupport
        } else {
            return None;
        };
        Some(Crash {
            reason,
            impact,
            elapsed: 0.0,
        })
    }

    /// First rider or frame proxy buried in the floor, if any.
    fn body_strike(&self) -> Option<Crash> {
        let w = self.rot * self.angular_velocity();
        self.collision_pose.bodies.iter().find_map(|s| {
            let r = self.rot * s.offset;
            let (pen, n) = sphere_gap(self.position + r, s.radius);
            (pen > BODY_STRIKE_DEPTH).then(|| Crash {
                reason: if s.rider {
                    CrashReason::RiderImpact
                } else {
                    CrashReason::FrameImpact
                },
                impact: (-(self.velocity + w.cross(r)).dot(n)).max(0.0),
                elapsed: 0.0,
            })
        })
    }

    /// Passive rigid-body step of the bike alone: gravity, wheel and frame spheres against the
    /// terrain with restitution and Coulomb friction. Rider spheres are skipped, the detached rider
    /// is its own ragdoll. The final vertical lift keeps every bike proxy above ground.
    fn crash_substep(&mut self, dt: f32) {
        if let Some(k) = &mut self.crash {
            k.elapsed += dt;
        }
        self.velocity.y -= GRAVITY * dt;
        self.velocity *= 1.0 - CRASH_AIR_DRAG * dt;
        self.rot = (self.rot * Quat::from_scaled_axis(self.angular_velocity() * dt)).normalize();
        self.position += self.velocity * dt;
        self.distance += Vec2::new(self.velocity.x, self.velocity.z).length() * dt;

        let pose = self.collision_pose;
        let mut shapes = [(Vec3::ZERO, WHEEL_RADIUS, CRASH_WHEEL_FRICTION); 14];
        for i in 0..2 {
            shapes[i].0 = pose.wheel_rest[i];
        }
        let mut count = 2;
        for b in pose.bodies.iter().filter(|b| !b.rider) {
            shapes[count] = (b.offset, b.radius, CRASH_FRICTION);
            count += 1;
        }

        let q = self.rot;
        let mut w = q * self.angular_velocity();
        let (mut lift, mut touching, mut wheel_touch) = (0.0_f32, false, [false; 2]);
        for pass in 0..CRASH_ITERATIONS {
            for (k, &(offset, radius, mu)) in shapes[..count].iter().enumerate() {
                let r = q * offset;
                let (pen, n) = sphere_gap(self.position + r, radius);
                if pen <= 0.0 {
                    continue;
                }
                if pass == 0 {
                    lift = lift.max(pen / n.y);
                    touching = true;
                    if k < 2 {
                        wheel_touch[k] = true;
                    }
                }
                let vp = self.velocity + w.cross(r);
                let vn = vp.dot(n);
                if vn >= 0.0 {
                    continue;
                }
                let e = if vn < -CRASH_BOUNCE_MIN {
                    CRASH_RESTITUTION
                } else {
                    0.0
                };
                let jn = -(1.0 + e) * vn / (1.0 + r.cross(n).length_squared() / GYRATION_SQ);
                self.velocity += n * jn;
                w += r.cross(n * jn) / GYRATION_SQ;
                let vt = vp - n * vn;
                let speed = vt.length();
                if speed > 1e-4 {
                    let t = vt / speed;
                    let jt =
                        (speed / (1.0 + r.cross(t).length_squared() / GYRATION_SQ)).min(mu * jn);
                    self.velocity -= t * jt;
                    w -= r.cross(t * jt) / GYRATION_SQ;
                }
            }
        }
        self.position.y += lift;
        if touching {
            w *= (-CRASH_SPIN_DAMP * dt).exp();
            if self.velocity.length() < SLEEP_SPEED && w.length() < SLEEP_SPIN {
                self.velocity = Vec3::ZERO;
                w = Vec3::ZERO;
            }
        }
        self.set_angular_velocity(q.inverse() * w);
        self.wheel_omega *= 1.0 - AIR_WHEEL_DECAY * dt;
        self.wheel_phase = (self.wheel_phase + self.wheel_omega * dt).rem_euclid(TAU);
        self.suspension = [0.0; 2];
        self.grounded = wheel_touch;
        self.air_time = if wheel_touch[0] || wheel_touch[1] {
            0.0
        } else {
            self.air_time + dt
        };
        self.sync_attitude();
    }

    /// How many of the two hands (or feet) are holding, 0..2, from their release amounts.
    fn holding(release: [f32; 2]) -> f32 {
        2.0 - release[0] - release[1]
    }

    /// Neither hands nor feet hold the bike: the rider can apply no torque to it.
    fn detached(&self) -> bool {
        Self::holding(self.collision_pose.hand_release) < SUPPORT_MIN
            && Self::holding(self.collision_pose.foot_release) < SUPPORT_MIN
    }

    fn substep(&mut self, c: &Controls, dt: f32, hop_edge: bool) {
        // A fully detached rider steers, pedals, hops and flips nothing; ballistics and landing
        // checks still run.
        let hop_edge = hop_edge && !self.detached();
        let c = &if self.detached() {
            Controls {
                pedal: 0.0,
                brake: 0.0,
                steering: 0.0,
                air_pitch: 0.0,
                air_roll: 0.0,
                air_yaw: 0.0,
                sprint: false,
                hop: false,
                wheelie: false,
                manual: false,
                nose_manual: false,
                flip: false,
                ..*c
            }
        } else {
            *c
        };
        if self.crash.is_none() {
            self.crash = self.body_strike();
        }
        let contacts = [self.contact(0), self.contact(1)];
        let p = self.discipline.profile();
        let grounded = [contacts[0].compression > 0.0, contacts[1].compression > 0.0];
        let any_ground = grounded[0] || grounded[1];
        if self.crash.is_none() && any_ground && !(self.grounded[0] || self.grounded[1]) {
            self.crash = self.assess_landing(&contacts, grounded);
        }
        if self.crash.is_some() {
            self.crash_substep(dt);
            return;
        }

        // Struts push along the terrain normal; that also yields slope gravity and ramp thrust.
        // Torque is the true lever arm (root to tyre patch) crossed with the force, and only its
        // pitch component is used: roll is servoed and yaw is steered.
        let mut accel = Vec3::new(0.0, -GRAVITY, 0.0);
        let mut force = [0.0_f32; 2];
        let mut torque = Vec3::ZERO;
        for (i, k) in contacts.iter().enumerate().filter(|(i, _)| grounded[*i]) {
            let f = (p.spring * k.compression
                + p.damper * k.rate
                + STOP_K * (k.compression - STOP_START).max(0.0))
            .clamp(0.0, MAX_WHEEL_FORCE);
            force[i] = f;
            let push = f * Vec3::new(-k.slope.x, 1.0, -k.slope.y);
            accel += push;
            // On one wheel the rider's legs and arms soak up a landing's impact: only up to
            // `ONE_WHEEL_TORQUE_G` of the strut force pitches the bike, so landing deep on a
            // wheel does not slam or launch it.
            let share = if grounded[0] != grounded[1] {
                (ONE_WHEEL_TORQUE_G * GRAVITY / f.max(1e-6)).min(1.0)
            } else {
                1.0
            };
            torque += k.lever.cross(push * share);
        }
        let inv = self.rot.inverse();
        let to_pitch = move |t: Vec3| (inv * t).x / GYRATION_SQ;
        let mut pitch_acc = to_pitch(torque);
        let drag = (p.drag * self.velocity.length() * dt).min(0.5);
        self.velocity += accel * dt - self.velocity * drag;

        let heading = Vec2::new(-self.yaw.sin(), -self.yaw.cos());
        let right = Vec2::new(self.yaw.cos(), -self.yaw.sin());
        let mut v_h = Vec2::new(self.velocity.x, self.velocity.z);
        let mut v_f = v_h.dot(heading);
        let mut v_l = v_h.dot(right);

        let steer_target = -c.steering * MAX_STEER_ANGLE / (1.0 + (v_f / STEER_SPEED_REF).powi(2));
        self.steering += (steer_target - self.steering) * (1.0 - (-STEER_RESPONSE * dt).exp());

        // World-vertical heading change from steering (grounded only).
        let mut steer_yaw = 0.0_f32;
        if any_ground {
            // Drive and brake forces act at the tyre patches, below the centre of mass: drive
            // pitches up, braking pitches down. Each wheel's share is limited by its strut load,
            // so an unloading wheel (rear under hard braking) fades out continuously.
            let along = Vec3::new(heading.x, 0.0, heading.y);
            let mut ground_torque = Vec3::ZERO;
            if c.pedal > 0.0 {
                let power = if c.sprint { p.sprint_power } else { p.power };
                let cap = PEDAL_FORCE * p.power / PEDAL_POWER;
                let ceiling = if v_f > 0.0 {
                    cap.min(power / v_f.max(MIN_POWER_SPEED))
                } else {
                    cap
                };
                let drive = (ceiling * c.pedal).min(GROUND_MU * force[1]);
                v_f += drive * dt;
                ground_torque += contacts[1].lever.cross(along * drive);
            }
            // Resistances clamp at zero: a settled bike stays settled and brakes never reverse it.
            v_f -= v_f.signum() * (ROLLING_RESISTANCE * p.tire_width * dt).min(v_f.abs());
            let held = [
                (BRAKE_FRONT * BRAKE_DECEL * c.brake).min(GROUND_MU * force[0]),
                (BRAKE_REAR * BRAKE_DECEL * c.brake).min(GROUND_MU * force[1]),
            ];
            let total = held[0] + held[1];
            let sign = v_f.signum();
            let dv = (total * dt).min(v_f.abs());
            v_f -= sign * dv;
            if total > 0.0 {
                let used = -sign * dv / (total * dt);
                for (k, h) in contacts.iter().zip(held) {
                    ground_torque += k.lever.cross(along * (used * h));
                }
            }
            pitch_acc += to_pitch(ground_torque);

            v_l -= v_l * (TIRE_GRIP * p.tire_width * dt).min(1.0);
            let yaw_limit = MAX_LATERAL_ACCEL / v_f.abs().max(1.0);
            let yaw_rate = (v_f * self.steering.tan() / WHEELBASE).clamp(-yaw_limit, yaw_limit);
            steer_yaw = yaw_rate;
            // Velocity turns with the heading, preserving speed.
            v_h = heading * v_f + right * v_l;
            let (s, co) = (yaw_rate * dt).sin_cos();
            v_h = Vec2::new(v_h.x * co + v_h.y * s, -v_h.x * s + v_h.y * co);
            self.velocity.x = v_h.x;
            self.velocity.z = v_h.y;

            let lean = (LEAN_GAIN * (v_f * yaw_rate / GRAVITY).atan()).clamp(-MAX_LEAN, MAX_LEAN);
            self.roll_rate = (lean - self.roll) * (1.0 - (-LEAN_RESPONSE * dt).exp()) / dt;

            let balance = c.manual || c.nose_manual;
            if c.wheelie && grounded[1] && !balance {
                pitch_acc += WHEELIE_TORQUE
                    * ((WHEELIE_PITCH - self.pitch) / WHEELIE_SOFT).clamp(0.0, 1.0)
                    - WHEELIE_DAMP * self.pitch_rate;
            }
            // Manual balances on the rear wheel, nose manual on the front one; neither needs
            // pedaling and both hold a fixed pitch relative to the ground slope.
            let (wheel, target) = if c.manual {
                (1, MANUAL_PITCH)
            } else {
                (0, -NOSE_PITCH)
            };
            let ground_pitch = |i: usize| contacts[i].slope.dot(heading).atan();
            if balance && grounded[wheel] {
                // The assist only holds a balance the rider has already found: far from its
                // target pitch (a flip landed deep on one wheel) it fades out, and the rider must
                // first steer the bike back with the arrows.
                let error = ground_pitch(wheel) + target - self.pitch;
                // How far past the target, towards tipping over (behind for a manual, over the
                // front for a nose manual); lifting into the balance from flat is always held.
                let deep = (-error * target.signum()).max(0.0);
                let strength = smoothstep(BALANCE_SPEED.0, BALANCE_SPEED.1, v_f)
                    * (1.0 - smoothstep(BALANCE_CAPTURE.0, BALANCE_CAPTURE.1, deep));
                let servo = BALANCE_K * error - BALANCE_D * self.pitch_rate;
                pitch_acc += (servo - pitch_acc) * strength;
            }
            // On one wheel the rider balances the bike by moving their weight over or behind the
            // contact wheel (up/down arrows): a limited torque, so past a critical angle gravity
            // wins and the bike loops out or goes over the bars.
            if grounded[0] != grounded[1] {
                pitch_acc += RIDER_BALANCE * c.air_pitch;
            }
            self.yaw_rate = 0.0;
            if hop_edge {
                self.velocity.y += p.hop_speed;
                self.pitch_rate += HOP_PITCH_KICK;
            }
            self.wheel_omega = v_f / WHEEL_RADIUS;
        } else {
            let authority = p.air_control * if c.flip { FLIP_AUTHORITY } else { 1.0 };
            // Assists fade out with tilt, and never run during a held flip, so a rotation keeps
            // its momentum and is never pulled upright.
            let tilt = (self.rot * Vec3::Y).y.clamp(-1.0, 1.0).acos();
            let assist = if c.flip {
                0.0
            } else {
                1.0 - smoothstep(ASSIST_TILT.0, ASSIST_TILT.1, tilt)
            };
            let push = |input: f32, rate: f32| {
                if input * rate > 0.0 && rate.abs() >= FLIP_MAX_RATE {
                    0.0
                } else {
                    input
                }
            };
            pitch_acc = AIR_PITCH_ACCEL * authority * push(c.air_pitch, self.pitch_rate)
                - assist * AIR_PITCH_DAMP * self.pitch_rate;
            // Air yaw is a real spin; the velocity keeps its direction, so landings sideways scrub.
            self.yaw_rate = (self.yaw_rate
                + (-AIR_YAW_ACCEL * p.air_control * c.air_yaw - AIR_YAW_DAMP * self.yaw_rate) * dt)
                .clamp(-MAX_AIR_YAW, MAX_AIR_YAW);
            self.roll_rate += (AIR_ROLL_ACCEL * authority * push(-c.air_roll, self.roll_rate)
                - AIR_ROLL_DAMP * assist * self.roll_rate)
                * dt;
            self.wheel_omega *= 1.0 - AIR_WHEEL_DECAY * dt;
        }

        self.pitch_rate += pitch_acc * dt;
        let mut q = self.rot * Quat::from_scaled_axis(self.angular_velocity() * dt);
        if steer_yaw != 0.0 {
            q = Quat::from_rotation_y(steer_yaw * dt) * q;
        }
        self.rot = q.normalize();
        self.position += self.velocity * dt;
        self.distance += Vec2::new(self.velocity.x, self.velocity.z).length() * dt;
        self.sync_attitude();

        // Bump-stop resolution: whatever the springs failed to stop, lift out of the ground so no
        // step (any speed or dt) can leave a wheel deeper than MAX_COMPRESSION.
        let contacts = [self.contact(0), self.contact(1)];
        // Contact may begin during integration. Judge that crossing before penetration
        // correction changes the impact velocity or the grounded flags hide the touchdown.
        let touching = [contacts[0].compression > 0.0, contacts[1].compression > 0.0];
        if !any_ground && (touching[0] || touching[1]) {
            self.crash = self.assess_landing(&contacts, touching);
        }
        if self.crash.is_none() {
            self.crash = self.body_strike();
        }
        if self.crash.is_some() {
            // Time was already integrated; resolve the impact without taking a second step.
            self.crash_substep(0.0);
            return;
        }
        let deepest = if contacts[0].compression >= contacts[1].compression {
            0
        } else {
            1
        };
        let excess = contacts[deepest].compression - MAX_COMPRESSION;
        if excess > 0.0 {
            self.position.y += excess;
            self.velocity.y = self.velocity.y.max(contacts[deepest].ground_vy);
        }
        let contacts = [self.contact(0), self.contact(1)];
        for i in 0..2 {
            self.suspension[i] = contacts[i].compression.clamp(0.0, MAX_COMPRESSION);
            self.grounded[i] = contacts[i].compression > 0.0;
        }
        // One wheel down and pitched past the point of no return: the rider falls off.
        if self.grounded[0] != self.grounded[1] {
            let i = usize::from(self.grounded[1]);
            let heading = Vec2::new(-self.yaw.sin(), -self.yaw.cos());
            let rel = self.pitch - contacts[i].slope.dot(heading).atan();
            let reason = if i == 1 && rel > TIP_OVER_PITCH {
                Some(CrashReason::LoopedOut)
            } else if i == 0 && rel < -TIP_OVER_PITCH {
                Some(CrashReason::OverTheBars)
            } else {
                None
            };
            if let Some(reason) = reason {
                self.crash = Some(Crash {
                    reason,
                    impact: 0.0,
                    elapsed: 0.0,
                });
                self.crash_substep(0.0);
                return;
            }
        }
        self.air_time = if self.grounded[0] || self.grounded[1] {
            0.0
        } else {
            self.air_time + dt
        };

        self.wheel_phase = (self.wheel_phase + self.wheel_omega * dt).rem_euclid(TAU);
        if c.pedal > 0.0 {
            let cadence = (self.wheel_omega.abs() * GEAR_RATIO).clamp(MIN_CADENCE, MAX_CADENCE);
            self.crank_phase = (self.crank_phase + cadence * dt).rem_euclid(TAU);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 120.0;

    fn pedal() -> Controls {
        Controls {
            pedal: 1.0,
            ..Controls::default()
        }
    }

    fn run(bike: &mut Bike, controls: Controls, seconds: f32) {
        for _ in 0..(seconds / DT) as usize {
            bike.step(&controls, DT);
        }
    }

    fn after(controls: Controls, seconds: f32) -> Bike {
        let mut b = Bike::default();
        run(&mut b, controls, seconds);
        b
    }

    /// Speed along the bike's heading.
    fn forward_speed(b: &Bike) -> f32 {
        -b.velocity.x * b.yaw.sin() - b.velocity.z * b.yaw.cos()
    }

    fn clearance(b: &Bike) -> f32 {
        b.position.y - terrain_height(b.position.x, b.position.z)
    }

    #[test]
    fn rests_on_struts_without_jitter() {
        let mut b = after(Controls::default(), 3.0);
        let y = b.position.y;
        run(&mut b, Controls::default(), 1.0);
        assert!(b.velocity.length() < 0.01 && (b.position.y - y).abs() < 1e-3);
        assert!(b.grounded == [true; 2] && b.air_time == 0.0);
        assert!(
            b.suspension.iter().all(|c| (0.03..0.08).contains(c)),
            "{:?}",
            b.suspension
        );
    }

    /// Pedal, coast, light brake, hard stop, rest, then a low-speed pedal and brake again.
    fn low_speed_script(t: f32) -> Controls {
        let (pedal, brake) = match t {
            t if t < 1.5 => (1.0, 0.0),
            t if t < 3.0 => (0.0, 0.0),
            t if t < 3.6 => (0.0, 0.4),
            t if t < 5.1 => (0.0, 1.0),
            t if t < 7.0 => (0.0, 0.0),
            t if t < 7.5 => (1.0, 0.0),
            _ => (0.0, 1.0),
        };
        Controls {
            pedal,
            brake,
            ..Controls::default()
        }
    }

    /// Every profile: the settled bike is still, and pedalling, coasting and braking at low speed
    /// never reverse the bike, never chatter the pitch (the pitch rate changes sign once per
    /// transient, not every few steps) and never flicker the wheel contacts (a hard stop may
    /// unload the rear once at most).
    #[test]
    fn low_speed_contact_is_stable_for_every_profile() {
        for &d in Discipline::ALL {
            let mut b = select(d);
            run(&mut b, Controls::default(), 2.0);
            let (y, pitch) = (b.position.y, b.pitch);
            run(&mut b, Controls::default(), 1.0);
            assert!(
                (b.position.y - y).abs() < 1e-5 && (b.pitch - pitch).abs() < 1e-5,
                "{d:?} rest drifts"
            );
            assert!(b.pitch_rate.abs() < 1e-4 && b.velocity.length() < 1e-4);

            let mut b = select(d);
            let (mut phase_flips, mut last_rate) = ([0_u32; 9], 0.0_f32);
            let (mut still_y, mut still_pitch) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
            let (mut toggles, mut was) = (0, [true; 2]);
            for n in 0..(9.0 / DT) as usize {
                let t = n as f32 * DT;
                b.step(&low_speed_script(t), DT);
                assert!(b.crash.is_none(), "{d:?} crashed at {t}");
                toggles += (b.grounded != was) as u32;
                was = b.grounded;
                assert!(forward_speed(&b) > -1e-3, "{d:?} reversed at {t}");
                assert!(b.pitch.abs() < 0.12, "{d:?} pitch {} at {t}", b.pitch);
                if b.pitch_rate.abs() > 1e-3 {
                    if b.pitch_rate * last_rate < 0.0 {
                        phase_flips[t as usize] += 1;
                    }
                    last_rate = b.pitch_rate;
                }
                if (5.8..7.0).contains(&t) {
                    still_y = (still_y.0.min(b.position.y), still_y.1.max(b.position.y));
                    still_pitch = (still_pitch.0.min(b.pitch), still_pitch.1.max(b.pitch));
                }
            }
            assert!(
                phase_flips.iter().all(|&f| f <= 4),
                "{d:?} pitch chatter {phase_flips:?}"
            );
            assert!(
                still_y.1 - still_y.0 < 1e-4 && still_pitch.1 - still_pitch.0 < 1e-4,
                "{d:?} stopped bike still moves: y {still_y:?} pitch {still_pitch:?}"
            );
            assert!(toggles <= 2, "{d:?} contact flicker: {toggles} changes");
            assert!(b.velocity.length() < 1e-3, "{d:?} {:?}", b.velocity);
        }
    }

    /// Full braking on the jump's rising face holds the bike still: static friction, no creep.
    #[test]
    fn brake_holds_a_bike_still_on_a_slope() {
        for &d in Discipline::ALL {
            let mut b = select(d);
            b.reset_at(0.0, -31.0, 0.0);
            let hold = Controls {
                brake: 1.0,
                ..Controls::default()
            };
            run(&mut b, hold, 2.0);
            let (p, pitch) = (b.position, b.pitch);
            run(&mut b, hold, 1.0);
            assert!(
                b.position.distance(p) < 1e-4 && (b.pitch - pitch).abs() < 1e-4,
                "{d:?} creeps {:?}",
                b.position - p
            );
            assert!(b.crash.is_none() && b.grounded == [true; 2]);
        }
    }

    #[test]
    fn pedals_up_and_brakes_without_reversing() {
        let mut b = after(pedal(), 4.0);
        let sprint = after(
            Controls {
                sprint: true,
                ..pedal()
            },
            4.0,
        );
        assert!(forward_speed(&b) > 4.5 && b.distance > 10.0);
        assert!(forward_speed(&sprint) > forward_speed(&b) + 0.5);
        for _ in 0..(3.0 / DT) as usize {
            b.step(
                &Controls {
                    brake: 1.0,
                    ..Controls::default()
                },
                DT,
            );
            assert!(forward_speed(&b) > -1e-3);
        }
        assert!(
            b.velocity.length() < 0.05 && b.grounded == [true; 2],
            "{:?}",
            b.velocity
        );
    }

    #[test]
    fn steering_turns_and_leans_into_the_corner() {
        for side in [1.0_f32, -1.0] {
            let mut b = after(pedal(), 3.0);
            run(
                &mut b,
                Controls {
                    steering: side,
                    ..pedal()
                },
                1.5,
            );
            assert!(
                b.position.x * side > 1.0 && b.velocity.x * side > 0.5,
                "x {}",
                b.position.x
            );
            assert!(b.yaw * side < -0.3 && b.roll * side < -0.1 && b.steering * side < 0.0);
        }
    }

    #[test]
    fn hop_fires_once_per_press_not_on_landing() {
        let held = Controls {
            hop: true,
            ..Controls::default()
        };
        let mut b = after(Controls::default(), 1.0);
        let (mut takeoffs, mut peak, mut was_air) = (0, 0.0_f32, false);
        for _ in 0..(2.5 / DT) as usize {
            b.step(&held, DT);
            takeoffs += (b.air_time > 0.0 && !was_air) as u32;
            was_air = b.air_time > 0.0;
            peak = peak.max(clearance(&b));
        }
        assert_eq!(takeoffs, 1, "held hop must not re-fire on landing");
        assert!(peak > 1.0 && b.grounded == [true; 2], "peak {peak}");
        // Release then press: a second hop.
        b.step(&Controls::default(), DT);
        run(&mut b, held, 0.1);
        assert!(b.air_time > 0.0);
        // Release then press while airborne: no extra impulse.
        let vy = b.velocity.y;
        b.step(&Controls::default(), DT);
        b.step(&held, DT);
        assert!(b.velocity.y < vy);
    }

    #[test]
    fn jumps_leave_crests_and_land_on_suspension() {
        let mut b = Bike::default();
        let (mut longest, mut deepest, mut lowest) = (0.0_f32, 0.0_f32, f32::MAX);
        for _ in 0..(22.0 / DT) as usize {
            b.step(
                &Controls {
                    sprint: true,
                    ..pedal()
                },
                DT,
            );
            longest = longest.max(b.air_time);
            deepest = deepest.max(b.suspension[0].max(b.suspension[1]));
            lowest = lowest.min(clearance(&b));
            assert!(b.position.is_finite() && b.velocity.is_finite());
        }
        assert!(longest > 0.5, "air {longest}");
        assert!(
            (0.1..MAX_COMPRESSION + 1e-4).contains(&deepest),
            "landing travel {deepest}"
        );
        assert!(lowest > 0.45, "tunnelled: {lowest}");
        assert!(b.grounded == [true; 2] && b.position.z < -JUMP_CRESTS[2]);
    }

    #[test]
    fn course_has_rising_ramps_and_falling_faces() {
        for d in JUMP_CRESTS {
            let crest = terrain_height(0.0, -d);
            assert!(crest - terrain_height(0.0, -d + JUMP_RISE) > 1.2);
            assert!(crest - terrain_height(0.0, -d - JUMP_FALL) > 1.2);
            // There is flat space beside each ramp to steer around it.
            assert!((terrain_height(4.0, -d) - terrain_height(5.0, -d)).abs() < 0.5);
        }
    }

    #[test]
    fn showcase_floor_is_flat_outside_the_jumps() {
        for x in [-120.0, -80.0, -10.0, -4.0, 4.0, 10.0, 120.0] {
            for z in [-200.0, -135.0, -85.0, -35.0, 0.0, 40.0] {
                assert_eq!(terrain_height(x, z), 0.0, "floor at ({x}, {z})");
            }
        }
        for z in [-220.0, -180.0, -120.0, -60.0, -20.0, 0.0, 45.0] {
            assert_eq!(terrain_height(0.0, z), 0.0, "flat between ramps at z={z}");
        }
    }

    #[test]
    fn hill_and_kicker_match_the_course_contract() {
        for z in [HILL_START_Z, 20.0, 30.0, 45.0] {
            assert!(
                (terrain_height(HILL_X, z) - HILL_HEIGHT).abs() < 1e-3,
                "summit z={z}"
            );
            assert!((terrain_height(HILL_X + 12.0, z) - HILL_HEIGHT).abs() < 1e-3);
        }
        assert!((terrain_height(HILL_X, -20.0) - HILL_HEIGHT / 2.0).abs() < 1e-3);
        for z in [-60.0, -66.0, -74.0] {
            assert_eq!(terrain_height(HILL_X, z), 0.0, "foot flat at z={z}");
        }
        let lip = terrain_height(HILL_X, HILL_LIP_Z);
        assert!((5.5..7.0).contains(&lip), "lip {lip}");
        assert!(terrain_height(HILL_X, HILL_LIP_Z + 5.0) < lip);
        assert_eq!(terrain_height(HILL_X, -88.0), 0.0);
        // Relief stays inside the agreed bounds; the jump lane and the hill do not overlap.
        for x in (-120..=180).step_by(6) {
            for z in (-200..=120).step_by(5) {
                let (x, z) = (x as f32, z as f32);
                if !(-4.0..=108.0).contains(&x) || !(-150.0..=65.0).contains(&z) {
                    assert_eq!(
                        terrain_height(x, z),
                        0.0,
                        "relief outside bounds at ({x}, {z})"
                    );
                }
            }
        }
    }

    #[test]
    fn wheelie_lifts_front_and_air_controls_rotate() {
        let mut b = after(pedal(), 3.0);
        run(
            &mut b,
            Controls {
                wheelie: true,
                ..pedal()
            },
            1.5,
        );
        assert!(
            b.pitch > 0.3 && !b.grounded[0] && b.grounded[1],
            "pitch {}",
            b.pitch
        );
        run(&mut b, pedal(), 1.5);
        assert!(b.pitch.abs() < 0.2 && b.grounded == [true; 2]);

        let flight = |p, r| {
            let mut a = Bike::default();
            a.position.y += 10.0;
            a.velocity = Vec3::new(0.0, 0.0, -8.0);
            run(
                &mut a,
                Controls {
                    air_pitch: p,
                    air_roll: r,
                    ..Controls::default()
                },
                0.3,
            );
            assert!(a.grounded == [false; 2]);
            a
        };
        let (neutral, up, down) = (flight(0.0, 0.0), flight(1.0, 1.0), flight(-1.0, -1.0));
        assert!(up.pitch > neutral.pitch + 0.2 && up.roll < -0.1);
        assert!(down.pitch < neutral.pitch - 0.2 && down.roll > 0.1);
    }

    #[test]
    fn stays_finite_and_never_buried_for_supported_dt() {
        for (dt, d) in [DT, 1.0 / 60.0, 1.0 / 30.0, 1.0]
            .into_iter()
            .flat_map(|dt| Discipline::ALL.iter().map(move |&d| (dt, d)))
        {
            let (mut b, mut seed) = (Bike::default(), 7_u32);
            b.select_discipline(d);
            let mut rnd = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32
            };
            let mut c = Controls::default();
            for i in 0..(60.0 / dt) as usize {
                if i % (0.25 / dt).max(1.0) as usize == 0 {
                    c = Controls {
                        pedal: (rnd() > 0.2) as u8 as f32,
                        sprint: rnd() > 0.5,
                        brake: (rnd() > 0.8) as u8 as f32,
                        steering: rnd() * 2.0 - 1.0,
                        hop: rnd() > 0.8,
                        wheelie: rnd() > 0.8,
                        air_pitch: rnd() * 2.0 - 1.0,
                        air_roll: rnd() * 2.0 - 1.0,
                        air_yaw: rnd() * 2.0 - 1.0,
                        manual: rnd() > 0.8,
                        nose_manual: rnd() > 0.9,
                        flip: rnd() > 0.8,
                        ..Controls::default()
                    };
                }
                b.step(&c, dt);
                assert!(b.position.is_finite() && b.velocity.is_finite() && b.pitch.is_finite());
                match b.crash {
                    None => assert!(deepest(&b, false) < 0.2, "dt {dt} {d:?}"),
                    Some(_) => assert!(deepest(&b, true) < 1e-3, "crashed body in floor, dt {dt}"),
                }
            }
        }
    }

    #[test]
    fn zero_duration_does_not_consume_a_hop_press() {
        let mut bike = Bike::default();
        let controls = Controls {
            hop: true,
            ..Controls::default()
        };
        bike.step(&controls, 0.0);
        bike.step(&controls, DT);
        assert!(bike.velocity.y > 1.0);
    }

    #[test]
    fn reset_clears_airborne_and_input_latch() {
        let held = Controls {
            hop: true,
            ..Controls::default()
        };
        let mut b = after(held, 0.3);
        assert!(b.air_time > 0.0);
        b.reset();
        assert!(b.velocity == Vec3::ZERO && b.air_time == 0.0 && b.grounded == [true; 2]);
        // Latch cleared: a still-held hop counts as a fresh press.
        b.step(&held, DT);
        assert!(b.velocity.y > 1.0);
    }

    fn select(d: Discipline) -> Bike {
        let mut b = Bike::default();
        b.select_discipline(d);
        b
    }

    /// A bike falling from well above the ground at 8 m/s.
    fn airborne(d: Discipline) -> Bike {
        let mut b = select(d);
        b.position.y += 30.0;
        b.velocity = Vec3::new(0.0, 0.0, -8.0);
        b
    }

    #[test]
    fn every_profile_rests_and_stays_contact_bound_on_the_ramps() {
        for &d in Discipline::ALL {
            let mut b = select(d);
            run(&mut b, Controls::default(), 3.0);
            let y = b.position.y;
            run(&mut b, Controls::default(), 1.0);
            let sag = GRAVITY / (2.0 * d.profile().spring);
            assert!(
                b.velocity.length() < 0.01 && (b.position.y - y).abs() < 1e-3,
                "{d:?}"
            );
            assert!(b.grounded == [true; 2] && b.air_time == 0.0, "{d:?}");
            assert!(
                b.suspension.iter().all(|c| (c - sag).abs() < 0.006),
                "{d:?} {:?}",
                b.suspension
            );

            let (mut b, mut longest) = (select(d), 0.0_f32);
            let sprint = Controls {
                sprint: true,
                ..pedal()
            };
            for _ in 0..(22.0 / DT) as usize {
                b.step(&sprint, DT);
                longest = longest.max(b.air_time);
                assert!(b.position.is_finite() && b.velocity.is_finite(), "{d:?}");
                assert!(clearance(&b) > 0.3, "{d:?} tunnelled {}", clearance(&b));
                for i in 0..2 {
                    assert!(b.contact(i).compression <= MAX_COMPRESSION + 1e-3, "{d:?}");
                }
            }
            assert!(longest > 0.2, "{d:?} never left a crest: {longest}");
        }
    }

    #[test]
    fn profiles_differ_in_stiffness_hop_and_pace() {
        let rest_sag = |d| {
            let mut b = select(d);
            run(&mut b, Controls::default(), 3.0);
            b.suspension[0]
        };
        let (road, slope, down, free) = (
            rest_sag(Discipline::Road),
            rest_sag(Discipline::Slopestyle),
            rest_sag(Discipline::Downhill),
            rest_sag(Discipline::Freeride),
        );
        assert!(
            road < slope && slope < down && down < free,
            "{road} {slope} {down} {free}"
        );

        let hop_peak = |d| {
            let mut b = select(d);
            run(&mut b, Controls::default(), 1.0);
            let rest = clearance(&b);
            let held = Controls {
                hop: true,
                ..Controls::default()
            };
            let mut peak = 0.0_f32;
            for _ in 0..(1.0 / DT) as usize {
                b.step(&held, DT);
                peak = peak.max(clearance(&b));
            }
            peak - rest
        };
        let (road, slope, down, free) = (
            hop_peak(Discipline::Road),
            hop_peak(Discipline::Slopestyle),
            hop_peak(Discipline::Downhill),
            hop_peak(Discipline::Freeride),
        );
        assert!(
            road < down && down < slope && road < free,
            "{road} {slope} {down} {free}"
        );

        // Road accelerates harder and coasts with less drag and rolling resistance.
        let pace = |d| {
            forward_speed(&{
                let mut b = select(d);
                run(&mut b, pedal(), 5.0);
                b
            })
        };
        assert!(pace(Discipline::Road) > pace(Discipline::Downhill) + 0.8);
        let coast = |d| {
            let mut b = select(d);
            b.velocity = Vec3::new(0.0, 0.0, -8.0);
            run(&mut b, Controls::default(), 3.0);
            forward_speed(&b)
        };
        assert!(coast(Discipline::Road) > coast(Discipline::Downhill) + 0.4);
    }

    #[test]
    fn manual_and_nose_manual_balance_without_pedaling_and_release_cleanly() {
        for &d in Discipline::ALL {
            let rolling = || {
                let mut b = select(d);
                b.velocity = Vec3::new(0.0, 0.0, -6.0);
                b
            };
            let mut m = rolling();
            run(
                &mut m,
                Controls {
                    manual: true,
                    ..Controls::default()
                },
                2.0,
            );
            assert!(
                (0.35..0.65).contains(&m.pitch) && !m.grounded[0] && m.grounded[1],
                "{d:?} {}",
                m.pitch
            );
            assert!(forward_speed(&m) > 4.0, "{d:?} {}", forward_speed(&m));
            run(&mut m, Controls::default(), 1.5);
            assert!(
                m.pitch.abs() < 0.2 && m.grounded == [true; 2],
                "{d:?} {}",
                m.pitch
            );

            let mut n = rolling();
            run(
                &mut n,
                Controls {
                    nose_manual: true,
                    ..Controls::default()
                },
                2.0,
            );
            assert!(
                (-0.6..-0.25).contains(&n.pitch) && n.grounded[0] && !n.grounded[1],
                "{d:?} {}",
                n.pitch
            );

            // Standing still there is no balance point: the wheel stays down.
            let mut s = select(d);
            run(
                &mut s,
                Controls {
                    manual: true,
                    ..Controls::default()
                },
                1.0,
            );
            assert!(s.pitch.abs() < 0.2, "{d:?} {}", s.pitch);
        }
    }

    #[test]
    fn air_yaw_spins_in_the_air_only_and_scales_with_the_profile() {
        let spin = |d, yaw| {
            let mut b = airborne(d);
            run(
                &mut b,
                Controls {
                    air_yaw: yaw,
                    ..Controls::default()
                },
                0.4,
            );
            assert!(b.grounded == [false; 2]);
            assert!(
                b.velocity.x.abs() < 1e-4,
                "spin must not steer the velocity"
            );
            b.yaw
        };
        assert_eq!(spin(Discipline::Downhill, 0.0), 0.0);
        let (right, left) = (
            spin(Discipline::Downhill, 1.0),
            spin(Discipline::Downhill, -1.0),
        );
        assert!(right < -0.2 && left > 0.2, "{right} {left}");
        assert!(spin(Discipline::Slopestyle, 1.0) < right && spin(Discipline::Road, 1.0) > right);

        let mut g = Bike::default();
        run(
            &mut g,
            Controls {
                air_yaw: 1.0,
                ..Controls::default()
            },
            1.0,
        );
        assert!(g.yaw.abs() < 1e-3 && g.grounded == [true; 2]);
    }

    #[test]
    fn selection_survives_reset_and_switching_resets_the_pose() {
        let mut b = Bike::default();
        assert_eq!(b.discipline, Discipline::Downhill);
        b.select_discipline(Discipline::Road);
        run(
            &mut b,
            Controls {
                hop: true,
                ..pedal()
            },
            0.5,
        );
        b.reset();
        assert_eq!(b.discipline, Discipline::Road);
        let sag = GRAVITY / (2.0 * Discipline::Road.profile().spring);
        assert!(b.velocity == Vec3::ZERO && b.suspension == [sag; 2] && b.air_time == 0.0);
        for &d in Discipline::ALL {
            b.select_discipline(d);
            assert_eq!(b.discipline, d);
            assert!(b.velocity == Vec3::ZERO && b.grounded == [true; 2]);
        }
    }

    /// Deepest proxy below the terrain, metres. `wreck`: the wheels and frame spheres the crashed
    /// bike collides with (the rider is a separate ragdoll); otherwise every rider/frame proxy of a
    /// bike still being ridden.
    fn deepest(b: &Bike, wreck: bool) -> f32 {
        let q = b.orientation();
        let pose = &b.collision_pose;
        let wheel_spheres = pose.wheel_rest.map(|r| (r, WHEEL_RADIUS));
        let body_spheres = pose
            .bodies
            .into_iter()
            .filter(|s| !wreck || !s.rider)
            .map(|s| (s.offset, s.radius));
        wheel_spheres
            .into_iter()
            .skip(if wreck { 0 } else { 2 })
            .chain(body_spheres)
            .map(|(o, r)| sphere_gap(b.position + q * o, r).0)
            .fold(f32::MIN, f32::max)
    }

    /// Drops an upright bike 1.5 m onto flat ground, then rides 3 s.
    #[test]
    fn upright_landings_survive_on_flat_ground_and_on_the_hill_slope() {
        for &d in Discipline::ALL {
            let mut b = select(d);
            b.position.y += 1.5;
            b.velocity = Vec3::new(0.0, 0.0, -5.0);
            run(&mut b, Controls::default(), 3.0);
            assert!(b.crash.is_none(), "{d:?} flat: {:?}", b.crash);

            let mut h = select(d);
            h.reset_at(HILL_X, -20.0, 10.0);
            assert!(
                h.air_time == 0.0 && h.grounded == [true; 2] && h.velocity.y < -3.0,
                "{d:?}"
            );
            h.position.y += 0.5;
            run(&mut h, Controls::default(), 2.0);
            assert!(
                h.crash.is_none() && h.position.is_finite(),
                "{d:?} hill: {:?}",
                h.crash
            );
        }
    }

    #[test]
    fn rolling_hill_start_is_settled_not_dropped() {
        let mut b = Bike::default();
        b.reset_at(HILL_X, HILL_START_Z, 8.0);
        assert!(b.grounded == [true; 2] && b.crash.is_none());
        assert!((forward_speed(&b) - 8.0).abs() < 1e-3);
        let g = terrain_height(HILL_X, HILL_START_Z);
        assert!(
            (b.position.y - g - 0.75).abs() < 0.1,
            "root {} over {g}",
            b.position.y
        );
        run(&mut b, Controls::default(), 1.0);
        assert!(b.grounded == [true; 2] && b.crash.is_none());
    }

    #[test]
    fn misaligned_landings_crash_and_the_wreck_stays_above_ground() {
        let cases = [
            ("nose dive", Quat::from_rotation_x(-1.4)),
            ("on its side", Quat::from_rotation_z(1.2)),
            ("inverted", Quat::from_rotation_z(PI)),
            ("sideways", Quat::from_rotation_y(1.2)),
        ];
        for (name, q) in cases {
            let mut b = Bike::default();
            b.position.y += 1.5;
            b.velocity = Vec3::new(0.0, 0.0, -6.0);
            b.rot = q;
            b.sync_attitude();
            for _ in 0..(4.0 / DT) as usize {
                b.step(&Controls::default(), DT);
                if b.crash.is_some() {
                    assert!(deepest(&b, true) < 1e-3, "{name} buried");
                }
            }
            let crash = b.crash.expect(name);
            assert!(crash.elapsed > 1.0, "{name}");
            if name == "sideways" {
                assert_eq!(crash.reason, CrashReason::BadLanding);
            }
        }
    }

    #[test]
    fn tilt_and_spin_are_not_silently_righted_in_the_air() {
        let mut b = airborne(Discipline::Downhill);
        b.rot = Quat::from_rotation_z(2.0);
        b.pitch_rate = 3.0;
        b.sync_attitude();
        run(&mut b, Controls::default(), 0.4);
        assert!(b.crash.is_none() && b.grounded == [false; 2]);
        let expected = Quat::from_rotation_z(2.0) * Quat::from_rotation_x(3.0 * 0.4);
        assert!(
            b.orientation().angle_between(expected) < 1e-3,
            "free rotation was altered"
        );
        assert!((b.pitch_rate - 3.0).abs() < 1e-3, "rate {}", b.pitch_rate);
    }

    #[test]
    fn hard_impact_crashes_ignores_inputs_and_reset_clears() {
        let crashed = || {
            let mut b = airborne(Discipline::Downhill);
            b.position.y += 30.0;
            run(&mut b, Controls::default(), 7.0);
            b
        };
        let (mut a, mut b) = (crashed(), crashed());
        let crash = a.crash.expect("30 m drop");
        assert_eq!(crash.reason, CrashReason::HardImpact);
        assert!(crash.impact > HARD_IMPACT_SPEED && !crash.reason.label().is_empty());
        let held = Controls {
            hop: true,
            steering: 1.0,
            air_pitch: 1.0,
            flip: true,
            ..pedal()
        };
        run(&mut a, held, 3.0);
        run(&mut b, Controls::default(), 3.0);
        assert!(
            a.crash.is_some() && a.position == b.position && a.orientation() == b.orientation()
        );
        a.reset();
        assert!(
            a.crash.is_none() && a.velocity == Vec3::ZERO && a.angular_velocity() == Vec3::ZERO
        );
        a.step(
            &Controls {
                hop: true,
                ..Controls::default()
            },
            DT,
        );
        assert!(a.velocity.y > 1.0);
    }

    #[test]
    fn fast_wrecks_never_end_a_step_inside_the_terrain() {
        for dt in [DT, 1.0 / 30.0, 1.0] {
            let mut b = Bike::default();
            b.reset_at(HILL_X, -62.0, 0.0);
            b.crash = Some(Crash {
                reason: CrashReason::HardImpact,
                impact: 0.0,
                elapsed: 0.0,
            });
            b.velocity = Vec3::new(3.0, 5.0, 35.0);
            b.set_angular_velocity(Vec3::new(4.0, 3.0, 2.0));
            for _ in 0..(6.0 / dt.min(MAX_FRAME_DT)) as usize {
                b.step(&Controls::default(), dt);
                assert!(b.position.is_finite() && b.velocity.is_finite(), "dt {dt}");
                assert!(deepest(&b, true) < 1e-3, "dt {dt}: {}", deepest(&b, true));
            }
        }
    }

    /// Lands on one wheel still rotating (1.5 rad/s) and rides on with the rider's balance input
    /// `input` (applied while one wheel is down).
    fn one_wheel_landing(pitch: f32, input: f32) -> Bike {
        let mut b = Bike::default();
        b.position.y += 1.2;
        b.rot = Quat::from_rotation_x(pitch);
        b.sync_attitude();
        b.velocity = Vec3::new(0.0, 0.0, -6.0);
        b.pitch_rate = 1.5 * pitch.signum();
        for _ in 0..480 {
            let one = b.grounded[0] != b.grounded[1];
            let c = Controls {
                air_pitch: if one { input } else { 0.0 },
                ..Controls::default()
            };
            b.step(&c, DT);
            if b.crash.is_some() {
                break;
            }
        }
        b
    }

    #[test]
    fn one_wheel_landings_must_be_balanced_or_they_tip_over() {
        for (pitch, tip) in [
            (0.6, CrashReason::LoopedOut),
            (-0.6, CrashReason::OverTheBars),
        ] {
            let left = one_wheel_landing(pitch, 0.0);
            assert_eq!(
                left.crash.map(|c| c.reason),
                Some(tip),
                "{pitch} unbalanced"
            );
            let saved = one_wheel_landing(pitch, -f32::signum(pitch));
            assert!(
                saved.crash.is_none() && saved.grounded == [true; 2],
                "{pitch} balanced: {:?}",
                saved.crash
            );
        }
    }

    /// A bike launched 1.5 m up at 9 m/s, held in a flip and turned `turn` radians about its pitch
    /// axis by a bang-bang controller, then given 3 s in total.
    fn flip_flight(turn: f32) -> Bike {
        let mut b = Bike::default();
        b.position.y += 1.5;
        b.velocity = Vec3::new(0.0, 9.0, -6.0);
        let accel = AIR_PITCH_ACCEL * FLIP_AUTHORITY * b.discipline.profile().air_control;
        let mut turned = 0.0;
        for _ in 0..(3.0 / DT) as usize {
            let rate = b.pitch_rate;
            let stop_at = turned + rate * rate.abs() / (2.0 * accel);
            let input = if b.air_time == 0.0 || ((turned - turn).abs() < 0.05 && rate.abs() < 0.6) {
                0.0
            } else if stop_at < turn {
                1.0
            } else {
                -1.0
            };
            b.step(
                &Controls {
                    flip: b.air_time > 0.0,
                    air_pitch: input,
                    ..Controls::default()
                },
                DT,
            );
            if b.air_time > 0.0 {
                turned += b.pitch_rate * DT;
            }
        }
        b
    }

    #[test]
    fn a_completed_flip_lands_and_an_incomplete_one_crashes() {
        let done = flip_flight(TAU);
        assert!(
            done.crash.is_none() && done.grounded == [true; 2],
            "{:?}",
            done.crash
        );
        let half = flip_flight(PI);
        assert!(half.crash.is_some(), "landed upside down without crashing");
    }

    #[test]
    fn a_detached_rider_applies_no_torque_and_cannot_land_on_released_feet() {
        let fly = |release: f32| {
            let mut b = airborne(Discipline::Downhill);
            b.collision_pose.hand_release = [release; 2];
            b.collision_pose.foot_release = [release; 2];
            let c = Controls {
                flip: true,
                air_pitch: 1.0,
                air_roll: 1.0,
                air_yaw: 1.0,
                ..Controls::default()
            };
            run(&mut b, c, 0.4);
            b
        };
        let (held, off) = (fly(0.0), fly(1.0));
        assert!(held.pitch_rate > 3.0 && held.roll_rate < -3.0 && held.yaw_rate < -1.0);
        assert!(off.pitch_rate.abs() < 1.0 && off.roll_rate == 0.0 && off.yaw_rate == 0.0);

        let mut b = Bike::default();
        b.position.y += 1.0;
        b.collision_pose.foot_release = [1.0; 2];
        run(&mut b, Controls::default(), 2.0);
        assert_eq!(b.crash.map(|k| k.reason), Some(CrashReason::MissingSupport));
    }
    /// Drops an upright bike rolled by `roll` radians from `height` metres and returns how it lands.
    fn drop_landing(height: f32, roll: f32) -> Option<Crash> {
        let mut b = Bike::default();
        b.position.y += height;
        b.rot = Quat::from_rotation_z(roll);
        b.sync_attitude();
        for _ in 0..(3.0 / DT) as usize {
            b.step(&Controls::default(), DT);
            if b.crash.is_some() || b.grounded == [true; 2] {
                break;
            }
        }
        b.crash
    }

    #[test]
    fn flat_landings_crash_above_13_ms_and_marginal_ones_above_8_ms() {
        // Closing speeds sqrt(2 g h): 6 m -> 10.8, 10 m -> 14.0 m/s.
        assert!(drop_landing(6.0, 0.0).is_none(), "clean 6 m drop");
        let hard = drop_landing(10.0, 0.0).expect("10 m drop");
        assert_eq!(hard.reason, CrashReason::HardImpact);
        assert!(hard.impact > HARD_IMPACT_SPEED);
        // A 0.4 rad roll uses 69 % of the roll limit: fine at 5 m/s, a crash at 10.8 m/s.
        assert!(drop_landing(1.3, 0.4).is_none(), "gentle rolled drop");
        let marginal = drop_landing(6.0, 0.4).expect("hard rolled drop");
        assert_eq!(marginal.reason, CrashReason::HardImpact);
        assert!(marginal.impact < HARD_IMPACT_SPEED);
    }
}
