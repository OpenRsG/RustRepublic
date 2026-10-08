//! Scripted ski showcase: summit runs flown with real controls only (no teleports mid-run, no pose
//! writes, no forced landings). Each run rides the hill, takes the 7 m kicker, performs one trick
//! with the air-control servos and lands; the last run is a deliberately incomplete front flip
//! that crashes. The outcome is shown for a while, then the next run starts and the list loops.

use bevy::prelude::Resource;
use std::f32::consts::{PI, TAU};

use super::physics::{
    AIR_FLIP_RATE, AIR_PITCH_RATE, AIR_ROLL_RATE, AIR_YAW_RATE, Grab, SkiControls, Skier,
};
use crate::bike::{HILL_LIP_Z, HILL_X, terrain_height};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Stage {
    #[default]
    Approach,
    Flight,
    Outcome,
}

#[derive(Clone, Copy)]
struct Run {
    name: &'static str,
    /// Carve a slalom down the hill instead of the straight line.
    slalom: bool,
    /// Total yaw to spin in the air, rad (+ right).
    spin: f32,
    /// Total pitch rotation, rad (+ back flip).
    flip: f32,
    grab: Grab,
    side: f32,
    expect_crash: bool,
}

const RUNS: [Run; 5] = [
    Run {
        name: "Slalom, 360 Mute",
        slalom: true,
        spin: TAU,
        flip: 0.0,
        grab: Grab::Mute,
        side: 1.0,
        expect_crash: false,
    },
    Run {
        name: "Backflip Safety",
        slalom: false,
        spin: 0.0,
        flip: TAU,
        grab: Grab::Safety,
        side: -1.0,
        expect_crash: false,
    },
    Run {
        name: "540 Japan",
        slalom: false,
        spin: -3.0 * PI,
        flip: 0.0,
        grab: Grab::Japan,
        side: 1.0,
        expect_crash: false,
    },
    Run {
        name: "Daffy",
        slalom: false,
        spin: 0.0,
        flip: 0.0,
        grab: Grab::Daffy,
        side: -1.0,
        expect_crash: false,
    },
    Run {
        name: "Incomplete front flip",
        slalom: false,
        spin: 0.0,
        flip: -2.4,
        grab: Grab::None,
        side: 1.0,
        expect_crash: true,
    },
];

/// The run-in tucks below this speed so the kicker is always taken fast enough for the tricks
/// and never past the hard-landing limit.
const TUCK_BELOW: f32 = 19.0;
/// Pure-pursuit look-ahead along the hill, metres.
const LOOKAHEAD: f32 = 12.0;
const STEER_GAIN: f32 = 2.5;
const LEVEL_GAIN: f32 = 2.5;
/// Deceleration assumed by the air servos when planning the stop of a rotation, rad/s^2.
const SERVO_ACCEL: f32 = 14.0;
/// Seconds an outcome stays on screen.
const OUTCOME_TIME: f32 = 2.5;
/// A run that never leaves the kicker within this time is abandoned.
const APPROACH_TIMEOUT: f32 = 60.0;

#[derive(Resource, Debug, Default)]
pub struct SkiDemo {
    pub enabled: bool,
    pub index: usize,
    pub completed: usize,
    pub outcome: &'static str,
    pub(crate) stage: Stage,
    pub elapsed: f32,
}

/// Lateral path of a run at hill position `z`.
fn path_x(run: &Run, z: f32) -> f32 {
    if run.slalom && (-30.0..20.0).contains(&z) {
        HILL_X + 4.5 * (TAU * (20.0 - z) / 50.0).sin()
    } else {
        HILL_X
    }
}

/// Normalised rate command that rotates `err` radians and stops there: a rate profile that can
/// always be braked inside `SERVO_ACCEL`, with a linear zone so it settles without chatter.
fn servo(err: f32, max_rate: f32) -> f32 {
    let want = (2.0 * SERVO_ACCEL * err.abs())
        .sqrt()
        .min(6.0 * err.abs())
        .min(max_rate);
    (want.copysign(err) / max_rate).clamp(-1.0, 1.0)
}

impl SkiDemo {
    pub fn name(&self) -> &'static str {
        RUNS[self.index % RUNS.len()].name
    }

    pub fn stage(&self) -> &'static str {
        match self.stage {
            Stage::Approach => "approach",
            Stage::Flight => "flight",
            Stage::Outcome => "outcome",
        }
    }

    pub fn begin_run(&mut self, s: &mut Skier) {
        s.reset();
        self.index %= RUNS.len();
        self.stage = Stage::Approach;
        self.elapsed = 0.0;
        self.outcome = "";
    }

    /// Advances the controller; true asks the owner to reset animation and camera. Overwrites the
    /// controls only while enabled.
    pub fn drive(&mut self, s: &mut Skier, c: &mut SkiControls, dt: f32) -> bool {
        if !self.enabled {
            return false;
        }
        self.elapsed += dt;
        let run = RUNS[self.index % RUNS.len()];
        let crashed = s.crash.is_some();
        if crashed && !matches!(self.outcome, "CRASH" | "UNEXPECTED CRASH") {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = if run.expect_crash {
                "CRASH"
            } else {
                "UNEXPECTED CRASH"
            };
        }
        if self.stage == Stage::Approach
            && !s.grounded
            && s.position.z < HILL_LIP_Z + 4.0
            && s.air_time > 0.05
        {
            self.stage = Stage::Flight;
            self.elapsed = 0.0;
        } else if self.stage == Stage::Approach && self.elapsed > APPROACH_TIMEOUT {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = "LAUNCH MISSED";
        } else if self.stage == Stage::Flight && s.grounded {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = if run.expect_crash {
                "UNEXPECTED LANDING"
            } else {
                "LANDED"
            };
        }
        *c = SkiControls::default();
        match self.stage {
            Stage::Approach => approach(s, c, &run),
            Stage::Flight => flight(s, c, &run),
            Stage::Outcome => {
                // Scrub the landing speed with a hockey stop once the legs have settled.
                if !crashed && s.grounded && self.elapsed > 0.3 {
                    c.brake = 1.0;
                }
                if self.elapsed >= OUTCOME_TIME {
                    self.completed += 1;
                    self.index = (self.index + 1) % RUNS.len();
                    self.begin_run(s);
                    return true;
                }
            }
        }
        false
    }
}

/// Skate off the summit, tuck down the hill, follow the path, wind up the spin on the kicker.
fn approach(s: &Skier, c: &mut SkiControls, run: &Run) {
    let speed = s.speed();
    c.push = if speed < 5.0 { 1.0 } else { 0.0 };
    c.tuck = s.position.z < 12.0 && speed < TUCK_BELOW;
    let aim = path_x(run, s.position.z - LOOKAHEAD) - s.position.x;
    // Right turns lower the heading, so a heading left of the aim steers right.
    let want = (-aim).atan2(LOOKAHEAD);
    let err = (s.heading - want + PI).rem_euclid(TAU) - PI;
    c.steer = (err * STEER_GAIN).clamp(-1.0, 1.0);
    if run.spin != 0.0 && s.position.z < HILL_LIP_Z + 8.0 {
        c.air_yaw = run.spin.signum();
    }
}

/// One trick with the air servos, then level the skis for the landing (never on the crash run).
fn flight(s: &Skier, c: &mut SkiControls, run: &Run) {
    c.trick_side = run.side;
    let height = s.position.y - terrain_height(s.position.x, s.position.z);
    if !(s.velocity.y < 0.0 && height < 5.0) {
        c.grab = run.grab;
    }
    let rates = s.body_rates();
    let mut done = true;
    if run.spin != 0.0 {
        let err = run.spin - s.air_spin;
        c.air_yaw = servo(err, AIR_YAW_RATE);
        done &= err.abs() < 0.35 && rates.y.abs() < 2.0;
    }
    if run.flip != 0.0 {
        let err = run.flip - s.air_flip;
        c.flip = true;
        c.air_pitch = servo(err, AIR_FLIP_RATE);
        done &= err.abs() < 0.35 && rates.x.abs() < 2.0;
    }
    if done && !run.expect_crash {
        c.flip = false;
        let pitch = s.forward().y.clamp(-1.0, 1.0).asin();
        let roll = (-(s.rotation * bevy::math::Vec3::X).y)
            .clamp(-1.0, 1.0)
            .asin();
        c.air_pitch = (-LEVEL_GAIN * pitch / AIR_PITCH_RATE).clamp(-1.0, 1.0);
        c.air_roll = (-LEVEL_GAIN * roll / AIR_ROLL_RATE).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_flies_every_run_lands_or_crashes_as_scripted_and_loops() {
        let dt = 1.0 / 120.0;
        let mut s = Skier::default();
        let mut demo = SkiDemo {
            enabled: true,
            ..SkiDemo::default()
        };
        demo.begin_run(&mut s);
        let mut launched = [false; RUNS.len()];
        let mut spin = [0.0_f32; RUNS.len()];
        let mut flip = [0.0_f32; RUNS.len()];
        let mut outcomes = [None; RUNS.len()];
        for _ in 0..(400.0 / dt) as usize {
            let index = demo.index;
            let mut c = SkiControls::default();
            demo.drive(&mut s, &mut c, dt);
            s.step(&c, dt);
            if demo.stage == Stage::Flight {
                launched[index] = true;
                spin[index] = spin[index].max(s.air_spin.abs());
                flip[index] = flip[index].max(s.air_flip.abs());
            }
            if demo.stage == Stage::Outcome {
                outcomes[index] = Some((
                    s.crash.is_some(),
                    s.crash.map(|c| c.reason),
                    s.last_spin,
                    s.last_flip,
                ));
            }
            assert!(s.position.is_finite() && s.velocity.is_finite());
            if demo.completed == RUNS.len() {
                break;
            }
        }
        assert_eq!(demo.completed, RUNS.len(), "demo did not loop");
        for (i, run) in RUNS.iter().enumerate() {
            assert!(launched[i], "{} never launched", run.name);
            assert_eq!(
                outcomes[i].map(|o| o.0),
                Some(run.expect_crash),
                "{} unexpected outcome {:?}",
                run.name,
                outcomes[i]
            );
            assert!(
                spin[i] >= run.spin.abs() - 0.5,
                "{} spun {}",
                run.name,
                spin[i]
            );
            assert!(
                flip[i] >= run.flip.abs() - 0.5,
                "{} flipped {}",
                run.name,
                flip[i]
            );
        }
        assert_eq!(demo.index, 0);
        assert!(s.crash.is_none());
    }
}
