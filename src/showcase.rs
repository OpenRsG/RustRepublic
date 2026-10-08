use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

use crate::bike::{
    Bike, BikeTrick, Controls, HILL_LIP_Z, HILL_START_Z, HILL_X, HandTrick, LegTrick,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Approach,
    Flight,
    Outcome,
}

#[derive(Clone, Copy)]
struct Run {
    name: &'static str,
    hand: HandTrick,
    feet: LegTrick,
    bike: BikeTrick,
    pitch: f32,
    roll: f32,
    expect_crash: bool,
}

const RUNS: [Run; 8] = [
    Run {
        name: "Superman",
        hand: HandTrick::None,
        feet: LegTrick::Superman,
        bike: BikeTrick::None,
        pitch: 0.0,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Backflip",
        hand: HandTrick::None,
        feet: LegTrick::None,
        bike: BikeTrick::None,
        pitch: TAU,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Frontflip",
        hand: HandTrick::None,
        feet: LegTrick::None,
        bike: BikeTrick::None,
        pitch: -TAU,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Barrel roll",
        hand: HandTrick::None,
        feet: LegTrick::None,
        bike: BikeTrick::None,
        pitch: 0.0,
        roll: -TAU,
        expect_crash: false,
    },
    Run {
        name: "No hands + no feet",
        hand: HandTrick::NoHand,
        feet: LegTrick::NoFoot,
        bike: BikeTrick::None,
        pitch: 0.0,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Barspin + tailwhip",
        hand: HandTrick::Barspin,
        feet: LegTrick::Tailwhip,
        bike: BikeTrick::None,
        pitch: 0.0,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Table",
        hand: HandTrick::None,
        feet: LegTrick::None,
        bike: BikeTrick::Table,
        pitch: 0.0,
        roll: 0.0,
        expect_crash: false,
    },
    Run {
        name: "Incomplete flip - crash",
        hand: HandTrick::None,
        feet: LegTrick::None,
        bike: BikeTrick::None,
        pitch: PI,
        roll: 0.0,
        expect_crash: true,
    },
];

#[derive(Resource)]
pub(crate) struct Showcase {
    pub enabled: bool,
    pub index: usize,
    pub stage: Stage,
    pub elapsed: f32,
    pub outcome: &'static str,
    pub completed: u32,
    pub pitch_progress: f32,
    pub roll_progress: f32,
    landing_heading: f32,
}

impl Default for Showcase {
    fn default() -> Self {
        Self {
            enabled: false,
            index: 0,
            stage: Stage::Approach,
            elapsed: 0.0,
            outcome: "",
            completed: 0,
            pitch_progress: 0.0,
            roll_progress: 0.0,
            landing_heading: 0.0,
        }
    }
}

impl Showcase {
    pub fn name(&self) -> &'static str {
        RUNS[self.index].name
    }

    pub fn begin_run(&mut self, bike: &mut Bike) {
        bike.reset_at(HILL_X, HILL_START_Z, 8.0);
        self.stage = Stage::Approach;
        self.elapsed = 0.0;
        self.outcome = "";
        self.pitch_progress = bike.pitch;
        self.roll_progress = bike.roll;
        info!(
            "SHOWCASE_EVENT case={} stage=approach run={}",
            self.name(),
            self.completed
        );
    }

    /// Advances the real-input controller; true asks the owner to reset animation and camera.
    pub fn drive(&mut self, bike: &mut Bike, input: &mut Controls, dt: f32) -> bool {
        if !self.enabled {
            return false;
        }
        self.elapsed += dt;
        // Quaternion nose elevation folds at vertical; body angular rates do not.
        self.pitch_progress += bike.pitch_rate * dt;
        self.roll_progress += bike.roll_rate * dt;
        let grounded = bike.grounded.iter().any(|&g| g);
        if bike.crash.is_some() && !matches!(self.outcome, "CRASH" | "UNEXPECTED CRASH") {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = if RUNS[self.index].expect_crash {
                "CRASH"
            } else {
                "UNEXPECTED CRASH"
            };
            info!(
                "SHOWCASE_EVENT case={} stage=crash reason={:?}",
                self.name(),
                bike.crash
            );
        }
        if self.stage == Stage::Approach
            && !grounded
            && bike.position.z < HILL_LIP_Z + 4.0
            && bike.air_time > 0.05
        {
            self.stage = Stage::Flight;
            self.elapsed = 0.0;
            self.pitch_progress = bike.pitch;
            self.roll_progress = bike.roll;
            self.landing_heading = bike.yaw;
            info!(
                "SHOWCASE_EVENT case={} stage=flight position={:?} velocity={:?}",
                self.name(),
                bike.position,
                bike.velocity
            );
        } else if self.stage == Stage::Approach && self.elapsed > 20.0 {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = "LAUNCH MISSED";
            warn!("SHOWCASE_EVENT case={} stage=launch-missed", self.name());
        } else if self.stage == Stage::Flight && grounded {
            self.stage = Stage::Outcome;
            self.elapsed = 0.0;
            self.outcome = if RUNS[self.index].expect_crash {
                "UNEXPECTED LANDING"
            } else {
                "LANDED"
            };
            info!(
                "SHOWCASE_EVENT case={} stage=landed pitch_turn={:.3} roll_turn={:.3}",
                self.name(),
                self.pitch_progress,
                self.roll_progress
            );
        }
        *input = Controls::default();
        match self.stage {
            Stage::Approach => {
                input.pedal = 1.0;
                input.sprint = true;
            }
            Stage::Flight => {
                let run = RUNS[self.index];
                input.trick_side = -1.0;
                // Leave time for the rider to regrab and for the rotating assemblies to catch.
                if self.elapsed < 1.2 && crate::animation::predict_landing(bike).0 > 0.8 {
                    input.hand_trick = run.hand;
                    input.leg_trick = run.feet;
                    input.bike_trick = run.bike;
                }
                // Bounded rider torque steers towards an actual whole revolution. No pose writes,
                // airborne teleport, deadline rescue, collision bypass or forced safe landing.
                input.flip = run.pitch != 0.0 || run.roll != 0.0;
                input.air_pitch =
                    ((12.0 * (run.pitch - self.pitch_progress) - 4.0 * bike.pitch_rate) / 24.0)
                        .clamp(-1.0, 1.0);
                let roll_target = if self.elapsed < 0.65 { 0.0 } else { run.roll };
                input.air_roll =
                    -((12.0 * (roll_target - self.roll_progress) - 4.0 * bike.roll_rate) / 24.0)
                        .clamp(-1.0, 1.0);
                if !run.expect_crash
                    && (run.roll != 0.0 || run.pitch != 0.0)
                    && (run.pitch - self.pitch_progress).abs() < 0.8
                    && (run.roll - self.roll_progress).abs() < 0.8
                {
                    // Settle the actual attitude with rider torque after the revolution. Body-axis
                    // rotations do not commute; scalar turn counters alone cannot level a barrel.
                    let mut error = bike.orientation().conjugate()
                        * Quat::from_rotation_y(self.landing_heading);
                    if error.w < 0.0 {
                        error = -error;
                    }
                    let (axis, angle) = error.to_axis_angle();
                    let accel = axis * angle * 12.0
                        - Vec3::new(bike.pitch_rate, bike.yaw_rate, bike.roll_rate) * 4.0;
                    let gain = bike.discipline.profile().air_control;
                    input.air_pitch = (accel.x / (22.0 * gain)).clamp(-1.0, 1.0);
                    input.air_yaw = (-accel.y / (12.0 * gain)).clamp(-1.0, 1.0);
                    input.air_roll = (-accel.z / (17.6 * gain)).clamp(-1.0, 1.0);
                }
            }
            Stage::Outcome => {
                // Hard braking immediately after a heavy landing can pitch the bike onto its head.
                input.brake = 0.15;
                if self.elapsed >= 2.5 {
                    self.completed += 1;
                    self.index = (self.index + 1) % RUNS.len();
                    self.begin_run(bike);
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{AnimationState, Input};
    use crate::bike::Discipline;
    use crate::scene;

    #[test]
    fn showcase_runs_real_terrain_tricks_landings_crash_and_loops() {
        let dt = 1.0 / 120.0;
        let mut bike = Bike::default();
        bike.select_discipline(Discipline::Freeride);
        let mut demo = Showcase {
            enabled: true,
            ..default()
        };
        let mut anim = AnimationState::default();
        demo.begin_run(&mut bike);
        let mut launched = [false; RUNS.len()];
        let mut shown = [false; RUNS.len()];
        let mut outcomes = [None; RUNS.len()];
        let mut details = [None; RUNS.len()];
        for _ in 0..(200.0 / dt) as usize {
            let index = demo.index;
            let mut c = Controls::default();
            if demo.drive(&mut bike, &mut c, dt) {
                anim.reset();
            }
            if bike.crash.is_none() {
                anim.update(&Input::from_bike(&bike, &c), dt);
            }
            bike.collision_pose = scene::collision_pose(&bike, &anim);
            bike.step(&c, dt);
            if demo.stage == Stage::Flight {
                launched[index] = true;
                let run = RUNS[index];
                shown[index] |= if run.pitch != 0.0 {
                    demo.pitch_progress.abs() > if run.expect_crash { 2.5 } else { 5.8 }
                } else if run.roll != 0.0 {
                    demo.roll_progress.abs() > 5.8
                } else {
                    (run.hand == HandTrick::None || anim.hand == run.hand)
                        && (run.feet == LegTrick::None || anim.leg == run.feet)
                        && (run.bike == BikeTrick::None || anim.bike == run.bike)
                };
            }
            if demo.stage == Stage::Outcome {
                outcomes[index] = Some(bike.crash.is_some());
                details[index] = Some((
                    bike.crash,
                    demo.pitch_progress,
                    demo.roll_progress,
                    bike.orientation(),
                ));
            }
            assert!(bike.position.is_finite() && bike.velocity.is_finite());
            if demo.completed == RUNS.len() as u32 {
                break;
            }
        }
        assert_eq!(demo.completed, RUNS.len() as u32, "demo did not loop");
        for (index, run) in RUNS.iter().enumerate() {
            assert!(launched[index], "{} never launched", run.name);
            assert!(shown[index], "{} animation/rotation not shown", run.name);
            assert_eq!(
                outcomes[index],
                Some(run.expect_crash),
                "{} unexpected landing outcome: {:?}",
                run.name,
                details[index]
            );
        }
        assert_eq!(demo.index, 0);
        assert!(bike.crash.is_none());
    }
}
