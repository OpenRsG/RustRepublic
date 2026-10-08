//! Skier animation state: smoothed blend weights driven by the physics `Skier` and the controls.
//! `rig::solve` turns these weights plus the `Skier` into a world-space `SkierPose`.

use super::physics::{Grab, SkiControls, Skier};
use crate::bike::terrain_height;
use bevy::prelude::*;

const GRAVITY: f32 = 9.81;
/// Airtime before a grab may start, seconds.
const GRAB_MIN_AIR: f32 = 0.20;
/// A grab is let go this long before the predicted ballistic landing, seconds.
const GRAB_RELEASE: f32 = 0.35;
/// Airtime after takeoff during which a pop shows the leg extension, seconds.
const EXTEND_WINDOW: f32 = 0.22;
/// Chest counter-rotation per rad/s of ski yaw rate, rad.
const COUNTER_LAG: f32 = 0.12;
/// Free-hand lag: velocity (m/s) per m/s of body velocity change, kick limit, offset limit, damping.
const HAND_GAIN: f32 = 0.6;
const HAND_KICK_MAX: f32 = 0.8;
const HAND_MAX: f32 = 0.16;
const HAND_ZETA: f32 = 0.45;
/// Seconds a turn-change pole plant takes, and the speed above which the plants are made, m/s.
const PLANT_TIME: f32 = 0.9;
const PLANT_SPEED: f32 = 4.0;
/// Rates at which a touchdown's crouch and visual skid are released, 1/s. The crouch is back to
/// near neutral about half a second after the deepest point, as in the retail landing footage.
const LAND_RECOVER: f32 = 5.0;
const SKID_DECAY: f32 = 3.5;

/// Smoothed pose weights (0..1 unless noted). Read by `rig::solve`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Blend {
    /// Baseline stance flexion (grows with speed).
    pub crouch: f32,
    pub tuck: f32,
    /// Knees pulled up while airborne.
    pub air_tuck: f32,
    /// Signed carve side, -1 left .. +1 right (counter-rotation / angulation).
    pub carve: f32,
    pub plow: f32,
    pub skid: f32,
    pub skate: f32,
    pub pole: f32,
    /// Poles joining the skating (V2).
    pub v2: f32,
    pub preload: f32,
    /// Takeoff leg extension.
    pub extend: f32,
    pub air: f32,
    pub grab: f32,
    pub land: f32,
    pub switch: f32,
    /// Side the torso twists to in switch stance, +1 / -1.
    pub switch_side: f32,
    /// Side of the grabbed ski, -1 left / +1 right (latched when a grab starts).
    pub grab_side: f32,
    // Edge detection / memory.
    pub was_grounded: bool,
    /// Preload weight at the moment of takeoff.
    pub pop: f32,
    pub land_kick: f32,
    /// Visual touchdown skid: decaying kick, spring-smoothed weight and the side (-1 / +1) the skis
    /// slew to. Physics never reads it.
    pub skid_kick: f32,
    pub land_skid: f32,
    pub skid_side: f32,
    /// Torso fold: follows `land` a beat after the legs and rebounds with them.
    pub fold: f32,
    /// Chest yaw lagging the skis' turn rate, rad (+ left): the counter-rotation of a turn entry.
    pub counter: f32,
    /// Seconds of animation time: idle breathing and weight shifts.
    pub clock: f32,
    /// Free hands' offsets from their carried positions (world, m) and velocities; sprung by the
    /// root acceleration so the arms trail the body.
    pub hand: [Vec3; 2],
    pub hand_vel: [Vec3; 2],
    pub last_velocity: Vec3,
    /// Low-passed velocity change per tick feeding the hand springs.
    pub dv: Vec3,
    /// Pole-plant pass at a turn change, per hand: progress 0..1 (1 = none) and weight.
    pub plant: [f32; 2],
    pub plant_w: [f32; 2],
    /// Side of the edge the last turn was on, -1 / 0 / +1.
    pub edge_side: f32,
    /// Rates of the spring-smoothed weights above (`spring`), 1/s.
    pub rate: Rates,
}

/// Velocities of the weights that move the body; `Blend` field of the same name.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Rates {
    crouch: f32,
    tuck: f32,
    air_tuck: f32,
    carve: f32,
    preload: f32,
    extend: f32,
    land: f32,
    plow: f32,
    skid: f32,
    skate: f32,
    pole: f32,
    v2: f32,
    air: f32,
    switch: f32,
    grab: f32,
    fold: f32,
    land_skid: f32,
    counter: f32,
    plant_w: [f32; 2],
}

#[derive(Resource, Clone, Debug)]
pub struct SkiAnimation {
    /// Coarse label: "idle" "skate" "pole" "glide" "carve" "tuck" "plow" "stop" "preload" "air" "grab"
    /// "landing" "crashed".
    pub mode: &'static str,
    /// Finer label.
    pub phase: &'static str,
    /// Grab currently shown (`Grab::None` once it has blended out).
    pub grab: Grab,
    pub note: &'static str,
    /// Set by the integrator when paused / crashed; `update` does not advance.
    pub frozen: bool,
    pub(super) w: Blend,
}

impl Default for SkiAnimation {
    fn default() -> Self {
        Self {
            mode: "idle",
            phase: "stand",
            grab: Grab::None,
            note: "relaxed stance, poles trailing",
            frozen: false,
            w: Blend {
                was_grounded: true,
                switch_side: 1.0,
                grab_side: 1.0,
                plant: [1.0; 2],
                ..Blend::default()
            },
        }
    }
}

/// Spring step towards `target` with damping ratio `zeta` (1 = critical). The rate stays continuous
/// when the target jumps, so the pose eases in and out instead of snapping; `zeta` < 1 overshoots.
fn spring_z(cur: &mut f32, rate: &mut f32, target: f32, omega: f32, zeta: f32, dt: f32) {
    *rate += (omega * omega * (target - *cur) - 2.0 * zeta * omega * *rate) * dt;
    *cur += *rate * dt;
}

fn spring(cur: &mut f32, rate: &mut f32, target: f32, omega: f32, dt: f32) {
    spring_z(cur, rate, target, omega, 1.0, dt);
}

/// Smoothstep of `t` clamped to 0..1.
fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Strength of a turn-change pole plant over its progress 0..1: reaches out, touches, lifts off.
fn plant_bump(p: f32) -> f32 {
    ease(p / 0.35) * (1.0 - ease((p - 0.6) / 0.4))
}

/// Seconds until the ballistic path from the skier's state meets the terrain (3 s if it never does).
fn time_to_landing(s: &Skier) -> f32 {
    let mut t = 0.0;
    while t < 3.0 {
        t += 0.04;
        let p = s.position + s.velocity * t + Vec3::NEG_Y * (0.5 * GRAVITY * t * t);
        if p.y <= terrain_height(p.x, p.z) {
            return t;
        }
    }
    3.0
}

impl SkiAnimation {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(&mut self, s: &Skier, c: &SkiControls, dt: f32) {
        if self.frozen || dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.05);
        let speed = s.speed();
        let airborne = !s.grounded;
        let w = &mut self.w;

        if w.was_grounded && airborne {
            w.pop = w.preload.max(s.preload);
        }
        if !w.was_grounded && s.grounded {
            w.land_kick = (s.impact / 10.0).clamp(0.0, 1.0);
            w.skid_kick = ((s.impact - 3.0) / 8.0).clamp(0.0, 1.0);
            w.skid_side = if s.last_spin.abs() > 1.0 {
                s.last_spin.signum()
            } else if c.trick_side < 0.0 {
                -1.0
            } else {
                1.0
            };
        }
        w.was_grounded = s.grounded;
        if s.grounded {
            w.pop = 0.0;
        }
        w.land_kick *= (-LAND_RECOVER * dt).exp();
        w.skid_kick *= (-SKID_DECAY * dt).exp();

        let ground = if airborne { 0.0 } else { 1.0 };
        let r = &mut w.rate;
        spring(
            &mut w.crouch,
            &mut r.crouch,
            if airborne {
                0.3
            } else {
                0.1 + 0.45 * (speed / 20.0).min(1.0)
            },
            8.0,
            dt,
        );
        spring(
            &mut w.tuck,
            &mut r.tuck,
            f32::from(c.tuck).max(s.tuck),
            11.0,
            dt,
        );
        let air_tuck = if !airborne {
            0.0
        } else if c.tuck {
            1.0
        } else if s.air_time > 0.3 {
            0.35
        } else {
            0.0
        };
        spring(&mut w.air_tuck, &mut r.air_tuck, air_tuck, 9.0, dt);
        spring(
            &mut w.carve,
            &mut r.carve,
            ground * (s.lean / 0.8).clamp(-1.0, 1.0),
            9.0,
            dt,
        );
        spring(&mut w.plow, &mut r.plow, s.plow, 14.0, dt);
        spring(&mut w.skid, &mut r.skid, s.skid, 18.0, dt);
        spring(&mut w.skate, &mut r.skate, s.skate, 12.0, dt);
        spring(&mut w.pole, &mut r.pole, s.pole, 12.0, dt);
        spring(&mut w.v2, &mut r.v2, s.v2, 12.0, dt);
        spring(&mut w.preload, &mut r.preload, s.preload, 18.0, dt);
        let extend = if airborne && s.air_time < EXTEND_WINDOW {
            w.pop
        } else {
            0.0
        };
        let omega = if extend > w.extend { 15.0 } else { 10.0 };
        spring(&mut w.extend, &mut r.extend, extend, omega, dt);
        spring(
            &mut w.air,
            &mut r.air,
            f32::from(u8::from(airborne)),
            14.0,
            dt,
        );
        // Landing: legs overshoot slightly on the rebound, the torso folds a beat after them.
        let compression = s.compression.max(w.land_kick);
        spring_z(&mut w.land, &mut r.land, compression, 15.0, 0.75, dt);
        spring_z(&mut w.fold, &mut r.fold, w.land, 17.0, 0.7, dt);
        spring(
            &mut w.land_skid,
            &mut r.land_skid,
            w.skid_kick * ground,
            14.0,
            dt,
        );
        spring(
            &mut w.switch,
            &mut r.switch,
            f32::from(u8::from(s.switch)),
            8.0,
            dt,
        );
        if w.switch < 0.1 {
            w.switch_side = if c.trick_side < 0.0 { -1.0 } else { 1.0 };
        }

        // The chest lags the skis' turn: counter-rotation builds at a turn entry and relaxes.
        let counter = (-COUNTER_LAG * s.angular_velocity.y).clamp(-0.35, 0.35) * ground;
        spring(&mut w.counter, &mut r.counter, counter, 8.0, dt);

        // Free hands are damped masses on the body: they trail its acceleration (not gravity).
        if w.clock == 0.0 {
            w.last_velocity = s.velocity;
        }
        let mut dv = s.velocity - w.last_velocity;
        w.last_velocity = s.velocity;
        if airborne {
            dv.y += GRAVITY * dt;
        }
        // An impulse (pop, touchdown) reaches the hands over ~40 ms, not in one tick.
        w.dv += (dv - w.dv) * (1.0 - (-dt / 0.04).exp());
        let kick = (-HAND_GAIN * w.dv).clamp_length_max(HAND_KICK_MAX);
        for (h, omega) in [11.0_f32, 13.5].into_iter().enumerate() {
            let (x, v) = (w.hand[h], w.hand_vel[h]);
            w.hand_vel[h] = v + kick + (-omega * omega * x - 2.0 * HAND_ZETA * omega * v) * dt;
            w.hand[h] = (x + w.hand_vel[h] * dt).clamp_length_max(HAND_MAX);
        }
        w.clock += dt;

        // A pole plant opens each new turn: the pole on the new inside edge.
        let edge_side = if s.edge > 0.12 {
            1.0
        } else if s.edge < -0.12 {
            -1.0
        } else {
            w.edge_side
        };
        let plantable = !airborne
            && !s.switch
            && speed > PLANT_SPEED
            && s.skid.max(s.skate).max(s.pole).max(s.plow) < 0.3;
        if edge_side != w.edge_side {
            if plantable {
                w.plant[usize::from(edge_side > 0.0)] = 0.0;
            }
            w.edge_side = edge_side;
        }
        for h in 0..2 {
            w.plant[h] = (w.plant[h] + dt / PLANT_TIME).min(1.0);
            let plant = if plantable {
                plant_bump(w.plant[h])
            } else {
                0.0
            };
            spring(&mut w.plant_w[h], &mut r.plant_w[h], plant, 12.0, dt);
        }

        // Grabs: start after a short airtime, let go before the predicted landing.
        let ok = c.grab != Grab::None
            && airborne
            && s.crash.is_none()
            && s.air_time > GRAB_MIN_AIR
            && time_to_landing(s) > GRAB_RELEASE;
        let hold = ok && (self.grab == Grab::None || self.grab == c.grab);
        if self.grab == Grab::None && ok {
            self.grab = c.grab;
            w.grab_side = if c.trick_side < 0.0 { -1.0 } else { 1.0 };
        }
        spring(
            &mut w.grab,
            &mut w.rate.grab,
            f32::from(u8::from(hold)),
            12.0,
            dt,
        );
        if !hold && w.grab < 0.02 {
            w.grab = 0.0;
            w.rate.grab = 0.0;
            self.grab = Grab::None;
        }

        self.label(s);
    }

    fn label(&mut self, s: &Skier) {
        let w = &self.w;
        let (mode, phase, note) = if let Some(crash) = s.crash {
            ("crashed", crash.reason.label(), "ragdoll takes over")
        } else if !s.grounded {
            if w.grab > 0.3 {
                ("grab", self.grab.label(), "grab held, hand on the ski")
            } else if s.air_flip.abs() > 1.5 || s.air_spin.abs() > 1.5 {
                ("air", "rotating", "knees tucked, arms balancing")
            } else if w.extend > 0.3 {
                ("air", "takeoff", "legs fully extended")
            } else if s.velocity.y > 0.0 {
                ("air", "ascending", "knees tucked, arms balancing")
            } else {
                ("air", "descending", "preparing to land")
            }
        } else if w.land > 0.3 {
            ("landing", "absorb", "legs absorb the impact")
        } else if w.preload > 0.3 {
            ("preload", "crouch", "deep crouch before the pop")
        } else if w.skid > 0.35 {
            ("stop", "hockey stop", "skis sideways, edges set")
        } else if w.plow > 0.3 {
            ("plow", "wedge", "tips together, tails apart")
        } else if w.skate > 0.3 {
            (
                "skate",
                if s.skate_phase.sin() > 0.0 {
                    "push left"
                } else {
                    "push right"
                },
                if w.v2 > 0.5 {
                    "V2: poles push with the leg, weight on the glide ski"
                } else {
                    "free skate: pushing ski out, arms swing across"
                },
            )
        } else if w.pole > 0.3 {
            let u = s.pole_phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
            let phase = if u < 0.38 {
                "plant and push"
            } else if u < 0.6 {
                "release"
            } else if u < 0.85 {
                "swing"
            } else {
                "reach"
            };
            ("pole", phase, "double pole, tips sweeping back")
        } else if w.tuck > 0.5 {
            ("tuck", "aero", "poles tucked under the arms")
        } else if w.carve.abs() > 0.3 {
            (
                "carve",
                if w.carve < 0.0 { "left" } else { "right" },
                "inclined and angulated into the turn",
            )
        } else if s.speed() < 0.5 {
            ("idle", "stand", "relaxed stance, poles trailing")
        } else {
            ("glide", "neutral", "soft knees, poles trailing")
        };
        let (w_switch, switch_note) = (
            w.switch > 0.5 && s.grounded,
            "switch stance, looking over the shoulder",
        );
        self.mode = mode;
        self.phase = phase;
        self.note = if w_switch { switch_note } else { note };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn airborne_skier(height: f32, vy: f32) -> Skier {
        let mut s = Skier::default();
        s.position = Vec3::new(20.0, height, -250.0);
        s.velocity = Vec3::new(0.0, vy, 0.0);
        s.grounded = false;
        s
    }

    #[test]
    fn grab_blends_in_after_airtime_and_releases_before_landing() {
        let mut a = SkiAnimation::default();
        let c = SkiControls {
            grab: Grab::Mute,
            trick_side: 1.0,
            ..Default::default()
        };
        let mut s = airborne_skier(60.0, 0.0);
        for _ in 0..120 {
            s.air_time += 1.0 / 120.0;
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.grab, Grab::Mute);
        assert!(a.w.grab > 0.95, "grab weight {}", a.w.grab);
        assert_eq!(a.mode, "grab");

        // Now 0.2 s from the ground: the grab must have let go.
        let mut s = airborne_skier(0.2, -8.0);
        s.air_time = 1.0;
        assert!(time_to_landing(&s) < GRAB_RELEASE);
        for _ in 0..90 {
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.w.grab, 0.0);
        assert_eq!(a.grab, Grab::None);
    }

    #[test]
    fn no_grab_before_min_airtime_and_none_blends_out() {
        let mut a = SkiAnimation::default();
        let mut c = SkiControls {
            grab: Grab::Tail,
            ..Default::default()
        };
        let mut s = airborne_skier(60.0, 0.0);
        for _ in 0..12 {
            s.air_time += 1.0 / 120.0;
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.grab, Grab::None);
        s.air_time = 0.5;
        for _ in 0..60 {
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.grab, Grab::Tail);
        c.grab = Grab::None;
        for _ in 0..120 {
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.grab, Grab::None);
        assert_eq!(a.w.grab, 0.0);
    }

    #[test]
    fn free_hands_trail_a_push_and_settle() {
        let mut a = SkiAnimation::default();
        let mut s = Skier::default();
        s.grounded = true;
        let c = SkiControls::default();
        let mut trail = 0.0_f32;
        for i in 0..720 {
            s.velocity = if i < 60 {
                Vec3::ZERO
            } else {
                Vec3::new(0.0, 0.0, -8.0)
            };
            a.update(&s, &c, 1.0 / 120.0);
            assert!(a.w.hand[0].length() <= HAND_MAX + 1e-4);
            trail = trail.max(a.w.hand[1].z);
        }
        // Skier accelerates forward (-z): the hands lag behind (+z), then return to rest.
        assert!(trail > 0.03, "hands did not trail ({trail})");
        assert!(a.w.hand[0].length() < 1e-3 && a.w.hand[1].length() < 1e-3);
    }

    #[test]
    fn frozen_does_not_advance_and_labels_follow_state() {
        let mut a = SkiAnimation::default();
        let mut s = Skier::default();
        s.grounded = true;
        s.skid = 1.0;
        let c = SkiControls::default();
        a.frozen = true;
        a.update(&s, &c, 0.1);
        assert_eq!(a.w.skid, 0.0);
        a.frozen = false;
        for _ in 0..120 {
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.mode, "stop");
        s.skid = 0.0;
        s.tuck = 1.0;
        for _ in 0..240 {
            a.update(&s, &c, 1.0 / 120.0);
        }
        assert_eq!(a.mode, "tuck");
    }
}
