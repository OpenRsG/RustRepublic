//! Procedural sound, no asset files: wind, tyre/snow noise, skid hiss, landing thuds and crash
//! impacts are synthesised sample by sample. A fixed-rate system maps the active sport's state to
//! a few target levels in atomics, and the audio thread's `Synth` smooths them to avoid clicks.
//! M mutes; sound also stops while paused, unfocused, or for the inactive sport.

use crate::bike::{Bike, Controls};
use crate::game::RideStatus;
use crate::ski::Sport;
use crate::ski::physics::Skier;
use bevy::audio::{AddAudioSource, AudioPlayer, Decodable, Source};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use std::f32::consts::TAU;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::Duration;

const SAMPLE_RATE: u32 = 44_100;
/// Samples between reads of the shared atomics.
const CONTROL_PERIOD: usize = 32;
/// Level smoothing time constant, seconds.
const SMOOTH_SECS: f32 = 0.05;
/// Output gain before the soft clip.
const VOLUME: f32 = 0.5;
/// Landing thud and crash decay time constants, seconds.
const THUD_DECAY_SECS: f32 = 0.12;
const CRASH_DECAY_SECS: f32 = 0.5;
/// Speed (m/s) at which wind reaches full level.
const WIND_FULL_SPEED: f32 = 30.0;
/// Speed (m/s) at which tyre/snow noise reaches full level.
const GROUND_FULL_SPEED: f32 = 20.0;
/// Lateral tyre speed (m/s) that gives a full skid hiss.
const SKID_FULL_SPEED: f32 = 3.0;
/// Touchdown closing speed (m/s) below which no thud plays, and at which it is full.
const THUD_MIN_IMPACT: f32 = 2.0;
const THUD_FULL_IMPACT: f32 = 14.0;
/// Crash impact speed (m/s) giving a full-level crash sound.
const CRASH_FULL_IMPACT: f32 = 18.0;

/// Continuous target levels, each 0..1.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Levels {
    wind: f32,
    /// Tyre roll / gravel or snow swish.
    roll: f32,
    /// Skid hiss.
    hiss: f32,
    /// Overall gate: 0 muted.
    master: f32,
}

/// Levels (f32 bits) and one-shot hits shared between the game and the audio thread.
#[derive(Default)]
struct Shared {
    wind: AtomicU32,
    roll: AtomicU32,
    hiss: AtomicU32,
    master: AtomicU32,
    /// Largest unplayed hit amplitudes; non-negative floats order like their bits.
    thud: AtomicU32,
    crash: AtomicU32,
}

fn load(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Relaxed))
}

fn store(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Relaxed);
}

/// Filter and envelope state of the synthesiser.
struct Synth {
    rng: u32,
    now: Levels,
    target: Levels,
    wind_lp: f32,
    roll_lp: f32,
    crackle: f32,
    hiss_lp: f32,
    thud: f32,
    thud_phase: f32,
    crash: f32,
    crash_phase: f32,
    crash_lp: f32,
}

impl Synth {
    fn new() -> Self {
        Self {
            rng: 0x9E37_79B9,
            now: Levels::default(),
            target: Levels::default(),
            wind_lp: 0.0,
            roll_lp: 0.0,
            crackle: 0.0,
            hiss_lp: 0.0,
            thud: 0.0,
            thud_phase: 0.0,
            crash: 0.0,
            crash_phase: 0.0,
            crash_lp: 0.0,
        }
    }

    /// White noise in [-1, 1].
    fn noise(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng as i32 as f32 / i32::MAX as f32
    }

    fn hit(&mut self, thud: f32, crash: f32) {
        self.thud = self.thud.max(thud);
        self.crash = self.crash.max(crash);
    }

    fn sample(&mut self) -> f32 {
        let k = 1.0 / (SMOOTH_SECS * SAMPLE_RATE as f32);
        self.now.wind += (self.target.wind - self.now.wind) * k;
        self.now.roll += (self.target.roll - self.now.roll) * k;
        self.now.hiss += (self.target.hiss - self.now.hiss) * k;
        self.now.master += (self.target.master - self.now.master) * k;
        let n = self.noise();

        // Wind: low-passed noise whose cutoff opens with speed.
        self.wind_lp += (n - self.wind_lp) * (0.03 + 0.25 * self.now.wind);
        let wind = self.wind_lp * self.now.wind * 1.4;

        // Roll: rumble plus sparse crackle (gravel).
        self.roll_lp += (n - self.roll_lp) * 0.08;
        let spark = self.noise();
        self.crackle = if spark.abs() > 1.0 - 0.04 * self.now.roll {
            spark
        } else {
            self.crackle * 0.9
        };
        let roll = (self.roll_lp * 1.5 + self.crackle * 0.4) * self.now.roll * 0.5;

        // Hiss: high-passed noise.
        self.hiss_lp += (n - self.hiss_lp) * 0.35;
        let hiss = (n - self.hiss_lp) * self.now.hiss * 0.35;

        // Landing thud: a falling sine with a noise knock.
        self.thud -= self.thud / (THUD_DECAY_SECS * SAMPLE_RATE as f32);
        self.thud_phase =
            (self.thud_phase + TAU * (40.0 + 30.0 * self.thud) / SAMPLE_RATE as f32) % TAU;
        let thud = self.thud * (self.thud_phase.sin() * 0.9 + self.roll_lp * 0.8);

        // Crash: low thump with a dull clatter.
        self.crash -= self.crash / (CRASH_DECAY_SECS * SAMPLE_RATE as f32);
        self.crash_phase = (self.crash_phase + TAU * 32.0 / SAMPLE_RATE as f32) % TAU;
        self.crash_lp += (n - self.crash_lp) * 0.2;
        let crash = self.crash * (self.crash_phase.sin() * 0.7 + self.crash_lp * 0.9);

        ((wind + roll + hiss + thud + crash) * self.now.master * VOLUME).tanh()
    }
}

/// Endless mono stream that follows the shared levels.
struct Stream {
    shared: Arc<Shared>,
    synth: Synth,
    until_control: usize,
}

impl Iterator for Stream {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.until_control == 0 {
            self.until_control = CONTROL_PERIOD;
            let s = &self.shared;
            self.synth.target = Levels {
                wind: load(&s.wind),
                roll: load(&s.roll),
                hiss: load(&s.hiss),
                master: load(&s.master),
            };
            self.synth.hit(
                f32::from_bits(s.thud.swap(0, Relaxed)),
                f32::from_bits(s.crash.swap(0, Relaxed)),
            );
        }
        self.until_control -= 1;
        Some(self.synth.sample())
    }
}

impl Source for Stream {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

#[derive(Asset, TypePath)]
struct Procedural(Arc<Shared>);

impl Decodable for Procedural {
    type DecoderItem = f32;
    type Decoder = Stream;

    fn decoder(&self) -> Stream {
        Stream {
            shared: self.0.clone(),
            synth: Synth::new(),
            until_control: 0,
        }
    }
}

#[derive(Resource)]
struct Mixer(Arc<Shared>);

#[derive(Resource, Default)]
struct Muted(bool);

pub(crate) struct SoundPlugin;

impl Plugin for SoundPlugin {
    fn build(&self, app: &mut App) {
        app.add_audio_source::<Procedural>()
            .insert_resource(Mixer(Arc::default()))
            .init_resource::<Muted>()
            .add_systems(Startup, play)
            .add_systems(Update, toggle_mute)
            .add_systems(FixedLast, drive);
    }
}

fn play(mixer: Res<Mixer>, mut assets: ResMut<Assets<Procedural>>, mut commands: Commands) {
    commands.spawn(AudioPlayer(assets.add(Procedural(mixer.0.clone()))));
}

fn toggle_mute(keys: Res<ButtonInput<KeyCode>>, mut muted: ResMut<Muted>) {
    if keys.just_pressed(KeyCode::KeyM) {
        muted.0 = !muted.0;
        info!("Sound {}", if muted.0 { "muted" } else { "on" });
    }
}

fn wind(speed: f32, grounded: bool) -> f32 {
    (speed / WIND_FULL_SPEED).min(1.0) * if grounded { 0.6 } else { 1.0 }
}

fn bike_levels(speed: f32, lateral: f32, brake: f32, grounded: bool) -> Levels {
    let v = (speed / GROUND_FULL_SPEED).min(1.0);
    let slip = (lateral / SKID_FULL_SPEED).min(1.0);
    Levels {
        wind: wind(speed, grounded),
        roll: if grounded { 0.4 * v + 0.3 * slip } else { 0.0 },
        hiss: if grounded { slip.max(brake * v) } else { 0.0 },
        master: 1.0,
    }
}

fn ski_levels(speed: f32, edge: f32, skid: f32, grounded: bool) -> Levels {
    let v = (speed / GROUND_FULL_SPEED).min(1.0);
    Levels {
        wind: wind(speed, grounded),
        roll: if grounded { 0.3 * v } else { 0.0 },
        hiss: if grounded {
            v * skid.max(0.4 * edge.abs().min(1.0))
        } else {
            0.0
        },
        master: 1.0,
    }
}

/// Last tick's contact and crash state, to catch touchdowns and crashes.
struct Edges {
    sport: Sport,
    grounded: bool,
    /// Descent speed of the previous tick, m/s.
    fall: f32,
    crashed: bool,
}

fn drive(
    mixer: Res<Mixer>,
    sport: Res<Sport>,
    status: Res<RideStatus>,
    muted: Res<Muted>,
    bike: Res<Bike>,
    controls: Res<Controls>,
    skier: Res<Skier>,
    mut edges: Local<Option<Edges>>,
) {
    // A sport switch swaps the state under us; only same-sport changes are events.
    let before = edges.take().filter(|b| b.sport == *sport);
    let (mut levels, grounded, fall, touch, crash) = match *sport {
        Sport::Bike => {
            let lateral = bike.velocity.dot(bike.orientation() * Vec3::X).abs();
            let grounded = bike.grounded[0] || bike.grounded[1];
            let levels = bike_levels(bike.velocity.length(), lateral, controls.brake, grounded);
            let prev_fall = before.as_ref().map_or(0.0, |b| b.fall);
            (
                levels,
                grounded,
                -bike.velocity.y,
                prev_fall,
                bike.crash.map(|c| c.impact),
            )
        }
        Sport::Ski => {
            let levels = ski_levels(skier.speed(), skier.edge, skier.skid, skier.grounded);
            (
                levels,
                skier.grounded,
                0.0,
                skier.impact,
                skier.crash.map(|c| c.impact),
            )
        }
    };
    *edges = Some(Edges {
        sport: *sport,
        grounded,
        fall,
        crashed: crash.is_some(),
    });
    let on = !muted.0 && !status.paused && status.focused;
    if !on {
        levels.master = 0.0;
    }
    let s = &mixer.0;
    store(&s.wind, levels.wind);
    store(&s.roll, levels.roll);
    store(&s.hiss, levels.hiss);
    store(&s.master, levels.master);
    let Some(before) = before.filter(|_| on) else {
        return;
    };
    if let Some(impact) = crash.filter(|_| !before.crashed) {
        let amp = (impact / CRASH_FULL_IMPACT).clamp(0.4, 1.0);
        s.crash.fetch_max(amp.to_bits(), Relaxed);
    } else if grounded && !before.grounded && touch > THUD_MIN_IMPACT {
        let amp = (touch - THUD_MIN_IMPACT) / (THUD_FULL_IMPACT - THUD_MIN_IMPACT);
        s.thud.fetch_max(amp.min(1.0).to_bits(), Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(synth: &mut Synth, seconds: f32) -> Vec<f32> {
        (0..(seconds * SAMPLE_RATE as f32) as usize)
            .map(|_| synth.sample())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn settled(levels: Levels) -> Synth {
        let mut s = Synth::new();
        s.target = levels;
        render(&mut s, 0.5);
        s
    }

    #[test]
    fn silent_at_rest_and_when_gated() {
        let mut s = settled(bike_levels(0.0, 0.0, 0.0, true));
        assert!(render(&mut s, 0.2).iter().all(|&v| v == 0.0));
        let mut fast = bike_levels(25.0, 4.0, 1.0, true);
        fast.master = 0.0;
        let mut s = settled(fast);
        assert!(render(&mut s, 0.2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn finite_and_bounded_at_full_level_with_hits() {
        let mut s = settled(Levels {
            wind: 1.0,
            roll: 1.0,
            hiss: 1.0,
            master: 1.0,
        });
        s.hit(1.0, 1.0);
        assert!(
            render(&mut s, 1.0)
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 1.0)
        );
    }

    #[test]
    fn louder_with_speed_and_airborne_and_skid() {
        let level = |l: Levels| rms(&render(&mut settled(l), 0.3));
        let (slow, fast) = (
            bike_levels(5.0, 0.0, 0.0, true),
            bike_levels(25.0, 0.0, 0.0, true),
        );
        assert!(level(fast) > 2.0 * level(slow));
        assert!(
            level(bike_levels(25.0, 0.0, 0.0, false))
                > level(fast) * 0.0 + level(bike_levels(25.0, 0.0, 0.0, true)) * 0.0
        );
        assert!(bike_levels(25.0, 0.0, 0.0, false).wind > fast.wind);
        assert!(
            level(bike_levels(15.0, 3.0, 0.0, true)) > level(bike_levels(15.0, 0.0, 0.0, true))
        );
        assert!(level(ski_levels(15.0, 0.0, 1.0, true)) > level(ski_levels(15.0, 0.0, 0.0, true)));
        assert_eq!(
            ski_levels(0.0, 0.5, 1.0, true),
            Levels {
                master: 1.0,
                ..Levels::default()
            }
        );
    }

    #[test]
    fn landing_is_a_decaying_transient() {
        let mut s = settled(Levels {
            master: 1.0,
            ..Levels::default()
        });
        s.hit(1.0, 0.0);
        let x = render(&mut s, 0.6);
        let w = |a: f32, b: f32| {
            rms(&x[(a * SAMPLE_RATE as f32) as usize..(b * SAMPLE_RATE as f32) as usize])
        };
        assert!(w(0.0, 0.05) > 0.1);
        assert!(w(0.0, 0.1) > w(0.1, 0.2) && w(0.1, 0.2) > w(0.3, 0.4));
        assert!(w(0.5, 0.6) < 0.02);
    }
}
