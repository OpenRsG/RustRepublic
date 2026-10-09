//! Ski mode: an independently authored skier (physics, animation rig, crash ragdoll, rendering)
//! sharing the bike game's arena, camera, pause/focus state and skeleton debug switches.
//! Key 5 selects it; 1–4 return to the bike.

mod anim;
mod demo;
pub(crate) mod physics;
pub(crate) mod pose;
mod ragdoll;
mod render;
mod rig;

use crate::animation::{GETUP_TIMEOUT, Rise};
use crate::game::{ChaseCamera, RideStatus};
use anim::SkiAnimation;
use bevy::ecs::system::SystemParam;
use bevy::input::InputSystems;
use bevy::prelude::*;
use demo::SkiDemo;
use physics::{Grab, SkiControls, Skier};
use pose::SkierPose;
use ragdoll::SkiRagdoll;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Sport {
    #[default]
    Bike,
    Ski,
}

pub(crate) fn ski_active(sport: Res<Sport>) -> bool {
    *sport == Sport::Ski
}

pub(crate) fn bike_active(sport: Res<Sport>) -> bool {
    *sport == Sport::Bike
}

/// Grab selection (U cycles, B holds) and trick side (J / L).
#[derive(Resource)]
struct SkiStatus {
    grab: Grab,
    side: f32,
}

impl Default for SkiStatus {
    fn default() -> Self {
        Self {
            grab: Grab::Mute,
            side: -1.0,
        }
    }
}

#[derive(Component)]
struct SkiHelp;

pub(crate) struct SkiPlugin;

impl Plugin for SkiPlugin {
    fn build(&self, app: &mut App) {
        // The browser build starts on skis (see `read_ski_controls`).
        let sport = if cfg!(target_arch = "wasm32") {
            Sport::Ski
        } else {
            Sport::Bike
        };
        app.insert_resource(sport)
            .init_resource::<SkiFrames>()
            .init_resource::<Skier>()
            .init_resource::<SkiControls>()
            .init_resource::<SkiAnimation>()
            .init_resource::<SkiRagdoll>()
            .init_resource::<SkiDemo>()
            .init_resource::<SkiStatus>()
            .add_systems(Startup, (render::spawn_skier, spawn_help))
            .add_systems(PreUpdate, read_ski_controls.after(InputSystems))
            .add_systems(FixedUpdate, simulate_ski.run_if(ski_active))
            .add_systems(Update, (render::animate_skier, show_panels));
    }
}

/// The pose every consumer draws: the ragdoll once crashed, otherwise the animated rig (blended
/// from the ragdoll pose while standing back up).
pub(crate) fn current_pose(s: &Skier, a: &SkiAnimation, r: &SkiRagdoll) -> SkierPose {
    r.pose().unwrap_or_else(|| {
        let riding = rig::solve(s, a);
        a.rise
            .as_ref()
            .map_or(riding, |up| lerp_pose(&up.from, &riding, up.weight()))
    })
}

/// Stands a crashed skier up on the snow under the ragdoll pelvis, facing the crash heading, once
/// the ragdoll sleeps or `GETUP_TIMEOUT` has passed. The drawn pose then blends from the ragdoll's.
fn stand_up(skier: &mut Skier, anim: &mut SkiAnimation, ragdoll: &mut SkiRagdoll) -> bool {
    let Some(crash) = skier.crash else {
        return false;
    };
    if !(ragdoll.sleeping() || crash.elapsed >= GETUP_TIMEOUT) {
        return false;
    }
    let (Some(pose), Some((pelvis, _))) = (ragdoll.pose(), ragdoll.centre()) else {
        return false;
    };
    skier.reset_at(pelvis.x, pelvis.z, skier.heading, 0.0);
    anim.reset();
    anim.rise = Some(Rise::new(pose));
    ragdoll.reset();
    true
}

/// Per-tick drawn pose plus skier root, kept for the last two fixed ticks so display frames can
/// interpolate between them.
#[derive(Clone, Copy)]
struct Frame {
    pose: SkierPose,
    position: Vec3,
}

#[derive(Resource, Default)]
pub(crate) struct SkiFrames {
    previous: Option<Frame>,
    current: Option<Frame>,
}

impl SkiFrames {
    /// Forget history (teleport / reset) so no frame blends across it.
    fn clear(&mut self) {
        *self = Self::default();
    }

    fn push(&mut self, pose: SkierPose, position: Vec3) {
        self.previous = self.current;
        self.current = Some(Frame { pose, position });
    }

    /// Pose and root position at `alpha` between the last two ticks; `None` before the first tick.
    pub(crate) fn blend(&self, alpha: f32) -> Option<(SkierPose, Vec3)> {
        let c = self.current?;
        let Some(p) = self.previous else {
            return Some((c.pose, c.position));
        };
        Some((
            lerp_pose(&p.pose, &c.pose, alpha),
            p.position.lerp(c.position, alpha),
        ))
    }
}

/// Per-joint lerp; `ski_up` / `facing` / `gaze` are nlerped back to unit length.
fn lerp_pose(a: &SkierPose, b: &SkierPose, t: f32) -> SkierPose {
    let mut out = *b;
    for i in 0..a.p.len() {
        out.p[i] = a.p[i].lerp(b.p[i], t);
    }
    for i in 0..2 {
        out.ski_up[i] = a.ski_up[i].lerp(b.ski_up[i], t).normalize_or(b.ski_up[i]);
    }
    out.facing = a.facing.lerp(b.facing, t).normalize_or(b.facing);
    out.gaze = a.gaze.lerp(b.gaze, t).normalize_or(b.gaze);
    out
}

/// What the shared chase camera follows in ski mode.
pub(crate) struct Follow {
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub crashed: bool,
    /// Ragdoll pelvis while crashed.
    pub ragdoll: Option<Vec3>,
    pub demo: bool,
}

/// Read-only ski state for the game's camera and HUD.
#[derive(SystemParam)]
pub(crate) struct SkiView<'w> {
    frames: Res<'w, SkiFrames>,
    fixed: Res<'w, Time<Fixed>>,
    sport: Res<'w, Sport>,
    skier: Res<'w, Skier>,
    anim: Res<'w, SkiAnimation>,
    ragdoll: Res<'w, SkiRagdoll>,
    demo: Res<'w, SkiDemo>,
    status: Res<'w, SkiStatus>,
}

impl SkiView<'_> {
    pub fn active(&self) -> bool {
        *self.sport == Sport::Ski
    }

    pub fn follow(&self) -> Option<Follow> {
        let (pose, position) = self
            .frames
            .blend(self.fixed.overstep_fraction())
            .map_or((None, self.skier.position), |(p, r)| (Some(p), r));
        self.active().then(|| Follow {
            position,
            velocity: self.skier.velocity,
            yaw: self.skier.heading,
            crashed: self.skier.crash.is_some(),
            ragdoll: self
                .ragdoll
                .centre()
                .map(|(p, _)| pose.map_or(p, |q| q[pose::J::Pelvis])),
            demo: self.demo.enabled,
        })
    }

    pub fn hud(&self, paused: bool, focused: bool, skeleton: bool, mesh: bool) -> Option<String> {
        if !self.active() {
            return None;
        }
        let s = &*self.skier;
        let a = &*self.anim;
        let mode = if paused {
            "PAUSED - Esc to resume".to_string()
        } else if !focused {
            "FOCUS LOST - click to ride".to_string()
        } else if let Some(c) = s.crash {
            let mut hurt = String::new();
            if !self.ragdoll.injuries().is_empty() {
                hurt += &format!(" | broke {}", self.ragdoll.injuries().join(", "));
            }
            match self.ragdoll.skis_off() {
                0 => {}
                1 => hurt += " | a ski came off",
                _ => hurt += " | both skis came off",
            }
            match self.ragdoll.poles_off() {
                0 => {}
                1 => hurt += " | dropped a pole",
                _ => hurt += " | dropped both poles",
            }
            format!(
                "CRASHED: {} at {:.1} m/s{hurt} - R to reset",
                c.reason.label(),
                c.impact
            )
        } else if !s.grounded {
            "AIRBORNE".to_string()
        } else if s.switch {
            "SKIING SWITCH".to_string()
        } else {
            "SKIING".to_string()
        };
        let on_off = |on: bool| if on { "on" } else { "off" };
        let demo = if self.demo.enabled {
            format!(
                "{} {} {}",
                self.demo.name(),
                self.demo.stage(),
                self.demo.outcome
            )
        } else {
            "F6 - ski showcase".to_string()
        };
        Some(format!(
            "Ski   {:>3.0} km/h   |   {}\n{}\nAnimation: {} / {}  {}\nGrab U: {} ({})   Active: {}\nEdge {:+.0} deg   Lean {:+.0} deg   Skid {:.0}%\nLast air: {}   impact {:.1} m/s\nSkeleton {} (F1)   Rider mesh {} (F2)",
            s.velocity.length() * 3.6,
            mode,
            demo,
            a.mode,
            a.phase,
            a.note,
            self.status.grab.label(),
            if self.status.side < 0.0 {
                "left"
            } else {
                "right"
            },
            a.grab.label(),
            s.edge.to_degrees(),
            s.lean.to_degrees(),
            s.skid * 100.0,
            trick_name(s.last_spin, s.last_flip),
            s.impact,
            on_off(skeleton),
            on_off(mesh),
        ))
    }
}

/// "540 + backflip" from accumulated air rotation, rounded to half turns / whole flips.
fn trick_name(spin: f32, flip: f32) -> String {
    let half_turns = (spin.abs() / std::f32::consts::PI).round() as u32;
    let flips = (flip.abs() / std::f32::consts::TAU).round() as u32;
    let mut parts = Vec::new();
    if half_turns > 0 {
        parts.push(format!(
            "{} {}",
            half_turns * 180,
            if spin > 0.0 { "right" } else { "left" }
        ));
    }
    if flips > 0 {
        let kind = if flip > 0.0 { "backflip" } else { "frontflip" };
        parts.push(if flips == 1 {
            kind.to_string()
        } else {
            format!("{flips}x {kind}")
        });
    }
    if parts.is_empty() {
        "straight".to_string()
    } else {
        parts.join(" + ")
    }
}

fn axis(keys: &ButtonInput<KeyCode>, positive: KeyCode, negative: KeyCode) -> f32 {
    u8::from(keys.pressed(positive)) as f32 - u8::from(keys.pressed(negative)) as f32
}

fn reset_camera(chase: &mut ChaseCamera, demo: bool) {
    *chase = ChaseCamera::default();
    if demo {
        chase.distance = crate::game::CAM_DEMO_DISTANCE;
    }
}

fn read_ski_controls(
    keys: Res<ButtonInput<KeyCode>>,
    sport: Res<Sport>,
    status: Res<RideStatus>,
    mut previous: Local<Sport>,
    mut controls: ResMut<SkiControls>,
    mut select: ResMut<SkiStatus>,
    mut skier: ResMut<Skier>,
    mut anim: ResMut<SkiAnimation>,
    mut ragdoll: ResMut<SkiRagdoll>,
    mut demo: ResMut<SkiDemo>,
    mut chase: ResMut<ChaseCamera>,
    mut frames: ResMut<SkiFrames>,
    mut autostarted: Local<bool>,
) {
    let entered = *sport == Sport::Ski && *previous != Sport::Ski;
    *previous = *sport;
    // The browser build opens straight into the looping showcase (browsers keep F-keys).
    // Only the very first frame counts: switching away first means no showcase on return.
    let autostart = cfg!(target_arch = "wasm32") && !*autostarted;
    *autostarted = true;
    if *sport != Sport::Ski {
        demo.enabled = false;
        *controls = SkiControls::default();
        return;
    }
    if entered {
        frames.clear();
        skier.reset();
        anim.reset();
        ragdoll.reset();
        reset_camera(&mut chase, false);
    }
    if autostart || status.focused && keys.just_pressed(KeyCode::F6) {
        demo.enabled = autostart || !demo.enabled;
        if demo.enabled {
            demo.index = 0;
            demo.completed = 0;
            frames.clear();
            demo.begin_run(&mut skier);
            anim.reset();
            ragdoll.reset();
            reset_camera(&mut chase, true);
        }
    }
    if !status.focused {
        *controls = SkiControls::default();
        return;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        if demo.enabled {
            frames.clear();
            demo.begin_run(&mut skier);
        } else {
            frames.clear();
            skier.reset();
        }
        anim.reset();
        ragdoll.reset();
        reset_camera(&mut chase, demo.enabled);
    }
    if keys.just_pressed(KeyCode::KeyU) {
        let i = Grab::ALL
            .iter()
            .position(|&g| g == select.grab)
            .unwrap_or(0);
        select.grab = Grab::ALL[(i + 1) % Grab::ALL.len()];
    }
    if keys.just_pressed(KeyCode::KeyJ) {
        select.side = -1.0;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        select.side = 1.0;
    }
    if keys.just_pressed(KeyCode::F3) {
        info!(
            "SKI_STATE position={:?} velocity={:?} speed={:.3} heading={:.3} edge={:.3} lean={:.3} grounded={} air_time={:.3} compression={:.3} preload={:.3} tuck={:.3} plow={:.3} skid={:.3} skate={:.3} pole={:.3} switch={} impact={:.3} last_spin={:.3} last_flip={:.3} crash={:?} ragdoll={} ragdoll_sleeping={} animation={} phase={} grab={:?} note={} demo={} demo_case={} demo_stage={} demo_completed={}",
            skier.position,
            skier.velocity,
            skier.speed(),
            skier.heading,
            skier.edge,
            skier.lean,
            skier.grounded,
            skier.air_time,
            skier.compression,
            skier.preload,
            skier.tuck,
            skier.plow,
            skier.skid,
            skier.skate,
            skier.pole,
            skier.switch,
            skier.impact,
            skier.last_spin,
            skier.last_flip,
            skier.crash,
            ragdoll.active(),
            ragdoll.sleeping(),
            anim.mode,
            anim.phase,
            anim.grab,
            anim.note,
            demo.enabled,
            demo.name(),
            demo.stage(),
            demo.completed,
        );
    }
    *controls = if status.paused {
        SkiControls::default()
    } else {
        SkiControls {
            push: u8::from(keys.pressed(KeyCode::KeyW)) as f32,
            brake: u8::from(keys.pressed(KeyCode::KeyS)) as f32,
            steer: axis(&keys, KeyCode::KeyD, KeyCode::KeyA),
            tuck: keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight),
            jump: keys.pressed(KeyCode::Space),
            air_pitch: axis(&keys, KeyCode::ArrowDown, KeyCode::ArrowUp),
            air_roll: axis(&keys, KeyCode::ArrowRight, KeyCode::ArrowLeft),
            air_yaw: axis(&keys, KeyCode::KeyE, KeyCode::KeyT),
            flip: keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight),
            grab: if keys.pressed(KeyCode::KeyB) {
                select.grab
            } else {
                Grab::None
            },
            trick_side: select.side,
        }
    };
}

fn simulate_ski(
    time: Res<Time<Fixed>>,
    status: Res<RideStatus>,
    mut controls: ResMut<SkiControls>,
    mut skier: ResMut<Skier>,
    mut anim: ResMut<SkiAnimation>,
    mut ragdoll: ResMut<SkiRagdoll>,
    mut demo: ResMut<SkiDemo>,
    mut chase: ResMut<ChaseCamera>,
    mut frames: ResMut<SkiFrames>,
    mut last: Local<Option<SkierPose>>,
) {
    anim.frozen = status.paused || !status.focused || skier.crash.is_some();
    if status.paused || !status.focused {
        return;
    }
    let dt = time.delta_secs();
    let mut input = *controls;
    if demo.drive(&mut skier, &mut input, dt) {
        anim.reset();
        ragdoll.reset();
        *last = None;
        frames.clear();
        reset_camera(&mut chase, true);
    }
    if let Some(up) = &mut anim.rise {
        up.elapsed += dt;
        input = SkiControls::default();
        if up.done() {
            anim.rise = None;
        }
    }
    if skier.crash.is_none() {
        anim.update(&skier, &input, dt);
    }
    // Pose before the step: the ragdoll inherits pre-impact limb momentum.
    let pose = rig::solve(&skier, &anim);
    skier.step(&input, dt);
    if skier.crash.is_some() {
        if !ragdoll.active() {
            ragdoll.activate(&last.unwrap_or(pose), &pose, dt);
        }
        ragdoll.step(dt);
        if demo.enabled || !stand_up(&mut skier, &mut anim, &mut ragdoll) {
            anim.mode = "crashed";
            anim.phase = "ragdoll";
            anim.frozen = true;
        }
    }
    *last = Some(pose);
    frames.push(current_pose(&skier, &anim, &ragdoll), skier.position);
    if demo.enabled {
        *controls = input;
    }
}

const HUD_SURFACE: Color = Color::srgba(0.035, 0.075, 0.068, 0.94);
const HUD_TEXT: Color = Color::srgb(0.91, 0.97, 0.92);

fn spawn_help(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(22),
                bottom: px(20),
                width: px(400),
                max_width: percent(45),
                padding: UiRect::all(px(14)),
                ..default()
            },
            BackgroundColor(HUD_SURFACE),
            Visibility::Hidden,
            SkiHelp,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new(
                    "Ski: F6 showcase  1-4 bike  R reset\nW skate / pole push  S plow / hockey stop\nA/D carve  Shift tuck  Space hold+release jump\nArrows flip/roll  E/T spin  Ctrl full flips\nU grab select  B hold grab  J/L side\nRMB orbit  Wheel zoom  C center\nF1 bones  F2 mesh  Esc pause  H help",
                ),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(HUD_TEXT),
            ));
        });
}

/// Ski help replaces the bike help; the bike assembly hides while skiing.
fn show_panels(
    sport: Res<Sport>,
    status: Res<RideStatus>,
    mut help: Query<&mut Visibility, With<SkiHelp>>,
    mut bike: Query<(&crate::scene::Part, &mut Visibility), Without<SkiHelp>>,
) {
    let ski = *sport == Sport::Ski;
    let shown = |on: bool| {
        if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    };
    for mut v in &mut help {
        v.set_if_neq(shown(ski && status.help));
    }
    for (part, mut v) in &mut bike {
        if matches!(part, crate::scene::Part::Root) {
            v.set_if_neq(shown(!ski));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bike::{HILL_START_Z, HILL_X};

    #[test]
    fn trick_names_round_to_half_turns_and_whole_flips() {
        use std::f32::consts::{PI, TAU};
        assert_eq!(trick_name(0.2, 0.3), "straight");
        assert_eq!(trick_name(-2.0 * PI - 0.3, 0.0), "360 left");
        assert_eq!(
            trick_name(3.0 * PI + 0.2, TAU - 0.4),
            "540 right + backflip"
        );
        assert_eq!(trick_name(0.0, -2.0 * TAU), "2x frontflip");
    }

    #[test]
    fn lerp_pose_endpoints_midpoint_and_unit_normals() {
        let mk = |x: f32, up: Vec3| {
            let mut p = SkierPose::from_points([Vec3::new(x, 0.0, 0.0); pose::JOINTS]);
            p.ski_up = [up; 2];
            p.facing = up;
            p.gaze = up;
            p
        };
        let a = mk(0.0, Vec3::Y);
        let b = mk(2.0, Vec3::X);
        let (l0, l1, m) = (
            lerp_pose(&a, &b, 0.0),
            lerp_pose(&a, &b, 1.0),
            lerp_pose(&a, &b, 0.5),
        );
        assert_eq!(l0.p, a.p);
        assert_eq!(l1.p, b.p);
        assert_eq!(m.p[0], Vec3::X);
        assert!((m.ski_up[0].length() - 1.0).abs() < 1e-5);
        assert!((m.facing.length() - 1.0).abs() < 1e-5);
        assert!((m.gaze.length() - 1.0).abs() < 1e-5);
    }

    /// The real ski systems, a skier dropped 2 m at 20 m/s onto flat snow at (-40, 0) (a hard-impact
    /// crash far from the summit start), the push key held throughout. `Update` has run once, so
    /// entering ski mode has already reset the skier.
    fn crashed_app() -> App {
        let mut clock = Time::<Fixed>::from_hz(120.0);
        clock.advance_by(std::time::Duration::from_secs_f64(1.0 / 120.0));
        let mut app = App::new();
        app.insert_resource(clock)
            .insert_resource(Sport::Ski)
            .init_resource::<RideStatus>()
            .init_resource::<ChaseCamera>()
            .init_resource::<SkiFrames>()
            .init_resource::<Skier>()
            .init_resource::<SkiControls>()
            .init_resource::<SkiAnimation>()
            .init_resource::<SkiRagdoll>()
            .init_resource::<SkiDemo>()
            .init_resource::<SkiStatus>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, read_ski_controls)
            .add_systems(FixedUpdate, simulate_ski);
        app.world_mut().run_schedule(Update);
        let mut skier = app.world_mut().resource_mut::<Skier>();
        skier.reset_at(-40.0, 0.0, 0.0, 8.0);
        skier.grounded = false;
        skier.position.y = 2.0;
        skier.velocity.y = -20.0;
        app.world_mut().resource_mut::<SkiControls>().push = 1.0;
        app
    }

    /// Largest per-tick world movement of a drawn body joint (pelvis to hands), m.
    const MAX_JOINT_STEP: f32 = 0.04;
    /// Same for skis and poles, which the crash may have thrown far away and which fly back to the
    /// rider over the blend (24 m in this test), m.
    const MAX_GEAR_STEP: f32 = 0.5;

    #[test]
    fn a_crashed_skier_stands_up_where_it_fell_blends_smoothly_and_skis_on() {
        let mut app = crashed_app();
        let (mut crashed, mut recovered, mut rising) = (0, 0, 0);
        let (mut pelvis, mut last) = (Vec3::ZERO, None::<SkierPose>);
        let (mut jump, mut gear_jump) = (0.0_f32, 0.0_f32);
        let mut standing = None;
        for _ in 0..2400 {
            app.world_mut().run_schedule(FixedUpdate);
            let world = app.world();
            let skier = world.resource::<Skier>();
            if let Some((p, _)) = world.resource::<SkiRagdoll>().centre() {
                pelvis = p;
            }
            let pose = world.resource::<SkiFrames>().current.unwrap().pose;
            let rise = world.resource::<SkiAnimation>().rise.is_some();
            if skier.crash.is_some() {
                crashed += 1;
            } else if crashed > 0 {
                recovered += 1;
                if rise {
                    rising += 1;
                    let before = last.unwrap();
                    let step = |range: std::ops::Range<usize>| {
                        range
                            .map(|i| pose.p[i].distance(before.p[i]))
                            .fold(0.0, f32::max)
                    };
                    let body = pose::J::BindL as usize;
                    jump = jump.max(step(0..body));
                    gear_jump = gear_jump.max(step(body..pose::JOINTS));
                } else if standing.is_none() {
                    // The blend is over: the controls were ignored until now.
                    assert!(skier.speed() < 0.05, "{:?}", skier.velocity);
                    standing = Some(recovered);
                    assert!(skier.up().y > 0.99);
                    assert!(
                        skier
                            .position
                            .xz()
                            .distance(Vec2::new(HILL_X, HILL_START_Z))
                            > 10.0
                    );
                    let rider = pose[pose::J::Pelvis];
                    assert!(Vec2::new(rider.x - pelvis.x, rider.z - pelvis.z).length() < 2.0);
                }
            }
            last = Some(pose);
            if standing.is_some_and(|s| recovered > s + 120) {
                break;
            }
        }
        let skier = app.world().resource::<Skier>();
        assert!(
            crashed > 0 && recovered > 0,
            "never crashed or never recovered"
        );
        assert!(
            (crashed as f32 / 120.0) <= GETUP_TIMEOUT + 0.05,
            "lay {crashed} ticks"
        );
        assert!(
            (rising as f32 / 120.0 - crate::animation::GETUP_BLEND).abs() < 0.05,
            "{rising} blend ticks"
        );
        assert!(jump < MAX_JOINT_STEP, "joint moved {jump} m in one tick");
        assert!(
            gear_jump < MAX_GEAR_STEP,
            "gear moved {gear_jump} m in one tick"
        );
        assert!(skier.crash.is_none() && skier.speed() > 0.5, "{skier:?}");
    }

    #[test]
    fn reset_during_the_blend_returns_to_the_summit() {
        let mut app = crashed_app();
        while app.world().resource::<SkiAnimation>().rise.is_none() {
            app.world_mut().run_schedule(FixedUpdate);
        }
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.world_mut().run_schedule(Update);
        let world = app.world();
        let skier = world.resource::<Skier>();
        assert!(skier.crash.is_none());
        assert!(
            skier
                .position
                .xz()
                .distance(Vec2::new(HILL_X, HILL_START_Z))
                < 0.5
        );
        assert!(world.resource::<SkiAnimation>().rise.is_none());
        assert!(!world.resource::<SkiRagdoll>().active());
    }
}
