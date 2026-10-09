use crate::animation::{GETUP_TIMEOUT, Input as AnimationInput, Rise};
use crate::bike::{Bike, BikeTrick, Controls, Discipline, HandTrick, LegTrick, terrain_height};
use crate::scene;
use crate::showcase::{Showcase, Stage};
use crate::ski;
use bevy::input::InputSystems;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::f32::consts::{PI, TAU};

const HUD_SURFACE: Color = Color::srgba(0.035, 0.075, 0.068, 0.94);
const HUD_TEXT: Color = Color::srgb(0.91, 0.97, 0.92);
const HUD_ACCENT: Color = Color::srgb(1.0, 0.63, 0.34);
const HUD_SECONDARY: Color = Color::srgb(0.73, 0.85, 0.77);

#[derive(Resource)]
pub(crate) struct RideStatus {
    pub paused: bool,
    pub focused: bool,
    pub help: bool,
    queued_hop: bool,
    hand: HandTrick,
    leg: LegTrick,
    bike: BikeTrick,
    side: f32,
}

impl Default for RideStatus {
    fn default() -> Self {
        Self {
            paused: false,
            focused: true,
            help: true,
            queued_hop: false,
            hand: HandTrick::NoHand,
            leg: LegTrick::NoFoot,
            bike: BikeTrick::Whip,
            side: -1.0,
        }
    }
}

const CAM_DEFAULT_PITCH: f32 = 0.22;
const CAM_DEMO_PITCH: f32 = 0.30;
const CAM_DEFAULT_DISTANCE: f32 = 5.5;
/// Showcase framing: closer than the old 9 m so the rider fills ~30% of the frame like retail.
pub(crate) const CAM_DEMO_DISTANCE: f32 = 6.5;
const CAM_PITCH: (f32, f32) = (0.0, 1.4);
const CAM_DISTANCE: (f32, f32) = (3.0, 14.0);
const CAM_MIN_ARM: f32 = 1.5;
const CAM_CLEARANCE: f32 = 0.5;
const CAM_ORBIT_RAD_PER_PX: f32 = 0.0035;
const CAM_ZOOM_PER_LINE: f32 = 0.12;
/// Horizontal speed (m/s) range over which the camera trusts travel direction over body heading.
const CAM_TRAVEL_SPEED: (f32, f32) = (2.0, 5.0);
const CAM_IDLE_BEFORE_RECENTER: f32 = 1.2;
const CAM_AUTO_RECENTER_MIN_SPEED: f32 = 1.5;
/// Focus height above the rider's root (bike root sits ~0.75 m up; the skier's is the snow).
const CAM_BIKE_FOCUS: f32 = 0.6;
const CAM_SKI_FOCUS: f32 = 1.0;
/// Critically damped rates (rad/s): focus horizontally / vertically, heading follow, C recenter,
/// terrain lift rising / settling. Focus lag at constant speed is cancelled by `lead`.
const CAM_FOCUS_RATE: (f32, f32) = (9.0, 5.0);
const CAM_YAW_RATE: f32 = 4.0;
const CAM_RECENTER_RATE: f32 = 8.0;
const CAM_LIFT_RATE: (f32, f32) = (8.0, 2.0);
const CAM_LIFT_STEP: f32 = 0.03;
const CAM_MAX_PITCH: f32 = 1.5;
/// Lowest the focus may sit above the terrain under it.
const CAM_FOCUS_FLOOR: f32 = 0.7;
/// Speed (m/s) at which FOV and arm length reach their widest.
const CAM_FAST: f32 = 28.0;
const CAM_FOV: (f32, f32) = (0.907, 1.1);
/// Vertical FOV cap when a portrait window widens the view (`hor_plus`).
const CAM_MAX_VFOV: f32 = 1.75;
const CAM_SPEED_ZOOM: f32 = 0.15;

/// Third-person chase state. `yaw` is the absolute azimuth of the camera arm (the camera sits
/// behind the focus when `yaw == bike.yaw`), `pitch` its manual elevation above the horizon.
#[derive(Resource)]
pub(crate) struct ChaseCamera {
    yaw: f32,
    pitch: f32,
    pub distance: f32,
    /// Collision-limited arm length actually used this frame.
    arm: f32,
    /// Seconds since the last orbit input.
    idle: f32,
    focus: Vec3,
    orbiting: bool,
    recentering: bool,
    snap: bool,
    yaw_vel: f32,
    pitch_vel: f32,
    focus_vel: Vec3,
    /// Previous frame's focus target and its smoothed velocity.
    target: Vec3,
    target_vel: Vec3,
    ragdolled: bool,
    /// Extra elevation that keeps the arm above rising terrain behind the rider.
    lift: f32,
    lift_vel: f32,
    /// Smoothed target speed driving FOV and arm length.
    speed: f32,
}

impl Default for ChaseCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: CAM_DEFAULT_PITCH,
            distance: CAM_DEFAULT_DISTANCE,
            arm: CAM_DEFAULT_DISTANCE,
            idle: f32::MAX,
            focus: Vec3::ZERO,
            orbiting: false,
            recentering: false,
            snap: true,
            yaw_vel: 0.0,
            pitch_vel: 0.0,
            focus_vel: Vec3::ZERO,
            target: Vec3::ZERO,
            target_vel: Vec3::ZERO,
            ragdolled: false,
            lift: 0.0,
            lift_vel: 0.0,
            speed: 0.0,
        }
    }
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// Windows narrower than 4:3 (portrait phones) keep the 4:3 horizontal field of view instead of
/// cropping the sides, by widening the vertical FOV up to `CAM_MAX_VFOV`.
fn hor_plus(fov: f32, aspect: f32) -> f32 {
    let wide = (fov * 0.5).tan() * (4.0 / 3.0) / aspect.max(0.1);
    (2.0 * wide.atan()).clamp(fov, CAM_MAX_VFOV.max(fov))
}

/// Exact critically damped step of an error `x` (value minus target) with rate `v` over `dt`,
/// independent of frame rate.
fn spring(x: &mut f32, v: &mut f32, omega: f32, dt: f32) {
    let k = *v + *x * omega;
    let e = (-omega * dt).exp();
    *x = (*x + k * dt) * e;
    *v = (*v - k * omega * dt) * e;
}

impl ChaseCamera {
    fn orbit(&mut self, pixels: Vec2) {
        if pixels == Vec2::ZERO {
            return;
        }
        self.yaw = wrap_angle(self.yaw - pixels.x * CAM_ORBIT_RAD_PER_PX);
        self.pitch = (self.pitch + pixels.y * CAM_ORBIT_RAD_PER_PX).clamp(CAM_PITCH.0, CAM_PITCH.1);
        self.yaw_vel = 0.0;
        self.pitch_vel = 0.0;
        self.idle = 0.0;
        self.recentering = false;
    }

    fn zoom(&mut self, lines: f32) {
        self.distance = (self.distance * (-lines * CAM_ZOOM_PER_LINE).exp())
            .clamp(CAM_DISTANCE.0, CAM_DISTANCE.1);
    }

    /// Eases yaw along the shortest arc to `heading` and pitch to `rest`.
    fn recenter(&mut self, heading: f32, rest: f32, omega: f32, dt: f32) {
        let mut yaw = wrap_angle(self.yaw - heading);
        spring(&mut yaw, &mut self.yaw_vel, omega, dt);
        self.yaw = wrap_angle(heading + yaw);
        let mut pitch = self.pitch - rest;
        spring(&mut pitch, &mut self.pitch_vel, omega, dt);
        self.pitch = rest + pitch;
    }

    fn centered(&self, heading: f32, rest: f32) -> bool {
        wrap_angle(heading - self.yaw).abs() < 0.01 && (rest - self.pitch).abs() < 0.01
    }
}

/// Unit vector from the focus to the camera.
fn arm_direction(yaw: f32, pitch: f32) -> Vec3 {
    Quat::from_rotation_y(yaw) * Vec3::new(0.0, pitch.sin(), pitch.cos())
}

/// Longest arm length up to `distance` along `dir` that keeps the whole segment above the terrain.
fn arm_reach(focus: Vec3, dir: Vec3, distance: f32) -> f32 {
    const STEPS: u32 = 32;
    let step = distance / STEPS as f32;
    for i in 1..=STEPS {
        let p = focus + dir * (step * i as f32);
        if p.y < terrain_height(p.x, p.z) + CAM_CLEARANCE {
            return (step * (i - 1) as f32).max(CAM_MIN_ARM.min(distance));
        }
    }
    distance
}

/// Smallest extra pitch that lets the arm keep its full `distance` above terrain.
fn arm_lift(focus: Vec3, yaw: f32, pitch: f32, distance: f32) -> f32 {
    let mut lift = 0.0;
    while pitch + lift < CAM_MAX_PITCH
        && arm_reach(focus, arm_direction(yaw, pitch + lift), distance) < distance
    {
        lift += CAM_LIFT_STEP;
    }
    lift
}

#[derive(Component)]
struct RideCamera;
#[derive(Component)]
struct Telemetry;
#[derive(Component)]
struct ControlHelp;

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "RiderRepRust - Bike rewrite".into(),
                resolution: (1280, 720).into(),
                // Browser build: draw into index.html's canvas and follow the page size.
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(120.0))
        .init_resource::<Bike>()
        .init_resource::<Controls>()
        .init_resource::<RideStatus>()
        .init_resource::<ChaseCamera>()
        .init_resource::<scene::SkeletonDebug>()
        .init_resource::<scene::AnimationState>()
        .init_resource::<scene::BikeFrames>()
        .init_resource::<crate::ragdoll::Ragdoll>()
        .init_resource::<Showcase>()
        .add_plugins(crate::ski::SkiPlugin)
        .add_plugins(crate::audio::SoundPlugin)
        .add_systems(Startup, (scene::setup_scene, setup_view))
        .add_systems(
            PreUpdate,
            (read_controls, read_camera_input, fit_ui).after(InputSystems),
        )
        .add_systems(
            FixedUpdate,
            (simulate, scene::record_frame)
                .chain()
                .run_if(ski::bike_active),
        )
        .add_systems(
            Update,
            (
                scene::animate_bike.run_if(ski::bike_active),
                follow_camera,
                update_hud,
            )
                .chain(),
        )
        .run();
}

fn setup_view(mut commands: Commands, bike: Res<Bike>) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 52.0_f32.to_radians(),
            far: 1200.0,
            ..default()
        }),
        Transform::from_translation(bike.position + Vec3::new(0.0, 2.0, 5.5))
            .looking_at(bike.position + Vec3::Y, Vec3::Y),
        Msaa::Off,
        RideCamera,
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(22),
                top: px(22),
                width: px(420),
                max_width: percent(70),
                padding: UiRect::all(px(18)),
                row_gap: px(8),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(HUD_SURFACE),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Movement showcase"),
                TextFont {
                    font_size: 25.0,
                    ..default()
                },
                TextColor(HUD_TEXT),
            ));
            panel.spawn((
                Text::new(""),
                TextFont {
                    font_size: 17.0,
                    ..default()
                },
                TextColor(HUD_ACCENT),
                Telemetry,
            ));
            panel.spawn((
                Text::new("Original Bevy simulation"),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(HUD_SECONDARY),
            ));
        });

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(22), bottom: px(20),
            width: px(400), max_width: percent(45),
            padding: UiRect::all(px(14)),
            ..default()
        },
        BackgroundColor(HUD_SURFACE),
        ControlHelp,
    )).with_children(|panel| {
        panel.spawn((
            Text::new("F6 hill showcase: Superman / flips / tricks / crash\n1 Downhill  2 Road  3 Slopestyle  4 Freeride  5 Ski\nW/S pedal/brake  A/D steer  Shift sprint\nSpace hop  Q wheelie  Z/X manuals\nArrows pitch/roll  E/T spin  Ctrl+arrows flip\nU/I/O hand/feet/bike tricks  B hold  J/L side\nRMB orbit  Wheel zoom  C center\nF1 bones  F2 mesh  R reset  Esc pause  H help"),
            TextFont { font_size: 14.0, ..default() },
            TextColor(HUD_TEXT),
        ));
    });
}

fn axis(keys: &ButtonInput<KeyCode>, positive: KeyCode, negative: KeyCode) -> f32 {
    u8::from(keys.pressed(positive)) as f32 - u8::from(keys.pressed(negative)) as f32
}

fn cycle<T: Copy + PartialEq>(current: T, choices: &[T]) -> T {
    let index = choices
        .iter()
        .position(|&item| item == current)
        .unwrap_or(0);
    choices[(index + 1) % choices.len()]
}

fn read_controls(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<&Transform, With<RideCamera>>,
    mut controls: ResMut<Controls>,
    mut bike: ResMut<Bike>,
    mut status: ResMut<RideStatus>,
    mut chase: ResMut<ChaseCamera>,
    mut skeleton: ResMut<scene::SkeletonDebug>,
    mut animation: ResMut<scene::AnimationState>,
    mut showcase: ResMut<Showcase>,
    mut ragdoll: ResMut<crate::ragdoll::Ragdoll>,
    mut sport: ResMut<ski::Sport>,
) {
    // A browser tab keeps running the showcase without a click; keys still need canvas focus.
    status.focused =
        cfg!(target_arch = "wasm32") || windows.single().is_ok_and(|window| window.focused);
    if !status.focused {
        *controls = Controls::default();
        status.queued_hop = false;
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        status.paused = !status.paused;
    }
    let biking = *sport == ski::Sport::Bike;
    if biking && keys.just_pressed(KeyCode::F6) {
        showcase.enabled = !showcase.enabled;
        status.queued_hop = false;
        if showcase.enabled {
            showcase.index = 0;
            showcase.completed = 0;
            bike.select_discipline(Discipline::Freeride);
            showcase.begin_run(&mut bike);
            animation.reset();
            ragdoll.reset();
            *chase = ChaseCamera::default();
            chase.distance = CAM_DEMO_DISTANCE;
            status.paused = false;
        }
    }
    let selection = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
    ]
    .into_iter()
    .zip(Discipline::ALL.iter().copied())
    .find(|(key, _)| keys.just_pressed(*key));
    if let Some((_, discipline)) = selection {
        *sport = ski::Sport::Bike;
        showcase.enabled = false;
        bike.select_discipline(discipline);
        animation.reset();
        ragdoll.reset();
        *chase = ChaseCamera::default();
        status.queued_hop = false;
    }
    if keys.just_pressed(KeyCode::Digit5) {
        // Ski mode resets the skier and camera itself on entry.
        *sport = ski::Sport::Ski;
        showcase.enabled = false;
        status.queued_hop = false;
    }
    if biking && keys.just_pressed(KeyCode::KeyR) {
        if showcase.enabled {
            showcase.begin_run(&mut bike);
        } else {
            bike.reset();
        }
        animation.reset();
        ragdoll.reset();
        *chase = ChaseCamera::default();
        if showcase.enabled {
            chase.distance = CAM_DEMO_DISTANCE;
        }
        status.queued_hop = false;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        chase.recentering = true;
    }
    if keys.just_pressed(KeyCode::F1) {
        skeleton.enabled = !skeleton.enabled;
    }
    if keys.just_pressed(KeyCode::F2) {
        skeleton.rider_mesh = !skeleton.rider_mesh;
    }
    if keys.just_pressed(KeyCode::KeyH) {
        status.help = !status.help;
    }
    if *sport == ski::Sport::Ski {
        // Riding keys belong to the skier (`ski::read_ski_controls`); the bike stands still.
        *controls = Controls::default();
        status.queued_hop = false;
        return;
    }
    if keys.just_pressed(KeyCode::KeyU) {
        status.hand = cycle(status.hand, &HandTrick::ALL);
    }
    if keys.just_pressed(KeyCode::KeyI) {
        status.leg = cycle(status.leg, &LegTrick::ALL);
    }
    if keys.just_pressed(KeyCode::KeyO) {
        status.bike = cycle(status.bike, &BikeTrick::ALL);
    }
    if keys.just_pressed(KeyCode::KeyJ) {
        status.side = -1.0;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        status.side = 1.0;
    }
    if status.paused {
        status.queued_hop = false;
    } else {
        status.queued_hop |= keys.just_pressed(KeyCode::Space);
    }
    *controls = if status.paused {
        Controls::default()
    } else {
        Controls {
            pedal: u8::from(keys.pressed(KeyCode::KeyW)) as f32,
            brake: u8::from(keys.pressed(KeyCode::KeyS)) as f32,
            steering: axis(&keys, KeyCode::KeyD, KeyCode::KeyA),
            sprint: keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight),
            hop: keys.pressed(KeyCode::Space),
            wheelie: keys.pressed(KeyCode::KeyQ),
            air_pitch: axis(&keys, KeyCode::ArrowDown, KeyCode::ArrowUp),
            air_roll: axis(&keys, KeyCode::ArrowRight, KeyCode::ArrowLeft),
            manual: keys.pressed(KeyCode::KeyZ),
            nose_manual: keys.pressed(KeyCode::KeyX),
            air_yaw: axis(&keys, KeyCode::KeyE, KeyCode::KeyT),
            flip: keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight),
            hand_trick: if keys.pressed(KeyCode::KeyB) {
                status.hand
            } else {
                HandTrick::None
            },
            leg_trick: if keys.pressed(KeyCode::KeyB) {
                status.leg
            } else {
                LegTrick::None
            },
            bike_trick: if keys.pressed(KeyCode::KeyB) {
                status.bike
            } else {
                BikeTrick::None
            },
            trick_side: status.side,
        }
    };
    if keys.just_pressed(KeyCode::F3) {
        let camera = cameras.single().copied().unwrap_or_default();
        info!(
            "BIKE_STATE position={:?} velocity={:?} speed={:.3} yaw={:.3} pitch={:.3} roll={:.3} steering={:.3} wheel={:.3} crank={:.3} suspension={:?} grounded={:?} air_time={:.3} distance={:.3} paused={} skeleton_enabled={} rider_mesh={} cam_position={:?} cam_yaw={:.3} cam_rel_yaw={:.3} cam_pitch={:.3} cam_distance={:.3} cam_arm={:.3} cam_idle={:.2} cam_orbiting={} cam_recentering={} discipline={:?} animation={} phase={} hand={:?} leg={:?} bike_trick={:?} selected_hand={:?} selected_leg={:?} selected_bike={:?} side={} note={} frozen={} hand_release={:?} foot_release={:?} bar_angle={:.3} tail_angle={:.3} pitch_rate={:.3} roll_rate={:.3} landing_weight={:.3} crash={:?} showcase={} demo_case={} demo_stage={:?} demo_completed={}",
            bike.position,
            bike.velocity,
            bike.velocity.length(),
            bike.yaw,
            bike.pitch,
            bike.roll,
            bike.steering,
            bike.wheel_phase,
            bike.crank_phase,
            bike.suspension,
            bike.grounded,
            bike.air_time,
            bike.distance,
            status.paused,
            skeleton.enabled,
            skeleton.rider_mesh,
            camera.translation,
            chase.yaw,
            wrap_angle(chase.yaw - bike.yaw),
            chase.pitch,
            chase.distance,
            chase.arm,
            chase.idle.min(999.0),
            chase.orbiting,
            chase.recentering,
            bike.discipline,
            animation.mode,
            animation.phase,
            animation.hand,
            animation.leg,
            animation.bike,
            status.hand,
            status.leg,
            status.bike,
            status.side,
            animation.note,
            animation.frozen,
            animation.hand_rel,
            animation.foot_rel,
            animation.bars.angle,
            animation.tail.angle,
            bike.pitch_rate,
            bike.roll_rate,
            animation.land,
            bike.crash,
            showcase.enabled,
            showcase.name(),
            showcase.stage,
            showcase.completed
        );
        info!(
            "RAGDOLL_STATE active={} sleeping={} hip={:?} head={:?}",
            ragdoll.body.is_some(),
            ragdoll.body.as_ref().is_some_and(|b| b.sleeping),
            ragdoll
                .body
                .as_ref()
                .map(|b| b.positions[crate::ragdoll::index(scene::P::Hip)]),
            ragdoll
                .body
                .as_ref()
                .map(|b| b.positions[crate::ragdoll::index(scene::P::Head)]),
        );
    }
}

/// Stands a crashed rider up on the spot once the ragdoll sleeps or `GETUP_TIMEOUT` has passed:
/// the bike is placed upright under the ragdoll hip facing the crash heading, at rest, and the
/// drawn rider then blends from the ragdoll pose to the riding pose (`AnimationState::rise`).
fn stand_up(
    bike: &mut Bike,
    animation: &mut scene::AnimationState,
    ragdoll: &mut crate::ragdoll::Ragdoll,
) {
    let (Some(crash), Some(body)) = (bike.crash, &ragdoll.body) else {
        return;
    };
    if !(body.sleeping || crash.elapsed >= GETUP_TIMEOUT) {
        return;
    }
    let from = body.positions;
    let hip = from[crate::ragdoll::index(scene::P::Hip)];
    bike.reset_facing(hip.x, hip.z, bike.yaw);
    animation.reset();
    animation.rise = Some(Rise::new(from));
    ragdoll.reset();
}

fn simulate(
    time: Res<Time<Fixed>>,
    mut controls: ResMut<Controls>,
    mut status: ResMut<RideStatus>,
    mut bike: ResMut<Bike>,
    mut animation: ResMut<scene::AnimationState>,
    mut showcase: ResMut<Showcase>,
    mut chase: ResMut<ChaseCamera>,
    mut ragdoll: ResMut<crate::ragdoll::Ragdoll>,
) {
    animation.frozen = status.paused || !status.focused || bike.crash.is_some();
    if status.paused || !status.focused {
        return;
    }
    let dt = time.delta_secs();
    let mut input = *controls;
    input.hop |= status.queued_hop;
    if showcase.drive(&mut bike, &mut input, dt) {
        animation.reset();
        ragdoll.reset();
        *chase = ChaseCamera::default();
        chase.distance = CAM_DEMO_DISTANCE;
    }
    if let Some(rise) = &mut animation.rise {
        rise.elapsed += dt;
        input = Controls::default();
        if rise.done() {
            animation.rise = None;
        }
    }
    if bike.crash.is_none() {
        animation.update(&AnimationInput::from_bike(&bike, &input), dt);
    }
    bike.collision_pose = scene::collision_pose(&bike, &animation);
    let seed = if bike.crash.is_none() {
        Some(ragdoll.sample(&bike, scene::rider_points(&bike, &animation), dt))
    } else {
        None
    };
    bike.step(&input, dt);
    if bike.crash.is_some() {
        if let Some(seed) = seed {
            ragdoll.activate(seed);
        }
        ragdoll.step(&bike, dt);
        if !showcase.enabled {
            stand_up(&mut bike, &mut animation, &mut ragdoll);
        }
    }
    if bike.crash.is_some() {
        animation.mode = "crashed";
        animation.phase = "ragdoll";
        animation.frozen = true;
    }
    if showcase.enabled {
        *controls = input;
    }
    status.queued_hop = false;
}

fn read_camera_input(
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    touches: Res<Touches>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut chase: ResMut<ChaseCamera>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    // Only react to the focused window, and only start a drag / zoom with the cursor over it.
    let inside = window.focused && window.cursor_position().is_some();
    if inside && mouse.just_pressed(MouseButton::Right) {
        chase.orbiting = true;
    }
    if !window.focused || !mouse.pressed(MouseButton::Right) {
        chase.orbiting = false;
    }
    if chase.orbiting {
        chase.orbit(motion.delta);
    }
    // Touch: one finger orbits, two fingers pinch-zoom.
    let fingers: Vec<_> = touches.iter().collect();
    match fingers.as_slice() {
        [one] => {
            chase.orbiting = true;
            chase.orbit(one.delta());
        }
        [a, b] => {
            let now = a.position().distance(b.position());
            let before = a.previous_position().distance(b.previous_position());
            if now > 1.0 && before > 1.0 {
                chase.zoom((now / before).ln() / CAM_ZOOM_PER_LINE);
            }
        }
        _ => {}
    }
    if inside {
        let lines = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
        };
        chase.zoom(lines);
    }
}

/// Small or portrait windows (phones): shrink the HUD, and hide the key help on turning portrait.
fn fit_ui(
    windows: Query<&Window, (With<PrimaryWindow>, Changed<Window>)>,
    mut ui: ResMut<UiScale>,
    mut status: ResMut<RideStatus>,
    mut was_portrait: Local<bool>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let (w, h) = (window.width(), window.height());
    let scale = (w / 1280.0).min(h / 720.0).clamp(0.6, 1.0);
    if ui.0 != scale {
        ui.0 = scale;
    }
    let portrait = w < h;
    if portrait && !*was_portrait {
        status.help = false;
    }
    *was_portrait = portrait;
}

fn follow_camera(
    time: Res<Time>,
    bike: Res<Bike>,
    status: Res<RideStatus>,
    mut chase: ResMut<ChaseCamera>,
    showcase: Res<Showcase>,
    frames: Res<scene::BikeFrames>,
    fixed: Res<Time<Fixed>>,
    ski: ski::SkiView,
    mut cameras: Query<(&mut Transform, &mut Projection), With<RideCamera>>,
) {
    let Ok((mut camera, mut projection)) = cameras.single_mut() else {
        return;
    };
    let chase = &mut *chase;
    let dt = time.delta_secs().min(0.1);
    let alpha = fixed.overstep_fraction();
    let follow = ski.follow().unwrap_or_else(|| ski::Follow {
        position: frames.root(alpha).unwrap_or(bike.position),
        velocity: bike.velocity,
        yaw: frames.yaw(alpha).unwrap_or(bike.yaw),
        crashed: bike.crash.is_some(),
        ragdoll: bike.crash.and_then(|_| frames.hip(alpha)),
        demo: showcase.enabled,
    });
    let horizontal = Vec3::new(follow.velocity.x, 0.0, follow.velocity.z);
    let look_ahead = if follow.demo {
        Vec3::ZERO
    } else {
        (horizontal * 0.15).clamp_length_max(3.0)
    };
    let height = if ski.active() {
        CAM_SKI_FOCUS
    } else {
        CAM_BIKE_FOCUS
    };
    let target = if let Some(hip) = follow.ragdoll {
        hip + Vec3::Y * 0.35
    } else {
        follow.position + Vec3::Y * height + look_ahead
    };
    // Follow where the rider travels, not how the body spins: flips and spins must not orbit the
    // camera. Below walking pace the body heading takes over.
    let travel = wrap_angle((-horizontal.x).atan2(-horizontal.z) - follow.yaw);
    let trust = ((horizontal.length() - CAM_TRAVEL_SPEED.0)
        / (CAM_TRAVEL_SPEED.1 - CAM_TRAVEL_SPEED.0))
        .clamp(0.0, 1.0);
    let course = follow.yaw + travel * trust;
    // Portrait: look down more so the tall frame shows ground around the rider, not sky.
    let aspect = match &*projection {
        Projection::Perspective(lens) => lens.aspect_ratio,
        _ => 16.0 / 9.0,
    };
    let portrait_tilt = (1.0 - aspect).clamp(0.0, 0.5) * 0.5;
    let (heading, rest_pitch) = if follow.demo {
        (course + 1.0, CAM_DEMO_PITCH + portrait_tilt)
    } else {
        (course, CAM_DEFAULT_PITCH + portrait_tilt)
    };
    let ragdolled = follow.ragdoll.is_some();
    let snap = chase.snap;
    chase.snap = false;
    if snap {
        chase.focus = target;
        chase.focus_vel = Vec3::ZERO;
        chase.target = target;
        chase.target_vel = Vec3::ZERO;
        chase.ragdolled = ragdolled;
        chase.yaw = heading;
        chase.pitch = rest_pitch;
        chase.recentering = false;
    } else {
        // The target's velocity (skipped on the root -> ragdoll jump) lets the focus lead by its
        // steady-state lag, so it settles on a moving rider without trailing it.
        if ragdolled == chase.ragdolled {
            let measured = (target - chase.target) / dt.max(1e-4);
            chase.target_vel += (measured - chase.target_vel) * (1.0 - (-dt / 0.05).exp());
        }
        chase.ragdolled = ragdolled;
        chase.target = target;
        for axis in 0..3 {
            let omega = if axis == 1 {
                CAM_FOCUS_RATE.1
            } else {
                CAM_FOCUS_RATE.0
            };
            let lead = chase.target_vel[axis] * 2.0 / omega;
            let mut error = chase.focus[axis] - target[axis] - lead;
            spring(&mut error, &mut chase.focus_vel[axis], omega, dt);
            chase.focus[axis] = target[axis] + lead + error;
        }
        // Landing dips and ramp-entry lag must never sink the focus into the snow or ramp.
        let floor = terrain_height(chase.focus.x, chase.focus.z) + CAM_FOCUS_FLOOR;
        if chase.focus.y < floor {
            chase.focus.y = floor;
            chase.focus_vel.y = chase.focus_vel.y.max(0.0);
        }
        chase.idle = if chase.orbiting { 0.0 } else { chase.idle + dt };
        let riding = status.focused
            && !status.paused
            && !follow.crashed
            && (follow.demo || horizontal.length() > CAM_AUTO_RECENTER_MIN_SPEED);
        if chase.recentering || riding && chase.idle > CAM_IDLE_BEFORE_RECENTER {
            let omega = if chase.recentering {
                CAM_RECENTER_RATE
            } else {
                CAM_YAW_RATE
            };
            chase.recenter(heading, rest_pitch, omega, dt);
            if chase.recentering && chase.centered(heading, rest_pitch) {
                chase.recentering = false;
            }
        } else {
            chase.yaw_vel = 0.0;
            chase.pitch_vel = 0.0;
        }
        chase.speed += (chase.target_vel.length() - chase.speed) * (1.0 - (-dt / 0.6).exp());
    }
    let fast = (chase.speed / CAM_FAST).clamp(0.0, 1.0);
    if let Projection::Perspective(lens) = &mut *projection {
        lens.fov = hor_plus(
            CAM_FOV.0 + (CAM_FOV.1 - CAM_FOV.0) * fast,
            lens.aspect_ratio,
        );
    }
    let distance = chase.distance * (1.0 + CAM_SPEED_ZOOM * fast);
    // Rising ground behind the rider lifts the camera over it instead of shortening the arm.
    let want = arm_lift(chase.focus, chase.yaw, chase.pitch, distance);
    if snap {
        chase.lift = want;
        chase.lift_vel = 0.0;
    } else {
        let omega = if want > chase.lift {
            CAM_LIFT_RATE.0
        } else {
            CAM_LIFT_RATE.1
        };
        let mut error = chase.lift - want;
        spring(&mut error, &mut chase.lift_vel, omega, dt);
        chase.lift = (want + error).max(0.0);
    }
    let dir = arm_direction(chase.yaw, (chase.pitch + chase.lift).min(CAM_MAX_PITCH));
    let reach = arm_reach(chase.focus, dir, distance);
    // Shrink instantly (never clip a ramp), regrow smoothly.
    chase.arm = if snap || reach < chase.arm {
        reach
    } else {
        chase.arm + (reach - chase.arm) * (1.0 - (-6.0 * dt).exp())
    };
    let mut position = chase.focus + dir * chase.arm;
    position.y = position
        .y
        .max(terrain_height(position.x, position.z) + CAM_CLEARANCE);
    *camera = Transform::from_translation(position).looking_at(chase.focus, Vec3::Y);
}

fn update_hud(
    time: Res<Time>,
    bike: Res<Bike>,
    status: Res<RideStatus>,
    chase: Res<ChaseCamera>,
    skeleton: Res<scene::SkeletonDebug>,
    animation: Res<scene::AnimationState>,
    showcase: Res<Showcase>,
    ragdoll: Res<crate::ragdoll::Ragdoll>,
    ski: ski::SkiView,
    mut text: Query<&mut Text, With<Telemetry>>,
    mut help: Query<&mut Visibility, With<ControlHelp>>,
    mut cooldown: Local<f32>,
) {
    if let Ok(mut visible) = help.single_mut() {
        *visible = if status.help && !ski.active() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    *cooldown -= time.delta_secs();
    if *cooldown > 0.0 {
        return;
    }
    *cooldown = 0.1;
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    if let Some(hud) = ski.hud(
        status.paused,
        status.focused,
        skeleton.enabled,
        skeleton.rider_mesh,
    ) {
        text.0 = format!(
            "{hud}\nCamera {:+.0} deg   {:.1} m{}",
            wrap_angle(chase.yaw - ski.follow().map_or(0.0, |f| f.yaw)).to_degrees(),
            chase.distance,
            if chase.orbiting { "   orbiting" } else { "" }
        );
        return;
    }
    let mode = if status.paused {
        "PAUSED - Esc to resume"
    } else if !status.focused {
        "FOCUS LOST - click to ride"
    } else if bike.crash.is_some() {
        "CRASHED - R to reset"
    } else if !bike.grounded.iter().any(|&contact| contact) {
        "AIRBORNE"
    } else if bike.grounded[1] && !bike.grounded[0] {
        "REAR WHEEL"
    } else {
        "RIDING"
    };
    let mut mode = mode.to_string();
    if bike.crash.is_some() {
        if let Some((hands, feet)) = ragdoll.let_go() {
            if hands == [true; 2] {
                mode += " | let go of the bars";
            }
            if feet == [true; 2] {
                mode += " | feet off the pedals";
            }
        }
    }
    let on_off = |on: bool| if on { "on" } else { "off" };
    text.0 = format!(
        "{}   {:>3.0} km/h   |   {}\n{} {} {}\nAnimation: {} / {}\nTricks U/I/O: {} + {} + {} ({})\nActive: {} + {} + {} {}\nFront {}   Rear {}   |   {:.0} m traveled\nSkeleton {} (F1)   Rider mesh {} (F2)\nCamera {:+.0} deg   {:.1} m{}",
        bike.discipline.label(),
        bike.velocity.x.hypot(bike.velocity.z) * 3.6,
        mode,
        if showcase.enabled {
            showcase.name()
        } else {
            "F6 - hill showcase"
        },
        if showcase.enabled {
            match showcase.stage {
                Stage::Approach => "approach",
                Stage::Flight => "flight",
                Stage::Outcome => "result",
            }
        } else {
            ""
        },
        if let Some(crash) = bike.crash {
            crash.reason.label()
        } else {
            showcase.outcome
        },
        animation.mode,
        animation.phase,
        status.hand.label(),
        status.leg.label(),
        status.bike.label(),
        if status.side < 0.0 { "left" } else { "right" },
        animation.hand.label(),
        animation.leg.label(),
        animation.bike.label(),
        animation.note,
        if bike.grounded[0] { "ground" } else { "air" },
        if bike.grounded[1] { "ground" } else { "air" },
        bike.distance,
        on_off(skeleton.enabled),
        on_off(skeleton.rider_mesh),
        wrap_angle(chase.yaw - bike.yaw).to_degrees(),
        chase.distance,
        if chase.orbiting { "   orbiting" } else { "" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn short_hop_is_kept_until_fixed_update_and_pause_freezes_motion() {
        let mut bike = Bike::default();
        for _ in 0..360 {
            bike.step(&Controls::default(), 1.0 / 120.0);
        }
        assert!(bike.grounded.iter().any(|&contact| contact));
        let mut clock = Time::<Fixed>::from_hz(120.0);
        clock.advance_by(Duration::from_secs_f64(1.0 / 120.0));
        let mut app = App::new();
        app.insert_resource(clock)
            .insert_resource(bike)
            .init_resource::<Controls>()
            .init_resource::<scene::AnimationState>()
            .init_resource::<Showcase>()
            .init_resource::<ChaseCamera>()
            .init_resource::<crate::ragdoll::Ragdoll>()
            .insert_resource(RideStatus {
                queued_hop: true,
                ..default()
            })
            .add_systems(FixedUpdate, simulate);

        app.world_mut().run_schedule(FixedUpdate);
        assert!(app.world().resource::<Bike>().velocity.y > 1.0);
        assert!(!app.world().resource::<RideStatus>().queued_hop);
        app.world_mut().resource_mut::<RideStatus>().paused = true;
        let position = app.world().resource::<Bike>().position;
        let velocity = app.world().resource::<Bike>().velocity;
        for _ in 0..120 {
            app.world_mut().run_schedule(FixedUpdate);
        }
        assert_eq!(app.world().resource::<Bike>().position, position);
        assert_eq!(app.world().resource::<Bike>().velocity, velocity);
    }

    /// The real bike systems, a bike dropped 10 m onto flat ground at (-20, 8) (a hard-impact
    /// crash 28 m from the course start) and the pedal held down throughout.
    fn crashed_app() -> App {
        let mut clock = Time::<Fixed>::from_hz(120.0);
        clock.advance_by(Duration::from_secs_f64(1.0 / 120.0));
        let mut bike = Bike::default();
        bike.reset_at(-20.0, 8.0, 8.0);
        bike.position.y += 10.0;
        let mut app = App::new();
        app.insert_resource(clock)
            .insert_resource(bike)
            .insert_resource(Controls {
                pedal: 1.0,
                ..default()
            })
            .init_resource::<RideStatus>()
            .init_resource::<ChaseCamera>()
            .init_resource::<scene::SkeletonDebug>()
            .init_resource::<scene::AnimationState>()
            .init_resource::<scene::BikeFrames>()
            .init_resource::<crate::ragdoll::Ragdoll>()
            .init_resource::<Showcase>()
            .init_resource::<ski::Sport>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, read_controls)
            .add_systems(FixedUpdate, (simulate, scene::record_frame).chain());
        app.world_mut().spawn((Window::default(), PrimaryWindow));
        app
    }

    /// Largest per-tick world movement of any drawn rider joint, m.
    const MAX_JOINT_STEP: f32 = 0.04;

    #[test]
    fn a_crashed_rider_stands_up_where_it_fell_blends_smoothly_and_rides_on() {
        use crate::ragdoll::{COUNT, Ragdoll, index};
        let mut app = crashed_app();
        let (mut crashed, mut recovered, mut rising) = (0, 0, 0);
        let (mut hip, mut last, mut jump) = (Vec3::ZERO, None::<[Vec3; COUNT]>, 0.0_f32);
        let mut standing = None;
        for _ in 0..1800 {
            app.world_mut().run_schedule(FixedUpdate);
            let world = app.world();
            let bike = world.resource::<Bike>();
            if let Some(body) = &world.resource::<Ragdoll>().body {
                hip = body.positions[index(scene::P::Hip)];
            }
            let joints = world.resource::<scene::BikeFrames>().joints().unwrap();
            let rise = world.resource::<scene::AnimationState>().rise.is_some();
            if bike.crash.is_some() {
                crashed += 1;
            } else if crashed > 0 {
                recovered += 1;
                if rise {
                    rising += 1;
                    let step = (0..COUNT).map(|i| joints[i].distance(last.unwrap()[i]));
                    jump = jump.max(step.fold(0.0, f32::max));
                } else if standing.is_none() {
                    // The blend is over: the controls were ignored until now.
                    assert!(bike.velocity.length() < 0.05, "{:?}", bike.velocity);
                    standing = Some(recovered);
                    assert!((bike.orientation() * Vec3::Y).y > 0.99);
                    assert!(bike.position.xz().distance(Vec2::new(0.0, 8.0)) > 10.0);
                    let rider = joints[index(scene::P::Hip)];
                    assert!(Vec2::new(rider.x - hip.x, rider.z - hip.z).length() < 2.0);
                }
            }
            last = Some(joints);
            if standing.is_some_and(|s| recovered > s + 120) {
                break;
            }
        }
        let bike = app.world().resource::<Bike>();
        assert!(
            crashed > 0 && recovered > 0,
            "never crashed or never recovered"
        );
        assert!(
            (crashed as f32 / 120.0) <= crate::animation::GETUP_TIMEOUT + 0.05,
            "lay {crashed} ticks"
        );
        assert!(
            (rising as f32 / 120.0 - crate::animation::GETUP_BLEND).abs() < 0.05,
            "{rising} blend ticks"
        );
        assert!(jump < MAX_JOINT_STEP, "joint moved {jump} m in one tick");
        assert!(
            bike.crash.is_none() && bike.velocity.length() > 1.0,
            "{bike:?}"
        );
    }

    #[test]
    fn reset_during_the_blend_returns_to_the_start() {
        use crate::ragdoll::Ragdoll;
        let mut app = crashed_app();
        while app
            .world()
            .resource::<scene::AnimationState>()
            .rise
            .is_none()
        {
            app.world_mut().run_schedule(FixedUpdate);
        }
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.world_mut().run_schedule(Update);
        let world = app.world();
        let bike = world.resource::<Bike>();
        assert!(bike.crash.is_none() && bike.position.xz().distance(Vec2::new(0.0, 8.0)) < 0.5);
        assert!(world.resource::<scene::AnimationState>().rise.is_none());
        assert!(world.resource::<Ragdoll>().body.is_none());
    }

    #[test]
    fn portrait_widens_the_view_but_landscape_is_untouched() {
        let fov = CAM_FOV.0;
        assert_eq!(hor_plus(fov, 16.0 / 9.0), fov);
        let tall = hor_plus(fov, 9.0 / 16.0);
        let horizontal = 2.0 * ((tall * 0.5).tan() * 9.0 / 16.0).atan();
        let landscape_4_3 = 2.0 * ((fov * 0.5).tan() * 4.0 / 3.0).atan();
        assert!(
            (horizontal - landscape_4_3).abs() < 1e-4,
            "{horizontal} vs {landscape_4_3}"
        );
        assert_eq!(hor_plus(fov, 9.0 / 21.0), CAM_MAX_VFOV);
    }

    #[test]
    fn orbit_wraps_yaw_and_clamps_pitch_and_zoom() {
        let mut chase = ChaseCamera::default();
        chase.orbit(Vec2::new(-1.5 * PI / CAM_ORBIT_RAD_PER_PX, 1.0e6));
        assert!((-PI..=PI).contains(&chase.yaw));
        assert!(
            (chase.yaw - 1.5 * PI + TAU).abs() < 1e-3,
            "yaw {}",
            chase.yaw
        );
        assert_eq!(chase.pitch, CAM_PITCH.1);
        chase.orbit(Vec2::new(0.0, -1.0e6));
        assert_eq!(chase.pitch, CAM_PITCH.0);
        chase.zoom(1000.0);
        assert_eq!(chase.distance, CAM_DISTANCE.0);
        chase.zoom(-1000.0);
        assert_eq!(chase.distance, CAM_DISTANCE.1);
    }

    #[test]
    fn recenter_takes_the_short_arc_and_converges() {
        let mut chase = ChaseCamera {
            yaw: 3.0,
            pitch: 1.0,
            ..default()
        };
        chase.recenter(-3.0, CAM_DEFAULT_PITCH, CAM_RECENTER_RATE, 1.0 / 60.0);
        assert!(
            chase.yaw > 3.0 || chase.yaw < -3.0,
            "went the long way: {}",
            chase.yaw
        );
        for _ in 0..200 {
            chase.recenter(-3.0, CAM_DEFAULT_PITCH, CAM_RECENTER_RATE, 1.0 / 60.0);
        }
        assert!(chase.centered(-3.0, CAM_DEFAULT_PITCH));
    }

    #[test]
    fn spring_is_critically_damped_and_frame_rate_independent() {
        let run = |steps: u32| {
            let (mut x, mut v) = (1.0, 0.0);
            let mut crossed = false;
            for _ in 0..steps {
                spring(&mut x, &mut v, 6.0, 1.0 / steps as f32);
                crossed |= x < 0.0;
            }
            assert!(!crossed, "overshot with {steps} steps");
            x
        };
        assert!((run(30) - run(240)).abs() < 1e-5);
        assert!(run(60) < 0.02);
    }

    #[test]
    fn camera_rises_over_the_slope_behind_the_rider() {
        // Down the big hill the snow behind the rider climbs steeper than the default arm.
        use crate::bike::HILL_X;
        let z = -20.0;
        let focus = Vec3::new(HILL_X, terrain_height(HILL_X, z) + CAM_SKI_FOCUS, z);
        let (distance, yaw) = (CAM_DEFAULT_DISTANCE, 0.0);
        assert!(arm_reach(focus, arm_direction(yaw, CAM_DEFAULT_PITCH), distance) < distance);
        let lift = arm_lift(focus, yaw, CAM_DEFAULT_PITCH, distance);
        assert!(lift > 0.1);
        let dir = arm_direction(yaw, CAM_DEFAULT_PITCH + lift);
        assert_eq!(arm_reach(focus, dir, distance), distance);
    }

    #[test]
    fn arm_never_cuts_into_ramps() {
        let distance = 8.0;
        let mut shortened = false;
        for z in (-450..=-250).step_by(5).map(|z| z as f32 / 10.0) {
            let focus = Vec3::new(0.0, terrain_height(0.0, z) + 1.0, z);
            for yaw in (0..16).map(|i| i as f32 * TAU / 16.0) {
                for pitch in [0.0, CAM_DEFAULT_PITCH, 1.0] {
                    let dir = arm_direction(yaw, pitch);
                    let reach = arm_reach(focus, dir, distance);
                    shortened |= reach < distance - 0.1;
                    if reach <= CAM_MIN_ARM {
                        continue;
                    }
                    for i in 1..=200 {
                        let p = focus + dir * (reach * i as f32 / 200.0);
                        assert!(
                            p.y >= terrain_height(p.x, p.z) + CAM_CLEARANCE - 0.1,
                            "arm clips terrain at z={z} yaw={yaw} pitch={pitch}"
                        );
                    }
                }
            }
        }
        assert!(shortened, "ramps never shortened the arm");
    }
}
