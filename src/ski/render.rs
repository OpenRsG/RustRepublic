//! Skier rendering: world-space body, ski and pole meshes driven by the solved `SkierPose`, the
//! debug skeleton overlay (same colours as the bike) and gizmo snow spray.
//!
//! Every mesh is a unit cube / cylinder / sphere stretched each frame; nothing is parented.

use bevy::prelude::*;

use super::pose::{BONES, GEAR, J, SKI_THICKNESS, SKI_WIDTH, Side, SkierPose, side};

// ---------------------------------------------------------------------------------------------
// Authored look
// ---------------------------------------------------------------------------------------------

const JACKET: [f32; 3] = [1.0, 0.42, 0.05];
/// Crashed skier: jacket turns red (same constants as the bike rider).
const CRASH_JACKET: [f32; 3] = [0.85, 0.04, 0.04];
const CRASH_BONE: [f32; 3] = [1.0, 0.1, 0.1];
const CRASH_BONE_ISOLATED: [f32; 3] = [0.75, 0.0, 0.0];
/// Spine lime, left cyan, right magenta; darker hues when only the rig is shown on the white floor.
const SKELETON: [[f32; 3]; 3] = [[0.55, 1.0, 0.1], [0.1, 0.85, 1.0], [1.0, 0.2, 0.75]];
const SKELETON_ISOLATED: [[f32; 3]; 3] =
    [[0.06, 0.34, 0.12], [0.05, 0.27, 0.70], [0.65, 0.06, 0.34]];
const GEAR_LINE: [f32; 3] = [0.25, 0.25, 0.28];
const JOINT_RADIUS: f32 = 0.03;
const HEAD_RADIUS: f32 = 0.09;
const SPRAY: [f32; 3] = [0.40, 0.68, 1.0];

/// Length of the upturned shovel segment, measured back from the ski tip.
const SHOVEL_LEN: f32 = 0.15;
/// Binding plate: width, length along the ski, height above the top surface.
const BINDING: Vec3 = Vec3::new(0.075, 0.30, 0.04);
const BASKET_DIAMETER: f32 = 0.12;
const BASKET_BACK: f32 = 0.08;
/// Snow spray needs real speed; below this the skis only shuffle.
const SPRAY_MIN_SPEED: f32 = 1.5;

// ---------------------------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------------------------

/// One animated mesh; `part_tf` turns it and the pose into a world transform.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) enum SkierPart {
    /// Unit mesh (extent 1, long axis Y) stretched from `a` to `b`; round meshes ignore roll.
    Link {
        a: J,
        b: J,
        dx: f32,
        dz: f32,
    },
    /// Unit cube from `a` to `b` whose Z axis follows `pose.facing` (or the ski's top normal when
    /// `ski` is set), raised by `lift` along that axis. Torso, hips and boots.
    Slab {
        a: J,
        b: J,
        dx: f32,
        dz: f32,
        lift: f32,
        ski: Option<bool>,
    },
    /// `len` long tube from `from` towards `toward`.
    Stub {
        from: J,
        toward: J,
        len: f32,
        dx: f32,
        dz: f32,
    },
    /// Sphere of the given radius centred on a joint.
    Ball(J, f32),
    Helmet,
    Goggles,
    /// Flat part of a ski, tail to the start of the shovel.
    Ski {
        left: bool,
    },
    /// Upturned front segment, shovel start to tip.
    Shovel {
        left: bool,
    },
    Binding {
        left: bool,
    },
    /// Pole basket disc near the tip.
    Basket {
        top: J,
        tip: J,
    },
}

impl SkierPart {
    /// Joints this part reads from the pose.
    #[cfg(test)]
    fn joints(self) -> Vec<J> {
        use SkierPart::*;
        match self {
            Link { a, b, .. } | Slab { a, b, .. } => vec![a, b],
            Stub { from, toward, .. } => vec![from, toward],
            Ball(j, _) => vec![j],
            Helmet | Goggles => vec![J::Head, J::Neck],
            Ski { left } | Shovel { left } => ski_joints(left).to_vec(),
            Binding { left } => ski_joints(left).to_vec(),
            Basket { top, tip } => vec![top, tip],
        }
    }
}

/// Body meshes: hidden by the F2 rider-mesh switch. Skis and poles are gear and stay visible.
#[derive(Component)]
pub(crate) struct SkierBody;

/// Handle of the jacket material (shared by torso and sleeves) for the crash tint.
#[derive(Resource)]
pub(crate) struct SkierPaint {
    jacket: Handle<StandardMaterial>,
}

fn ski_joints(left: bool) -> [J; 3] {
    if left {
        [J::TailL, J::BindL, J::TipL]
    } else {
        [J::TailR, J::BindR, J::TipR]
    }
}

fn ski_index(left: bool) -> usize {
    if left { 0 } else { 1 }
}

// ---------------------------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------------------------

/// Unit mesh stretched along Y from `a` to `b` (exact endpoints at local y = ±0.5).
fn tube_tf(a: Vec3, b: Vec3, dx: f32, dz: f32) -> Transform {
    let d = b - a;
    Transform {
        translation: (a + b) * 0.5,
        rotation: Quat::from_rotation_arc(Vec3::Y, d.try_normalize().unwrap_or(Vec3::Y)),
        scale: Vec3::new(dx, d.length().max(1e-5), dz),
    }
}

/// Rotation with local Y along `y` and local Z as close to `z_hint` as orthogonality allows.
fn frame(y: Vec3, z_hint: Vec3) -> Quat {
    let y = y.normalize_or(Vec3::Y);
    let z = (z_hint - y * z_hint.dot(y))
        .try_normalize()
        .unwrap_or_else(|| y.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(y.cross(z), y, z))
}

/// Like `tube_tf`, but the cross-section is oriented: `dx` along `y × hint`, `dz` along `hint`.
fn oriented_tf(a: Vec3, b: Vec3, hint: Vec3, dx: f32, dz: f32) -> Transform {
    let d = b - a;
    Transform {
        translation: (a + b) * 0.5,
        rotation: frame(d, hint),
        scale: Vec3::new(dx, d.length().max(1e-5), dz),
    }
}

/// Start of the shovel: on the flat plane through the binding, `SHOVEL_LEN` before the tip.
fn shovel_start(tail: Vec3, bind: Vec3, tip: Vec3, up: Vec3) -> Vec3 {
    let along = (tip - tail)
        .reject_from(up)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    bind + along * ((tip - bind).dot(along) - SHOVEL_LEN).max(0.0)
}

fn part_tf(part: SkierPart, pose: &SkierPose) -> Transform {
    use SkierPart::*;
    let head_up = (pose[J::Head] - pose[J::Neck]).normalize_or(Vec3::Y);
    match part {
        Link { a, b, dx, dz } => tube_tf(pose[a], pose[b], dx, dz),
        Slab {
            a,
            b,
            dx,
            dz,
            lift,
            ski,
        } => {
            let hint = ski.map_or(pose.facing, |left| pose.ski_up[ski_index(left)]);
            let mut tf = oriented_tf(pose[a], pose[b], hint, dx, dz);
            tf.translation += tf.rotation * Vec3::Z * lift;
            tf
        }
        Stub {
            from,
            toward,
            len,
            dx,
            dz,
        } => {
            let a = pose[from];
            let d = (pose[toward] - a).normalize_or(Vec3::Y);
            tube_tf(a, a + d * len, dx, dz)
        }
        Ball(j, r) => Transform::from_translation(pose[j]).with_scale(Vec3::splat(2.0 * r)),
        Helmet => {
            let rotation = frame(head_up, -pose.gaze);
            Transform {
                translation: pose[J::Head] + head_up * 0.03 + rotation * Vec3::Z * 0.01,
                rotation,
                scale: Vec3::new(0.20, 0.18, 0.215),
            }
        }
        Goggles => {
            let rotation = frame(head_up, -pose.gaze);
            Transform {
                translation: pose[J::Head] + head_up * 0.015 + rotation * Vec3::NEG_Z * 0.075,
                rotation,
                scale: Vec3::new(0.175, 0.06, 0.05),
            }
        }
        Ski { left } | Shovel { left } | Binding { left } => {
            let [tail, bind, tip] = ski_joints(left).map(|j| pose[j]);
            let up = pose.ski_up[ski_index(left)];
            // Joints sit on the top surface; the box is centred half a thickness below.
            let mid = -up * (SKI_THICKNESS * 0.5);
            let start = shovel_start(tail, bind, tip, up);
            match part {
                Ski { .. } => oriented_tf(tail + mid, start + mid, up, SKI_WIDTH, SKI_THICKNESS),
                Shovel { .. } => oriented_tf(start + mid, tip + mid, up, SKI_WIDTH, SKI_THICKNESS),
                _ => Transform {
                    translation: bind + up * (BINDING.z * 0.5),
                    rotation: frame((tip - tail).reject_from(up), up),
                    scale: BINDING,
                },
            }
        }
        Basket { top, tip } => {
            let dir = (pose[tip] - pose[top]).normalize_or(Vec3::NEG_Y);
            Transform {
                translation: pose[tip] - dir * BASKET_BACK,
                rotation: Quat::from_rotation_arc(Vec3::Y, dir),
                scale: Vec3::new(BASKET_DIAMETER, 0.01, BASKET_DIAMETER),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------

fn mat(
    m: &mut Assets<StandardMaterial>,
    c: [f32; 3],
    rough: f32,
    metal: f32,
) -> Handle<StandardMaterial> {
    m.add(StandardMaterial {
        base_color: Color::srgb(c[0], c[1], c[2]),
        perceptual_roughness: rough,
        metallic: metal,
        ..default()
    })
}

pub fn spawn_skier(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use SkierPart::*;
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let cyl = meshes.add(Cylinder::new(0.5, 1.0).mesh().resolution(14).build());
    let ball = meshes.add(Sphere::new(0.5).mesh().uv(18, 12));

    let jacket = mat(&mut materials, JACKET, 0.6, 0.0);
    let pants = mat(&mut materials, [0.08, 0.10, 0.20], 0.7, 0.0);
    let boot = mat(&mut materials, [0.10, 0.10, 0.12], 0.4, 0.1);
    let glove = mat(&mut materials, [0.06, 0.06, 0.07], 0.8, 0.0);
    let skin = mat(&mut materials, [0.93, 0.70, 0.58], 0.8, 0.0);
    let helmet = mat(&mut materials, [0.0, 0.62, 0.68], 0.3, 0.1);
    let lens = mat(&mut materials, [0.05, 0.12, 0.20], 0.15, 0.6);
    let ski = mat(&mut materials, [0.0, 0.70, 0.78], 0.3, 0.2);
    let black = mat(&mut materials, [0.04, 0.04, 0.05], 0.5, 0.3);
    let pole = mat(&mut materials, [0.20, 0.20, 0.23], 0.35, 0.6);
    let basket = mat(&mut materials, [1.0, 0.5, 0.1], 0.6, 0.0);
    commands.insert_resource(SkierPaint {
        jacket: jacket.clone(),
    });

    let mut put =
        |mesh: &Handle<Mesh>, m: &Handle<StandardMaterial>, part: SkierPart, body: bool| {
            let mut e = commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(m.clone()),
                Transform::default(),
                Visibility::Hidden,
                part,
            ));
            if body {
                e.insert(SkierBody);
            }
        };
    let link = |a, b, dx, dz| Link { a, b, dx, dz };

    // Pelvis, torso and neck.
    put(
        &cube,
        &pants,
        Slab {
            a: J::Pelvis,
            b: J::Waist,
            dx: 0.30,
            dz: 0.20,
            lift: 0.0,
            ski: None,
        },
        true,
    );
    put(
        &cube,
        &jacket,
        Slab {
            a: J::Waist,
            b: J::Chest,
            dx: 0.34,
            dz: 0.21,
            lift: 0.0,
            ski: None,
        },
        true,
    );
    put(
        &cube,
        &jacket,
        Slab {
            a: J::Chest,
            b: J::Neck,
            dx: 0.40,
            dz: 0.22,
            lift: 0.0,
            ski: None,
        },
        true,
    );
    put(&cyl, &skin, link(J::Neck, J::Head, 0.08, 0.08), true);
    put(&ball, &skin, Ball(J::Head, 0.0825), true);
    put(&ball, &helmet, Helmet, true);
    put(&cube, &lens, Goggles, true);

    for left in [true, false] {
        let (hip, knee, ankle, heel, toe, shoulder, elbow, wrist, hand, top, tip) = if left {
            (
                J::HipL,
                J::KneeL,
                J::AnkleL,
                J::HeelL,
                J::ToeL,
                J::ShoulderL,
                J::ElbowL,
                J::WristL,
                J::HandL,
                J::PoleTopL,
                J::PoleTipL,
            )
        } else {
            (
                J::HipR,
                J::KneeR,
                J::AnkleR,
                J::HeelR,
                J::ToeR,
                J::ShoulderR,
                J::ElbowR,
                J::WristR,
                J::HandR,
                J::PoleTopR,
                J::PoleTipR,
            )
        };
        // Legs and boots.
        put(&cyl, &pants, link(J::Pelvis, hip, 0.14, 0.14), true);
        put(&cyl, &pants, link(hip, knee, 0.15, 0.15), true);
        put(&ball, &pants, Ball(knee, 0.075), true);
        put(&cyl, &pants, link(knee, ankle, 0.115, 0.115), true);
        put(
            &cyl,
            &boot,
            Stub {
                from: ankle,
                toward: knee,
                len: 0.20,
                dx: 0.14,
                dz: 0.14,
            },
            true,
        );
        put(
            &cube,
            &boot,
            Slab {
                a: ankle,
                b: toe,
                dx: 0.115,
                dz: 0.10,
                lift: 0.05,
                ski: Some(left),
            },
            true,
        );
        put(
            &cube,
            &boot,
            Slab {
                a: heel,
                b: ankle,
                dx: 0.115,
                dz: 0.10,
                lift: 0.05,
                ski: Some(left),
            },
            true,
        );
        // Arms and gloves.
        put(&cyl, &jacket, link(J::Chest, shoulder, 0.10, 0.10), true);
        put(&ball, &jacket, Ball(shoulder, 0.06), true);
        put(&cyl, &jacket, link(shoulder, elbow, 0.095, 0.095), true);
        put(&ball, &jacket, Ball(elbow, 0.05), true);
        put(&cyl, &jacket, link(elbow, wrist, 0.08, 0.08), true);
        put(&cyl, &glove, link(wrist, hand, 0.075, 0.075), true);
        put(&ball, &glove, Ball(hand, 0.055), true);
        // Skis, bindings and poles.
        put(&cube, &ski, Ski { left }, false);
        put(&cube, &ski, Shovel { left }, false);
        put(&cube, &black, Binding { left }, false);
        put(&cyl, &pole, link(top, tip, 0.014, 0.014), false);
        put(
            &cyl,
            &black,
            Stub {
                from: top,
                toward: tip,
                len: 0.14,
                dx: 0.032,
                dz: 0.032,
            },
            false,
        );
        put(&cyl, &basket, Basket { top, tip }, false);
    }
}

// ---------------------------------------------------------------------------------------------
// Per-frame animation
// ---------------------------------------------------------------------------------------------

fn draw_skeleton(g: &mut Gizmos, pose: &SkierPose, over_mesh: bool, crashed: bool) {
    let tint = |group: usize| {
        let [r, gr, b] = match (crashed, over_mesh) {
            (true, true) => CRASH_BONE,
            (true, false) => CRASH_BONE_ISOLATED,
            (false, true) => SKELETON[group],
            (false, false) => SKELETON_ISOLATED[group],
        };
        Color::srgb(r, gr, b)
    };
    g.sphere(pose[J::Pelvis], JOINT_RADIUS * 1.5, tint(0))
        .resolution(12);
    for &(a, b) in BONES {
        let color = tint(match side(b) {
            Side::Centre => 0,
            Side::Left => 1,
            Side::Right => 2,
        });
        g.line(pose[a], pose[b], color);
        let radius = if b == J::Head {
            HEAD_RADIUS
        } else {
            JOINT_RADIUS
        };
        g.sphere(pose[b], radius, color).resolution(12);
    }
    let [r, gr, b] = GEAR_LINE;
    for &(a, b2) in GEAR {
        g.line(pose[a], pose[b2], Color::srgb(r, gr, b));
    }
}

/// Deterministic pseudo-random value in 0..1.
fn hash(k: u32, seed: f32) -> f32 {
    ((k as f32 * 12.9898 + seed * 78.233).sin() * 43_758.547)
        .fract()
        .abs()
}

/// Offset from the emitter of snow particle `k` at normalised age `u`: thrown along `dir`
/// (unit, horizontal-ish), arcing under gravity over a 0.6 s life.
fn spray_offset(dir: Vec3, k: u32, seed: f32, u: f32) -> Vec3 {
    const LIFE: f32 = 0.6;
    let side = dir.cross(Vec3::Y).normalize_or_zero();
    let v = dir * (1.5 + 2.5 * hash(k + 97, seed))
        + Vec3::Y * (1.2 + 1.5 * hash(k + 31, seed))
        + side * (hash(k + 53, seed) - 0.5) * 1.5;
    let t = u * LIFE;
    v * t + Vec3::NEG_Y * (0.5 * 9.81 * t * t)
}

fn emit_spray(g: &mut Gizmos, origin: Vec3, dir: Vec3, amount: f32, time: f32, seed: f32) {
    let count = 3 + (amount.clamp(0.0, 1.0) * 7.0) as u32;
    let color = Color::srgb(SPRAY[0], SPRAY[1], SPRAY[2]);
    for k in 0..count {
        let u = (time * 1.7 + k as f32 / count as f32 + hash(k, seed) * 0.2).fract();
        let p = origin + spray_offset(dir, k, seed, u);
        g.line(
            p,
            origin + spray_offset(dir, k, seed, (u + 0.05).min(1.0)),
            color,
        );
        g.sphere(p, 0.012 + 0.02 * (1.0 - u), color).resolution(6);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn animate_skier(
    sport: Res<super::Sport>,
    skier: Res<super::physics::Skier>,
    anim: Res<super::anim::SkiAnimation>,
    ragdoll: Res<super::ragdoll::SkiRagdoll>,
    debug: Res<crate::scene::SkeletonDebug>,
    paint: Res<SkierPaint>,
    time: Res<Time>,
    fixed: Res<Time<Fixed>>,
    frames: Res<super::SkiFrames>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut crash_tint: Local<Option<bool>>,
    mut parts: Query<(&SkierPart, &mut Transform, &mut Visibility, Has<SkierBody>)>,
    mut gizmos: Gizmos,
) {
    if *sport != super::Sport::Ski {
        for (_, _, mut v, _) in &mut parts {
            v.set_if_neq(Visibility::Hidden);
        }
        return;
    }
    let pose = frames
        .blend(fixed.overstep_fraction())
        .map_or_else(|| super::current_pose(&skier, &anim, &ragdoll), |(p, _)| p);
    let crashed = skier.crash.is_some();
    if *crash_tint != Some(crashed) {
        *crash_tint = Some(crashed);
        let [r, g, b] = if crashed { CRASH_JACKET } else { JACKET };
        if let Some(m) = materials.get_mut(&paint.jacket) {
            m.base_color = Color::srgb(r, g, b);
        }
    }
    for (part, mut tf, mut v, body) in &mut parts {
        *tf = part_tf(*part, &pose);
        v.set_if_neq(if body && !debug.rider_mesh {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        });
    }
    if debug.enabled {
        draw_skeleton(&mut gizmos, &pose, debug.rider_mesh, crashed);
    }

    // Snow spray: sideways from the tails in a skid, outward from the wedge when snowploughing.
    if !skier.grounded || crashed || ragdoll.active() || skier.speed() < SPRAY_MIN_SPEED {
        return;
    }
    let t = time.elapsed_secs();
    let right = skier.rotation * Vec3::X;
    let forward = skier.forward();
    if skier.skid > 0.2 {
        let slide = if skier.velocity.dot(right) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let dir = (right * slide + forward * 0.4).normalize();
        for (i, tail) in [J::TailL, J::TailR].into_iter().enumerate() {
            emit_spray(&mut gizmos, pose[tail], dir, skier.skid, t, i as f32);
        }
    }
    if skier.plow > 0.2 {
        for (i, (bind, tip, out)) in [(J::BindL, J::TipL, -1.0), (J::BindR, J::TipR, 1.0)]
            .into_iter()
            .enumerate()
        {
            let dir = (right * out - forward * 0.5).normalize();
            let origin = pose[bind].lerp(pose[tip], 0.5);
            emit_spray(&mut gizmos, origin, dir, skier.plow, t, 2.0 + i as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pose::JOINTS;
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    fn spawned_parts() -> Vec<SkierPart> {
        let mut world = World::new();
        world.init_resource::<Assets<Mesh>>();
        world.init_resource::<Assets<StandardMaterial>>();
        world.run_system_once(spawn_skier).unwrap();
        let parts: Vec<_> = world.query::<&SkierPart>().iter(&world).copied().collect();
        // Everything starts hidden; only body meshes carry the F2 marker.
        let mut q = world.query::<(&Visibility, Has<SkierBody>, &SkierPart)>();
        assert!(q.iter(&world).all(|(v, _, _)| *v == Visibility::Hidden));
        for (_, body, part) in q.iter(&world) {
            // Equipment parts reference only ski/pole joints (all declared after the body).
            let gear = part
                .joints()
                .into_iter()
                .all(|j| j as usize >= J::BindL as usize);
            assert_eq!(body, !gear, "{part:?}");
        }
        assert!(world.contains_resource::<SkierPaint>());
        parts
    }

    /// A pose with distinct, well separated points and a tilted-ski frame.
    fn pose() -> SkierPose {
        let mut p = [Vec3::ZERO; JOINTS];
        for (i, v) in p.iter_mut().enumerate() {
            let t = i as f32;
            *v = Vec3::new(
                (t * 0.7).sin(),
                1.0 + (t * 1.3).cos(),
                (t * 0.9).sin() * 2.0,
            );
        }
        // Skis: tail -> bind -> tip along -Z, tip raised, normals up.
        for (left, x) in [(true, -0.12), (false, 0.12)] {
            let [tail, bind, tip] = ski_joints(left);
            p[tail as usize] = Vec3::new(x, 0.0, 0.8);
            p[bind as usize] = Vec3::new(x, 0.0, 0.0);
            p[tip as usize] = Vec3::new(x, 0.07, -0.95);
        }
        let mut pose = SkierPose::from_points(p);
        pose.facing = Vec3::NEG_Z;
        pose.ski_up = [Vec3::Y; 2];
        pose
    }

    #[test]
    fn every_joint_is_drawn_by_some_part() {
        let mut seen = [false; JOINTS];
        for part in spawned_parts() {
            for j in part.joints() {
                assert!((j as usize) < JOINTS, "{part:?} reads the sentinel joint");
                seen[j as usize] = true;
            }
        }
        let all = [
            J::Pelvis,
            J::HipL,
            J::HipR,
            J::KneeL,
            J::KneeR,
            J::AnkleL,
            J::AnkleR,
            J::HeelL,
            J::HeelR,
            J::ToeL,
            J::ToeR,
            J::Waist,
            J::Chest,
            J::Neck,
            J::Head,
            J::ShoulderL,
            J::ShoulderR,
            J::ElbowL,
            J::ElbowR,
            J::WristL,
            J::WristR,
            J::HandL,
            J::HandR,
            J::BindL,
            J::BindR,
            J::TipL,
            J::TipR,
            J::TailL,
            J::TailR,
            J::PoleTopL,
            J::PoleTopR,
            J::PoleTipL,
            J::PoleTipR,
        ];
        assert_eq!(all.len(), JOINTS);
        for j in all {
            assert!(seen[j as usize], "{j:?} has no mesh");
        }
    }

    #[test]
    fn tube_maps_unit_segment_onto_endpoints() {
        let (a, b) = (Vec3::new(1.0, 2.0, -3.0), Vec3::new(-0.5, 4.0, 0.25));
        let tf = tube_tf(a, b, 0.3, 0.2);
        assert!(tf.transform_point(Vec3::new(0.0, -0.5, 0.0)).distance(a) < 1e-5);
        assert!(tf.transform_point(Vec3::new(0.0, 0.5, 0.0)).distance(b) < 1e-5);
        let tf = oriented_tf(a, b, Vec3::Y, 0.3, 0.2);
        assert!(tf.transform_point(Vec3::new(0.0, -0.5, 0.0)).distance(a) < 1e-5);
        assert!(tf.transform_point(Vec3::new(0.0, 0.5, 0.0)).distance(b) < 1e-5);
        // The cross-section follows the hint: local Z is the closest axis to world Y.
        let z = tf.rotation * Vec3::Z;
        assert!(z.dot(Vec3::Y) > 0.5 && z.dot((b - a).normalize()).abs() < 1e-5);
    }

    #[test]
    fn links_span_their_joints_and_skis_keep_flat_then_rise() {
        let pose = pose();
        for part in spawned_parts() {
            let tf = part_tf(part, &pose);
            assert!(
                tf.translation.is_finite() && tf.rotation.is_finite() && tf.scale.is_finite(),
                "{part:?}"
            );
            if let SkierPart::Link { a, b, .. } = part {
                assert!(
                    tf.transform_point(Vec3::new(0.0, -0.5, 0.0))
                        .distance(pose[a])
                        < 1e-4
                );
                assert!(
                    tf.transform_point(Vec3::new(0.0, 0.5, 0.0))
                        .distance(pose[b])
                        < 1e-4
                );
            }
        }
        // Ski + shovel run continuously from tail to the (raised) tip, shovel tilted upward.
        let tail = Vec3::new(-0.12, 0.0, 0.8) - Vec3::Y * 0.0125;
        let ski = part_tf(SkierPart::Ski { left: true }, &pose);
        let shovel = part_tf(SkierPart::Shovel { left: true }, &pose);
        let end = |tf: Transform, s: f32| tf.transform_point(Vec3::new(0.0, s, 0.0));
        assert!(end(ski, -0.5).distance(tail) < 1e-4);
        assert!(end(ski, 0.5).distance(end(shovel, -0.5)) < 1e-4);
        assert!(end(shovel, 0.5).distance(pose[J::TipL] - Vec3::Y * 0.0125) < 1e-4);
        assert!((end(shovel, 0.5) - end(shovel, -0.5)).y > 0.0);
        assert!(((end(ski, 0.5) - end(ski, -0.5)).y).abs() < 1e-4);
        let len = (end(shovel, 0.5) - end(shovel, -0.5)).length();
        assert!(len > SHOVEL_LEN - 1e-3 && len < SHOVEL_LEN + 0.1);
    }

    #[test]
    fn spray_is_deterministic_and_bounded() {
        let dir = Vec3::X;
        for k in 0..10 {
            for step in 0..=10 {
                let u = step as f32 / 10.0;
                let o = spray_offset(dir, k, 1.0, u);
                assert_eq!(o, spray_offset(dir, k, 1.0, u));
                assert!(o.is_finite() && o.length() < 3.0, "{o:?}");
            }
            assert_eq!(spray_offset(dir, k, 1.0, 0.0), Vec3::ZERO);
        }
    }
}
