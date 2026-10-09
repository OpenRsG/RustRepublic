//! Detached rider: fifteen rigid capsule segments (`crate::rigid`) pinned at the skeleton's joints
//! with anatomical swing/twist limits, colliding with each other, the terrain and the crashed bike.
//! The bike is a body of the same solver, so hands and feet holding it, and every contact between
//! rider and bike, push the bike as hard as the bike pushes the rider.
use bevy::prelude::*;

use crate::bike::{Bike, CRASH_MASS, CollisionPose, GYRATION_SQ, TIRE_TUBE, WHEEL_RADIUS};
use crate::rigid::*;
use crate::scene::P;

pub(crate) const COUNT: usize = P::N as usize - P::Hip as usize;
pub(crate) const fn index(p: P) -> usize {
    p as usize - P::Hip as usize
}

/// The rider's rig points in `index` order.
const RIDER: [P; COUNT] = [
    P::Hip,
    P::HipL,
    P::HipR,
    P::KneeL,
    P::KneeR,
    P::AnkleL,
    P::AnkleR,
    P::HeelL,
    P::HeelR,
    P::ToeL,
    P::ToeR,
    P::Waist,
    P::Shoulder,
    P::ShoulderL,
    P::ShoulderR,
    P::ElbowL,
    P::ElbowR,
    P::WristL,
    P::WristR,
    P::Neck,
    P::Head,
    P::HandL,
    P::HandR,
];

/// Pull a hand can hold against before the fingers open, N: a sustained pull of a few hundred N
/// on a bar the fingers hook around.
const HAND_GRIP_FORCE: f32 = 600.0;
/// Pull a foot can hold against on a flat pedal (pins and friction only), N.
const FOOT_GRIP_FORCE: f32 = 200.0;
/// Wrists and ankles with the strength each can hold.
const GRIPS: [(P, f32); 4] = [
    (P::WristL, HAND_GRIP_FORCE),
    (P::WristR, HAND_GRIP_FORCE),
    (P::AnkleL, FOOT_GRIP_FORCE),
    (P::AnkleR, FOOT_GRIP_FORCE),
];
/// Limb speed relative to the bike beyond which the animation's pose difference is a glitch, m/s.
const MAX_LIMB_SPEED: f32 = 12.0;
/// The same for a segment's spin, rad/s.
const MAX_LIMB_SPIN: f32 = 25.0;
/// Lumbar and thoracic spine as one joint (the rig has one waist point): forward bend is negative
/// flex, a right side-bend negative side, a right turn negative twist.
const SPINE: Limit = ball(
    [-85.0, 40.0],
    [-45.0, 45.0],
    [-47.0, 47.0],
    [600.0, 600.0, 300.0],
);
/// Ankle in a riding shoe: free flex both ways, some side play; the shoe breaks it like a boot does.
const SHOE_ANKLE: Limit = hinged(
    [-40.0, 40.0],
    [-20.0, 20.0],
    [-25.0, 25.0],
    [500.0, 375.0, 190.0],
);

/// Segments of the rider's body followed by the bike. Each has an anatomical frame: X right, Y up,
/// Z back when standing with the arms hanging; Y runs along the segment.
#[derive(Clone, Copy)]
enum S {
    Pelvis,
    Chest,
    Head,
    UpperArmL,
    UpperArmR,
    ForearmL,
    ForearmR,
    HandL,
    HandR,
    ThighL,
    ThighR,
    ShinL,
    ShinR,
    FootL,
    FootR,
    Bike,
}

impl From<S> for usize {
    fn from(s: S) -> usize {
        s as usize
    }
}

/// Segments of the rider alone.
const LIMBS: usize = S::Bike as usize;
const BIKE: usize = S::Bike as usize;

/// The segment a rig point is rigidly fixed to.
fn owner(p: P) -> S {
    match p {
        P::Hip | P::HipL | P::HipR => S::Pelvis,
        P::Waist | P::Shoulder => S::Chest,
        P::Neck | P::Head => S::Head,
        P::ShoulderL => S::UpperArmL,
        P::ShoulderR => S::UpperArmR,
        P::ElbowL => S::ForearmL,
        P::ElbowR => S::ForearmR,
        P::WristL | P::HandL => S::HandL,
        P::WristR | P::HandR => S::HandR,
        P::KneeL => S::ShinL,
        P::KneeR => S::ShinR,
        P::AnkleL | P::HeelL | P::ToeL => S::FootL,
        P::AnkleR | P::HeelR | P::ToeR => S::FootR,
        _ => unreachable!("not a rider point"),
    }
}

fn side_of(i: usize, l: S, r: S) -> S {
    if i == 0 { l } else { r }
}

/// Mass elements (kg, 80 kg of rider with helmet and shoes) and collision capsules, sized like the
/// rendered rider, from rig points in `at`.
fn parts(at: impl Fn(P) -> Vec3) -> Vec<Part> {
    let slab = |body: S, a: P, b: P, across: Vec3, dx: f32, dz: f32, mass: f32| {
        let (a, b) = (at(a), at(b));
        let x = unit(across.reject_from(b - a)) * (0.5 * (dx - dz));
        [
            part(body, a - x, b - x, 0.5 * dz, mass * 0.5),
            part(body, a + x, b + x, 0.5 * dz, mass * 0.5),
        ]
    };
    let mut out = Vec::new();
    let hips = at(P::HipR) - at(P::HipL);
    let shoulders = at(P::ShoulderR) - at(P::ShoulderL);
    out.extend(slab(S::Pelvis, P::Hip, P::Waist, hips, 0.30, 0.21, 9.5));
    out.extend(slab(
        S::Chest,
        P::Waist,
        P::Neck,
        shoulders,
        0.38,
        0.22,
        26.0,
    ));
    out.push(part(S::Head, at(P::Head), at(P::Head), 0.12, 5.4));
    out.push(part(S::Head, at(P::Neck), at(P::Head), 0.05, 1.1));
    for i in 0..2 {
        let pick = |l: P, r: P| at(if i == 0 { l } else { r });
        let [hip, knee, ankle, heel, toe] = [
            pick(P::HipL, P::HipR),
            pick(P::KneeL, P::KneeR),
            pick(P::AnkleL, P::AnkleR),
            pick(P::HeelL, P::HeelR),
            pick(P::ToeL, P::ToeR),
        ];
        let [shoulder, elbow, wrist, hand] = [
            pick(P::ShoulderL, P::ShoulderR),
            pick(P::ElbowL, P::ElbowR),
            pick(P::WristL, P::WristR),
            pick(P::HandL, P::HandR),
        ];
        // The upper arm collides from below the shoulder, which sits inside the chest slab.
        let upper = side_of(i, S::UpperArmL, S::UpperArmR);
        out.push(Part {
            collide: false,
            ..part(upper, shoulder, elbow, 0.05, 2.6)
        });
        out.push(part(upper, shoulder.lerp(elbow, 0.45), elbow, 0.05, 0.0));
        out.push(part(
            side_of(i, S::ForearmL, S::ForearmR),
            elbow,
            wrist,
            0.04,
            1.5,
        ));
        out.push(part(
            side_of(i, S::HandL, S::HandR),
            wrist,
            hand,
            0.045,
            0.5,
        ));
        out.push(part(
            side_of(i, S::ThighL, S::ThighR),
            hip,
            knee,
            0.075,
            9.0,
        ));
        out.push(part(
            side_of(i, S::ShinL, S::ShinR),
            knee,
            ankle,
            0.055,
            4.2,
        ));
        out.push(part(side_of(i, S::FootL, S::FootR), heel, toe, 0.045, 1.5));
    }
    out
}

/// Anatomical frame of each of the rider's segments, from rig points in `at`.
fn frames(at: impl Fn(P) -> Vec3) -> Vec<Quat> {
    let mut f = vec![Quat::IDENTITY; LIMBS];
    let pelvis = basis(at(P::HipR) - at(P::HipL), at(P::Waist) - at(P::Hip));
    let chest = basis(
        at(P::ShoulderR) - at(P::ShoulderL),
        at(P::Neck) - at(P::Waist),
    );
    f[S::Pelvis as usize] = pelvis;
    f[S::Chest as usize] = chest;
    f[S::Head as usize] = basis(chest * Vec3::X, at(P::Head) - at(P::Neck));
    for i in 0..2 {
        let pick = |l: P, r: P| at(if i == 0 { l } else { r });
        let [hip, knee, ankle, heel, toe] = [
            pick(P::HipL, P::HipR),
            pick(P::KneeL, P::KneeR),
            pick(P::AnkleL, P::AnkleR),
            pick(P::HeelL, P::HeelR),
            pick(P::ToeL, P::ToeR),
        ];
        let [shoulder, elbow, wrist, hand] = [
            pick(P::ShoulderL, P::ShoulderR),
            pick(P::ElbowL, P::ElbowR),
            pick(P::WristL, P::WristR),
            pick(P::HandL, P::HandR),
        ];
        let mut set = |l: S, r: S, q: Quat| f[side_of(i, l, r) as usize] = q;
        // Knees bend backwards: their flexion axis is the elbow's with root and end swapped.
        let knee_x = hinge(ankle, knee, hip, pelvis * Vec3::X);
        set(S::ThighL, S::ThighR, basis(knee_x, hip - knee));
        set(S::ShinL, S::ShinR, basis(knee_x, knee - ankle));
        set(
            S::FootL,
            S::FootR,
            basis(knee_x, (heel - toe).cross(knee_x)),
        );
        let elbow_x = hinge(shoulder, elbow, wrist, chest * Vec3::X);
        set(S::UpperArmL, S::UpperArmR, basis(elbow_x, shoulder - elbow));
        set(S::ForearmL, S::ForearmR, basis(elbow_x, elbow - wrist));
        set(S::HandL, S::HandR, basis(elbow_x, wrist - hand));
    }
    f
}

/// The bike's collision capsules, bike-local: the frame proxies, and each tyre as a ring of chords.
fn bike_shapes(pose: &CollisionPose) -> Vec<(Vec3, Vec3, f32)> {
    const CHORDS: usize = 8;
    let mut out: Vec<_> = (pose.bodies.iter().filter(|b| !b.rider))
        .map(|b| (b.offset, b.offset, b.radius))
        .collect();
    for (hub, axle) in pose.wheel_rest.into_iter().zip(pose.wheel_axes) {
        let (u, v) = (
            axle.any_orthonormal_vector(),
            axle.cross(axle.any_orthonormal_vector()),
        );
        let ring = |k: usize| {
            let a = k as f32 / CHORDS as f32 * std::f32::consts::TAU;
            hub + (u * a.cos() + v * a.sin()) * (WHEEL_RADIUS - TIRE_TUBE)
        };
        out.extend((0..CHORDS).map(|k| (ring(k), ring(k + 1), TIRE_TUBE)));
    }
    out
}

#[derive(Resource, Default)]
pub(crate) struct Ragdoll {
    pub body: Option<Body>,
    previous: Option<[Vec3; COUNT]>,
}

/// The rider and bike at the instant before the bike resolved its impact.
pub(crate) struct Seed {
    pub positions: [Vec3; COUNT],
    /// Rig points in the bike's frame now and one step earlier, and the step, s.
    local: [Vec3; COUNT],
    previous: Option<[Vec3; COUNT]>,
    dt: f32,
    pose: CollisionPose,
    bike: (Vec3, Quat),
    velocity: Vec3,
    omega: Vec3,
}

pub(crate) struct Body {
    pub positions: [Vec3; COUNT],
    pub sleeping: bool,
    solver: Solver,
    /// Owner segment and local position of every rig point.
    points: [(usize, Vec3); COUNT],
    /// Joint holding each of `GRIPS` to the bike, if the rider held it at the crash.
    grips: [Option<usize>; 4],
    /// Where the bike stood when the last step ended, which is where this step's bike body starts.
    bike: (Vec3, Quat),
}

impl Ragdoll {
    pub fn reset(&mut self) {
        self.body = None;
        self.previous = None;
    }

    /// Capture BEFORE the bike resolves its impact, preserving rider momentum and limb motion.
    pub fn sample(&mut self, bike: &Bike, local: [Vec3; COUNT], dt: f32) -> Seed {
        let q = bike.orientation();
        let seed = Seed {
            positions: local.map(|p| bike.position + q * p),
            local,
            previous: self.previous,
            dt,
            pose: bike.collision_pose,
            bike: (bike.position, q),
            velocity: bike.velocity,
            omega: bike.world_omega(),
        };
        self.previous = Some(local);
        seed
    }

    /// Which hands and feet have let go of the bike: `([left, right], [left, right])`.
    pub fn let_go(&self) -> Option<([bool; 2], [bool; 2])> {
        let b = self.body.as_ref()?;
        let open = |k: usize| b.grips[k].is_none_or(|j| b.solver.joints[j].broken);
        Some(([open(0), open(1)], [open(2), open(3)]))
    }

    pub fn activate(&mut self, seed: Seed) {
        self.body = Some(Body::new(&seed));
    }

    /// Steps the rider over `dt`; whatever it did to the crashed bike is applied to `bike`.
    pub fn step(&mut self, bike: &mut Bike, dt: f32) {
        if let Some(body) = &mut self.body {
            body.step(bike, dt);
        }
    }
}

impl Body {
    fn new(seed: &Seed) -> Self {
        let world = |p: P| seed.positions[index(p)];
        let mut parts = parts(world);
        let mut frame = frames(world);
        let mut com = centres(&parts, LIMBS);
        let (x, q) = seed.bike;
        let shapes = bike_shapes(&seed.pose);
        for &(a, b, r) in &shapes {
            parts.push(part(
                S::Bike,
                x + q * a,
                x + q * b,
                r,
                CRASH_MASS / shapes.len() as f32,
            ));
        }
        frame.push(q);
        com.push(x);
        let mut solver = Solver::new(&parts, &frame, &com);
        let bike = &mut solver.bodies[BIKE];
        bike.inv_inertia = Mat3::IDENTITY / (CRASH_MASS * GYRATION_SQ);
        bike.driven = true;
        let points = std::array::from_fn(|i| {
            let k = owner(RIDER[i]) as usize;
            (k, frame[k].inverse() * (world(RIDER[i]) - com[k]))
        });

        let mut join =
            |parent: S, child: S, at: P, limit: Limit, left: bool, name: &'static str| {
                let joint = Joint {
                    limit: Some(limit),
                    mirror: left,
                    frame: limit.frame(left),
                    name,
                    ..solver.pin(parent as usize, child as usize, world(at))
                };
                solver.joints.push(joint);
            };
        join(S::Pelvis, S::Chest, P::Waist, SPINE, false, "back");
        join(S::Chest, S::Head, P::Neck, NECK, false, "neck");
        for i in 0..2 {
            let s = |l: S, r: S| side_of(i, l, r);
            let j = |l: P, r: P| if i == 0 { l } else { r };
            let n = |l: &'static str, r: &'static str| if i == 0 { l } else { r };
            let left = i == 0;
            join(
                S::Chest,
                s(S::UpperArmL, S::UpperArmR),
                j(P::ShoulderL, P::ShoulderR),
                SHOULDER,
                left,
                n("left shoulder", "right shoulder"),
            );
            join(
                s(S::UpperArmL, S::UpperArmR),
                s(S::ForearmL, S::ForearmR),
                j(P::ElbowL, P::ElbowR),
                ELBOW,
                left,
                n("left elbow", "right elbow"),
            );
            join(
                s(S::ForearmL, S::ForearmR),
                s(S::HandL, S::HandR),
                j(P::WristL, P::WristR),
                WRIST,
                left,
                n("left wrist", "right wrist"),
            );
            join(
                S::Pelvis,
                s(S::ThighL, S::ThighR),
                j(P::HipL, P::HipR),
                HIP,
                left,
                n("left hip", "right hip"),
            );
            join(
                s(S::ThighL, S::ThighR),
                s(S::ShinL, S::ShinR),
                j(P::KneeL, P::KneeR),
                KNEE,
                left,
                n("left knee", "right knee"),
            );
            join(
                s(S::ShinL, S::ShinR),
                s(S::FootL, S::FootR),
                j(P::AnkleL, P::AnkleR),
                SHOE_ANKLE,
                left,
                n("left ankle", "right ankle"),
            );
        }
        // Hands and feet still on the bar and pedals are pinned to the bike by a joint that lets
        // go when the pull on it is more than the fingers or the shoe can hold.
        let release = [
            seed.pose.hand_release[0],
            seed.pose.hand_release[1],
            seed.pose.foot_release[0],
            seed.pose.foot_release[1],
        ];
        let grips = std::array::from_fn(|k| {
            (release[k] < 0.5).then(|| {
                let (p, strength) = GRIPS[k];
                let joint = Joint {
                    detachable: true,
                    grip: strength,
                    name: "grip",
                    ..solver.pin(BIKE, owner(p) as usize, world(p))
                };
                solver.joints.push(joint);
                solver.joints.len() - 1
            })
        });
        solver.collide_unjointed(&[]);

        // Velocities: the bike's, plus each segment's motion relative to the bike over the last step.
        let local = |p: P| seed.local[index(p)];
        let (now_com, now_f) = (centres(&self::parts(local), LIMBS), frames(local));
        let before = seed.previous.map(|old| {
            let old = |p: P| old[index(p)];
            (centres(&self::parts(old), LIMBS), frames(old))
        });
        for k in 0..LIMBS {
            let (dv, dw) = before.as_ref().map_or((Vec3::ZERO, Vec3::ZERO), |(c, g)| {
                let dq = now_f[k] * g[k].inverse();
                let dq = if dq.w < 0.0 { -dq } else { dq };
                (
                    ((now_com[k] - c[k]) / seed.dt).clamp_length_max(MAX_LIMB_SPEED),
                    (dq.to_scaled_axis() / seed.dt).clamp_length_max(MAX_LIMB_SPIN),
                )
            });
            let b = &mut solver.bodies[k];
            b.v = seed.velocity + seed.omega.cross(b.x - x) + q * dv;
            b.w = seed.omega + q * dw;
        }
        for k in 0..solver.joints.len() {
            solver.joints[k].allow = solver.excess(k) * 1.2 + DEG;
        }
        // The bike stays where it is while the seed overlaps are pushed out.
        let (inv_mass, inv_inertia) = (
            solver.bodies[BIKE].inv_mass,
            solver.bodies[BIKE].inv_inertia,
        );
        solver.bodies[BIKE].inv_mass = 0.0;
        solver.bodies[BIKE].inv_inertia = Mat3::ZERO;
        solver.settle();
        solver.bodies[BIKE].inv_mass = inv_mass;
        solver.bodies[BIKE].inv_inertia = inv_inertia;

        let mut body = Self {
            positions: [Vec3::ZERO; COUNT],
            sleeping: false,
            solver,
            points,
            grips,
            bike: seed.bike,
        };
        body.refresh();
        body
    }

    fn refresh(&mut self) {
        for (p, &(k, local)) in self.positions.iter_mut().zip(&self.points) {
            *p = self.solver.bodies[k].point(local);
        }
        self.sleeping = self.solver.sleeping;
    }

    /// Steps over `dt` with the bike body carried from where it stood to where `bike` is now.
    fn step(&mut self, bike: &mut Bike, dt: f32) {
        if self.sleeping || dt <= 0.0 {
            return;
        }
        let (x0, q0) = self.bike;
        let v = (bike.position - x0) / dt;
        let turn = bike.orientation() * q0.inverse();
        let w = (if turn.w < 0.0 { -turn } else { turn }).to_scaled_axis() / dt;
        let b = &mut self.solver.bodies[BIKE];
        (b.x, b.q, b.v, b.w) = (x0, q0, v, w);
        self.solver.step(dt);
        if !self.solver.sleeping {
            let b = self.solver.bodies[BIKE];
            let carried = Quat::from_scaled_axis(w * dt) * q0;
            bike.push(
                b.x - (x0 + v * dt),
                b.q * carried.inverse(),
                bike.velocity + b.v - v,
                bike.world_omega() + b.w - w,
            );
            self.bike = (bike.position, bike.orientation());
        }
        self.refresh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::AnimationState;
    use crate::bike::{Controls, CrashReason, terrain_height};
    use crate::scene::{SKELETON, collision_pose, rider_points};

    const DT: f32 = 1.0 / 120.0;

    /// Drives the bike and ragdoll like `game::simulate` until the bike crashes, then returns
    /// the wreck and rider.
    fn crash_from(mut bike: Bike, feet_off: bool) -> (Bike, Ragdoll) {
        let anim = AnimationState::default();
        let mut ragdoll = Ragdoll::default();
        for _ in 0..1200 {
            bike.collision_pose = collision_pose(&bike, &anim);
            if feet_off {
                bike.collision_pose.foot_release = [1.0; 2];
            }
            let seed = ragdoll.sample(&bike, rider_points(&bike, &anim), DT);
            bike.step(&Controls::default(), DT);
            if bike.crash.is_some() {
                ragdoll.activate(seed);
                return (bike, ragdoll);
            }
        }
        panic!("bike never crashed");
    }

    fn hard_crash() -> (Bike, Ragdoll) {
        let mut bike = Bike::default();
        bike.position.y += 40.0;
        crash_from(bike, false)
    }

    fn held(body: &Body, k: usize) -> bool {
        body.grips[k].is_some_and(|j| !body.solver.joints[j].broken)
    }

    /// Steps the wreck and the rider for `ticks`, asserting every tick that the rider stays finite
    /// and above the terrain, that bones keep their length, and that no two limbs (5 mm) and no
    /// limb and the bike (2 cm, a tyre's give) overlap.
    fn run_checked(bike: &mut Bike, ragdoll: &mut Ragdoll, ticks: usize) {
        let lengths = |p: &[Vec3; COUNT]| {
            SKELETON
                .iter()
                .flat_map(|(bones, _)| bones.iter())
                .map(|&(a, b)| p[index(a)].distance(p[index(b)]))
                .collect::<Vec<_>>()
        };
        let rest = lengths(&ragdoll.body.as_ref().unwrap().positions);
        for tick in 0..ticks {
            bike.step(&Controls::default(), DT);
            ragdoll.step(bike, DT);
            let body = ragdoll.body.as_ref().unwrap();
            for (i, p) in body.positions.iter().enumerate() {
                assert!(p.is_finite(), "tick {tick}: point {i} not finite");
                assert!(
                    p.y >= terrain_height(p.x, p.z) - 0.01,
                    "tick {tick}: point {i} sank to {p}"
                );
            }
            for (now, was) in lengths(&body.positions).into_iter().zip(&rest) {
                assert!(
                    (now - was).abs() < 0.03,
                    "tick {tick}: bone {was} stretched to {now}"
                );
            }
            for (with_bike, limit) in [(false, -0.005), (true, -0.02)] {
                let (gap, (a, b)) = body
                    .solver
                    .worst_overlap(|a, b| (a == BIKE || b == BIKE) != with_bike);
                assert!(
                    gap > limit,
                    "tick {tick}: segments {a}/{b} overlap {gap} (limit {limit})"
                );
            }
        }
    }

    #[test]
    fn penetration_repair_does_not_launch_a_stationary_rider() {
        let mut bike = Bike::default();
        bike.position.x = -50.0;
        let local = rider_points(&bike, &AnimationState::default());
        let mut ragdoll = Ragdoll::default();
        let mut seed = ragdoll.sample(&bike, local, DT);
        // The lowest point 10 cm under the terrain, nothing else moving.
        let low = local.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        seed.positions = local.map(|p| Vec3::new(20.0, -0.1 - low, 8.0) + p);
        (seed.velocity, seed.omega) = (Vec3::ZERO, Vec3::ZERO);
        (seed.pose.hand_release, seed.pose.foot_release) = ([1.0; 2], [1.0; 2]);
        ragdoll.activate(seed);
        ragdoll.step(&mut bike, DT);
        let body = ragdoll.body.as_ref().unwrap();
        // Free fall for one tick gives -0.08 m/s; repairing the overlap must not throw anything up.
        for b in &body.solver.bodies[..LIMBS] {
            assert!(b.v.y < 0.3, "launched at {}", b.v);
        }
        assert!(body.positions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn a_hard_crash_tumbles_intact_with_limbs_apart_and_settles_asleep() {
        let (mut bike, mut ragdoll) = hard_crash();
        let (crash, original) = (bike.position, ragdoll.body.as_ref().unwrap().positions);
        run_checked(&mut bike, &mut ragdoll, 600);
        let body = ragdoll.body.as_ref().unwrap();
        assert!(body.sleeping, "still moving after 5 s");
        let hip = body.positions[index(P::Hip)];
        assert!(
            hip.distance(crash) < 40.0,
            "rider ended {hip}, crash at {crash}"
        );
        let knee_to_shoulder =
            |p: &[Vec3; COUNT]| p[index(P::KneeL)].distance(p[index(P::Shoulder)]);
        assert!(
            (knee_to_shoulder(&original) - knee_to_shoulder(&body.positions)).abs() > 0.05,
            "pose must articulate, not rigidly tumble"
        );
        ragdoll.reset();
        assert!(ragdoll.body.is_none());
    }

    #[test]
    fn gentle_crashes_keep_the_hands_on_the_bars_hard_ones_lose_them() {
        let controls = Controls::default();
        let mut gentle = Bike::default();
        gentle.position.y += 0.15;
        let ((mut gb, mut gr), (mut hb, mut hr)) = (crash_from(gentle, true), hard_crash());
        assert_eq!(gb.crash.unwrap().reason, CrashReason::MissingSupport);
        assert_eq!(hb.crash.unwrap().reason, CrashReason::HardImpact);
        for _ in 0..(0.3 / DT) as usize {
            gb.step(&controls, DT);
            gr.step(&mut gb, DT);
        }
        for _ in 0..(0.1 / DT) as usize {
            hb.step(&controls, DT);
            hr.step(&mut hb, DT);
        }
        let (g, h) = (gr.body.as_ref().unwrap(), hr.body.as_ref().unwrap());
        let gap = |bike: &Bike, body: &Body, k: usize| {
            let anchor = body.solver.joints[body.grips[k].unwrap()].pa;
            body.positions[index(GRIPS[k].0)].distance(bike.position + bike.orientation() * anchor)
        };
        assert!(
            (0..2).any(|k| held(g, k) && gap(&gb, g, k) < 0.15),
            "gentle crash let go: {:?}",
            gr.let_go()
        );
        assert!(
            !held(h, 0) && !held(h, 1),
            "hard impact kept its hands: {:?}",
            hr.let_go()
        );
        assert_eq!(hr.let_go().unwrap().0, [true; 2]);
    }

    #[test]
    fn grips_already_open_stay_open_and_a_yanked_grip_lets_go() {
        let mut bike = Bike::default();
        bike.collision_pose.hand_release = [1.0, 0.0];
        let mut ragdoll = Ragdoll::default();
        let seed = ragdoll.sample(&bike, rider_points(&bike, &AnimationState::default()), DT);
        ragdoll.activate(seed);
        assert_eq!(ragdoll.let_go().unwrap().0, [true, false]);
        // The bike is torn away at 50 m/s.
        for _ in 0..3 {
            bike.position.x += 50.0 * DT;
            ragdoll.step(&mut bike, DT);
        }
        assert_eq!(ragdoll.let_go().unwrap(), ([true; 2], [true; 2]));
    }

    #[test]
    fn a_gripping_rider_drags_the_bike_and_momentum_is_conserved() {
        let anim = AnimationState::default();
        let mut bike = Bike::default();
        bike.position.y += 60.0;
        bike.velocity = Vec3::new(0.0, 0.0, -6.0);
        bike.collision_pose = collision_pose(&bike, &anim);
        let mut ragdoll = Ragdoll::default();
        let seed = ragdoll.sample(&bike, rider_points(&bike, &anim), DT);
        // The wreck stops dead; the rider, holding the bike, does not.
        bike.velocity = Vec3::ZERO;
        ragdoll.activate(seed);
        let momentum = |bike: &Bike, body: &Body| {
            body.solver.bodies[..LIMBS]
                .iter()
                .map(|b| b.v / b.inv_mass)
                .sum::<Vec3>()
                + bike.velocity * CRASH_MASS
        };
        let before = momentum(&bike, ragdoll.body.as_ref().unwrap());
        for _ in 0..24 {
            bike.position += bike.velocity * DT;
            ragdoll.step(&mut bike, DT);
        }
        assert!(
            bike.velocity.z < -0.5,
            "bike was not dragged: {}",
            bike.velocity
        );
        let after = momentum(&bike, ragdoll.body.as_ref().unwrap());
        // Air drag takes a few per cent off the rider.
        assert!(
            (after.z - before.z).abs() < 0.05 * before.z.abs() && after.x.abs() < 1.0,
            "momentum {before} -> {after}"
        );
    }
}
