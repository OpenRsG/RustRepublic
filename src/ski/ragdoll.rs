//! Detached skier: mass-weighted position constraints over every pose joint (body, skis, poles),
//! soft joint limits, point and segment terrain contact, anisotropic ski friction and sleeping.

use bevy::prelude::*;

use super::pose::{BONES, GEAR, J, JOINTS, SkierPose};
use crate::bike::terrain_height;

const GRAVITY: f32 = 9.81;
/// Nominal solver step; larger frame steps are split into substeps of at most this length.
const SUBSTEP: f32 = 1.0 / 120.0;
const ITERATIONS: usize = 16;
/// A substep moves any point at most this far, metres (fast tumbles get more substeps).
const MAX_STEP_TRAVEL: f32 = 0.06;
const MAX_SUBSTEPS: usize = 32;
/// Seed velocities beyond this (pose-difference glitches) are clamped, m/s.
const MAX_SEED_SPEED: f32 = 40.0;
/// Spacing of terrain probes along long links, metres.
const SAMPLE_SPACING: f32 = 0.08;
const AIR_DAMPING: f32 = 0.1;
/// Passive joint friction rate on rigid links; removes limb jitter without driving a pose.
const JOINT_DAMPING: f32 = 3.0;
const RESTITUTION: f32 = 0.1;
/// Coulomb coefficient of body, boots and pole baskets on snow.
const BODY_FRICTION: f32 = 0.6;
/// Ski friction along the ski axis (they glide) and across it (edges bite).
const SKI_ALONG: f32 = 0.05;
const SKI_ACROSS: f32 = 0.8;
const SLEEP_SPEED: f32 = 0.12;
const SLEEP_TIME: f32 = 0.6;

/// Every joint in enum order, to map array indices back to `J`.
const ALL: [J; JOINTS] = {
    use J::*;
    [
        Pelvis, HipL, HipR, KneeL, KneeR, AnkleL, AnkleR, HeelL, HeelR, ToeL, ToeR, Waist, Chest,
        Neck, Head, ShoulderL, ShoulderR, ElbowL, ElbowR, WristL, WristR, HandL, HandR, BindL,
        BindR, TipL, TipR, TailL, TailR, PoleTopL, PoleTopR, PoleTipL, PoleTipR,
    ]
};

/// Per-joint collision radius (m) and inverse mass (1/kg-ish; lighter points move more).
fn shape(j: J) -> (f32, f32) {
    use J::*;
    match j {
        Head => (0.11, 0.25),
        Pelvis => (0.12, 0.12),
        Waist => (0.12, 0.2),
        Chest => (0.12, 0.12),
        HipL | HipR => (0.06, 0.25),
        KneeL | KneeR => (0.06, 0.3),
        AnkleL | AnkleR => (0.06, 0.4),
        Neck => (0.06, 0.6),
        ShoulderL | ShoulderR => (0.06, 0.3),
        ElbowL | ElbowR => (0.06, 0.5),
        WristL | WristR | HandL | HandR => (0.06, 0.8),
        HeelL | HeelR | ToeL | ToeR => (0.02, 0.6),
        BindL | BindR => (0.02, 0.5),
        TipL | TipR | TailL | TailR => (0.02, 1.0),
        PoleTopL | PoleTopR => (0.02, 1.5),
        PoleTipL | PoleTipR => (0.01, 1.5),
        N => (0.0, 0.0),
    }
}

/// Leg hinges: hip, knee, ankle and the boot sole (heel, toe) that says which way the knee bends.
const KNEE_HINGES: [(J, J, J, J, J); 2] = [
    (J::HipL, J::KneeL, J::AnkleL, J::HeelL, J::ToeL),
    (J::HipR, J::KneeR, J::AnkleR, J::HeelR, J::ToeR),
];

/// Ski axis (tail to tip) a ski-mounted point slides along, if it belongs to a ski.
fn ski_ends(j: J) -> Option<(J, J)> {
    use J::*;
    match j {
        BindL | TipL | TailL | HeelL | ToeL => Some((TailL, TipL)),
        BindR | TipR | TailR | HeelR | ToeR => Some((TailR, TipR)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct Link {
    a: usize,
    b: usize,
    min: f32,
    max: f32,
}

impl Link {
    fn rigid(&self) -> bool {
        self.min == self.max
    }
}

#[derive(Resource, Default)]
pub(crate) struct SkiRagdoll {
    body: Option<Body>,
}

impl SkiRagdoll {
    pub(crate) fn reset(&mut self) {
        self.body = None;
    }

    pub(crate) fn active(&self) -> bool {
        self.body.is_some()
    }

    /// Detaches the skier from the rig: positions = `current`, velocities = finite difference.
    pub(crate) fn activate(&mut self, previous: &SkierPose, current: &SkierPose, dt: f32) {
        let dt = dt.max(1e-4);
        let velocities = std::array::from_fn(|i| {
            ((current.p[i] - previous.p[i]) / dt).clamp_length_max(MAX_SEED_SPEED)
        });
        self.body = Some(Body::new(current.p, velocities));
    }

    pub(crate) fn step(&mut self, dt: f32) {
        if let Some(body) = &mut self.body {
            body.step(dt);
        }
    }

    pub(crate) fn pose(&self) -> Option<SkierPose> {
        self.body
            .as_ref()
            .map(|b| SkierPose::from_points(b.positions))
    }

    /// Pelvis position and velocity (camera target).
    pub(crate) fn centre(&self) -> Option<(Vec3, Vec3)> {
        self.body.as_ref().map(|b| {
            (
                b.positions[J::Pelvis as usize],
                b.velocities[J::Pelvis as usize],
            )
        })
    }

    pub(crate) fn sleeping(&self) -> bool {
        self.body.as_ref().is_some_and(|b| b.sleeping)
    }
}

struct Body {
    positions: [Vec3; JOINTS],
    velocities: [Vec3; JOINTS],
    links: Vec<Link>,
    radius: [f32; JOINTS],
    inverse_mass: [f32; JOINTS],
    sleeping: bool,
    quiet_time: f32,
}

impl Body {
    fn new(positions: [Vec3; JOINTS], velocities: [Vec3; JOINTS]) -> Self {
        let mut b = Self {
            positions,
            velocities,
            links: Vec::new(),
            radius: [0.0; JOINTS],
            inverse_mass: [0.0; JOINTS],
            sleeping: false,
            quiet_time: 0.0,
        };
        for i in 0..JOINTS {
            let j = ALL[i];
            (b.radius[i], b.inverse_mass[i]) = shape(j);
        }
        for &(a, c) in BONES.iter().chain(GEAR) {
            b.fixed(a, c);
        }
        // Pelvis and chest stay rigid triangles/tetrahedra; the spine between them is soft.
        for (a, c) in [
            (J::HipL, J::HipR),
            (J::Waist, J::HipL),
            (J::Waist, J::HipR),
            (J::ShoulderL, J::ShoulderR),
            (J::Waist, J::ShoulderL),
            (J::Waist, J::ShoulderR),
            (J::Neck, J::ShoulderL),
            (J::Neck, J::ShoulderR),
        ] {
            b.fixed(a, c);
        }
        // Each ski, its binding and the boot form one rigid body.
        for set in [
            [J::BindL, J::TipL, J::TailL, J::HeelL, J::ToeL, J::AnkleL],
            [J::BindR, J::TipR, J::TailR, J::HeelR, J::ToeR, J::AnkleR],
        ] {
            for (n, &a) in set.iter().enumerate() {
                for &c in &set[n + 1..] {
                    b.fixed(a, c);
                }
            }
        }
        // A pole is a rigid shaft gripped at the fist, free to pivot about it.
        for (hand, top, tip) in [
            (J::HandL, J::PoleTopL, J::PoleTipL),
            (J::HandR, J::PoleTopR, J::PoleTipR),
        ] {
            b.fixed(hand, top);
            b.fixed(hand, tip);
        }
        // Soft chains: distance end to end stays between a collapsed fold and fully straight.
        for (a, m, c, ratio) in [
            (J::Pelvis, J::Waist, J::Chest, 0.85),
            (J::Waist, J::Chest, J::Neck, 0.85),
            (J::Chest, J::Neck, J::Head, 0.8),
            (J::HipL, J::KneeL, J::AnkleL, 0.3),
            (J::HipR, J::KneeR, J::AnkleR, 0.3),
            (J::ShoulderL, J::ElbowL, J::WristL, 0.25),
            (J::ShoulderR, J::ElbowR, J::WristR, 0.25),
        ] {
            b.soft_chain(a, m, c, ratio);
        }
        // Limited torso twist/bend: shoulder-to-hip distances stay within 15 % of the pose.
        for (s, h) in [
            (J::ShoulderL, J::HipL),
            (J::ShoulderR, J::HipR),
            (J::ShoulderL, J::HipR),
            (J::ShoulderR, J::HipL),
        ] {
            let d = b.distance(s, h);
            b.links.push(Link {
                a: s as usize,
                b: h as usize,
                min: d * 0.85,
                max: d * 1.15,
            });
        }
        b
    }

    fn distance(&self, a: J, b: J) -> f32 {
        self.positions[a as usize].distance(self.positions[b as usize])
    }

    fn fixed(&mut self, a: J, b: J) {
        let (a, c) = (a as usize, b as usize);
        if self
            .links
            .iter()
            .any(|l| l.rigid() && ((l.a, l.b) == (a, c) || (l.a, l.b) == (c, a)))
        {
            return;
        }
        let length = self.positions[a].distance(self.positions[c]);
        self.links.push(Link {
            a,
            b: c,
            min: length,
            max: length,
        });
    }

    fn soft_chain(&mut self, a: J, m: J, c: J, ratio: f32) {
        let max = self.distance(a, m) + self.distance(m, c);
        let min = (max * ratio).min(self.distance(a, c));
        self.links.push(Link {
            a: a as usize,
            b: c as usize,
            min,
            max,
        });
    }

    fn step(&mut self, dt: f32) {
        if self.sleeping || dt <= 0.0 {
            return;
        }
        let speed = self
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0_f32, f32::max);
        let count = (dt / SUBSTEP - 1e-3)
            .ceil()
            .max((speed * dt / MAX_STEP_TRAVEL).ceil())
            .clamp(1.0, MAX_SUBSTEPS as f32) as usize;
        let h = dt / count as f32;
        for _ in 0..count {
            self.substep(h);
            if self.sleeping {
                break;
            }
        }
    }

    fn substep(&mut self, h: f32) {
        let before = self.positions;
        let damping = (-AIR_DAMPING * h).exp();
        for i in 0..JOINTS {
            self.velocities[i].y -= GRAVITY * h;
            self.velocities[i] *= damping;
            self.positions[i] += self.velocities[i] * h;
        }
        let incoming = self.velocities;
        let mut normals = [Vec3::ZERO; JOINTS];
        for _ in 0..ITERATIONS {
            self.solve_contacts(&mut normals);
            self.solve_links();
            self.limit_knees();
        }
        for _ in 0..ITERATIONS {
            self.limit_knees();
            self.solve_links();
        }
        let impulses = self.solve_velocities(h, &incoming, &normals);
        self.apply_friction(&normals, &impulses);

        let supported = normals.iter().any(|&n| n != Vec3::ZERO);
        // Net displacement over the substep: a point that gravity pulls and a constraint pushes
        // straight back is at rest even though its solver velocity carries residual error.
        let quiet = (0..JOINTS).all(|i| self.positions[i].distance(before[i]) < SLEEP_SPEED * h);
        self.quiet_time = if supported && quiet {
            self.quiet_time + h
        } else {
            0.0
        };
        if self.quiet_time > SLEEP_TIME {
            self.sleeping = true;
            self.velocities.fill(Vec3::ZERO);
        }
    }

    /// A knee only bends forwards (the way the boot points): it is kept in front of the straight
    /// hip-ankle line, so a tumbling leg folds at the knee instead of hyperextending.
    fn limit_knees(&mut self) {
        for (hip, knee, ankle, heel, toe) in KNEE_HINGES {
            let f =
                (self.positions[toe as usize] - self.positions[heel as usize]).normalize_or_zero();
            let mid = (self.positions[hip as usize] + self.positions[ankle as usize]) * 0.5;
            let behind = (mid - self.positions[knee as usize]).dot(f);
            if behind > 0.0 {
                self.positions[knee as usize] += f * behind;
            }
        }
    }

    fn solve_links(&mut self) {
        for k in 0..self.links.len() {
            let c = self.links[k];
            let d = self.positions[c.b] - self.positions[c.a];
            let length = d.length();
            if length < 1e-6 {
                continue;
            }
            let weight = self.inverse_mass[c.a] + self.inverse_mass[c.b];
            let correction = d * ((length - length.clamp(c.min, c.max)) / length);
            self.positions[c.a] += correction * (self.inverse_mass[c.a] / weight);
            self.positions[c.b] -= correction * (self.inverse_mass[c.b] / weight);
        }
    }

    fn solve_contacts(&mut self, normals: &mut [Vec3; JOINTS]) {
        for i in 0..JOINTS {
            let (depth, n) = terrain_contact(self.positions[i], self.radius[i]);
            if depth > 0.0 {
                self.positions[i] += n * depth;
                normals[i] = n;
            }
        }
        // Long links (skis, poles, limbs) collide along their length, not only at the joints.
        for &(a, b) in BONES.iter().chain(GEAR) {
            let (a, b) = (a as usize, b as usize);
            let radius = self.radius[a].min(self.radius[b]);
            let samples =
                (self.positions[a].distance(self.positions[b]) / SAMPLE_SPACING).ceil() as usize;
            for k in 1..samples {
                let t = k as f32 / samples as f32;
                let (depth, n) =
                    terrain_contact(self.positions[a].lerp(self.positions[b], t), radius);
                if depth <= 0.0 {
                    continue;
                }
                let wa = self.inverse_mass[a] * (1.0 - t);
                let wb = self.inverse_mass[b] * t;
                let denom = wa * (1.0 - t) + wb * t;
                if denom <= 0.0 {
                    continue;
                }
                self.positions[a] += n * (depth * wa / denom);
                self.positions[b] += n * (depth * wb / denom);
                normals[a] = n;
                normals[b] = n;
            }
        }
    }

    /// Split impulses: penetration repair never becomes launch speed; only real incoming
    /// momentum supplies restitution. Returns each point's accumulated normal impulse.
    fn solve_velocities(
        &mut self,
        h: f32,
        incoming: &[Vec3; JOINTS],
        normals: &[Vec3; JOINTS],
    ) -> [f32; JOINTS] {
        let joint_damping = 1.0 - (-JOINT_DAMPING * h).exp();
        for c in self.links.iter().filter(|c| c.rigid()) {
            let relative = self.velocities[c.b] - self.velocities[c.a];
            let impulse =
                relative * (joint_damping / (self.inverse_mass[c.a] + self.inverse_mass[c.b]));
            self.velocities[c.a] += impulse * self.inverse_mass[c.a];
            self.velocities[c.b] -= impulse * self.inverse_mass[c.b];
        }
        let mut impulses = [0.0_f32; JOINTS];
        for _ in 0..ITERATIONS {
            for c in &self.links {
                let d = self.positions[c.b] - self.positions[c.a];
                let length = d.length();
                let n = d.normalize_or_zero();
                let relative = (self.velocities[c.b] - self.velocities[c.a]).dot(n);
                let constrained = c.rigid()
                    || (length <= c.min + 0.001 && relative < 0.0)
                    || (length >= c.max - 0.001 && relative > 0.0);
                if constrained {
                    let impulse = relative / (self.inverse_mass[c.a] + self.inverse_mass[c.b]);
                    self.velocities[c.a] += n * (impulse * self.inverse_mass[c.a]);
                    self.velocities[c.b] -= n * (impulse * self.inverse_mass[c.b]);
                }
            }
            for i in 0..JOINTS {
                let n = normals[i];
                if n != Vec3::ZERO {
                    let vn = incoming[i].dot(n);
                    let bounce = if vn < -1.0 { -vn * RESTITUTION } else { 0.0 };
                    let impulse = (bounce - self.velocities[i].dot(n)).max(0.0);
                    self.velocities[i] += n * impulse;
                    impulses[i] += impulse;
                }
            }
        }
        impulses
    }

    /// Coulomb friction from the normal impulse; ski points use separate along/across
    /// coefficients so skis glide forwards and resist sliding sideways.
    fn apply_friction(&mut self, normals: &[Vec3; JOINTS], impulses: &[f32; JOINTS]) {
        for i in 0..JOINTS {
            let n = normals[i];
            if n == Vec3::ZERO {
                continue;
            }
            let j = ALL[i];
            let normal = n * self.velocities[i].dot(n);
            let tangent = self.velocities[i] - normal;
            let grip = impulses[i];
            let (along, mu_along, mu_across) = match ski_ends(j) {
                Some((tail, tip)) => {
                    let axis = self.positions[tip as usize] - self.positions[tail as usize];
                    (
                        (axis - n * axis.dot(n)).normalize_or_zero(),
                        SKI_ALONG,
                        SKI_ACROSS,
                    )
                }
                None => (Vec3::ZERO, BODY_FRICTION, BODY_FRICTION),
            };
            let t_along = along * tangent.dot(along);
            self.velocities[i] = normal
                + shrink(t_along, mu_along * grip)
                + shrink(tangent - t_along, mu_across * grip);
        }
    }
}

/// Reduces `v` by `amount` (clamped at zero) without changing its direction.
fn shrink(v: Vec3, amount: f32) -> Vec3 {
    let speed = v.length();
    if speed <= amount {
        Vec3::ZERO
    } else {
        v * (1.0 - amount / speed)
    }
}

/// Penetration depth of a sphere at `p` into the terrain, and the surface normal.
fn terrain_contact(p: Vec3, radius: f32) -> (f32, Vec3) {
    let eps = 0.05;
    let dx = (terrain_height(p.x + eps, p.z) - terrain_height(p.x - eps, p.z)) / (2.0 * eps);
    let dz = (terrain_height(p.x, p.z + eps) - terrain_height(p.x, p.z - eps)) / (2.0 * eps);
    let n = Vec3::new(-dx, 1.0, -dz).normalize();
    (radius - (p.y - terrain_height(p.x, p.z)) * n.y, n)
}

#[cfg(test)]
mod tests {
    use super::super::pose::{
        ANKLE_ABOVE_SKI, CHEST_TO_NECK, FOREARM, HAND, HEEL_BACK, HIP_HALF_WIDTH, NECK_TO_HEAD,
        PELVIS_TO_WAIST, POLE_GRIP, POLE_LENGTH, SHIN, SHOULDER_HALF_WIDTH, SKI_BACK, SKI_FRONT,
        SKI_THICKNESS, SKI_TIP_RISE, STANCE_WIDTH, THIGH, TOE_FORWARD, UPPER_ARM, WAIST_TO_CHEST,
    };
    use super::*;
    use crate::bike::HILL_X;

    const DT: f32 = 1.0 / 120.0;

    /// Two-bone solution from `a` to `c` with the middle joint on the `hint` side.
    fn bend(a: Vec3, c: Vec3, l1: f32, l2: f32, hint: Vec3) -> Vec3 {
        let d = c - a;
        let dist = d.length();
        assert!(dist < l1 + l2, "unreachable test limb");
        let axis = d / dist;
        let x = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist);
        let h = (l1 * l1 - x * x).max(0.0).sqrt();
        a + axis * x + (hint - axis * hint.dot(axis)).normalize() * h
    }

    /// Plausible upright skier facing local -Z, ski bases on y = 0, placed by `rotation`/`origin`.
    fn standing(origin: Vec3, rotation: Quat) -> SkierPose {
        let mut p = [Vec3::ZERO; JOINTS];
        let fwd = Vec3::NEG_Z;
        let pelvis_y = SKI_THICKNESS + ANKLE_ABOVE_SKI + 0.8;
        let chest_y = pelvis_y + PELVIS_TO_WAIST + WAIST_TO_CHEST;
        p[J::Pelvis as usize] = Vec3::new(0.0, pelvis_y, 0.0);
        p[J::Waist as usize] = Vec3::new(0.0, pelvis_y + PELVIS_TO_WAIST, 0.0);
        p[J::Chest as usize] = Vec3::new(0.0, chest_y, 0.0);
        p[J::Neck as usize] = Vec3::new(0.0, chest_y + CHEST_TO_NECK, 0.0);
        p[J::Head as usize] = Vec3::new(0.0, chest_y + CHEST_TO_NECK + NECK_TO_HEAD, 0.0);
        let pole_dir = Vec3::new(0.0, 1.0, 0.3).normalize();
        for (sx, left) in [(-1.0, true), (1.0, false)] {
            let pick = |l: J, r: J| (if left { l } else { r }) as usize;
            let (hip, knee, ankle) = (
                pick(J::HipL, J::HipR),
                pick(J::KneeL, J::KneeR),
                pick(J::AnkleL, J::AnkleR),
            );
            let (shoulder, elbow, wrist) = (
                pick(J::ShoulderL, J::ShoulderR),
                pick(J::ElbowL, J::ElbowR),
                pick(J::WristL, J::WristR),
            );
            let hand = pick(J::HandL, J::HandR);
            let bind = Vec3::new(sx * STANCE_WIDTH / 2.0, SKI_THICKNESS, 0.0);
            p[pick(J::BindL, J::BindR)] = bind;
            p[pick(J::TipL, J::TipR)] = bind + fwd * SKI_FRONT + Vec3::Y * SKI_TIP_RISE;
            p[pick(J::TailL, J::TailR)] = bind - fwd * SKI_BACK;
            p[ankle] = bind + Vec3::Y * ANKLE_ABOVE_SKI;
            p[pick(J::HeelL, J::HeelR)] = bind - fwd * HEEL_BACK;
            p[pick(J::ToeL, J::ToeR)] = bind + fwd * TOE_FORWARD;
            p[hip] = Vec3::new(sx * HIP_HALF_WIDTH, pelvis_y, 0.0);
            p[knee] = bend(p[hip], p[ankle], THIGH, SHIN, fwd);
            p[shoulder] = Vec3::new(sx * SHOULDER_HALF_WIDTH, chest_y, 0.0);
            let pole_tip = Vec3::new(sx * 0.4, 0.01, -0.6);
            let top = pole_tip + pole_dir * POLE_LENGTH;
            p[pick(J::PoleTipL, J::PoleTipR)] = pole_tip;
            p[pick(J::PoleTopL, J::PoleTopR)] = top;
            p[hand] = top - pole_dir * POLE_GRIP;
            p[wrist] = p[hand] + Vec3::Z * HAND;
            p[elbow] = bend(
                p[shoulder],
                p[wrist],
                UPPER_ARM,
                FOREARM,
                Vec3::new(0.0, -1.0, 1.0),
            );
        }
        p.iter_mut().for_each(|q| *q = origin + rotation * *q);
        SkierPose::from_points(p)
    }

    /// Skier standing on the 32 m hill at `z`, skis along the fall line (`yaw` 0) or rotated.
    fn on_hill(z: f32, yaw: f32) -> SkierPose {
        let ground = Vec3::new(HILL_X, terrain_height(HILL_X, z), z);
        let n = terrain_contact(ground, 0.0).1;
        standing(
            ground + n * 0.03,
            Quat::from_rotation_arc(Vec3::Y, n) * Quat::from_rotation_y(yaw),
        )
    }

    fn rest_lengths(r: &SkiRagdoll) -> Vec<(usize, usize, f32)> {
        let b = r.body.as_ref().unwrap();
        b.links
            .iter()
            .filter(|l| l.rigid())
            .map(|l| (l.a, l.b, l.min))
            .collect()
    }

    fn assert_intact(r: &SkiRagdoll, rest: &[(usize, usize, f32)]) {
        let b = r.body.as_ref().unwrap();
        for &(a, c, len) in rest {
            let now = b.positions[a].distance(b.positions[c]);
            assert!(
                (now - len).abs() <= 0.02 * len,
                "{:?}-{:?} stretched {len} -> {now}",
                ALL[a],
                ALL[c]
            );
        }
    }

    fn assert_above_terrain(r: &SkiRagdoll) {
        for (i, p) in r.body.as_ref().unwrap().positions.iter().enumerate() {
            assert!(p.is_finite(), "{:?} not finite", ALL[i]);
            assert!(
                p.y >= terrain_height(p.x, p.z) - 0.01,
                "{:?} sank to {p}",
                ALL[i]
            );
        }
    }

    #[test]
    fn dropped_skier_comes_to_rest_intact() {
        let pose = standing(Vec3::new(-20.0, 2.0, 0.0), Quat::IDENTITY);
        let mut r = SkiRagdoll::default();
        r.activate(&pose, &pose, DT);
        let rest = rest_lengths(&r);
        for _ in 0..720 {
            r.step(DT);
            assert_above_terrain(&r);
            assert_intact(&r, &rest);
        }
        assert!(r.sleeping(), "not asleep after 6 s");
        assert!(r.centre().unwrap().0.y < 1.0, "pelvis must come down");
    }

    #[test]
    fn launched_skier_tumbles_downhill_without_penetrating() {
        let pose = on_hill(0.0, 0.0);
        let slope = terrain_contact(Vec3::new(HILL_X, 0.0, 0.0), 0.0).1;
        let velocity = Vec3::new(0.0, -slope.z.abs(), -1.0).normalize() * 15.0;
        let mut earlier = pose;
        earlier.p.iter_mut().for_each(|p| *p -= velocity * DT);
        let mut r = SkiRagdoll::default();
        r.activate(&earlier, &pose, DT);
        let start = r.centre().unwrap().0;
        for _ in 0..360 {
            r.step(DT);
            assert_above_terrain(&r);
        }
        let end = r.centre().unwrap().0;
        assert!(end.z < start.z - 10.0, "pelvis only moved {start} -> {end}");
        assert!(r.pose().unwrap().p.iter().all(|p| p.is_finite()));
    }

    /// Slide of the left binding over 2 s from standing on a slope, skis along vs across the fall line.
    fn slide(yaw: f32) -> f32 {
        let pose = on_hill(-20.0, yaw);
        let mut r = SkiRagdoll::default();
        r.activate(&pose, &pose, DT);
        let start = r.body.as_ref().unwrap().positions[J::BindL as usize];
        for _ in 0..240 {
            r.step(DT);
        }
        start.distance(r.body.as_ref().unwrap().positions[J::BindL as usize])
    }

    #[test]
    fn skis_slide_along_their_axis_more_than_across() {
        let along = slide(0.0);
        let across = slide(std::f32::consts::FRAC_PI_2);
        assert!(
            along > across * 1.5 + 0.1,
            "along {along} vs across {across}"
        );
    }

    #[test]
    fn knees_never_bend_backwards_in_a_tumble() {
        let pose = on_hill(0.0, 0.0);
        let slope = terrain_contact(Vec3::new(HILL_X, 0.0, 0.0), 0.0).1;
        let velocity = Vec3::new(0.0, -slope.z.abs(), -1.0).normalize() * 15.0;
        let mut earlier = pose;
        earlier.p.iter_mut().for_each(|p| *p -= velocity * DT);
        let mut r = SkiRagdoll::default();
        r.activate(&earlier, &pose, DT);
        let mut worst = 0.0_f32;
        for _ in 0..480 {
            r.step(DT);
            let p = r.body.as_ref().unwrap().positions;
            for (hip, knee, ankle, heel, toe) in KNEE_HINGES {
                let f = (p[toe as usize] - p[heel as usize]).normalize_or_zero();
                let mid = (p[hip as usize] + p[ankle as usize]) * 0.5;
                worst = worst.max((mid - p[knee as usize]).dot(f));
            }
        }
        assert!(worst < 0.02, "a knee bent {worst} m backwards");
    }
}
