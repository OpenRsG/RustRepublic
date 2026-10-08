//! Detached rider: mass-weighted position constraints, bounded joints and capsule terrain contact.
use bevy::prelude::*;

use crate::bike::{Bike, WHEEL_RADIUS, terrain_height};
use crate::scene::{P, SKELETON};

pub(crate) const COUNT: usize = P::N as usize - P::Hip as usize;
pub(crate) const fn index(p: P) -> usize {
    p as usize - P::Hip as usize
}

#[derive(Clone, Copy, Default)]
struct Constraint {
    a: usize,
    b: usize,
    min: f32,
    max: f32,
}

#[derive(Resource, Default)]
pub(crate) struct Ragdoll {
    pub body: Option<Body>,
    previous: Option<[Vec3; COUNT]>,
}

pub(crate) struct Seed {
    pub positions: [Vec3; COUNT],
    velocities: [Vec3; COUNT],
}

pub(crate) struct Body {
    pub positions: [Vec3; COUNT],
    velocities: [Vec3; COUNT],
    constraints: [Constraint; 48],
    constraint_count: usize,
    radii: [f32; COUNT],
    inverse_mass: [f32; COUNT],
    pub sleeping: bool,
    quiet_time: f32,
}

impl Ragdoll {
    pub fn reset(&mut self) {
        self.body = None;
        self.previous = None;
    }

    /// Capture BEFORE the bike resolves its impact, preserving rider momentum and limb motion.
    pub fn sample(&mut self, bike: &Bike, local: [Vec3; COUNT], dt: f32) -> Seed {
        let q = bike.orientation();
        let omega = q * Vec3::new(bike.pitch_rate, bike.yaw_rate, bike.roll_rate);
        let positions = local.map(|p| bike.position + q * p);
        let velocities = std::array::from_fn(|i| {
            let limb = self.previous.as_ref().map_or(Vec3::ZERO, |old| {
                ((local[i] - old[i]) / dt).clamp_length_max(12.0)
            });
            bike.velocity + omega.cross(q * local[i]) + q * limb
        });
        self.previous = Some(local);
        Seed {
            positions,
            velocities,
        }
    }

    pub fn activate(&mut self, seed: Seed) {
        self.body = Some(Body::new(seed));
    }

    pub fn step(&mut self, bike: &Bike, dt: f32) {
        if let Some(body) = &mut self.body {
            body.step(bike, dt);
        }
    }
}

impl Body {
    fn new(seed: Seed) -> Self {
        let mut b = Self {
            positions: seed.positions,
            velocities: seed.velocities,
            constraints: [Constraint::default(); 48],
            constraint_count: 0,
            radii: [0.055; COUNT],
            inverse_mass: [1.0; COUNT],
            sleeping: false,
            quiet_time: 0.0,
        };
        for p in [P::Hip, P::Waist, P::Shoulder] {
            b.radii[index(p)] = 0.13;
            b.inverse_mass[index(p)] = 0.15;
        }
        b.radii[index(P::Head)] = 0.14;
        b.inverse_mass[index(P::Head)] = 0.3;
        for p in [P::KneeL, P::KneeR, P::ElbowL, P::ElbowR] {
            b.radii[index(p)] = 0.065;
        }
        for (bones, _) in SKELETON {
            for &(a, c) in bones {
                b.fixed(a, c);
            }
        }
        // Pelvis, chest, feet and hands retain their shape; the spine and limbs remain articulated.
        for (a, c) in [
            (P::HipL, P::HipR),
            (P::ShoulderL, P::ShoulderR),
            (P::Waist, P::ShoulderL),
            (P::Waist, P::ShoulderR),
            (P::Neck, P::ShoulderL),
            (P::Neck, P::ShoulderR),
            (P::HeelL, P::ToeL),
            (P::HeelR, P::ToeR),
        ] {
            b.fixed(a, c);
        }
        for (a, joint, c) in [
            (P::HipL, P::KneeL, P::AnkleL),
            (P::HipR, P::KneeR, P::AnkleR),
            (P::ShoulderL, P::ElbowL, P::WristL),
            (P::ShoulderR, P::ElbowR, P::WristR),
            (P::Hip, P::Waist, P::Shoulder),
            (P::Shoulder, P::Neck, P::Head),
        ] {
            let upper = b.positions[index(a)].distance(b.positions[index(joint)]);
            let lower = b.positions[index(c)].distance(b.positions[index(joint)]);
            let min = if matches!(joint, P::Waist | P::Neck) {
                (upper + lower) * 0.8
            } else {
                (upper - lower).abs().max((upper + lower) * 0.2)
            };
            b.add(a, c, min, upper + lower);
        }
        // Ball-joint cone limits keep arms/legs from folding through the torso, without pose motors.
        for (a, c) in [
            (P::Hip, P::KneeL),
            (P::Hip, P::KneeR),
            (P::Shoulder, P::ElbowL),
            (P::Shoulder, P::ElbowR),
        ] {
            let rest = b.positions[index(a)].distance(b.positions[index(c)]);
            b.add(a, c, rest * 0.55, rest * 1.35);
        }
        b
    }

    fn add(&mut self, a: P, b: P, min: f32, max: f32) {
        self.constraints[self.constraint_count] = Constraint {
            a: index(a),
            b: index(b),
            min,
            max,
        };
        self.constraint_count += 1;
    }
    fn fixed(&mut self, a: P, b: P) {
        let length = self.positions[index(a)].distance(self.positions[index(b)]);
        self.add(a, b, length, length);
    }

    fn step(&mut self, bike: &Bike, dt: f32) {
        if self.sleeping || dt <= 0.0 {
            return;
        }
        let speed = self
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0_f32, f32::max);
        let count = (dt * 240.0)
            .ceil()
            .max((speed * dt / 0.04).ceil())
            .clamp(1.0, 32.0) as usize;
        let h = dt / count as f32;
        for _ in 0..count {
            self.substep(bike, h);
        }
    }

    fn substep(&mut self, bike: &Bike, dt: f32) {
        for i in 0..COUNT {
            self.velocities[i].y -= 9.81 * dt;
            self.velocities[i] *= (-0.2 * dt).exp();
            self.positions[i] += self.velocities[i] * dt;
        }
        let incoming = self.velocities;
        let mut normals = [Vec3::ZERO; COUNT];
        let mut bike_normals = [Vec3::ZERO; COUNT];
        let mut bike_velocities = [Vec3::ZERO; COUNT];
        let bike_q = bike.orientation();
        let bike_omega = bike_q * Vec3::new(bike.pitch_rate, bike.yaw_rate, bike.roll_rate);
        for _ in 0..20 {
            for k in 0..self.constraint_count {
                let c = self.constraints[k];
                let d = self.positions[c.b] - self.positions[c.a];
                let length = d.length();
                if length < 1e-6 {
                    continue;
                }
                let correction = d * ((length - length.clamp(c.min, c.max)) / length);
                let weight = self.inverse_mass[c.a] + self.inverse_mass[c.b];
                self.positions[c.a] += correction * (self.inverse_mass[c.a] / weight);
                self.positions[c.b] -= correction * (self.inverse_mass[c.b] / weight);
            }
            for i in 0..COUNT {
                let (depth, n) = terrain_contact(self.positions[i], self.radii[i]);
                if depth > 0.0 {
                    self.positions[i] += n * depth;
                    normals[i] = n;
                }
            }
            // Capsule interiors also collide: long thighs/forearms cannot cut through a ramp.
            for (bones, _) in SKELETON {
                for &(a, b) in bones {
                    let (a, b) = (index(a), index(b));
                    let radius = self.radii[a].min(self.radii[b]);
                    let samples =
                        (self.positions[a].distance(self.positions[b]) / 0.07).ceil() as usize;
                    for k in 1..samples {
                        let t = k as f32 / samples as f32;
                        let centre = self.positions[a].lerp(self.positions[b], t);
                        let (depth, n) = terrain_contact(centre, radius);
                        if depth <= 0.0 {
                            continue;
                        }
                        let wa = self.inverse_mass[a] * (1.0 - t);
                        let wb = self.inverse_mass[b] * t;
                        let denom = wa * (1.0 - t) + wb * t;
                        self.positions[a] += n * (depth * wa / denom);
                        self.positions[b] += n * (depth * wb / denom);
                        normals[a] = n;
                        normals[b] = n;
                    }
                }
            }
            // Bike contacts act on the detached rider, never a hidden attachment to the saddle.
            for i in 0..COUNT {
                for shape in bike.collision_pose.bodies.iter().filter(|s| !s.rider) {
                    let centre = bike.position + bike_q * shape.offset;
                    if let Some(n) = self.separate(i, centre, shape.radius) {
                        bike_normals[i] = n;
                        bike_velocities[i] =
                            bike.velocity + bike_omega.cross(centre - bike.position);
                    }
                }
                for (k, hub) in bike.collision_pose.wheel_rest.iter().enumerate() {
                    let centre = bike.position + bike_q * *hub;
                    // Tire ring, not a solid wheel disk.
                    let axle = bike_q * bike.collision_pose.wheel_axes[k];
                    let d = self.positions[i] - centre;
                    let radial = d - axle * d.dot(axle);
                    let ring = centre + radial.normalize_or_zero() * (WHEEL_RADIUS - 0.05);
                    if let Some(n) = self.separate(i, ring, 0.05) {
                        bike_normals[i] = n;
                        bike_velocities[i] = bike.velocity + bike_omega.cross(ring - bike.position);
                    }
                }
                let (depth, n) = terrain_contact(self.positions[i], self.radii[i]);
                if depth > 0.0 {
                    self.positions[i] += n * depth;
                    normals[i] = n;
                }
            }
        }
        // Split impulses: penetration repair must never become launch velocity. Solve joint and
        // contact velocities separately; only real incoming momentum supplies restitution.
        let mut normal_impulses = [0.0_f32; COUNT];
        let mut directions = [Vec3::ZERO; 48];
        let mut lengths = [0.0_f32; 48];
        for (k, c) in self.constraints[..self.constraint_count].iter().enumerate() {
            let d = self.positions[c.b] - self.positions[c.a];
            lengths[k] = d.length();
            directions[k] = d.normalize_or_zero();
        }
        // Passive joint friction removes limb oscillation without driving a rest pose or changing
        // the centre-of-mass velocity.
        let joint_damping = 1.0 - (-3.0 * dt).exp();
        for c in &self.constraints[..self.constraint_count] {
            if c.min != c.max {
                continue;
            }
            let relative = self.velocities[c.b] - self.velocities[c.a];
            let impulse =
                relative * (joint_damping / (self.inverse_mass[c.a] + self.inverse_mass[c.b]));
            self.velocities[c.a] += impulse * self.inverse_mass[c.a];
            self.velocities[c.b] -= impulse * self.inverse_mass[c.b];
        }
        for _ in 0..100 {
            let mut residual = 0.0_f32;
            for (k, c) in self.constraints[..self.constraint_count].iter().enumerate() {
                let length = lengths[k];
                let n = directions[k];
                let relative = (self.velocities[c.b] - self.velocities[c.a]).dot(n);
                let constrained = c.min == c.max
                    || (length <= c.min + 0.001 && relative < 0.0)
                    || (length >= c.max - 0.001 && relative > 0.0);
                if constrained {
                    residual = residual.max(relative.abs());
                    let impulse = relative / (self.inverse_mass[c.a] + self.inverse_mass[c.b]);
                    self.velocities[c.a] += n * (impulse * self.inverse_mass[c.a]);
                    self.velocities[c.b] -= n * (impulse * self.inverse_mass[c.b]);
                }
            }
            for i in 0..COUNT {
                let n = normals[i];
                if n != Vec3::ZERO {
                    let vn = incoming[i].dot(n);
                    let bounce = if vn < -1.0 { -vn * 0.12 } else { 0.0 };
                    let impulse = (bounce - self.velocities[i].dot(n)).max(0.0);
                    residual = residual.max(impulse);
                    self.velocities[i] += n * impulse;
                    normal_impulses[i] += impulse;
                }
                let n = bike_normals[i];
                if n != Vec3::ZERO {
                    let closing = (self.velocities[i] - bike_velocities[i]).dot(n);
                    residual = residual.max((-closing).max(0.0));
                    self.velocities[i] -= n * closing.min(0.0);
                }
            }
            if residual < 1e-4 {
                break;
            }
        }
        for i in 0..COUNT {
            let n = normals[i];
            if n != Vec3::ZERO {
                let normal = n * self.velocities[i].dot(n);
                let tangent = self.velocities[i] - normal;
                let speed = tangent.length();
                self.velocities[i] = normal
                    + tangent * (1.0 - (0.65 * normal_impulses[i] / speed.max(1e-6)).min(1.0));
            }
        }
        let supported = normals.iter().any(|&n| n != Vec3::ZERO);
        let quiet = self.velocities.iter().all(|v| v.length() < 0.12);
        self.quiet_time = if supported && quiet {
            self.quiet_time + dt
        } else {
            0.0
        };
        if self.quiet_time > 0.5 {
            self.sleeping = true;
            self.velocities.fill(Vec3::ZERO);
        }
    }

    fn separate(&mut self, i: usize, centre: Vec3, radius: f32) -> Option<Vec3> {
        let d = self.positions[i] - centre;
        let reach = self.radii[i] + radius;
        let length = d.length();
        if length < reach {
            let n = d.try_normalize().unwrap_or(Vec3::Y);
            self.positions[i] = centre + n * reach;
            Some(n)
        } else {
            None
        }
    }
}

fn terrain_contact(p: Vec3, radius: f32) -> (f32, Vec3) {
    let eps = 0.05;
    let dx = (terrain_height(p.x + eps, p.z) - terrain_height(p.x - eps, p.z)) / (2.0 * eps);
    let dz = (terrain_height(p.x, p.z + eps) - terrain_height(p.x, p.z - eps)) / (2.0 * eps);
    let n = Vec3::new(-dx, 1.0, -dz).normalize();
    (radius - (p.y - terrain_height(p.x, p.z)) * n.y, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::AnimationState;

    #[test]
    fn penetration_repair_does_not_launch_a_stationary_rider() {
        let mut bike = Bike::default();
        let local = crate::scene::rider_points(&bike, &AnimationState::default());
        let positions = local
            .map(|p| Vec3::new(20.0, 0.4, 8.0) + Quat::from_rotation_z(std::f32::consts::PI) * p);
        let mut body = Body::new(Seed {
            positions,
            velocities: [Vec3::ZERO; COUNT],
        });
        bike.position.x = -50.0;
        let dt = 1.0 / 120.0;
        body.step(&bike, dt);
        let energy = (0..COUNT)
            .map(|i| body.velocities[i].length_squared() / body.inverse_mass[i])
            .sum::<f32>();
        let freefall_energy = body
            .inverse_mass
            .iter()
            .map(|m| (9.81 * dt).powi(2) / m)
            .sum::<f32>();
        assert!(
            energy <= freefall_energy * 1.01,
            "penetration injected kinetic energy: {energy} > {freefall_energy}"
        );
        assert!(body.positions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn crash_detaches_articulates_and_keeps_bones_above_ground() {
        let mut bike = Bike::default();
        bike.position += Vec3::new(20.0, 2.0, 0.0);
        bike.velocity = Vec3::new(3.0, -5.0, -4.0);
        bike.pitch_rate = 2.0;
        let local = crate::scene::rider_points(&bike, &AnimationState::default());
        let mut ragdoll = Ragdoll::default();
        let seed = ragdoll.sample(&bike, local, 1.0 / 120.0);
        let original = seed.positions;
        ragdoll.activate(seed);
        bike.position.x = -50.0; // The wreck has its own trajectory, not a rider parent constraint.
        for _ in 0..1200 {
            ragdoll.step(&bike, 1.0 / 120.0);
        }
        let body = ragdoll.body.as_ref().unwrap();
        assert!(
            body.positions[index(P::Hip)].distance(bike.position) > 20.0,
            "rider must move independently of the wreck"
        );
        let before = original[index(P::KneeL)].distance(original[index(P::Shoulder)]);
        let after = body.positions[index(P::KneeL)].distance(body.positions[index(P::Shoulder)]);
        assert!(
            (before - after).abs() > 0.05,
            "pose must articulate, not rigidly tumble"
        );
        for c in &body.constraints[..body.constraint_count] {
            let length = body.positions[c.a].distance(body.positions[c.b]);
            assert!(
                length >= c.min - 0.012 && length <= c.max + 0.012,
                "joint length {length} outside {}..{}",
                c.min,
                c.max
            );
        }
        for i in 0..COUNT {
            assert!(body.positions[i].is_finite());
            assert!(terrain_contact(body.positions[i], body.radii[i]).0 < 0.008);
        }
        assert!(
            body.velocities.iter().all(|v| v.length() < 0.2),
            "wreck must settle; joint velocities: {:?}",
            body.velocities
        );
        ragdoll.reset();
        assert!(ragdoll.body.is_none());
    }
}
