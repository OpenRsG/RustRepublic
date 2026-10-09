//! Articulated rigid-body ragdolls solved with extended position-based dynamics (Müller et al.
//! 2020, one iteration per 1/2400 s substep): segments with mass and inertia pinned at joints with
//! anatomical swing/twist ranges that break under load, collision capsules for every pair of
//! segments that is not jointed, Coulomb friction on the terrain, and bodies that are *driven*
//! (a bike carrying its own gravity and terrain contact) which segments can push and be held by.

use bevy::math::{Mat3, Quat, Vec3};

use crate::bike::terrain_height;

pub(crate) const GRAVITY: f32 = 9.81;
/// Longest solver substep, s.
pub(crate) const SUBSTEP: f32 = 1.0 / 2400.0;
pub(crate) const MAX_SUBSTEPS: usize = 64;
/// Spacing of terrain probes along a capsule, m.
pub(crate) const SAMPLE_SPACING: f32 = 0.08;
pub(crate) const AIR_DAMPING: f32 = 0.1;
/// Passive joint friction (relative spin decay rate, 1/s): muscle tone without driving a pose.
pub(crate) const JOINT_DAMPING: f32 = 12.0;
/// Spin decay of a segment lying in the snow, 1/s.
pub(crate) const ROLL_DAMPING: f32 = 5.0;
pub(crate) const RESTITUTION: f32 = 0.1;
/// Coulomb coefficients: body and boots on snow, limb on limb, skis along and across their axis.
pub(crate) const BODY_FRICTION: f32 = 0.6;
pub(crate) const LIMB_FRICTION: f32 = 0.4;
pub(crate) const SKI_ALONG: f32 = 0.05;
pub(crate) const SKI_ACROSS: f32 = 0.8;
pub(crate) const SLEEP_SPEED: f32 = 0.12;
pub(crate) const SLEEP_TIME: f32 = 0.6;
/// Averaging time of the torque loading a joint at its limit: shorter spikes are absorbed.
pub(crate) const LOAD_TIME: f32 = 0.02;
pub(crate) const DEG: f32 = std::f32::consts::PI / 180.0;
/// How far past its anatomical range a broken joint goes before soft tissue stops it.
pub(crate) const BROKEN_SLACK: f32 = 50.0 * DEG;
/// Position-only passes that remove seed overlaps before the first step.
pub(crate) const SETTLE_PASSES: usize = 40;
/// Rate at which a joint handed over outside its range is eased back into it, rad/s.
pub(crate) const SEED_RELAX: f32 = 4.0;
/// Past the end of its range a joint gives like a stiff spring (ligaments, the boot shell), so the
/// torque holding it is physical rather than a one-substep impulse; this much further on it stops
/// dead.
pub(crate) const HARD_STOP: f32 = 6.0 * DEG;

/// Anatomical range of a joint for the right side (left mirrors it), degrees, about the child
/// segment's frame relative to the parent's: `flex` swings about X, `side` about Z, `twist` turns
/// about the segment's own Y. A ball joint's swing stays inside the ellipse through the four
/// extremes; a `hinge` limits flex and side independently. `centre` (X, Z swing) shifts the middle
/// of the range off the upright stance. `strength`: torque about the flex, side and twist axes
/// that breaks the joint, N m (combined as an ellipsoid). `stiffness`: how hard the end of range
/// pushes back, N m per rad.
#[derive(Clone, Copy)]
pub(crate) struct Limit {
    pub centre: [f32; 2],
    pub flex: [f32; 2],
    pub side: [f32; 2],
    pub twist: [f32; 2],
    pub hinge: bool,
    pub strength: [f32; 3],
    pub stiffness: f32,
}

impl Limit {
    /// Joint frame in the parent's: the centre of the range.
    pub fn frame(&self, mirror: bool) -> Quat {
        let m = if mirror { -1.0 } else { 1.0 };
        Quat::from_scaled_axis(Vec3::new(self.centre[0], 0.0, self.centre[1] * m) * DEG)
    }
}

/// Ligament end-of-range stiffness, N m per rad.
pub(crate) const LIGAMENT: f32 = 2500.0;

pub(crate) const fn ball(
    flex: [f32; 2],
    side: [f32; 2],
    twist: [f32; 2],
    strength: [f32; 3],
) -> Limit {
    Limit {
        centre: [0.0, 0.0],
        flex,
        side,
        twist,
        hinge: false,
        strength,
        stiffness: LIGAMENT,
    }
}

pub(crate) const fn hinged(
    flex: [f32; 2],
    side: [f32; 2],
    twist: [f32; 2],
    strength: [f32; 3],
) -> Limit {
    Limit {
        hinge: true,
        ..ball(flex, side, twist, strength)
    }
}

/// Spine: forward bend is negative flex, a right side-bend negative side, a right turn negative twist.
/// The trunk's muscles and ribcage share the load, so it is far stronger than a single limb joint.
pub(crate) const LUMBAR: Limit = ball(
    [-50.0, 25.0],
    [-25.0, 25.0],
    [-12.0, 12.0],
    [600.0, 600.0, 300.0],
);
pub(crate) const THORACIC: Limit = ball(
    [-35.0, 15.0],
    [-20.0, 20.0],
    [-35.0, 35.0],
    [600.0, 600.0, 300.0],
);
pub(crate) const NECK: Limit = ball(
    [-50.0, 60.0],
    [-40.0, 40.0],
    [-75.0, 75.0],
    [225.0, 180.0, 120.0],
);
/// Shoulder (humerus on thorax), measured about an arm raised 45 deg sideways and 20 deg forward:
/// flexion to about 170 deg, extension 60, abduction 180, adduction across the chest; humeral
/// rotation about +-80.
pub(crate) const SHOULDER: Limit = Limit {
    centre: [20.0, 45.0],
    ..ball(
        [-90.0, 150.0],
        [-90.0, 140.0],
        [-80.0, 80.0],
        [120.0, 120.0, 70.0],
    )
};
/// Elbow: a hinge from straight to 150 deg; pronation/supination as twist.
pub(crate) const ELBOW: Limit = hinged(
    [-3.0, 150.0],
    [-5.0, 5.0],
    [-80.0, 80.0],
    [160.0, 90.0, 80.0],
);
pub(crate) const WRIST: Limit = ball(
    [-70.0, 70.0],
    [-25.0, 25.0],
    [-10.0, 10.0],
    [110.0, 90.0, 70.0],
);
/// Hip: flexion 120, extension 20, abduction 45, adduction 30, internal 35 / external 45 rotation.
pub(crate) const HIP: Limit = ball(
    [-20.0, 120.0],
    [-30.0, 45.0],
    [-45.0, 35.0],
    [420.0, 420.0, 280.0],
);
/// Knee: flexion (negative) to 150 deg, barely any hyperextension, side play or tibial rotation.
/// Varus/valgus and rotation tear it first.
pub(crate) const KNEE: Limit = hinged(
    [-150.0, 3.0],
    [-5.0, 5.0],
    [-20.0, 20.0],
    [250.0, 150.0, 100.0],
);

#[derive(Clone, Copy)]
pub(crate) struct Joint {
    pub parent: usize,
    pub child: usize,
    /// Anchor in each segment's local frame.
    pub pa: Vec3,
    pub pc: Vec3,
    /// Joint frame in the parent's local frame (the centre of the range).
    pub frame: Quat,
    pub limit: Option<Limit>,
    pub mirror: bool,
    pub name: &'static str,
    /// Averaged torque at the limit about the joint frame's flex, side and twist axes, N m; of a
    /// gripped pole, in the first slot, the averaged pull on the grip, N.
    pub load: [f32; 3],
    /// Position constraint impulse of `attach` this substep, kg m.
    pub impulse: f32,
    /// A broken joint's range opens by `BROKEN_SLACK`; a released detachable joint lets go.
    pub broken: bool,
    /// Gear (binding, pole grip) that lets go of what it holds instead of breaking.
    pub detachable: bool,
    /// Pull at which a detachable joint without a range lets go, N; infinite: it releases only
    /// through its range limit.
    pub grip: f32,
    /// Extra range the joint starts with when the animation handed it over outside its anatomical
    /// range; it closes at `SEED_RELAX`, so the limb is eased in rather than snapped.
    pub allow: f32,
}

impl Joint {
    pub fn slack(&self) -> f32 {
        if self.broken { BROKEN_SLACK } else { 0.0 }.max(self.allow)
    }

    pub fn holds(&self) -> bool {
        !(self.detachable && self.broken)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Rigid {
    pub x: Vec3,
    pub q: Quat,
    pub v: Vec3,
    pub w: Vec3,
    /// Constraint displacement and rotation this substep; velocities come from these rather than
    /// from position differences, which f32 rounds away far from the origin.
    pub dx: Vec3,
    pub dphi: Vec3,
    pub inv_mass: f32,
    /// Inverse inertia in the local frame.
    pub inv_inertia: Mat3,
    /// Radius of a sphere round `x` holding every collision capsule.
    pub bound: f32,
    /// Carried by something else (a bike with its own gravity, drag and impacts): it moves with its
    /// velocity alone and the terrain only holds it up, while segments push and hold it.
    pub driven: bool,
}

impl Rigid {
    pub fn inv_inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.q);
        r * self.inv_inertia * r.transpose()
    }

    pub fn point(&self, local: Vec3) -> Vec3 {
        self.x + self.q * local
    }

    pub fn velocity_at(&self, r: Vec3) -> Vec3 {
        self.v + self.w.cross(r)
    }

    /// Small rotation by the rotation vector `phi`.
    pub fn turn(&mut self, phi: Vec3) {
        self.q = (self.q + Quat::from_xyzw(phi.x, phi.y, phi.z, 0.0) * self.q * 0.5).normalize();
        self.dphi += phi;
    }

    pub fn shift_by(&mut self, d: Vec3) {
        self.x += d;
        self.dx += d;
    }

    /// Inverse mass felt along `n` at offset `r` from the centre of mass.
    pub fn weight(&self, r: Vec3, n: Vec3) -> f32 {
        let rn = r.cross(n);
        self.inv_mass + rn.dot(self.inv_inertia_world() * rn)
    }
}

/// Collision capsule fixed to a segment, endpoints in its local frame. `ignore`: a segment it may
/// touch (the grip section of a pole lies along its own wrist).
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub body: usize,
    pub a: Vec3,
    pub b: Vec3,
    pub r: f32,
    pub ski: bool,
    pub ignore: Option<usize>,
}

impl Shape {
    pub fn meets(&self, other: &Shape) -> bool {
        self.ignore != Some(other.body) && other.ignore != Some(self.body)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Contact {
    pub a: usize,
    /// `None`: the snow.
    pub b: Option<usize>,
    pub ra: Vec3,
    pub rb: Vec3,
    /// Pushes `a` away from `b`.
    pub n: Vec3,
    /// Normal impulse of the position solve (kg m).
    pub lambda: f32,
    pub mu: f32,
    pub restitution: f32,
    /// World ski axis for anisotropic friction.
    pub ski: Option<Vec3>,
}

/// A segment's mass element or collision capsule in world space, from the pose.
pub(crate) struct Part {
    pub body: usize,
    pub a: Vec3,
    pub b: Vec3,
    pub r: f32,
    pub mass: f32,
    pub collide: bool,
    pub ski: bool,
    pub ignore: Option<usize>,
}

pub(crate) fn part(body: impl Into<usize>, a: Vec3, b: Vec3, r: f32, mass: f32) -> Part {
    Part {
        body: body.into(),
        a,
        b,
        r,
        mass,
        collide: true,
        ski: false,
        ignore: None,
    }
}

pub(crate) fn unit(v: Vec3) -> Vec3 {
    v.normalize_or_zero()
}

/// Rotation with X along `x` made perpendicular to `y`, Y along `y`, Z = X x Y.
pub(crate) fn basis(x: Vec3, y: Vec3) -> Quat {
    let y = y.normalize_or(Vec3::Y);
    let x = x
        .reject_from_normalized(y)
        .try_normalize()
        .unwrap_or_else(|| y.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y))).normalize()
}

pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Flexion axis (X) of a two-bone limb bent at `mid`: `(end - mid) x (root - mid)`, blended towards
/// `fallback` as the limb straightens and its bend plane becomes undefined.
pub(crate) fn hinge(root: Vec3, mid: Vec3, end: Vec3, fallback: Vec3) -> Vec3 {
    let (a, b) = (root - mid, end - mid);
    let c = b.cross(a);
    let bend = c.length() / (a.length() * b.length()).max(1e-6);
    let w = smoothstep(0.05, 0.3, bend);
    (unit(c) * w + unit(fallback) * (1.0 - w)).normalize_or(fallback)
}

/// Parameters `(s, t)` of the closest points between segments `p0`-`p1` and `q0`-`q1`.
pub(crate) fn closest_params(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> (f32, f32) {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.length_squared(), d2.length_squared(), d2.dot(r));
    if a < 1e-9 && e < 1e-9 {
        return (0.0, 0.0);
    }
    if a < 1e-9 {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    if e < 1e-9 {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(d2);
    let den = a * e - b * b;
    let mut s = if den > 1e-9 {
        ((b * f - c * e) / den).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

/// Centre of mass of each of `n` segments.
pub(crate) fn centres(parts: &[Part], n: usize) -> Vec<Vec3> {
    let mut sum = vec![(Vec3::ZERO, 0.0_f32); n];
    for q in parts {
        let s = &mut sum[q.body];
        s.0 += (q.a + q.b) * (0.5 * q.mass);
        s.1 += q.mass;
    }
    sum.into_iter().map(|(m, w)| m / w).collect()
}

pub(crate) fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Inertia of a rod or ball `q` about the point `about`, world axes.
pub(crate) fn inertia(q: &Part, about: Vec3) -> Mat3 {
    let (m, r) = (q.mass, q.r);
    let d = q.b - q.a;
    let l = d.length();
    let own = if l < 1e-6 {
        Mat3::IDENTITY * (0.4 * m * r * r)
    } else {
        let u = d / l;
        let perp = m * (3.0 * r * r + l * l) / 12.0;
        Mat3::IDENTITY * perp + outer(u, u) * (0.5 * m * r * r - perp)
    };
    let o = (q.a + q.b) * 0.5 - about;
    own + (Mat3::IDENTITY * o.length_squared() - outer(o, o)) * m
}

/// Clamps the joint rotation `q` (child frame in the parent's joint frame) into `l`, widened by
/// `slack` rad. A ball joint's swing (about X and Z) is pulled radially into the ellipse through
/// the four range extremes, a hinge's flex and side are clamped separately; twist (about Y) is
/// clamped into its interval. `mirror` for the left side.
pub(crate) fn clamp_rotation(q: Quat, l: &Limit, mirror: bool, slack: f32) -> Quat {
    let q = if q.w < 0.0 { -q } else { q };
    let m = if mirror { -1.0 } else { 1.0 };
    let len = (q.y * q.y + q.w * q.w).sqrt();
    let twist = if len < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_xyzw(0.0, q.y / len, 0.0, q.w / len)
    };
    let swing = (q * twist.inverse()).to_scaled_axis();
    let (fx, sz, tw) = (swing.x, swing.z * m, 2.0 * twist.y.atan2(twist.w) * m);
    let open = |r: [f32; 2], cap: f32| {
        (
            (r[0] * DEG - slack).max(-cap),
            (r[1] * DEG + slack).min(cap),
        )
    };
    let swing_cap = 160.0 * DEG;
    let (f0, f1) = open(l.flex, swing_cap);
    let (s0, s1) = open(l.side, swing_cap);
    let (t0, t1) = open(l.twist, 170.0 * DEG);
    let (fx, sz) = if l.hinge {
        (fx.clamp(f0, f1), sz.clamp(s0, s1))
    } else {
        let ax = (if fx >= 0.0 { f1 } else { -f0 }).max(DEG);
        let az = (if sz >= 0.0 { s1 } else { -s0 }).max(DEG);
        let e = (fx / ax).powi(2) + (sz / az).powi(2);
        let k = if e > 1.0 { e.sqrt().recip() } else { 1.0 };
        (fx * k, sz * k)
    };
    Quat::from_scaled_axis(Vec3::new(fx, 0.0, sz * m)) * Quat::from_rotation_y(tw.clamp(t0, t1) * m)
}

/// Segments, joints, collision capsules and contacts of one ragdoll.
pub(crate) struct Solver {
    pub bodies: Vec<Rigid>,
    pub shapes: Vec<Shape>,
    /// Shape index range of each segment.
    pub range: Vec<(usize, usize)>,
    /// Segment pairs that collide (not jointed).
    pub pairs: Vec<(usize, usize)>,
    pub joints: Vec<Joint>,
    pub contacts: Vec<Contact>,
    /// Velocities at the start of the substep; kept between substeps to avoid reallocating.
    before: Vec<(Vec3, Vec3)>,
    pub injuries: Vec<&'static str>,
    pub sleeping: bool,
    pub quiet_time: f32,
}

impl Solver {
    /// One segment per entry of `f` (its frame) and `com` (its centre of mass), made of the `parts`
    /// that name it. Joints are added with `pin`, then `collide_unjointed`.
    pub fn new(parts: &[Part], f: &[Quat], com: &[Vec3]) -> Self {
        let mut bodies = Vec::with_capacity(f.len());
        for k in 0..f.len() {
            let own = parts.iter().filter(|q| q.body == k);
            let mass: f32 = own.clone().map(|q| q.mass).sum();
            let world = own.fold(Mat3::ZERO, |sum, q| sum + inertia(q, com[k]));
            let r = Mat3::from_quat(f[k]);
            // Floor keeps thin poles and skis from spinning arbitrarily fast about their axis.
            let local = r.transpose() * world * r + Mat3::IDENTITY * (mass * 4e-4);
            bodies.push(Rigid {
                x: com[k],
                q: f[k],
                v: Vec3::ZERO,
                w: Vec3::ZERO,
                dx: Vec3::ZERO,
                dphi: Vec3::ZERO,
                inv_mass: 1.0 / mass,
                inv_inertia: local.inverse(),
                bound: 0.0,
                driven: false,
            });
        }
        let mut shapes = Vec::new();
        let mut range = vec![(0, 0); f.len()];
        for (k, rk) in range.iter_mut().enumerate() {
            let start = shapes.len();
            let inv = f[k].inverse();
            for q in parts.iter().filter(|q| q.body == k && q.collide) {
                let (a, b) = (inv * (q.a - com[k]), inv * (q.b - com[k]));
                bodies[k].bound = bodies[k].bound.max(a.length() + q.r).max(b.length() + q.r);
                shapes.push(Shape {
                    body: k,
                    a,
                    b,
                    r: q.r,
                    ski: q.ski,
                    ignore: q.ignore,
                });
            }
            *rk = (start, shapes.len());
        }
        Self {
            before: Vec::with_capacity(bodies.len()),
            bodies,
            shapes,
            range,
            pairs: Vec::new(),
            joints: Vec::new(),
            contacts: Vec::new(),
            injuries: Vec::new(),
            sleeping: false,
            quiet_time: 0.0,
        }
    }

    /// A joint between `parent` and `child` at world point `at`, free and unbreakable; the caller
    /// fills in its range.
    pub fn pin(&self, parent: usize, child: usize, at: Vec3) -> Joint {
        let (bp, bc) = (&self.bodies[parent], &self.bodies[child]);
        Joint {
            parent,
            child,
            pa: bp.q.inverse() * (at - bp.x),
            pc: bc.q.inverse() * (at - bc.x),
            frame: Quat::IDENTITY,
            limit: None,
            mirror: false,
            name: "",
            load: [0.0; 3],
            impulse: 0.0,
            broken: false,
            detachable: false,
            grip: f32::INFINITY,
            allow: 0.0,
        }
    }

    /// Every pair of segments that is not jointed, and not in `joined` either, collides.
    pub fn collide_unjointed(&mut self, joined: &[(usize, usize)]) {
        let n = self.bodies.len();
        let touching = |a: usize, b: usize| {
            let pair = |p: (usize, usize)| p == (a, b) || p == (b, a);
            self.joints.iter().any(|j| pair((j.parent, j.child))) || joined.iter().any(|&p| pair(p))
        };
        self.pairs = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .filter(|&(a, b)| !touching(a, b))
            .collect();
    }

    /// Position-only passes over the seed pose: overlaps and out-of-range joints left by the
    /// animation are removed without turning the correction into speed.
    pub fn settle(&mut self) {
        for _ in 0..SETTLE_PASSES {
            self.solve_positions(1.0, false);
        }
        self.contacts.clear();
    }

    pub fn step(&mut self, dt: f32) {
        if self.sleeping || dt <= 0.0 {
            return;
        }
        let count = ((dt / SUBSTEP - 1e-3).ceil() as usize).clamp(1, MAX_SUBSTEPS);
        let h = dt / count as f32;
        for _ in 0..count {
            self.substep(h);
            if self.sleeping {
                break;
            }
        }
    }

    pub fn substep(&mut self, h: f32) {
        let mut before = std::mem::take(&mut self.before);
        before.clear();
        before.extend(self.bodies.iter().map(|b| (b.v, b.w)));
        let damping = (-AIR_DAMPING * h).exp();
        for b in &mut self.bodies {
            if !b.driven {
                b.v.y -= GRAVITY * h;
                b.v *= damping;
                b.w *= damping;
            }
            b.x += b.v * h;
            let w = b.w;
            b.turn(w * h);
            b.dx = Vec3::ZERO;
            b.dphi = Vec3::ZERO;
        }
        for j in &mut self.joints {
            j.allow = (j.allow - SEED_RELAX * h).max(0.0);
        }
        self.contacts.clear();
        self.solve_positions(h, true);
        for b in &mut self.bodies {
            b.v += b.dx / h;
            b.w += b.dphi / h;
        }
        self.solve_velocities(&before, h);
        self.before = before;

        let supported = self.contacts.iter().any(|c| c.b.is_none());
        // At rest when every capsule end is slow: a thin pole may keep spinning about its own
        // axis, which moves nothing.
        let quiet = self.shapes.iter().all(|s| {
            let b = &self.bodies[s.body];
            [s.a, s.b]
                .iter()
                .all(|&p| b.velocity_at(b.q * p).length() < SLEEP_SPEED)
        });
        self.quiet_time = if supported && quiet {
            self.quiet_time + h
        } else {
            0.0
        };
        if self.quiet_time > SLEEP_TIME {
            self.sleeping = true;
            for b in &mut self.bodies {
                b.v = Vec3::ZERO;
                b.w = Vec3::ZERO;
            }
        }
    }

    /// One pass over every constraint. `h` scales joint loads; `live` false (settling) neither
    /// loads nor breaks joints. Segment and snow contacts go last: whatever the joints leave,
    /// nothing ends a substep inside anything else.
    pub fn solve_positions(&mut self, h: f32, live: bool) {
        for j in &mut self.joints {
            j.impulse = 0.0;
        }
        for k in 0..self.joints.len() {
            self.attach(k);
        }
        for k in 0..self.joints.len() {
            self.limit(k, h, live);
        }
        for k in 0..self.joints.len() {
            self.attach(k);
            if live {
                self.pull(k, h);
            }
        }
        for k in 0..self.joints.len() {
            let joint = self.joints[k];
            if let Some(l) = joint.limit.filter(|_| joint.holds()) {
                let (_, stop) = self.overshoot(&joint, &l, HARD_STOP);
                self.twist_apart(joint.child, joint.parent, stop, 0.0, h);
            }
        }
        self.collide_segments();
        self.collide_snow();
    }

    /// Moves the point at offset `ra` of segment `a` by `delta` relative to the point at `rb` of
    /// `b` (`None`: the snow), split by generalized inverse mass. Returns the impulse, kg m.
    pub fn shift(&mut self, a: usize, b: Option<usize>, ra: Vec3, rb: Vec3, delta: Vec3) -> f32 {
        let c = delta.length();
        if c < 1e-9 {
            return 0.0;
        }
        let n = delta / c;
        let wa = self.bodies[a].weight(ra, n);
        let wb = b.map_or(0.0, |b| self.bodies[b].weight(rb, n));
        if wa + wb <= 0.0 {
            return 0.0;
        }
        let p = n * (c / (wa + wb));
        let body = &mut self.bodies[a];
        body.shift_by(p * body.inv_mass);
        let turn = body.inv_inertia_world() * ra.cross(p);
        body.turn(turn);
        if let Some(b) = b {
            let body = &mut self.bodies[b];
            body.shift_by(-p * body.inv_mass);
            let turn = body.inv_inertia_world() * rb.cross(p);
            body.turn(-turn);
        }
        c / (wa + wb)
    }

    /// Turns segment `a` by the rotation vector `phi` relative to `b`, split by inertia, against a
    /// spring of `compliance` (rad per N m) over the substep `h`. Returns the angular impulse,
    /// kg m^2.
    pub fn twist_apart(&mut self, a: usize, b: usize, phi: Vec3, compliance: f32, h: f32) -> f32 {
        let angle = phi.length();
        if angle < 1e-7 {
            return 0.0;
        }
        let n = phi / angle;
        let (ia, ib) = (
            self.bodies[a].inv_inertia_world(),
            self.bodies[b].inv_inertia_world(),
        );
        let lambda = angle / (n.dot(ia * n) + n.dot(ib * n) + compliance / (h * h));
        self.bodies[a].turn(ia * n * lambda);
        self.bodies[b].turn(-(ib * n * lambda));
        lambda
    }

    pub fn attach(&mut self, k: usize) {
        let j = self.joints[k];
        if !j.holds() {
            return;
        }
        let (bp, bc) = (self.bodies[j.parent], self.bodies[j.child]);
        let (wp, wc) = (bp.point(j.pa), bc.point(j.pc));
        let lambda = self.shift(j.child, Some(j.parent), wc - bc.x, wp - bp.x, wp - wc);
        self.joints[k].impulse += lambda;
    }

    /// Joint rotation: child joint frame in the parent's.
    pub fn relative(&self, j: &Joint) -> (Quat, Quat) {
        let qp = self.bodies[j.parent].q * j.frame;
        (qp, qp.inverse() * self.bodies[j.child].q)
    }

    /// Rotation vector turning the child of `j` back to its range widened by `extra` rad.
    pub fn overshoot(&self, j: &Joint, l: &Limit, extra: f32) -> (Quat, Vec3) {
        let (qp, rel) = self.relative(j);
        let target = clamp_rotation(rel, l, j.mirror, j.slack() + extra);
        let fix = qp * target * rel.inverse() * qp.inverse();
        (qp, if fix.w < 0.0 { -fix } else { fix }.to_scaled_axis())
    }

    /// Range limit: a spring of the joint's stiffness past the end of range, and a hard stop
    /// `HARD_STOP` further on.
    pub fn limit(&mut self, k: usize, h: f32, live: bool) {
        let j = self.joints[k];
        let Some(l) = j.limit.filter(|_| j.holds()) else {
            return;
        };
        let (qp, phi) = self.overshoot(&j, &l, 0.0);
        let soft = self.twist_apart(j.child, j.parent, phi, 1.0 / l.stiffness, h);
        let (_, stop) = self.overshoot(&j, &l, HARD_STOP);
        let hard = self.twist_apart(j.child, j.parent, stop, 0.0, h);
        if !live {
            return;
        }
        // Torque holding the limit, about the joint frame's flex (X), side (Z) and twist (Y) axes.
        let torque = qp.inverse()
            * (phi.normalize_or_zero() * soft + stop.normalize_or_zero() * hard)
            / (h * h);
        let j = &mut self.joints[k];
        let mut over = 0.0;
        // A binding's heel releases only when the boot levers forward off the ski (-X).
        let flex = if j.detachable {
            (-torque.x).max(0.0)
        } else {
            torque.x.abs()
        };
        for (a, t) in [flex, torque.z.abs(), torque.y.abs()]
            .into_iter()
            .enumerate()
        {
            j.load[a] += (t - j.load[a]) * (h / LOAD_TIME).min(1.0);
            over += (j.load[a] / l.strength[a]).powi(2);
        }
        if !j.broken && over > 1.0 {
            self.fail(k);
        }
    }

    /// Pulls on a gripped joint (pole in the fist): the constraint force averaged over `LOAD_TIME`
    /// pulls it out above `grip`.
    pub fn pull(&mut self, k: usize, h: f32) {
        let j = &mut self.joints[k];
        if !j.holds() || !j.grip.is_finite() {
            return;
        }
        let force = j.impulse / (h * h);
        j.load[0] += (force - j.load[0]) * (h / LOAD_TIME).min(1.0);
        if j.load[0] > j.grip {
            self.fail(k);
        }
    }

    /// A detachable joint lets go and its two segments collide from now on; any other breaks.
    pub fn fail(&mut self, k: usize) {
        let j = &mut self.joints[k];
        j.broken = true;
        if j.detachable {
            self.pairs.push((j.parent, j.child));
        } else {
            self.injuries.push(j.name);
        }
    }

    pub fn collide_segments(&mut self) {
        for k in 0..self.pairs.len() {
            let (a, b) = self.pairs[k];
            if self.bodies[a].x.distance(self.bodies[b].x)
                > self.bodies[a].bound + self.bodies[b].bound
            {
                continue;
            }
            for i in self.range[a].0..self.range[a].1 {
                for j in self.range[b].0..self.range[b].1 {
                    let (sa, sb) = (self.shapes[i], self.shapes[j]);
                    if !sa.meets(&sb) {
                        continue;
                    }
                    let (ba, bb) = (self.bodies[a], self.bodies[b]);
                    let (a0, a1, b0, b1) = (
                        ba.point(sa.a),
                        ba.point(sa.b),
                        bb.point(sb.a),
                        bb.point(sb.b),
                    );
                    let (s, t) = closest_params(a0, a1, b0, b1);
                    let (pa, pb) = (a0.lerp(a1, s), b0.lerp(b1, t));
                    let d = pa - pb;
                    let gap = d.length() - sa.r - sb.r;
                    if gap >= 0.0 {
                        continue;
                    }
                    let n = d.try_normalize().unwrap_or_else(|| unit(ba.x - bb.x));
                    let (ra, rb) = (pa - n * sa.r - ba.x, pb + n * sb.r - bb.x);
                    let lambda = self.shift(a, Some(b), ra, rb, n * -gap);
                    self.contacts.push(Contact {
                        a,
                        b: Some(b),
                        ra,
                        rb,
                        n,
                        lambda,
                        mu: LIMB_FRICTION,
                        restitution: 0.0,
                        ski: None,
                    });
                }
            }
        }
    }

    pub fn collide_snow(&mut self) {
        for i in 0..self.shapes.len() {
            let s = self.shapes[i];
            let len = (s.b - s.a).length();
            let samples = (len / SAMPLE_SPACING).ceil() as usize;
            for k in 0..=samples {
                let t = if samples == 0 {
                    0.0
                } else {
                    k as f32 / samples as f32
                };
                let body = self.bodies[s.body];
                let p = body.point(s.a.lerp(s.b, t));
                if p.y - terrain_height(p.x, p.z) > 2.0 * s.r + 0.1 {
                    continue;
                }
                let (depth, n) = terrain_contact(p, s.r);
                if depth <= 0.0 {
                    continue;
                }
                let r = p - n * s.r - body.x;
                let lambda = self.shift(s.body, None, r, Vec3::ZERO, n * depth);
                // A driven body resolves its own impact; the ground here only holds it up.
                if body.driven {
                    continue;
                }
                let q = self.bodies[s.body].q;
                self.contacts.push(Contact {
                    a: s.body,
                    b: None,
                    ra: r,
                    rb: Vec3::ZERO,
                    n,
                    lambda,
                    mu: BODY_FRICTION,
                    restitution: RESTITUTION,
                    ski: s.ski.then(|| unit(q * (s.b - s.a))),
                });
            }
        }
    }

    pub fn relative_velocity(&self, c: &Contact) -> Vec3 {
        let vb = c.b.map_or(Vec3::ZERO, |b| self.bodies[b].velocity_at(c.rb));
        self.bodies[c.a].velocity_at(c.ra) - vb
    }

    /// Applies impulse `p` at the contact: `+p` to `a`, `-p` to `b`.
    pub fn impulse(&mut self, c: &Contact, p: Vec3) {
        let a = &mut self.bodies[c.a];
        a.v += p * a.inv_mass;
        a.w += a.inv_inertia_world() * c.ra.cross(p);
        if let Some(b) = c.b {
            let b = &mut self.bodies[b];
            b.v -= p * b.inv_mass;
            b.w -= b.inv_inertia_world() * c.rb.cross(p);
        }
    }

    pub fn contact_weight(&self, c: &Contact, n: Vec3) -> f32 {
        self.bodies[c.a].weight(c.ra, n) + c.b.map_or(0.0, |b| self.bodies[b].weight(c.rb, n))
    }

    /// Coulomb friction along unit `dir` against relative speed `speed`, at most `max` impulse.
    pub fn friction(&mut self, c: &Contact, dir: Vec3, speed: f32, max: f32) {
        if dir == Vec3::ZERO {
            return;
        }
        let j = (speed / self.contact_weight(c, dir)).clamp(-max, max);
        self.impulse(c, -dir * j);
    }

    /// Restitution (from the pre-step approach speed) and friction from each contact's normal
    /// impulse, then passive joint damping. Position repair never becomes launch speed: the
    /// normal speed is reset to the bounce alone.
    pub fn solve_velocities(&mut self, before: &[(Vec3, Vec3)], h: f32) {
        for i in 0..self.contacts.len() {
            let c = self.contacts[i];
            let at = |k: usize, r: Vec3| before[k].0 + before[k].1.cross(r);
            let approach = (at(c.a, c.ra) - c.b.map_or(Vec3::ZERO, |b| at(b, c.rb))).dot(c.n);
            let bounce = if approach < -1.0 {
                -c.restitution * approach
            } else {
                0.0
            };
            let vn = self.relative_velocity(&c).dot(c.n);
            let p = (bounce - vn) / self.contact_weight(&c, c.n);
            self.impulse(&c, c.n * p);
            let v = self.relative_velocity(&c);
            let vt = v - c.n * v.dot(c.n);
            let max = c.lambda / h;
            match c.ski {
                Some(axis) => {
                    let along = unit(axis.reject_from(c.n));
                    let slide = vt.dot(along);
                    self.friction(&c, along, slide, SKI_ALONG * max);
                    let across = vt - along * slide;
                    self.friction(&c, unit(across), across.length(), SKI_ACROSS * max);
                }
                None => self.friction(&c, unit(vt), vt.length(), c.mu * max),
            }
        }
        // Rolling resistance: whatever lies in the snow digs in, so it cannot roll on forever.
        let roll = (-ROLL_DAMPING * h).exp();
        let mut rolled = 0_u64;
        for c in self.contacts.iter().filter(|c| c.b.is_none()) {
            if rolled & (1 << c.a) == 0 {
                rolled |= 1 << c.a;
                self.bodies[c.a].w *= roll;
            }
        }
        let keep = 1.0 - (-JOINT_DAMPING * h).exp();
        for k in 0..self.joints.len() {
            let j = self.joints[k];
            let (ip, ic) = (
                self.bodies[j.parent].inv_inertia_world(),
                self.bodies[j.child].inv_inertia_world(),
            );
            let spin = self.bodies[j.child].w - self.bodies[j.parent].w;
            let rate = spin.length();
            if rate < 1e-6 {
                continue;
            }
            let n = spin / rate;
            let p = n * (rate * keep / (n.dot(ip * n) + n.dot(ic * n)));
            self.bodies[j.parent].w += ip * p;
            self.bodies[j.child].w -= ic * p;
        }
    }

    /// Angle (rad) by which joint `k` is outside its (possibly broken) range.
    pub fn excess(&self, k: usize) -> f32 {
        let j = &self.joints[k];
        let Some(l) = j.limit.filter(|_| j.holds()) else {
            return 0.0;
        };
        let rel = self.relative(j).1;
        rel.angle_between(clamp_rotation(rel, &l, j.mirror, j.slack()))
    }

    /// Deepest overlap (m, negative) between two segments that should not touch, and which, among
    /// the colliding pairs `skip` does not exclude.
    #[cfg(test)]
    pub fn worst_overlap(&self, skip: impl Fn(usize, usize) -> bool) -> (f32, (usize, usize)) {
        let mut worst = (f32::MAX, (0, 0));
        for &(a, b) in &self.pairs {
            if skip(a, b) {
                continue;
            }
            for i in self.range[a].0..self.range[a].1 {
                for j in self.range[b].0..self.range[b].1 {
                    let (sa, sb) = (self.shapes[i], self.shapes[j]);
                    if !sa.meets(&sb) {
                        continue;
                    }
                    let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                    let (a0, a1, b0, b1) = (
                        ba.point(sa.a),
                        ba.point(sa.b),
                        bb.point(sb.a),
                        bb.point(sb.b),
                    );
                    let (s, t) = closest_params(a0, a1, b0, b1);
                    let gap = a0.lerp(a1, s).distance(b0.lerp(b1, t)) - sa.r - sb.r;
                    if gap < worst.0 {
                        worst = (gap, (a, b));
                    }
                }
            }
        }
        worst
    }
}

/// Penetration depth of a sphere at `p` into the terrain, and the surface normal.
pub(crate) fn terrain_contact(p: Vec3, radius: f32) -> (f32, Vec3) {
    let eps = 0.05;
    let dx = (terrain_height(p.x + eps, p.z) - terrain_height(p.x - eps, p.z)) / (2.0 * eps);
    let dz = (terrain_height(p.x, p.z + eps) - terrain_height(p.x, p.z - eps)) / (2.0 * eps);
    let n = Vec3::new(-dx, 1.0, -dz).normalize();
    (radius - (p.y - terrain_height(p.x, p.z)) * n.y, n)
}
