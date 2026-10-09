//! Tyre and ski tracks: a ring buffer of ground quads in one mesh. Marks are laid every `STEP`
//! metres while a wheel or ski is on the ground, fade with age and are written in place, so the
//! memory is fixed at `CAPACITY` quads whatever the distance ridden.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use crate::bike::{Bike, SUSPENSION_REST, WHEELBASE, terrain_height};
use crate::ski::{Sport, bike_active};

/// Quads in the ring; the oldest is overwritten first (400 m of one track).
const CAPACITY: usize = 4096;
/// Distance between marks, m.
const STEP: f32 = 0.10;
/// Seconds until a mark has faded out.
const LIFETIME: f32 = 45.0;
/// Opacity of a fresh mark.
const OPACITY: f32 = 0.5;
/// Height above the terrain, m.
const LIFT: f32 = 0.02;
/// Colour of a mark (linear): snow packed darker by the tyre or ski.
const MARK: [f32; 3] = [0.26, 0.27, 0.30];
/// Seconds between colour refreshes while marks fade.
const FADE_EVERY: f32 = 0.25;
/// A jump this long between two samples is a reset, not travel.
const TELEPORT: f32 = 2.0;
const TYRE_WIDTH: f32 = 0.07;
pub(crate) const SKI_TRACK_WIDTH: f32 = 0.08;

#[derive(Resource)]
pub(crate) struct Tracks {
    mesh: Handle<Mesh>,
    pos: Vec<[f32; 3]>,
    /// Birth time of each quad, s; `NEG_INFINITY` while empty.
    born: Vec<f32>,
    next: usize,
    /// Last laid point of each of the two tracks.
    last: [Option<Vec3>; 2],
    newest: f32,
    dirty: bool,
}

impl Tracks {
    /// Lays marks of `width` from the track's previous point to `at` in `STEP` pieces; `None`
    /// (contact lost) ends the track.
    pub(crate) fn lay(&mut self, track: usize, at: Option<Vec3>, width: f32, now: f32) {
        let Some(at) = at else {
            self.last[track] = None;
            return;
        };
        let Some(mut from) = self.last[track] else {
            self.last[track] = Some(at);
            return;
        };
        if from.distance(at) > TELEPORT {
            self.last[track] = Some(at);
            return;
        }
        let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
        while flat(at - from).length() >= STEP {
            let dir = flat(at - from).normalize();
            let to = from + dir * STEP;
            let half = Vec3::new(dir.z, 0.0, -dir.x) * (width * 0.5);
            let slot = self.next * 4;
            for (i, corner) in [from - half, from + half, to + half, to - half]
                .into_iter()
                .enumerate()
            {
                let y = terrain_height(corner.x, corner.z) + LIFT;
                self.pos[slot + i] = [corner.x, y, corner.z];
            }
            self.born[self.next] = now;
            self.next = (self.next + 1) % CAPACITY;
            from = to;
        }
        self.last[track] = Some(from);
        self.newest = now;
        self.dirty = true;
    }

    fn clear(&mut self) {
        self.born.fill(f32::NEG_INFINITY);
        self.last = [None; 2];
        self.next = 0;
        self.dirty = true;
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let pos = vec![[0.0; 3]; CAPACITY * 4];
    let idx = (0..CAPACITY as u32)
        .flat_map(|q| [0, 1, 2, 0, 2, 3].map(|i| q * 4 + i))
        .collect();
    let mesh = meshes.add(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; CAPACITY * 4])
        .with_inserted_indices(Indices::U32(idx)),
    );
    commands.insert_resource(Tracks {
        mesh: mesh.clone(),
        pos,
        born: vec![f32::NEG_INFINITY; CAPACITY],
        next: 0,
        last: [None; 2],
        newest: f32::NEG_INFINITY,
        dirty: false,
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            double_sided: true,
            depth_bias: 4.0,
            ..default()
        })),
        // The marks roam the whole map; the mesh bounds are those of the empty buffer.
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Transform::default(),
    ));
}

/// R and a sport switch wipe the marks.
fn clear(keys: Res<ButtonInput<KeyCode>>, sport: Res<Sport>, mut tracks: ResMut<Tracks>) {
    if keys.just_pressed(KeyCode::KeyR) || sport.is_changed() {
        tracks.clear();
    }
}

/// Bike tyres: the front and rear hub under the wheel, while that wheel is on the ground.
fn bike_tracks(bike: Res<Bike>, time: Res<Time>, mut tracks: ResMut<Tracks>) {
    let rot = bike.orientation();
    for (i, z) in [-0.5 * WHEELBASE, 0.5 * WHEELBASE].into_iter().enumerate() {
        let hub = bike.position + rot * Vec3::new(0.0, -SUSPENSION_REST + bike.suspension[i], z);
        let contact = (bike.grounded[i] && bike.crash.is_none()).then_some(hub);
        tracks.lay(i, contact, TYRE_WIDTH, time.elapsed_secs());
    }
}

/// Copies new marks to the mesh and ages the opacity of all of them.
fn flush(
    time: Res<Time>,
    mut tracks: ResMut<Tracks>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut since: Local<f32>,
) {
    let now = time.elapsed_secs();
    *since += time.delta_secs();
    let fading = now - tracks.newest < LIFETIME;
    if !(tracks.dirty || fading && *since >= FADE_EVERY) {
        return;
    }
    *since = 0.0;
    tracks.dirty = false;
    let Some(mesh) = meshes.get_mut(&tracks.mesh) else {
        return;
    };
    if let Some(VertexAttributeValues::Float32x3(pos)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        pos.copy_from_slice(&tracks.pos);
    }
    if let Some(VertexAttributeValues::Float32x4(color)) = mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR)
    {
        for (quad, born) in color.chunks_exact_mut(4).zip(&tracks.born) {
            let alpha = OPACITY * (1.0 - (now - born) / LIFETIME).clamp(0.0, 1.0);
            quad.fill([MARK[0], MARK[1], MARK[2], alpha]);
        }
    }
}

pub(crate) struct TracksPlugin;

impl Plugin for TracksPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(
            Update,
            (clear, bike_tracks.run_if(bike_active), flush).chain(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracks() -> Tracks {
        Tracks {
            mesh: Handle::default(),
            pos: vec![[0.0; 3]; CAPACITY * 4],
            born: vec![f32::NEG_INFINITY; CAPACITY],
            next: 0,
            last: [None; 2],
            newest: f32::NEG_INFINITY,
            dirty: false,
        }
    }

    fn marks(t: &Tracks) -> usize {
        t.born.iter().filter(|b| b.is_finite()).count()
    }

    #[test]
    fn marks_are_laid_every_step_and_the_ring_stays_bounded() {
        let mut t = tracks();
        t.lay(0, Some(Vec3::ZERO), 0.1, 0.0);
        assert_eq!(marks(&t), 0, "the first sample only starts the track");
        t.lay(0, Some(Vec3::new(0.0, 0.0, -1.05)), 0.1, 1.0);
        assert_eq!(marks(&t), 10);
        assert!(
            (t.last[0].unwrap().z + 1.0).abs() < 1e-5,
            "the remainder carries over"
        );
        // Far more than the capacity: still CAPACITY quads, the newest bornlast.
        let mut z = 0.0;
        for k in 0..(CAPACITY * 3) {
            z -= STEP;
            t.lay(0, Some(Vec3::new(0.0, 0.0, z)), 0.1, 2.0 + k as f32 * 1e-3);
        }
        assert_eq!(marks(&t), CAPACITY);
        assert!(t.pos.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn lost_contact_and_teleports_break_the_track() {
        let mut t = tracks();
        t.lay(1, Some(Vec3::ZERO), 0.1, 0.0);
        t.lay(1, None, 0.1, 0.1);
        t.lay(1, Some(Vec3::new(0.0, 0.0, -0.5)), 0.1, 0.2);
        assert_eq!(marks(&t), 0, "no mark across a gap");
        t.lay(1, Some(Vec3::new(0.0, 0.0, -50.0)), 0.1, 0.3);
        assert_eq!(marks(&t), 0, "no mark across a teleport");
        t.clear();
        assert_eq!(t.last, [None; 2]);
    }
}
