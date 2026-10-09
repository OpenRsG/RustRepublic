# RiderRepRust: development notes

Detailed design, verification history and the optional asset-research tools.
The player-facing overview is in [README.md](../README.md).
The simulation, bike model, skier and rider rigs are independently authored in this
repository; this is **not verified original-game physics or animation parity**.

## Play

```sh
cargo run --locked
```

Requires Rust 1.88+, a C compiler, Linux X11/Wayland development libraries and a
working Vulkan graphics driver. Development dependencies are optimized; a release
build is optional (`cargo run --release --locked`).

The game runs without a Riders Republic installation. The white arena retains
three small jumps and adds a **32 m hill with a 7 m kicker** at x = 60 m. Behind the kicker a
6 m table rises to a crest 18 m past the lip and falls away over 26 m, so flights land on a
downslope instead of the flat.
The forest, mountains, gates, sky dome and dirt course remain removed.
Keys **1–4** select Downhill, Road, Slopestyle and Freeride. Switching resets to the
flat start; R resets without changing the selected discipline.
Each profile has authored spring/damper, power, drag, tire grip, hop and air-control
settings. Bars, saddle height, tire width, frame and jersey colors change too;
Road has narrow drop grips, smooth tires and a visually rigid frame/fork.
All four retain the shared wheel radius, wheelbase and bounded two-strut contact model.
Each tyre has one grip budget (`GROUND_MU` 0.65 × strut load) shared by drive, brake and
cornering; demand beyond it stays as sideways slip, and a front tyre overloaded and sliding
for 0.5 s is a `Washout` crash. Above 6 m/s steering sets a target lean (roll rate limited)
and yaw follows `g·tan(lean)/v` minus a countersteer term, so the bars dip the other way
first; below 3 m/s the bars steer kinematically, blended in between.
The fixed 120 Hz simulation supports pedaling, sprinting, braking, steering/lean,
hops, wheelies, rear-wheel manuals, nose manuals and physical flips, rolls and yaw spins.
Wheel spin, drivetrain and suspension movement feed the same animated rig.

On one wheel the rider balances the bike: Up/Down shift the rider's weight and give a limited
pitch torque (4 rad/s²). Landing a flip on the rear wheel (or the nose) leaves the bike to its
own momentum and gravity; past the balance point gravity soon outgrows the rider's torque, and
beyond 72 degrees to the ground the rider loops out (or goes over the bars). The manual and nose
manual assists only hold a balance the rider has already found: deeper than 0.2–0.4 rad past
their target they let go. On one wheel the rider's legs and arms absorb a landing's impact, so
only up to 1.5 g of the strut force pitches the bike and landing deep on a wheel no longer
slams or launches it.

In the air the rider uses body English. Head and shoulders are thrown back for a backflip (elbows
flaring as the bars are pulled) or over the bars for a front flip. The shoulders drop into a
barrel roll with the hips countering and the head tilted into it, and turn ahead of the hips to
lead a spin with the head further still. The bike is pulled in while it rotates, and the body
opens up to spot the landing. Skiers do the same: the chest and head lead a spin (wound up the
other way in the preload), the outside arm sweeps across the chest while the other opens
behind, the torso arches back into a backflip with the arms reaching up or curls for a front
flip, and the shoulders drop into a roll; a held grab takes the arms and torso over.

Driving the pedals hard (a sprint, or pedalling while the speed still climbs) the rider gets low
and forward, throws the bike from side to side under them (up to three times the standing
rock), drops the pelvis onto each pushing pedal, pulls the bar up on that side so the shoulders
turn with every stroke, and flares the elbows. Once the speed stops climbing in a sprint the
rider is spun out at top speed: lower, chest down, rocking less. All of it rides on springs, so
starting, holding and easing off blend.

No two landings are absorbed alike. The touchdown's direction sets a jolt held through the
compression and released on a spring: the body is thrown to the low side of a leaned bike (on
skis, the way the skis were sliding across their length), back as the rear wheel lands first,
over the bars on a nose-first landing or one still rotating forward (on skis, over landed-on
tips or back onto the tails), and the shoulders twist. A deterministic per-landing jitter varies
the compression depth, each jolt axis and, on skis, how hard each arm flails. At speed the
skier's free hands, and the poles they carry, wander a centimetre or two on incommensurate
periods (snow chatter), fore-aft and sideways only, so the poles never ride still or in step.

The live skeleton overlay is enabled at launch and uses the same solved joints
as the rider animation. Blue marks the left side, pink the right and green the
spine/head; skeleton-only mode uses darker hues for contrast on the white floor.
Hands and feet release for tricks and blend back to the bike before landing.
Released hands have their own joint positions, not debug lines falsely tied to grips.
Crashed riders turn red and release the bike into a **23-joint ragdoll**.
The rider's limbs articulate, collide with the floor, tumble and slide independently
of the bike; the camera follows the fallen rider's pelvis.

The GTA-style third-person camera follows behind the direction of travel (not the
spinning body) with critically damped focus, heading and elevation, velocity
look-ahead, and a wider FOV and longer arm at speed. Landings dip and settle
instead of snapping. Hold the right mouse button to orbit, scroll to zoom (3–14 m),
or press C to recenter. It also recenters automatically while riding after orbit
input stops. Rising ground behind the rider (hill descents, ramp faces) lifts the
camera over it instead of pulling it down to the rider; the horizon stays upright.
Braking and drive forces act at the actual tire patches and are capped by wheel
load, so unloading a wheel fades its force instead of switching brake torque abruptly.

| Keys | Action |
| --- | --- |
| 1 / 2 / 3 / 4 | Select Downhill / Road / Slopestyle / Freeride; reset to start |
| W / S | Pedal / brake |
| A / D | Steer left / right |
| Shift | Sprint while pedaling |
| Space | Hop; holding does not re-hop on landing |
| Q | Assisted wheelie |
| Up / Down | Nose down / up while airborne; shift weight to balance on one wheel |
| Left / Right | Airborne roll |
| Z / X | Rear-wheel manual / nose manual while moving; no pedaling required |
| E / T | Airborne yaw spin right / left |
| Ctrl + Up / Down | Higher-authority physical forward / backward flip; counter-input brakes rotation |
| Ctrl + Left / Right | Higher-authority physical barrel rotation |
| U / I / O | Cycle selected hand / foot / bike trick, including None |
| B | Hold selected trick combination while airborne |
| J / L | Select left / right trick side |
| Right mouse drag | Orbit the third-person camera |
| Mouse wheel | Zoom camera |
| C | Recenter behind the bike |
| F1 | Toggle skeleton overlay |
| F2 | Toggle rider mesh / skeleton-only view; bike stays visible |
| F6 | Toggle looping hill showcase; starting it selects Freeride and the summit |
| R | Clear crash/reset bike and animation; during F6, restart the current demonstration |
| Esc | Pause/resume physics and animation; camera remains operable |
| H | Show/hide controls |
| F3 | Log physics, crash reason/impact, animation layers, angular rates, demo stage and camera |

Losing window focus suspends physics and animation. Terrain contact is a shared
analytic heightfield, not a general rigid-body collision world. Normal balance,
steering and rate damping remain arcade-assisted; air motion has no target-attitude
torque. Full flips and rolls are now the bike's **physical quaternion orientation**,
with momentum and ordinary collision tests—not independent visual layers.

### Browser (WebGPU)

```sh
scripts/web.sh   # writes target/www: index.html, JS glue, wasm and a gzipped copy (~8.5 MB)
```

Needs the `wasm32-unknown-unknown` standard library (Arch: `rust-wasm`) and
`wasm-bindgen-cli` at the `wasm-bindgen` version in `Cargo.lock`
(`cargo install --locked wasm-bindgen-cli --version 0.2.129`). It uses the `web` profile
(opt-level 3, thin LTO, stripped): the physics runs at full speed and gzip keeps the
download near 8.5 MB. `wasm-opt -O3` runs when binaryen is on `PATH` or unpacked under
`.tools/binaryen/`. The page fetches `rider-rep-rust_bg.wasm.gz` and unpacks it with
`DecompressionStream` while a progress bar shows the stage and bytes, falling back to the
plain wasm without that API. It also resumes the sound's `AudioContext` on the first
pointer or key press. The page needs WebGPU, so serve it over HTTPS or `localhost`.
It opens in ski mode with the F6 showcase looping and keeps running without focus;
click the canvas to use the keys. WebGL is not built. On portrait screens the camera keeps a
4:3 horizontal field of view (vertical FOV up to 100 degrees, `hor_plus` in `src/game.rs`)
and tilts down; the HUD scales with `UiScale`, key help hides on turning portrait (and on
load on a landscape touch screen), one finger orbits and two pinch-zoom.
The web build starts with the skeleton overlay off.

`web/index.html` adds a menu and touch controls on top of the canvas. Both only dispatch
synthetic `KeyboardEvent`s (`code` = the game's key) to the canvas, so the game sees ordinary
keys and needs no web-specific input code. The ride stick maps to W/A/S/D and the lean stick
to the arrows, eight-way like a d-pad (diagonals hold two keys); buttons hold their key while
touched, with pointer capture so a sliding thumb doesn't drop it. Taps are spaced by
animation frames rather than milliseconds, so slow phones still see each press on its own
frame. `pointerdown` is cancelled so the canvas keeps focus (Bevy releases held keys on
blur). The page mirrors the mode keys (1–5, F6) to label the menu; touching any control
during a showcase sends F6 first to take over. The browser build autostarts the ski showcase
on the first frame only, so switching away before it runs doesn't start it later.

To share it on a tailnet, `sudo tailscale serve --bg --https=8443 <repo>/target/www`
serves it at `https://<host>.<tailnet>.ts.net:8443/`. Rebuilding updates it in place.

## Ski mode

Press **5** for a skier; **1–4** return to the bike. The code is in `src/ski/` and was written for this project. It is not decoded retail behaviour.
- Skeleton (`pose.rs`): 23 body joints plus ski tail/binding/upturned tip and pole top/tip, with fixed bone lengths. Segment lengths, hip and shoulder width and the 0.366 m ski stance follow the decoded retail rider rig's proportions; the ski is 1.5 m, longer than the 1.26 m between the retail ski end bones. F1/F2 and the overlay colours work the same as for the bike.
- Physics (`physics.rs`, 120 Hz): slope gravity, snow friction, drag with a lighter tuck, sidecut carving (radius `SIDECUT_RADIUS`·cos edge, fading in from about 3 to 23 degrees of edge) and edge grip, skidding once v²/R exceeds the grip, skating (below 4 m/s), double-poling, snowplow and hockey stop, and switch riding. Hold Space to crouch and release to jump; the legs absorb landings. In the air you get free-quaternion spins and flips. Landings are judged before any penetration is corrected: a bad angle, crossed skis or too much spin, a hard impact, a body strike, or a caught edge is a crash.
- Animation (`anim.rs`, `rig.rs`): athletic stance, carve inclination and angulation with knee drive, tuck, plow with knees in, hockey stop, V-skate and double-pole with stride rates that rise with speed, jump preload, landing absorption, and switch stance. Grabs: Mute, Safety, Japan, Tail, Tip, Truck Driver, Daffy, Spread Eagle and Iron Cross. Grabs release about 0.35 s before the predicted landing. Blend weights are springs, so poses never snap between ticks; a touchdown eases the pose from its in-air attitude onto the snow over about 0.15 s. The drawn skier and camera are interpolated between the 120 Hz physics ticks.
- Secondary motion: breathing and slow weight shifts, a chest that lags the skis' turn (counter-rotation), a head that stays level and keeps looking downhill, free hands that trail the body's acceleration, a deep preload crouch whose arms swing back before the pop, a race tuck with the back about 25 degrees above horizontal and fists at the knees, a touchdown that reaches its deepest point about 0.2 s after contact (torso folded about 65 degrees, wider stance, brief visual skid across the line of travel) and is back near neutral about 0.5 s later, and an inside-pole plant at each carve change. Timings and postures were tuned against retail reference video, clip durations and decoded retail air/ollie curves, not copied from retail data.
- Flat ground: skating is a stroke cycle. On each stroke one ski edges and pushes out and back while the hips cross over to the other ski; the pushing ski then lifts, swings in and lands as the next stroke begins. Both skis sit about 12 degrees off the travel line, a V of roughly 23 degrees with the tails close. The skier travels along the ski it glides on, so that ski never slides, and the body weaves slightly as the weight moves. The pushing foot ends about 0.5 m from the glide foot with a nearly straight leg; the glide knee is flexed 50–60 degrees, with the trunk leaning 30–40 degrees forward over it. Strokes take about 0.95 s from standstill and 0.6 s at 4 m/s, and start short and upright. From about 2 m/s the poles push on every stroke (V2); below that it is free skate with the arms swinging across. Double poling runs a cycle of 1.2 s down to 0.9 s, faster with speed: poles plant ahead of the binding, the trunk crunches from near upright to about 55 degrees off vertical with the hips and knees flexing in step, and the hands finish past the hips. Speed is gained only in the push window of each stroke or cycle. No retail footage of skating or poling exists, so these follow real cross-country technique rather than a game reference.
- Poles and hands never enter the body. Carried poles trail back with their baskets splayed outwards, tuned against measured reference poses. Plants land wide of the boots, and the fists pass outside the thighs during a double-pole push. The trailing poles follow the chest's turn in stops and switch riding but not a carve's counter-rotation, so the inside basket stays outside the legs. A clearance pass in `rig.rs` then wraps the head, torso, thighs, knees, shins and boots in capsules sized like the rendered meshes. It moves the fist and the end of the pole next to it out of those capsules, swings the elbow (or, if a leg or the trunk pins the arm, moves the wrist) so neither forearm nor upper arm enters the body, and turns the pole about the wrist so the shaft and basket clear. A pole always passes outside its own leg instead of flipping to the inner side, and a final pass keeps each pole out of the other arm, glove and pole. A hand reaching across the body for a Mute or Japan grab leads with its basket ahead of the body and the elbow up, then swings the pole round the outside of the grabbed ski. The regression test sweeps every scene, grab and side, plus the F6 loop, skating, carving, stops, the tuck and jumps. Before this pass, poles ran through the legs in 6,545 of 10,800 F6 ticks. Known limits: the Japan reach brushes the glove up to 3 mm into the thigh for a few frames, and in the Mute grab the cross-held pole flicks round the grabbed thigh for one tick.
- The legs keep clear of each other and the trunk: after the leg IK each knee swings about its hip-ankle line (at most 60 degrees, bone lengths unchanged) until thigh, knee and shin clear the other leg, its boot, the belly, chest and head. Before this, the snowplow crossed the knees by 15 cm. In the air outside a held grab, each ski turns with its shin no further than a ski boot lets the ankle (10 degrees back, 40 forward, 6 sideways and twisting).
- Crashes hand over to an articulated rigid-body ragdoll (`ragdoll.rs`, extended position-based dynamics at 2,400 substeps a second). Twenty segments with anthropometric masses and inertias (pelvis, abdomen, thorax, head, upper arms, forearms, hands, thighs, shins, boots, skis, poles) are pinned at the skeleton's joints. Every joint has an anatomical range measured from the upright stance, for example hip flexion 120 / extension 20 / abduction 45 / adduction 30 degrees, knee 0-150 degrees with 5 degrees of side play, elbow 0-150, shoulder flexion to about 170 and abduction to 180 about a raised-arm centre, ski-boot ankle 10 back / 40 forward. Past the end of range the joint gives like a stiff spring (ligaments, the boot shell) and stops dead 6 degrees further. The torque holding each joint at its limit, averaged over 20 ms, is compared per axis with the joint's strength (knee: 250 N m hyperextension, 150 sideways, 100 twisting; neck 225/180/120; wrist 110/90/70 and so on). A joint over its strength breaks and its range opens by 50 degrees; it stays attached and still collides. The bindings release the skis forward (heel) or in twist (toe), never sideways, usually before the legs give in moderate falls, at about 2.5 times DIN 7 torques because the limp body levers the boots harder than a skier's working legs. Strengths are tuned so falling over, a stiff 2 m drop or a 1 m drop with forward speed break nothing, while 12-22 m/s tumbles and dives release skis and break joints. The HUD lists what broke and whether a ski came off.
- Every pair of segments that is not jointed collides as rendered-size capsules, broken or not, so arms stay out of the torso, head and legs, legs out of each other, skis out of boots and poles out of limbs to within a few millimetres. Only the grip section of a pole may lie along its own wrist, and the upper arm collides from just below the shoulder, which sits inside the chest slab. Skis glide along their length (friction 0.05) and bite across it (0.8); anything lying in the snow has rolling resistance. Out-of-range or overlapping animated poses are eased into range at 230 degrees a second instead of snapped. The tests throw the skier down the hill straight, flipping, cartwheeling and corkscrewing, dive it head-first at 22 m/s, drop it and run the F6 crash, checking every tick that nothing sinks into the snow, no bone stretches, no two segments overlap by more than 5 mm (1 cm for a pole on a ski) and no joint passes its hard stop by more than a few degrees.
- Hard impacts: closing speed into the snow above 14 m/s (a 10 m drop onto flat snow) always crashes, and above 0.65 of that a touchdown that also uses more than half of its tilt, crossed-ski or spin limit crashes too. The crash hands the ragdoll the pre-impact momentum (seed within a few percent of the velocity before the step). The fists hold the poles until the pull on a grip passes 900 N (low-passed over 20 ms): gentle drops and topples keep both poles, a head-first dive or a 20 m/s flat landing tears at least one away, after which the loose pole tumbles and collides like any other segment. The HUD adds "dropped a pole / dropped both poles" after the bindings. Poles are single rigid shafts and do not bend or snap.
- F6 in ski mode loops 14 runs (about 3½ minutes): slalom 360 Mute, backflip Safety, 540 Japan (lands switch), frontflip Mute, Daffy, rodeo Mute (a 360 spin and a backflip flown together, which lands switch), backflip Iron cross, slalom 360 Spread eagle, switch 180 Tip, and five crashes: sideways 270, a 720 stopped at 675 degrees, an under-rotated backflip (about 100 degrees, onto the back), an over-rotated backflip (1.3 turns) and an incomplete front flip. A full 720 doesn't fit in the kicker's air time, so it is shown as the crash it would be.

Keys: W skate/pole, S plow/hockey stop, A/D carve, Shift tuck, Space jump, arrows flip/roll, E/T spin, Ctrl full flips, U picks the grab, B holds it, J/L picks the side, R resets.

## Rendering

- `src/body.rs`: each rider is one procedural mesh per material (top, bottom, skin, shoe,
  glove) skinned to 17 bones, two weights per vertex. Every bone is aimed from its parent
  point to its child point on the solved rig (minimal-arc twist for limbs; pelvis/shoulder,
  heel/toe and head frames for torso, feet and head) and stretches to the rig's spacing,
  so riding, interpolated frames, the get-up blend and the ragdoll all draw through it.
  F2 hides it; helmet and goggles stay rigid parts.
- `src/surface.rs`: a tileable 256×256 grain albedo + normal map built at startup with
  hand-built mips, repeat sampler and 8× anisotropy; the floor has world-space UVs (5 m tile).
- `src/tracks.rs`: 4096 quads in one mesh updated in place, one mark per 10 cm of grounded
  travel (tyres and both skis), unlit and alpha-blended 2 cm above the terrain, fading over
  45 s. Gaps, teleports over 2 m, R and sport switches break or clear them.
- Camera: SMAA; SSAO on native builds only (WebGPU's 4 storage textures per stage are
  fewer than it needs). `FrameTimeDiagnosticsPlugin` feeds the HUD fps line. On the
  RTX 3090 / Xvfb setup this costs about 8% (59 → 55 fps, mostly SSAO); the WebGPU page
  ran the showcases at 52–61 fps after the opt-level 3 build.

## Crashes and landing rules

Touchdown is judged **before** the bump stop changes velocity or contact flags.
The checks use terrain normals, actual posed wheel centres/axles and angular rates:
- Excess nose angle, sideways tilt, travel/heading mismatch or spin at contact
  causes a bad landing. Inverted rider/frame impacts can occur before a wheel lands.
- Both feet still released at touchdown cause missing-support failure.
- A fully detached rider cannot apply steering, pedaling, hop or air-control torque.
- Twelve posed rider/frame sphere proxies detect head, torso, knees, hands and bike
  strikes. They follow the same rig and trick assembly transforms as the meshes.
- Closing speed along the surface normal above 13 m/s (a flat landing from an 8.6 m
  drop) is a hard-impact crash. Above 0.65 of that, a touchdown that also uses more
  than half of any pitch, roll, tilt, slip or spin limit is one too. Thresholds are
  authored constants in `src/bike.rs`, relative to the local surface, not retail
  measurements.

A crashed bike ignores riding inputs and falls, bounces, slides and tumbles with
gravity, friction and angular impulses. The detached rider inherits linear/angular
momentum and limb motion from the impact pose. It is 15 rigid capsule segments on the
XPBD solver shared with the ski ragdoll (`src/rigid.rs`), with anatomical joint limits,
collision between every non-jointed pair (limb vs limb and limb vs bike frame and tyres)
and the bike as a coupled 14 kg body: grips and contacts push the wreck through
`Bike::push`. The hands keep hold
of the bars and the feet stay on the pedals until the wreck pulls harder than they can
hold (600 N per hand, 200 N per flat-pedal foot, sustained for 20 ms): a slow topple keeps
the rider attached to the bike for a moment, while a hard impact tears hands and feet away
at once (grips are detachable joints with those strengths). The HUD reports "let go of
the bars" / "feet off the pedals".
Penetration repair is separate from contact velocity, so getting out of the floor
does not launch the rider. Resting bodies sleep instead of continually jittering.
Outside the F6 showcase the rider gets up where they fell once the ragdoll sleeps or
after 3 s (`GETUP_TIMEOUT`): bike or skis are placed upright at rest under the hip, and
the drawn pose blends from the ragdoll to the riding pose over 0.7 s (`GETUP_BLEND`,
smoothstep) with controls ignored. **R** still returns to the start; F6 resets between
runs. A ski or pole thrown far lerps back to the feet during the blend.
Pause/focus loss freezes both bodies; reset restores riding and clears the ragdoll.

These are proxy contacts, not triangle-perfect collisions or retail ragdoll tracks.
The crashed bike has no friction inside the rider solver (its own crash model keeps tyre
and ground friction) and feels the rider one step late. The spine is one joint.
Extreme airborne poses while riding can still intersect.

## F6 hill showcase

Press **F6** to start an automatic, repeating sequence of 17 runs:
**Superman → backflip → 360 X-up → frontflip → backflip Superman → barrel roll →
nose dive (crash) → no hands + no feet → barspin + tailwhip → sideways landing (crash) →
table → 360 tuck no-hander → frontflip can-can → half barrel (crash) →
barrel roll no-hander → over-rotated backflip, 1.3 turns (crash) → incomplete flip (crash)**.
Spins use a yaw servo on the real E/T air-yaw input; combined runs hold their trick while
the rotation servo flies the flip, roll or spin. Each run rolls from the summit without
pedalling, launches off the real ramp, shows its landing or crash for 2.5 seconds, then
resets for the next run. The loop takes about 3½ minutes. It uses ordinary brake and
bounded air-control inputs;
there are no airborne teleports, direct orientation writes or collision exemptions.

The camera takes a side/rear angle and centres the bike for the longer released
poses; RMB orbit and zoom still work. Esc pauses the whole demonstration.
R restarts its current run. F6 again returns control at the current position;
1–4 stop the demo and return to the flat start. To ride the hill manually, start
F6 and immediately turn it off.

## Animation coverage

`src/animation.rs` blends posture and independent hand/foot/bike trick layers at
120 Hz; `src/scene.rs` solves both visible geometry and collision proxies.
These are **authored poses inspired by observed retail names**, not decoded clips.

| Layer | Implemented families |
| --- | --- |
| Riding | Idle, seated pedaling, standing sprint, coast, braking, turning, wheelie/manual, nose manual |
| Jump transitions | Hop preload/takeoff, air tuck/extension, impact-weighted landing compression/recovery |
| Hands | One hand, no hands, tuck no-hand, barspin, tire grab, seat grab, toboggan |
| Feet | One foot, no feet, can-can, no-foot can, Superman, tailwhip, nac-nac, Indian |
| Bike | Whip, table, X-up, turndown, Euro table, invert, crankflip |
| Rotations | Physical quaternion flips, barrel rolls and airborne yaw |

Fluidity (tuned against retail reference video timings, not retail clips):
- The bike is drawn from the last two fixed ticks (`BikeFrames`: joints lerp, rotations slerp,
  wheel/crank angles blend the short way) at `Time<Fixed>::overstep_fraction`; the chase camera reads
  the same interpolated root. Resets/teleports and discipline changes never blend.
- Posture, trick weights, release, landing, lean, pedalling weight and steering/crank followers are
  critically damped springs (no velocity steps). The drawn crank follows the physical one on a spring
  and, freewheeling, settles level (outside pedal down in a corner).
- Pedalling: pelvis rocks over the pushing pedal (more when standing) with the shoulders against it;
  the ankle pitches with the stroke (toe down at the bottom, heel dropped coasting). The hip reach is
  a soft minimum over both pedals, so legs never snap or fully straighten.
- Cornering: the rider's pelvis goes outboard and the torso stays more upright than the bike; the inside
  knee and elbow flare; the head stays level and near the bike centreline. Descents add an attack
  crouch, terrain compression pushes the rider into it, and the torso lags the bike's acceleration.
- Landing: contact to deepest compression about 0.25 s, then about 0.3 s more to recover (small
  overshoot); the pelvis drops about 12-20 cm, knees reach about 80-100 deg flexion and elbows about
  100 deg with the upper arms flared. The compression rises from rest on a spring, with no velocity kick.
- Standing strokes rock the drawn bike (not the physics) about the tyre contact line, up to 3 deg, against
  the pelvis sway; hands and feet follow the drawn grips and pedals.
- Rider bone lengths (thigh, shin, arm, hip/shoulder width) are our own constants near the decoded retail
  rider's. Pedalling cadence, knee range and pelvis sway are within the decoded retail ranges (see the
  test probes); the torso still leans a few degrees more than the retail pedal clips.
- Not done: hop preload is limited by the physical hop firing on the key press.

Hand, foot and bike selections combine, with left/right variants. Barspin turns the
front assembly, tailwhip swings the frame/rear assembly about the steering axis,
and crankflip turns the cranks while feet release. Bone lengths are preserved;
unreachable free-limb targets are projected into reach rather than stretching limbs.
Extreme mixed poses can move attached hands slightly off the grips; the matrix
currently bounds that gap to 10 cm and attached feet to 5 cm, versus 5 mm on normal
riding poses. There is no rider/bike self-collision; extreme tricks can intersect.

The HUD shows selected versus active layers and any priority override:
- Tricks start only airborne, after the first 0.05 s.
- New holds stop 0.45 s before predicted wheel contact (0.75 s for Superman, 0.65 s for
  tailwhip); rider release/regrab and bar/tail/crank completion prepare the landing. Short
  flights may have no safe trick window. Hands and feet regrab in about 0.3 s, faster when
  contact is closer than that.
- Barspin owns the bars and suppresses table, X-up and Euro table.
- Invert, Euro table and crankflip force feet clear; an explicit foot trick
  still determines their pose.
- Bar/tail/crank spins that are moving always finish forward, easing out, rather than
  reversing. Their timing remains authored assistance. **Root flips/rolls are not completed
  for you**; an incorrect attitude or unfinished physical rotation can crash.

Superman and tailwhip timing is tuned against measurements; the poses are ours.
- Superman: the feet leave at once, and the straight legs swing from the pedals to straight
  back over 0.1–0.45 s. The body lies flat behind the bars, arms straightening from 0.2 s,
  while the bike swings nose-up under the gripped bars to about 90 degrees (easing out over
  1.1 s). The pose is levelled against the bike's physical pitch, so the body stays flat in
  the world. The return takes 0.63 s: bike and body swing back first, the knees fold (peak
  about 150 degrees) and the feet come down onto the pedals last, easing out.
- Tailwhip: the trick-side foot kicks out and back over the passing rear wheel first; the
  other foot follows 0.1 s later and goes across and forward. The rider sinks behind the
  bars with the trunk nearly upright. The frame spins at about 720 degrees/s. On release it
  keeps turning forward, then eases out; the catch starts 100 degrees before the frame comes
  round, with the trick-side foot back in about 0.2 s, the other in about 0.4 s, and the
  rider lunging forward over the bars.

Observed but **not implemented as separate trick families**: Bikeflip, Briflip,
HDflip, Grizzair, Cannonball and Tsunami. Retail walk-back, dismount, menu/taunt,
crash and other opaque clip variants are not reproduced. Opposite-side procedural
poses do not certify the retail `OPPO` tracks, and PS prefixes are not verified
discipline bindings. Complete original-game animation playback still requires the
track schemas, bone bindings and blend graph described under parity limits.

## Retail asset tools

These can build without the Bevy renderer using `--no-default-features`.

```sh
cargo run --locked --no-default-features --bin rider-assets -- inspect
cargo run --locked --no-default-features --bin rider-assets -- find pedal
cargo run --locked --no-default-features --bin rider-assets -- check ID07
```

The game directory defaults to the Flatpak Steam installation under `$HOME`.
Override it with `RIDERS_REPUBLIC_DIR` or the final directory argument:

```sh
cargo run --locked --no-default-features --bin rider-assets -- inspect /path/to/RidersRepublic
cargo run --locked --no-default-features --bin rider-assets -- find pedal /path/to/RidersRepublic
cargo run --locked --no-default-features --bin rider-assets -- check pedal /path/to/RidersRepublic
cargo run --locked --no-default-features --bin rider-assets -- extract /path/to/DataPC.forge RESOURCE_ID
```

`find` searches index names, case-insensitively for ASCII. `check` additionally
decodes every matching entry and validates checksums, lengths and resource
identities. No matches, unsupported formats and corrupt entries fail explicitly.
IDs accept decimal or `0x`-prefixed hexadecimal.

`extract` writes only to `.local/extracted/<hex-id>/`, refuses an existing output
directory, and retains the stored container, decompressed metadata/files, and each
native record/header/payload. It does **not** convert payloads into animation
keyframes, meshes, or physics settings. Proprietary evidence/extracts and build
outputs are ignored by Git. Original game files are opened read-only.

## GamePort2Rust

The existing sibling checkout is installed through `.tools/GamePort2Rust`; its
already-installed REA/Ghidra/Java stack is reused. No global package or unrelated
OMP configuration was changed during this integration.

```sh
bash scripts/gameport.sh install
bash scripts/gameport.sh rea --help
bash scripts/gameport.sh query /path/to/RidersRepublic.exe PROCEDURE
```

For another checkout, set `GAMEPORT2RUST_ROOT` when installing. GamePort2Rust is
native reverse-engineering tooling, not an asset loader or Rust engine. This
project supplies the Rust Forge/container reader; the toolkit supplied the static
executable inspection. Its own query wrapper retains evidence in the toolkit's
ignored `.local/re/` directory.

## Local physics/showcase verification

`cargo test --locked --bins` runs the gameplay/rig/physics/camera tests plus 3
archive/container tests (86 + 3 at the realism pass; this section's revision passed 45).
Physical checks cover safe flat/slope landings for all disciplines, nose/side/
inverted/slipping failures, hard impacts, detached support, free angular momentum,
completed versus incomplete flips, ignored post-crash controls, reset and bounded
proxy penetration at supported time steps. A posed table crashes where the same
upright bike can land. The floor grid is checked against collision height.
Ground-contact checks cover low-speed pedal/coast/brake transitions and slope holds
for all four profiles. Ragdoll checks cover detached articulation, bounded bones,
terrain clearance, settling/reset and penetration repair without kinetic-energy injection.

The 4,608-scenario hand × foot × bike × discipline × side matrix remains.
Its root-flip assumptions were removed: full rotations and the eight-case demo
now have physical integration checks, including seven successful runs followed
by a deliberate crash and a loop reset.

The actual app ran locally on a private Xvfb display using RTX 3090 Vulkan.
All four bikes pedaled, coasted, braked and settled without measured resting
height drift, pitch rate or speed in the sampled state captures.
An impossible short-hop flip detached the ragdoll; its joints settled and slept,
pause froze both bodies, and R restored riding. The full F6 loop retained seven
safe landings and activated the ragdoll for its deliberate incomplete-flip crash.
The high-impact check also caught and corrected an initial penetration-induced
launch; that failed observation is preserved separately from final evidence.
Screenshots, input states, regression/build output and final recordings are in
`.local/verification/ragdoll-contact/`; no owner's-desktop input was injected.

Earlier physical-flip/hill evidence, before the articulated ragdoll and contact
stability fix, remains in `.local/verification/crash-showcase/`.

Earlier authored-animation evidence (28 tests at that revision, including the
now-removed visual flip layer) remains in `.local/verification/disciplines/`.

Earlier camera/white-arena verification (21 tests at that revision) remains in
`.local/verification/showcase/`: orbit/zoom, manual/automatic recenter, reset,
natural ramp jumps, and a controlled 14 m camera arm shortened to about 8.75 m.
Earlier bike-only evidence remains in `.local/verification/bevy/`.

The preserved `rider-assets` binary was previously run locally against the actual
installed game: all six indexes loaded (1,301,684 entries), and 23 pedaling
containers decoded without failures. Original files were opened read-only;
whole-source-file before/after hashes were not taken.

This revision also read the actual indexes for ID07 names (1,502 animation-tagged
entries) and checked all 275 `AIR_UNGRAB`-named containers with zero decode failures.
The name census and raw checks are in `.local/verification/disciplines/`.
Container integrity is not track decoding, and index names alone do not establish
which discipline uses a clip.

Use `cargo test --locked --bins` for the regression checks.
**AISandbox is prohibited for this project** by [AGENTS.md](AGENTS.md);
build, test and debug locally.

## Prior retail extraction verification

The earlier, pre-rule sandbox verification used exact index bytes from all six installed archives and
exact selected retail payloads, with SHA-256 checks for every copied segment.
The verification files were sparse: **unsampled payloads were absent**, so this is
not a claim that every resource in the installation was decoded.

| Check | Observed result |
| --- | --- |
| Forge v27 indexes | 6 archives, 1,301,684 entries |
| Animation-tagged index entries | 9,111 across all archives; not deduplicated |
| Selected ID07-named resource containers | 1,505 decoded, zero failures |
| Pedaling animation containers | 23 decoded, zero failures |
| Selected bike puppet skeleton container | 1 decoded, zero failures |
| Selected bike air-trick settings containers | 4 decoded, zero failures |
| Selected bike collision/physics-property containers | 2 decoded, zero failures |
| Parser regression checks | 3 passed |
| Existing-output, empty-search and malformed-ID rejection | Passed |
| Copied retail index/payload bytes after checks | Unchanged |
| Installed toolkit command dispatch | Query and REA help succeeded with Node 24 |

The observed data format uses two v2/algorithm-5 Zstandard block streams and
seed-zero Adler-32 checksums. Decode size is bounded to 256 MiB per container;
Zstandard window size is bounded to 8 MiB. Sidecars, other codecs/versions and
unrecognized record layouts are rejected, not silently skipped. Archives are
examined separately; original-game patch precedence is not reproduced.

Evidence and selected extracted resources are in `.local/verification/`,
`.local/extracted/`, `.local/retail-samples.zip` and `.local/re/`. Run the permanent
parser checks alone with `cargo test --locked --no-default-features --bin rider-assets`.

## Original-game parity limits

GamePort2Rust's shipped Ghidra bridge successfully imported the selected executable,
identified its native mappings, and read its entry-point bytes without executing
it. The executable has no on-disk bytes in its original code/data sections and
stores bytes in packing sections. Exposed strings/exports did not yield a bike
physics or animation function contract. That observation does not prove that such
code does not exist after runtime loading.

Animation, skeleton and bike settings **containers are recoverable**. Their inner
Riders-specific track/object schemas, rig bindings, units, blend/IK transitions,
and bike simulation algorithms are not established. Public older-Anvil animation
documentation does not certify compatibility with Riders Republic. The Bevy
model and procedural animation do not claim to reproduce those opaque schemas.

A faithful full port needs recoverable bike runtime contracts (or a documented,
measurable behavior reference), verified Riders-specific skeletal/animation
layouts, and an original-game parity reference. No retail executable, anti-cheat
service, runtime attachment or protection-bypass operation was run.

## Format references

Independent reader implementation informed by these public descriptions, then
checked against actual Riders Republic index/container bytes. Cross-title resource
semantics remain unverified.

- [Forge v27 layout](https://github.com/dataterminals/grb-modding-knowledgebase/blob/9f51464a47011f1bf7c006ce7b3277ef26622c29/docs/02-forge-file-format.md)
- [Resource records and extended headers](https://github.com/dataterminals/grb-modding-knowledgebase/blob/9f51464a47011f1bf7c006ce7b3277ef26622c29/docs/03-data-and-resources.md)
- [Older Anvil animation support limits](https://github.com/Kamzik123/AnvilToolkit-Resources/wiki/Creating-Custom-Animations)
