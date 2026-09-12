<div align="center">

# streamlib-doom

**The first level of DOOM (1993), rendered by [StreamLib](https://github.com/tatolab/streamlib) processors and playable from your phone.**

Not a port of the Doom engine. The real E1M1 from the shareware WAD — its geometry, textures, sprites, sounds and score — rendered by a GPU column-caster written as a StreamLib processor, driven by a game simulation in another, composited by a third, and streamed to a browser by a fourth. Every one of them is Python in its own helper process on top of the Rust engine.

<img src="https://gh-artifact.tatolab.com/streamlib-doom/e1m1-demo.gif" alt="E1M1 rendered by StreamLib" width="640">

[Watch the LEGO reel](https://gh-artifact.tatolab.com/streamlib-doom/doom-lego.mp4) · [Play it](#play-it-on-your-phone) · [Watch the console live](#watch-the-console-live-while-someone-else-plays) · [The robot console](#the-robot-console-sensors-autonomy-and-claude-all-live) · [How it works](#how-it-works) · [WebRTC](#webrtc-h264-over-whip-and-whep)

</div>

---

## What this is

"Can it run Doom?" is computing's oldest litmus test. StreamLib is a real-time perception and control runtime, so the honest way to answer was not to port the 1993 engine but to build the level out of the runtime's own parts:

- **The game** is a processor ticking at 35 tics a second: sliding collision against the level's 475 linedefs with Doom's 24-unit step and 56-unit clearance, the four doors, the lift, the secret floor, the exit switch, every pickup with its original message and palette flash, hitscan pistol and 7-pellet shotgun at their real cadences, zombiemen and imps that wake on line of sight, chase, shoot, throw fireballs, take pain and die through their own animation frames.
- **The renderer** is a GPU compute kernel, one invocation per screen column, the way the 1993 engine walked the screen: it casts through every linedef, sorts the hits, walks them front to back drawing walls, flats and sky into a shrinking clip window, then billboards sprites against the depth it recorded. Every pixel goes through COLORMAP, so what it publishes is the engine's own 8-bit framebuffer.
- **The status bar** is the real STBAR with the real digits, face, arms panel and ammo table at id's own coordinates, and the weapon lit like the room it's in.
- **The browser bridge** serves the page and sends each 320×200 indexed frame over a WebSocket, about 20 KB deflated at 35 fps. The page applies PLAYPAL itself on a canvas — exactly the indexed buffer Doom kept. Touch controls come back over a second WebSocket straight into the game processor.
- **The sound** is a two-operator FM synthesizer reading the GENMIDI patches from the WAD, the way a Sound Blaster played "At Doom's Gate", with the original effects triggered by game events.

The graph is four processors and three links. The phone is two more links.

## Requirements

- Linux x86_64 with a Vulkan GPU. The renderer, compositor and upscaler are compute kernels; recording and WebRTC also need Vulkan Video H.264 encode (NVIDIA, or AMD with Mesa's encoder).
- Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/).
- A phone or laptop on the same network. Any browser from 2023 on; the page uses `DecompressionStream`.

The shareware WAD is downloaded from the Internet Archive on first run and verified by hash. It is id Software's freely redistributable 1993 shareware episode and is not part of this repository.

## Watch the console live, while someone else plays

The console is not only a recording: point it at a WHIP endpoint and the whole 1920×1080
dashboard — the live graph, the game, the re-render, the sensor strip and the telemetry —
publishes as H.264 and Opus, so anyone on the network can watch it in a browser while
someone plays on their phone.

```bash
# a MediaMTX (or any WHIP/WHEP server) on the same box
export STREAMLIB_WHIP_URL=http://127.0.0.1:8889/doom/whip             # the game, for the phone's WebRTC option
export STREAMLIB_DOOM_CONSOLE_WHIP_URL=http://127.0.0.1:8889/console/whip   # the console itself
uv run streamlib run -f showcase.py
```

Then, from any device on the network:

| what | where |
|---|---|
| play | `http://<LAN address>:8666/` |
| watch the console, low latency | `http://<LAN address>:8889/console` |
| watch it on anything, ~5 s behind | `http://<LAN address>:8888/console/index.m3u8` |
| open it in VLC or ffplay | `rtsp://<LAN address>:8554/console` |

The two publishes are independent: the phone's stream is the 1280×960 game, the console's is
the dashboard. Both come out of the same graph, from the same frames, encoded by the engine's
own H.264 encoder — nothing is screen-captured.

The model panes read *no diffusion node yet · add one over MCP* until something adds them,
because the four networks own most of the GPU and the node does not assume you want them.
`scripts/neural-setup.py` adds and proves them in about a minute. They are dynamic processors, so a
node restart leaves them behind; `scripts/neural-setup.py --watch` stays running and puts them back
whenever the graph comes up without them, which is what the long-lived service runs.

## Play it on your phone

```bash
git clone https://github.com/tatolab/streamlib-doom
cd streamlib-doom
uv sync                          # streamlib comes from tatolab's wheel index
uv run streamlib run             # boots the node; the first run fetches the WAD
```

Then open `http://<this machine's LAN address>:8666/` on your phone and tap the screen. The moment you
touch the controls the marine is yours and stays yours — through every pause to aim, wait out a door or
read the room — until you put the phone down for half a minute or hand it back with
`{"command":"control","mode":"auto"}`. `{"command":"control","mode":"stop"}` switches every machine
driver off, the planner and the built-in autopilot alike. Left pad moves and strafes, right pad drags to turn, FIRE shoots, USE opens doors and flips the exit switch, WPN swaps to the shotgun once you've picked it up in the courtyard. When you die, FIRE restarts. On a laptop: WASD, arrows, space, E.

Ports: 8666 serves the page and frames, 8667 takes controls, 9200 is the node's own control plane and MCP endpoint.

To keep it running as a service that survives your terminal:

```bash
scripts/install-service.sh       # a user-level systemd unit named streamlib-doom
```


## A transcript, so an agent never has to look at a frame

An agent watching this game directly would pull frames, and a frame costs orders of magnitude more
tokens than a line of text and arrives thirty-five times a second. `streamlib_doom/transcript.py`
watches instead: it reads the world and the perception node's detections at full rate, keeps the
state a narrator would keep, and writes a line only when something changes.

```
[67]  78.1s  damage    took 4 damage, health 96
[69]  79.9s  damage    took 18 damage, health 82
[70]  80.3s  pickup    picked up 4 shells
[71]  80.4s  move      moved about 1307 units, now in sector 56
```

Measured over one 90-second run: **2537 world updates watched, 75 lines written**, and the entire
session's transcript is 6.3 kB. An agent reads it with a cursor, so it asks only for what it has
not seen:

```shell
curl "http://127.0.0.1:8670/transcript?since=40"   # the lines after 40
curl "http://127.0.0.1:8670/transcript?tail=10"    # the last ten
```

It reads the game's own state — health, kills, position — not the picture, so it is a transcript of
a simulation the graph is inside of, not video analysis. That is the right call for a game, where
the state is simply available and far more accurate than perception on a 22-pixel sprite. It is not
the right call as a claim about watching video. The honest version of this takes its detections from
the Grounding DINO node and its ranges from the depth network, which read only pixels and would work
the same on a real camera, and uses the game's state to score them rather than to write the lines.
Both nodes are already in the graph doing exactly that grading; the transcript just does not use
them yet.

Getting the noise out was most of the work. Sector changes fire every few steps in a corridor, so
movement is reported by distance travelled instead; the autonomy regenerates two health a tic, so
only a gain of ten or more counts as a pickup; a target stepping behind a pillar changes the
contact count twice a second, so a count has to hold for half a second before it is written down.
Without those three rules the same run produced 466 lines instead of 75.

## The monsters, swapped for something else, tracked live

The renderer already writes what every pixel is and how far away it is, and the game already
publishes where every monster stands. That is everything needed to put something else there:
`streamlib_doom/models.py` marches a signed distance field of a robot at each monster's own world
position in a compute kernel, scales it by range, and depth-tests it against the renderer's depth
channel so a wall in front still hides it. The original sprite is painted out with the wall behind
it first. No model is loaded and no network runs — one GPU kernel on data the graph was already
carrying, at 30 Hz, tracking whatever the game does.

### The diffusion re-render is off by default

It stays in the tree and one environment variable brings it back:

```shell
STREAMLIB_DOOM_RERENDER=sdturbo scripts/neural-setup.py   # sd-turbo with two ControlNets
STREAMLIB_DOOM_RERENDER=video   scripts/neural-setup.py   # StreamDiffusionV2, a causal video model
```

It is off because a 320x200 palette frame gives a diffusion model very little to build on, and the
result reads as mush however it is tuned. Everything below is what was learned trying, and it all
still applies to a source with real detail in it — a camera, or a 1080p video — where the same
pipeline looks far better. Left in for anyone who wants to play with it.

## One graph, three worlds — LEGO, neon, and whatever Claude picks

<a href="https://gh-artifact.tatolab.com/streamlib-doom/doom-worlds.mp4"><img src="https://gh-artifact.tatolab.com/streamlib-doom/worlds-cyberpunk.png" alt="the same corridor as LEGO, as blade runner neon, and as stained glass" width="900"></a>

**[Watch it](https://gh-artifact.tatolab.com/streamlib-doom/doom-worlds.mp4)** · [GIF](https://gh-artifact.tatolab.com/streamlib-doom/doom-worlds.gif) · stills: [LEGO](https://gh-artifact.tatolab.com/streamlib-doom/worlds-lego.png) · [neon](https://gh-artifact.tatolab.com/streamlib-doom/worlds-cyberpunk.png) · [stained glass, the director's pick](https://gh-artifact.tatolab.com/streamlib-doom/worlds-stained-glass.png) · [the border flash](https://gh-artifact.tatolab.com/streamlib-doom/worlds-flash.png)

The level in bricks, then one `style` command turns it into Blade Runner without restarting or
re-wiring anything, then `claude -p` picks a look of its own, adds the processor that applies it and
sends monsters to meet the robot in it. The sensors never notice: depth, detection and the map read
the game's frame, not the dream.

A pane's border flashes for a couple of seconds when what feeds it was reconfigured — a new style
on the re-render, a screen effect on the game, a swapped model, a node added to the graph — and
stays dark otherwise, so the eye goes to what changed and nothing else blinks.

## DOOM, rebuilt in LEGO while you play — a diffusion model, graded by its own renderer

<a href="https://gh-artifact.tatolab.com/streamlib-doom/doom-lego.mp4"><img src="https://gh-artifact.tatolab.com/streamlib-doom/lego-minifig.png" alt="the game beside its LEGO re-render, with neural depth, a detector and the level repainted" width="900"></a>

**[Watch the reel with audio](https://gh-artifact.tatolab.com/streamlib-doom/doom-lego.mp4)** · [the uncut take](https://gh-artifact.tatolab.com/streamlib-doom/doom-lego-full.mp4) · [GIF](https://gh-artifact.tatolab.com/streamlib-doom/doom-lego.gif)

Four neural networks, each its own process on the same GPU, all reading the renderer's frame through the surface's DLPack door — one engine-side blit to a CUDA tensor, no CPU hop — while the game keeps rendering at 60 fps beside them:

- **A diffusion re-render, held to one look.** sd‑turbo with a depth ControlNet. The renderer already writes log depth into the view's green channel, so the ControlNet's conditioning costs nothing and the bricks land on the game's real geometry. What keeps them *there* is reprojection: every model frame starts from the model's own previous output, moved into the new camera pose through the game's depth and the pose the renderer publishes, and only refined — the same trick a game's temporal anti-aliasing plays. The first frame of a look is re-imagined hard; every frame after it is a small step from the last, so a corridor stays the same corridor of bricks as you walk it. Monsters come out as minifigures.
- **12 Hz model, 60 fps picture.** The console reprojects the newest neural frame to the current camera on every frame it draws, per pixel, in its own kernel, using the depth in the frame it already has. The model runs at 12 frames a second; the pane carries about 34 new frames a second in the recording.
- **A depth network, graded against truth.** Depth Anything V2 on the RGB, aligned to the renderer's true depth in log space, with the abs‑rel error on screen every frame.
- **A detector, graded against labels.** Grounding DINO asked for monsters, scored live against the renderer's own per-pixel labels. Mid-reel its detections replace the oracle on the planner's link, and the robot fights from a learned detector.
- **A repainter, driven by Claude.** Claude names a material; the model generates wall and floor textures for it, they are contrast-stretched and Floyd–Steinberg dithered into the 1993 palette and written into the atlas the running renderer samples. The level changes under your feet and the brick view follows.

```bash
uv sync --extra neural                    # CUDA torch (cu126), diffusers, transformers
uv run streamlib run -f showcase.py       # the console
scripts/neural-setup.py                   # adds the four models and the lidar map over MCP and proves each is delivering
scripts/demo-reel-neural.py               # the reel
STREAMLIB_DOOM_STYLE=claymation …         # any preset, or a free-text prompt, as the held look
```

Holding a look steady is its own problem, and `streamlib_doom/capture.py` exists to settle it with
measurements rather than taste: add it over MCP, let it write the renderer's view and the pose that
drew it for a few dozen frames, remove it, and replay that one scene offline against as many settings
as you like. Two numbers matter — how much a frame changes from the last one reprojected into its
camera, and how much high-frequency detail survives — and they pull against each other. Carrying more
of the previous frame and refining it gently halves the change, but the carry is a resample, so the
studs bleed away over a few seconds and a perfectly steady picture is usually a washed-out one.
Carrying most of the previous frame is a trap, and the change metric alone will walk you straight
into it. A loop that starts each frame from 82% of its own last output holds beautifully still and
scores best on paper, but the carry is a resample: the colour and the studs drain away within
seconds, and the picture slowly stops being about the game at all. The rendered frame is the camera
and it has to drive every frame. What ships starts each denoise from the current frame at 65%, keeps
a minority carry to damp the shimmer, and replaces the lost stability with conditioning rather than
recycling — two ControlNets, the renderer's depth and its own per-pixel surface classes, painted in
ADE20K's colours so a segmentation ControlNet reads sky, wall, floor and monster as the things they
are. That costs about 5 ms a frame. The class map is free: the renderer already writes it into the blue
channel for the sensors.

With the geometry pinned by those two ControlNets the denoise can run at strength 0.85 instead of
0.5, which is what finally puts studs on every surface: the model replaces the flat 1993 texture
outright rather than tinting it, and the ControlNets stop the shapes drifting. A negative prompt
naming what to avoid — `3d render, video game graphics, low-poly, cell-shaded, flat shading` —
needs guidance above 1 to apply at all, and that second pass per step is worth it. Measured over
the captured walk, going from strength 0.5 to 0.85 with a negative prompt took detail from 5.4 to
14.3, nearly three times, at 6 Hz instead of 12.

Only the level is rebuilt. The same class channel that conditions the model also masks it: the
re-render is composited back over the game's own pixels everywhere the renderer says a pixel is not
a wall, floor or ceiling, so monsters, barrels, pickups, projectiles and the sky stay as 1993 drew
them and only the world around them turns to brick. The mask is feathered by one box blur so the
edge does not read as a cut-out. `STREAMLIB_DOOM_RESTYLED_CLASSES` takes the class codes to restyle,
default `1,2,3`; set it to `0,1,2,3,4,5,6,7` to put everything through the model as before.

Loading two ControlNets and compiling them takes about 95 s and the engine caps `setup()` at 60 s,
so the load runs on a worker thread and `process()` publishes nothing until it reports ready. A longer, more specific prompt measured
slightly worse on both counts; a reference image would need a base model IP-Adapter supports, and
sd-turbo's SD 2.1 is not one.

What made it fast was not resolution: at this size the models are launch-bound, so the UNet costs the same on 64×40 latents as on 44×28. Encoding each prompt once and letting inductor replay the denoising step as a CUDA graph took a frame from 98 ms to 52 ms under full load. The models pace themselves — diffusion 12 Hz, depth 4 Hz, detector 1.5 Hz — so the renderer keeps 60 fps and the console around 50 with the GPU at 50 to 80 percent. Telemetry shows the GPU load, each model's frame time and score, and the console's own rate, so the picture never claims more than it measures.

### A causal video model, measured against the image model — and not shipped as the default

The obvious objection to a per-frame image model is that temporal consistency is bolted on. So
[StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2) (Wan 2.1 1.3B, DMD-distilled,
rolling KV cache over four-frame chunks) was benchmarked as a replacement. It runs, it is genuinely
steadier frame to frame, and it is **not** the default, because it is much blurrier and it costs the
rest of the graph too much.

Every number below is on one RTX 3090. Quality is the same 48-frame capture and the same two metrics
`streamlib_doom/capture.py` exists for, scored at the 512×320 the console samples, with the shipping
sd-turbo settings as the control. `noise_scale` 0.75 except the last row.

| size | step | TAEHV | out fps | ms/chunk | VRAM peak | first frame | change, reprojected | change, plain | detail |
|---|---|---|---|---|---|---|---|---|---|
| 480×832 | 1 | no | 5.4 | 746 | 13.3 GB | 1.35 s | 16.02 | 17.78 | 1.63 |
| 480×832 | 1 | yes | 9.0 | 446 | 11.1 GB | 0.91 s | 19.51 | 17.29 | 1.74 |
| 480×832 | 2 | no | 4.5 | 885 | 15.1 GB | 1.55 s | 18.38 | 18.48 | 1.47 |
| 480×832 | 2 | yes | 6.7 | 585 | 12.7 GB | 1.10 s | 21.81 | 18.36 | 1.59 |
| 320×544 | 1 | no | 11.8 | 340 | 8.1 GB | 0.82 s | 16.41 | 15.44 | 2.06 |
| 320×544 | 1 | yes | 19.3 | 207 | 7.1 GB | 0.62 s | 18.59 | 14.79 | 1.91 |
| 320×544 | 2 | no | 9.9 | 397 | 8.8 GB | 0.92 s | 19.03 | 15.69 | 1.80 |
| 320×544 | 2 | yes | 14.8 | 264 | 7.8 GB | 0.71 s | 20.34 | 15.86 | 1.61 |
| 256×448 | 1 | no | 17.4 | 230 | 6.7 GB | 0.66 s | 16.02 | 14.28 | 1.67 |
| 256×448 | 1 | yes | 27.9 | 144 | 6.1 GB | 0.55 s | 17.10 | 13.45 | 1.63 |
| 256×448 | 2 | no | 14.2 | 277 | 7.2 GB | 0.74 s | 18.51 | 13.87 | 1.69 |
| 256×448 | 2 | yes | 20.4 | 191 | 6.7 GB | 0.67 s | 19.91 | 13.87 | 1.60 |
| 320×544, noise 0.85 | 1 | yes | 19.4 | 206 | 7.1 GB | 0.61 s | 19.66 | 13.45 | **2.33** |
| **sd-turbo, what ships** | — | — | 19.0 | 53 per frame | 3.6 GB | — | **10.28** | 19.30 | **6.36** |

Throughput is no obstacle: TAEHV is worth 1.6–1.7× everywhere, and 256×448 at one step sustains 27.9
fps. Quality is. On plain consecutive-frame change the video model wins clearly — 13.5 against 19.3,
30% less raw flicker, which is exactly what a causal model with a KV cache should buy. But once the
camera motion is compensated for, the image model's reprojected feedback loop is steadier still
(10.28 against 16–22), and the video model carries a third of the detail at best. That is not an
artefact of scoring at 512×320: at its own 256×448 the best configuration measures 2.98 against
sd-turbo's 6.36. Raising `noise_scale` from 0.75 to 0.85 buys 45% more detail for no extra time and
is the default here, and it still is not close.

On the full graph — renderer at 60, depth, detector, lidar, map and the console, mission `patrol`,
60 s each, measured from the bags' own publish stamps and the console's `MARKER:CONSOLE_TIMING`:

| | sd-turbo | video, 320×544 step 1 TAEHV, paced 16 |
|---|---|---|
| re-render output | 11.9 fps | 17.4 fps |
| render → re-render lag, median / p90 | 486 / 514 ms | 1286 / 1400 ms |
| GPU | 63% | 97% |
| GPU memory peak | 9.9 GB | 12.4 GB |
| console compositor | 21.4 fps | 14.9 fps |
| neural pane, frame-to-frame change | 5.75 | **3.48** |
| neural pane, detail | **9.03** | 3.06 |

The bar set before any of this was measured: switch only if the video model holds ≥ 8 output fps,
≤ 1.2 s glass to glass, less frame-to-frame change at ≥ 80% of sd-turbo's detail, and the console
keeps ≥ 45 fps. It meets two of five. It is steadier and it is fast enough; it is 1.29 s behind the
game, it carries 34% of the detail, and at 97% GPU it drags the console from 21 fps to 15. So
sd-turbo stays the default and the video model is opt-in:

```bash
STREAMLIB_DOOM_RERENDER=video scripts/neural-setup.py     # sdturbo is the default
```

`STREAMLIB_DOOM_VIDEO_DIFFUSION_HEIGHT` / `_WIDTH` / `_STEP` / `_NOISE` / `_TAEHV` / `_FPS` tune it;
`STREAMLIB_DOOM_VIDEO_DIFFUSION_CKPT` and `STREAMDIFFUSIONV2_ROOT` say where the 23 GB of weights
live. Install it with `--no-deps` — the published package pins torch 2.6 and numpy 1.24, and the
engine's DLPack door needs a cu126 torch — plus `av einops ftfy imageio imageio-ffmpeg omegaconf
sentencepiece scikit-image`. It ran unmodified against torch 2.14, numpy 2.5, transformers 5.17 and
diffusers 0.40; the only thing it takes from transformers is `AutoTokenizer`.

Two things are worth knowing if you try it. The pipeline's `to(device)` moves the 11 GB umt5-xxl text
encoder onto the GPU with everything else, which on a 24 GB card is most of the budget for a prompt
that changes only when someone asks for a new style; keeping it on the CPU drops resident VRAM from
13.8 GB to 3.0 GB and costs 10 s per new style, off the frame thread and cached. And the engine gives
`setup()` 60 s while this checkpoint needs 87 s, so the load runs on the processor's worker thread and
the node simply publishes nothing until it is ready — which is exactly the case `scripts/neural-setup.py`
already proves with `tap` and re-wires.

One engine finding worth knowing if you build on this: a link wired over MCP into a helper that is still loading its model can fail to open its port on either end, and the runtime reports the helper as running before its setup has finished. `scripts/neural-setup.py` therefore proves each node with `tap` on its output and the console's own `/panes` status before it returns, and re-wires whichever link stays silent.

## The robot console: sensors, autonomy, and Claude, all live

<a href="https://gh-artifact.tatolab.com/streamlib-doom/doom-console.mp4"><img src="https://gh-artifact.tatolab.com/streamlib-doom/console-thermal.png" alt="the robot console: live graph, operator view, depth, segmentation, lidar, map, telemetry, Claude directing" width="900"></a>

**[Watch the 2‑minute reel with audio](https://gh-artifact.tatolab.com/streamlib-doom/doom-console.mp4)** · [the uncut take](https://gh-artifact.tatolab.com/streamlib-doom/doom-console-full.mp4) · [GIF](https://gh-artifact.tatolab.com/streamlib-doom/doom-console.gif)

Treat the marine as a robot and the renderer as its camera, and the game becomes a robotics stack made of StreamLib processors — every one in its own process, fanned out from the same simulation and the same GPU frame with no copies:

- **Perception** reads the rendered frame back and reports monsters by bearing and range from the segmentation and depth channels alone. It has no access to the simulation.
- **The planner** builds a costmap of the level at 16‑unit cells, runs A* to the mission's goal, string‑pulls the route, faces and USEs doors, and turns to fight what perception reports. Its controls enter the game over a link — the same port a phone's controls arrive on. A hand on the phone overrides it for 700 ms after every touch; the badge under the operator view flips between AUTONOMY and TELEOP.
- **The sensors** are added to the running graph over MCP, and their panes light up as the nodes appear: **depth** and **segmentation** are lookup‑table kernels over the renderer's own surface (the view is rgba8: palette index in red, log depth in green, surface class in blue); the **lidar** casts 360 rays a tic through the level's linedefs at eye height; the **occupancy map** folds the scans into free and occupied cells with the robot's trajectory and the planner's route — the level as the robot has discovered it, never the map file.
- **Telemetry** shows the loop: sim tics a second, console frames a second, the render → HUD → console latency chain in milliseconds, and the frame witness — which surface id is in which process right now.
- **The live graph** is the hero pane, drawn from the node's own `/api/graph`: every box a process, links pulsing with flow, a node added live scaling in gold with a chime from the mixer, a cut link flashing red.

```bash
uv run streamlib run -f showcase.py                       # the console: phone-playable, nothing recorded
STREAMLIB_DOOM_RECORDING=console.mp4 uv run streamlib run -f showcase.py
scripts/demo-reel.py                                      # adds the sensors live, brings Claude in, runs the story
./director.sh "send it to the courtyard and ambush it with four imps"
```

The reel is driven over the same MCP tools an agent uses — `add_processor`, `connect`, `disconnect`, `remove_processor` on the live node — and the beat that says *Claude* runs `claude -p` for real: it reads the snapshot and the state, adds `DirectorCommand` nodes named `Claude: …` (they draw in gold), and its two‑sentence reply lands on the caption bar. Mid‑reel the HUD → console link is cut, a thermal effect node is spliced in and the picture turns thermal while the sensors stay untouched, then the node is removed and the link reconnected. Frames never stop; the recorder never notices.

The 1920×1080 picture is composited at 60 fps by one kernel from textures the other processes published and recorded through the engine's own H.264 and Opus encoders into its MP4 writer — the runtime records this picture of itself. The game still simulates at Doom's 35 tics a second; the renderer runs at 60 and interpolates the camera and every sprite between tics, the way a modern source port does, so the picture is sixty distinct frames a second. The sensors pace themselves at 20 Hz and the lidar at 15, because every frame a sensor skips is a GPU request nothing else in the node waits on.

## How it works

```
                controls (WebSocket :8667)
  phone  ───────────────────────────────────►  DoomGame  ──world──►  E1M1GameRenderer
    ▲                                            35 Hz               one invocation per column
    │                                                                        │ 320x200 indexed
    │           frames (WebSocket :8666)                                     ▼
    └─────────────────────────────────────  BrowserFrameSender  ◄──  GameStatusBarCompositor
                 20 KB deflated, 35 fps        HTTP :8666 too          weapon, STBAR, face, messages
```

Each box is one `@processor` class in `streamlib_doom/`, running in its own child interpreter. The renderer and compositor bind the previous stage's published GPU surface directly into their compute dispatch — zero copies across the process boundary. `app.py` declares the graph in `setup(rt)`; `assemble.py` builds the same graph on a running node over its MCP endpoint instead, which is how an agent would do it.

| module | what it is |
|---|---|
| `wad.py` | the WAD reader: lumps, palettes, COLORMAP, patches, composed textures, flats, sprites, map lumps, BSP, DMX sounds, MUS events, GENMIDI |
| `atlas.py` | packs every picture into one texture with a rectangle table the shaders index |
| `shaders.py` | the column renderer, the compositor and the palette upscaler, GLSL compute |
| `game.py` | the simulation: movement, doors, lifts, pickups, weapons, monsters, damage |
| `synth.py` | MUS to PCM through GENMIDI's OPL2 patches, approximated in floating point |
| `processors.py` | the playable graph: game, renderer, compositor, browser bridge, audio mixer |
| `demo_processors.py` | the scripted seventeen-second demo and the recording graph |
| `wsserver.py` | RFC 6455 on the standard library, small enough to live in a processor |
| `web/index.html` | the page: canvas, palettes, touch pads, Web Audio, optional WHEP |
| `game.py` (director) | the verbs Claude drives: spawn, give, lights, effect, message, god, heal, autopilot |
| `director.py` | `DirectorCommand`, a processor that fires one instruction into the game when added |
| `effects.py` | the screen filters, as index remaps built from PLAYPAL and COLORMAP |
| `sensors.py` | depth and segmentation kernels over the renderer's surface, perception from the frame, the lidar, the occupancy mapper |
| `planner.py` | the costmap, A*, string pulling, door handling, and the mission planner processor whose controls enter the game over a link |
| `showcase.py` | the live-graph panel, the caption bar, telemetry with the frame witness, and the 1920x1080 console kernel the engine records |
| `neural.py` | the diffusion re-render, the depth network graded against truth, the detector graded against labels, and the repainter that patches the atlas |
| `capture.py` | writes the renderer's view and the pose that drew it, so a scene replays offline against a model as often as a parameter sweep needs |

## Record the demo

The recorded video at the top is the engine's own MP4 — the same renderer fed by a scripted world, through StreamLib's H.264 and Opus encoders into its fragmented MP4 writer, audio and video in one file:

```bash
uv run streamlib run -f demo.py          # Ctrl-C after ~25 s
```

## WebRTC: H.264 over WHIP and WHEP

The WebSocket transport needs nothing but the node. To stream the 1280×960 palette-upscaled picture as H.264 with its Opus mix instead, install the `streamlib-webrtc` extension and point the node at a WHIP endpoint:

```bash
uv sync --extra webrtc
export STREAMLIB_WHIP_URL=http://127.0.0.1:8889/doom/whip     # a MediaMTX on the same box, say
export STREAMLIB_WHEP_URL=http://<LAN address>:8889/doom/whep  # where the page plays it back from
uv run streamlib run
```

The page then offers "switch to WebRTC", plays the stream in a `<video>` over WHEP, and keeps sending controls over its own WebSocket. Cloudflare Stream works the same way with its `.../webRTC/publish` URL; the URL is a credential, keep it in the environment.

## Tests

```bash
uv sync --extra test
uv run pytest
```

The tests read the real WAD: the parser against id's byte layouts, the score's opening riff decoded to E2 E2 E3 E2 E2 D3, the simulation tic by tic — walls that hold, a pistol that fires every fourteen tics, a door that opens and closes on its own — and the planner driving that simulation to the courtyard through a door and a firefight without an engine in the loop.

## Credits

DOOM is a trademark of id Software LLC. The shareware episode is redistributed under the terms id published in 1993; nothing from it is committed here. Byte layouts follow [id's released source](https://github.com/id-Software/DOOM) and [Chocolate Doom](https://github.com/chocolate-doom/chocolate-doom)'s `mus2mid.c`. Built with [StreamLib](https://github.com/tatolab/streamlib).

## License

BUSL-1.1, the same as StreamLib. See [LICENSE](LICENSE).
