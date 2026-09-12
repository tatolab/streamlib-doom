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

What made it fast was not resolution: at this size the models are launch-bound, so the UNet costs the same on 64×40 latents as on 44×28. Encoding each prompt once and letting inductor replay the denoising step as a CUDA graph took a frame from 98 ms to 52 ms under full load. The models pace themselves — diffusion 12 Hz, depth 4 Hz, detector 1.5 Hz — so the renderer keeps 60 fps and the console around 50 with the GPU at 50 to 80 percent. Telemetry shows the GPU load, each model's frame time and score, and the console's own rate, so the picture never claims more than it measures.

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
