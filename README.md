<div align="center">

# streamlib-doom

**The first level of DOOM (1993), rendered by [StreamLib](https://github.com/tatolab/streamlib) processors and playable from your phone.**

Not a port of the Doom engine. The real E1M1 from the shareware WAD — its geometry, textures, sprites, sounds and score — rendered by a GPU column-caster written as a StreamLib processor, driven by a game simulation in another, composited by a third, and streamed to a browser by a fourth. Every one of them is Python in its own helper process on top of the Rust engine.

<img src="https://gh-artifact.tatolab.com/streamlib-doom/e1m1-demo.gif" alt="E1M1 rendered by StreamLib" width="640">

[Watch the recorded demo with audio](https://gh-artifact.tatolab.com/streamlib-doom/e1m1-demo.mp4) · [Play it](#play-it-on-your-phone) · [How it works](#how-it-works) · [Record the demo](#record-the-demo) · [WebRTC](#webrtc-h264-over-whip-and-whep)

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

## Play it on your phone

```bash
git clone https://github.com/tatolab/streamlib-doom
cd streamlib-doom
uv sync                          # streamlib comes from tatolab's wheel index
uv run streamlib run             # boots the node; the first run fetches the WAD
```

Then open `http://<this machine's LAN address>:8666/` on your phone and tap the screen. Left pad moves and strafes, right pad drags to turn, FIRE shoots, USE opens doors and flips the exit switch, WPN swaps to the shotgun once you've picked it up in the courtyard. When you die, FIRE restarts. On a laptop: WASD, arrows, space, E.

Ports: 8666 serves the page and frames, 8667 takes controls, 9200 is the node's own control plane and MCP endpoint.

To keep it running as a service that survives your terminal:

```bash
scripts/install-service.sh       # a user-level systemd unit named streamlib-doom
```

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

The tests read the real WAD: the parser against id's byte layouts, the score's opening riff decoded to E2 E2 E3 E2 E2 D3, and the simulation tic by tic — walls that hold, a pistol that fires every fourteen tics, a door that opens and closes on its own.

## Credits

DOOM is a trademark of id Software LLC. The shareware episode is redistributed under the terms id published in 1993; nothing from it is committed here. Byte layouts follow [id's released source](https://github.com/id-Software/DOOM) and [Chocolate Doom](https://github.com/chocolate-doom/chocolate-doom)'s `mus2mid.c`. Built with [StreamLib](https://github.com/tatolab/streamlib).

## License

BUSL-1.1, the same as StreamLib. See [LICENSE](LICENSE).
