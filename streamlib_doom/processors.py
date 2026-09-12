"""The interactive graph: the game, the two renderers from the demo, and the
browser bridge. The phone is two links — controls in over one WebSocket into
the game, frames out over another from the compositor's 8-bit framebuffer.
"""
from __future__ import annotations

import json
import math
import os
import queue as _queue
import socket
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from . import shaders
from . import synth
from . import wsserver
from .demo_processors import _Shared, _log, _video_bag, _write_floats, RING_USAGE, VIEW_W, VIEW_H, SPRITE_PREFIXES as DEMO_PREFIXES
from .game import Game, TICRATE, WEAPON_SHOTGUN
from .game import THING_CLASS_DECORATION
from .wad import Wad

CONTROLS_PORT = 8667
PAGE_PORT = 8666
DIRECTOR_PORT = 8668
GAME_PREFIXES = sorted(set(DEMO_PREFIXES) | {"BAL1", "BLUD", "PUFF", "BEXP", "ARM2", "ROCK", "CBRA", "CAND", "COLU", "SBOX", "AMMO", "TFOG"})


# ---------------------------------------------------------------------------
TIC_NS = 1_000_000_000 // TICRATE


@processor(execution="continuous", interval_ms=10)
class DoomGame:
    """E1M1 at 35 tics a second, driven by whatever the browser last sent."""

    @output()
    def world_to_downstream(self) -> None: ...

    @input(delivery_profile="ordered")
    def director_from_upstream(self) -> None: ...

    @input(delivery_profile="ordered")
    def controls_from_upstream(self) -> None: ...

    def __init__(self) -> None:
        self.controls = {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "fire": 0, "use": 0, "weapon": 0}
        self.turn_accumulated = 0.0
        self.lock = threading.Lock()
        self.clients = 0
        self.frames = 0
        self.next_tick_ns = 0
        # A phone touching the screen owns the marine for 700 ms; otherwise a planner on the link does.
        self.last_touch_ns = 0
        self.planner_controls: dict | None = None
        self.planner_ns = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        wad = Wad()
        self.game = Game(wad, wad.level("E1M1"))
        self.director_queue: "_queue.Queue[dict]" = _queue.Queue()
        self._start_control_server()

        def on_message(message: dict) -> None:
            with self.lock:
                if any(message.get(key) for key in ("f", "s", "t", "fire", "use", "w")):
                    self.last_touch_ns = clock.monotonic_now_ns()
                self.controls["forward"] = float(message.get("f", 0.0))
                self.controls["strafe"] = float(message.get("s", 0.0))
                self.turn_accumulated += float(message.get("t", 0.0))
                self.controls["fire"] = int(bool(message.get("fire", 0)))
                self.controls["use"] = int(bool(message.get("use", 0)))
                if message.get("w"):
                    self.controls["weapon"] = int(message["w"])

        def on_client_change(delta: int) -> None:
            self.clients += delta
            if self.clients <= 0:
                with self.lock:
                    self.controls.update(forward=0.0, strafe=0.0, fire=0, use=0)

        wsserver.serve_controls(CONTROLS_PORT, on_message, on_client_change)
        _log("GAME_SETUP", controls_port=CONTROLS_PORT, director_port=DIRECTOR_PORT, monsters=len(self.game.monsters))

    def _start_control_server(self) -> None:
        """An HTTP endpoint the director drives with one curl — the reliable path,
        needing no link wired after this processor's setup. Every command the
        game's `director()` understands is a POST /director with that JSON body,
        or a plain `GET /director?command=spawn&kind=imp&count=3`."""
        director_queue = self.director_queue
        import json as _json
        import urllib.parse as _urlparse

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, code: int, body: dict) -> None:
                payload = _json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _enqueue(self, command: dict) -> None:
                director_queue.put(command)
                self._reply(200, {"accepted": command})

            def do_GET(self) -> None:
                parsed = _urlparse.urlparse(self.path)
                if parsed.path == "/director":
                    fields = {k: v[0] for k, v in _urlparse.parse_qs(parsed.query).items()}
                    for numeric in ("count", "factor"):
                        if numeric in fields:
                            fields[numeric] = float(fields[numeric]) if numeric == "factor" else int(float(fields[numeric]))
                    if "on" in fields:
                        fields["on"] = fields["on"].lower() in ("1", "true", "yes", "on")
                    self._enqueue(fields)
                else:
                    self._reply(404, {"error": "POST or GET /director"})

            def do_POST(self) -> None:
                if _urlparse.urlparse(self.path).path != "/director":
                    self._reply(404, {"error": "POST /director"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    command = _json.loads(self.rfile.read(length).decode() or "{}")
                except ValueError:
                    self._reply(400, {"error": "body must be JSON"})
                    return
                self._enqueue(command)

        server = ThreadingHTTPServer(("0.0.0.0", DIRECTOR_PORT), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        # 35 tics a second against the monotonic clock, whatever cadence the helper calls at.
        now = clock.monotonic_now_ns()
        if self.next_tick_ns == 0:
            self.next_tick_ns = now
        while True:
            command = ctx.inputs.read("director_from_upstream")
            if command is None:
                break
            did = self.game.director(command)
            if did:
                _log("DIRECTOR", did=did, via="link")
        while not self.director_queue.empty():
            try:
                command = self.director_queue.get_nowait()
            except _queue.Empty:
                break
            command.setdefault("nonce", "http-" + str(clock.monotonic_now_ns()))
            did = self.game.director(command)
            if did:
                _log("DIRECTOR", did=did, via="http")
        while True:
            planned = ctx.inputs.read("controls_from_upstream")
            if planned is None:
                break
            self.planner_controls, self.planner_ns = planned, now
        ticked = 0
        while now >= self.next_tick_ns and ticked < 4:
            touching = now - self.last_touch_ns < 700_000_000
            planning = self.planner_controls is not None and now - self.planner_ns < 500_000_000
            if planning and not touching:
                controls = {key: self.planner_controls.get(key, 0) for key in ("forward", "strafe", "turn", "fire", "use", "weapon")}
                self.planner_controls["use"] = 0  # one plan, one press
                self.planner_controls["weapon"] = 0
                self.game.control_source = "autonomy"
            else:
                with self.lock:
                    controls = dict(self.controls)
                    controls["turn"] = self.turn_accumulated
                    self.turn_accumulated = 0.0
                    self.controls["weapon"] = 0
                self.game.control_source = "teleop" if touching else ("autopilot" if self.game.autopilot else "idle")
            self.game.tick(controls)
            self.next_tick_ns += TIC_NS
            ticked += 1
        if ticked == 0:
            return
        if now - self.next_tick_ns > 10 * TIC_NS:
            self.next_tick_ns = now
        world = self.game.snapshot()
        world["clients"] = self.clients
        world["route"] = (self.planner_controls or {}).get("route") or []
        world["stage_ns"] = {"game": clock.monotonic_now_ns()}
        ctx.outputs.write("world_to_downstream", world)
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            _log("GAME_FRAME", frames=self.frames, clients=self.clients)


# ---------------------------------------------------------------------------
class _GpuStage:
    def _upload_shared(self, gpu) -> None:
        self.shared = _Shared.__new__(_Shared)
        self.shared.wad = Wad()
        self.shared.level = self.shared.wad.level("E1M1")
        textures, flats = set(), set()
        for sd in self.shared.level.sidedefs:
            for name in sd[2:5]:
                if name != "-":
                    textures.add(name)
        for sector in self.shared.level.sectors:
            flats.add(sector[2])
            flats.add(sector[3])
        from .atlas import Atlas
        self.shared.atlas = Atlas(self.shared.wad, textures, flats, GAME_PREFIXES)
        atlas = self.shared.atlas
        self._atlas_texture = gpu.acquire_texture(atlas.image.shape[1], atlas.image.shape[0], "rgba8_unorm", ["texture_binding"])
        self._atlas_texture.lock(read_only=False)
        self._atlas_texture.as_numpy()[:, :, :] = atlas.image
        self._atlas_texture.unlock()
        self._level_texture = gpu.acquire_texture(4096, 4, "rgba32_float", ["texture_binding"])
        _write_floats(self._level_texture, self.shared.level_rows())
        self._colormap_texture = gpu.acquire_texture(256, 34, "rgba8_unorm", ["texture_binding"])
        self._colormap_texture.lock(read_only=False)
        cm = self._colormap_texture.as_numpy()
        cm[:, :, 0] = self.shared.wad.colormap()
        cm[:, :, 3] = 255
        self._colormap_texture.unlock()


@processor
class E1M1GameRenderer(_GpuStage):
    """The demo's column renderer, reading moving sector heights from the game."""

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def view_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        self._upload_shared(gpu)
        self._dynamic_texture = gpu.acquire_texture(512, 3, "rgba32_float", ["texture_binding"])
        self._kernel = gpu.create_compute_kernel(
            source=shaders.RENDERER_GLSL, push_constant_size=48,
            bindings={"atlas": "sampled_texture", "level": "sampled_texture", "dynamic": "sampled_texture",
                      "colormap": "sampled_texture", "view_image": "storage_image"})
        self.sky_entry = self.shared.atlas.ids["T:SKY1"]
        _log("RENDERER_SETUP", atlas_entries=len(self.shared.atlas.entries))

    def _dynamic_rows(self, world: dict) -> tuple[numpy.ndarray, int]:
        import math
        rows = numpy.zeros((3, 512, 4), dtype=numpy.float32)
        px, py, view_angle = world["x"], world["y"], world["angle"]
        f = (math.cos(view_angle), math.sin(view_angle))
        visible = []
        frames_by_prefix = self.shared.atlas.sprite_frames
        for x, y, z, prefix, frame, thing_angle, fullbright, light, *tail in world["things"]:
            kind = tail[0] if tail else THING_CLASS_DECORATION
            depth = (x - px) * f[0] + (y - py) * f[1]
            if depth < 4.0 or depth > 4000.0:
                continue
            to_viewer = math.degrees(math.atan2(py - y, px - x))
            rotation = int(((to_viewer - thing_angle + 202.5) % 360.0) / 45.0) & 7
            frames = frames_by_prefix.get(prefix, {})
            hit = frames.get((frame, rotation)) or frames.get((frame, 0))
            if hit is None:
                continue
            visible.append((depth, x, y, z, hit[0], hit[1], light, fullbright, kind))
        visible.sort(key=lambda v: -v[0])
        for n, (depth, x, y, z, entry, mirrored, light, fullbright, kind) in enumerate(visible[:256]):
            rows[0, 2 * n] = (x, y, z, entry)
            rows[0, 2 * n + 1] = (1.0 if mirrored else 0.0, light, float(fullbright), float(kind))
        tick, lights, floors, ceilings = world["tick"], world["lights"], world["floors"], world["ceilings"]
        for k, (floor, ceiling, floor_flat, ceiling_flat, light, special, tag) in enumerate(self.shared.level.sectors):
            rows[1, k] = (lights[k], self.shared.flat_entry(floor_flat, tick), self.shared.flat_entry(ceiling_flat, tick), 1.0 if ceiling_flat == "F_SKY1" else 0.0)
            rows[2, k] = (floors[k], ceilings[k], 0.0, 0.0)
        return rows, min(len(visible), 256)

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        world = ctx.inputs.read("world_from_upstream")
        if world is None:
            return
        rows, thing_count = self._dynamic_rows(world)
        _write_floats(self._dynamic_texture, rows)
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, VIEW_W, VIEW_H)
        push = struct.pack("<12f", world["x"], world["y"], world["z"], world["angle"],
                           float(world["tick"]), float(thing_count), float(world["sector"]), float(world["extralight"]),
                           float(len(self.shared.level.linedefs)), float(self.sky_entry), 0.0, 0.0)
        self._kernel.dispatch(
            bindings={"atlas": self._atlas_texture, "level": self._level_texture, "dynamic": self._dynamic_texture,
                      "colormap": self._colormap_texture, "view_image": slot},
            group_count=(VIEW_W // 32, 1, 1), push_constants=push)
        ctx.outputs.write("view_to_downstream", _video_bag(slot, VIEW_W, VIEW_H, hud=world["hud"], palette=world["palette"], state=world["state"],
                                                           event_log=world["event_log"], sector_light=world["lights"][world["sector"]],
                                                           extralight=world["extralight"], tick=world["tick"],
                                                           stage_ns={**(world.get("stage_ns") or {}), "render": clock.monotonic_now_ns()}, pid=os.getpid()))
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            _log("RENDERER_FRAME", frames=self.frames, things=thing_count)


# ---------------------------------------------------------------------------
ST_AMMO_X, ST_AMMO_Y = 44, 171
ST_HEALTH_X, ST_ARMOR_X = 90, 221
ST_FACE_X, ST_FACE_Y = 143, 168
ST_ARMS_X, ST_ARMS_Y = 111, 172
ST_AMMO_TABLE_X, ST_AMMO_TABLE_MAX_X, ST_AMMO_TABLE_Y = 288, 314, 173
VIEWPORT_H = 168


@processor
class GameStatusBarCompositor(_GpuStage):
    """Weapon, flash, status bar, face and message over the game's view."""

    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def frame_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        self._upload_shared(gpu)
        self._hud_texture = gpu.acquire_texture(256, 1, "rgba32_float", ["texture_binding"])
        from .effects import build_all_remaps
        remaps = build_all_remaps(self.shared.wad)
        self._effect_remap = gpu.acquire_texture(256, remaps.shape[0], "rgba8_unorm", ["texture_binding"])
        self._effect_remap.lock(read_only=False)
        er = self._effect_remap.as_numpy()
        er[:, :, 0] = remaps
        er[:, :, 3] = 255
        self._effect_remap.unlock()
        self._kernel = gpu.create_compute_kernel(
            source=shaders.GAME_COMPOSITOR_GLSL, push_constant_size=16,
            bindings={"view_from_renderer": "sampled_texture", "atlas": "sampled_texture", "level": "sampled_texture",
                      "hud": "sampled_texture", "colormap": "sampled_texture", "effect_remap": "sampled_texture", "frame_image": "storage_image"})
        self.anchor = {2: self.shared.wad.patch("PISGA0"), 3: self.shared.wad.patch("SHTGA0")}
        _log("COMPOSITOR_SETUP")

    def _entry(self, name: str) -> int:
        return self.shared.atlas.ids["P:" + name]

    def _weapon_draws(self, hud: dict) -> list:
        ready = hud["ready"]
        anchor = self.anchor[ready]
        gun_x0 = int(160 - anchor.width / 2 + hud["bob_x"] * 0.6)
        gun_top = int(VIEWPORT_H - anchor.height + 4 + hud["bob_y"] * 0.5 + hud["raise"])
        prefix = "PISG" if ready == 2 else "SHTG"
        frame_name = f"{prefix}{hud['weapon_frame']}0"
        if "P:" + frame_name not in self.shared.atlas.ids:
            frame_name = f"{prefix}A0"
        gun = self.shared.wad.patch(frame_name)
        draws = [(self._entry(frame_name), gun_x0 + (anchor.left_offset - gun.left_offset), gun_top + (anchor.top_offset - gun.top_offset), 1)]
        if hud["flash"] > 0 and not hud["dead"]:
            flash_name = "PISFA0" if ready == 2 else ("SHTFA0" if hud["flash"] > 3 else "SHTFB0")
            flash = self.shared.wad.patch(flash_name)
            draws.append((self._entry(flash_name), gun_x0 + (anchor.left_offset - flash.left_offset), gun_top + (anchor.top_offset - flash.top_offset), 1))
        return draws if not hud["dead"] else []

    def _number(self, value: int, right_x: int, y: int, font: str, width: int) -> list:
        draws, x = [], right_x
        for digit in reversed(str(max(0, int(value)))):
            x -= width
            draws.append((self._entry(f"{font}{digit}"), x, y, 0))
        return draws

    def _text(self, text: str, x: int, y: int) -> list:
        draws = []
        for ch in text.upper():
            code = ord(ch)
            if 33 <= code <= 95 and f"P:STCFN{code:03d}" in self.shared.atlas.ids:
                draws.append((self._entry(f"STCFN{code:03d}"), x, y, 0))
                x += self.shared.wad.patch(f"STCFN{code:03d}").width
            else:
                x += 4
        return draws

    def _status_bar_draws(self, hud: dict) -> list:
        draws = [(self._entry("STBAR"), 0, 168, 0), (self._entry("STARMS"), 104, 168, 0)]
        ammo = hud["bullets"] if hud["ready"] == 2 else hud["shells"]
        draws += self._number(ammo, ST_AMMO_X, ST_AMMO_Y, "STTNUM", 14)
        draws += self._number(hud["health"], ST_HEALTH_X, ST_AMMO_Y, "STTNUM", 14)
        draws.append((self._entry("STTPRCNT"), ST_HEALTH_X, ST_AMMO_Y, 0))
        draws += self._number(hud["armor"], ST_ARMOR_X, ST_AMMO_Y, "STTNUM", 14)
        draws.append((self._entry("STTPRCNT"), ST_ARMOR_X, ST_AMMO_Y, 0))
        owned = set(hud["weapons"])
        for slot in range(6):
            weapon = slot + 2
            font = "STYSNUM" if weapon in owned else "STGNUM"
            draws.append((self._entry(f"{font}{weapon}"), ST_ARMS_X + (slot % 3) * 12, ST_ARMS_Y + (slot // 3) * 10, 0))
        face = hud["face"] if "P:" + hud["face"] in self.shared.atlas.ids else "STFST01"
        draws.append((self._entry(face), ST_FACE_X, ST_FACE_Y, 0))
        for row, (have, maximum) in enumerate(((hud["bullets"], 200), (hud["shells"], 50), (hud["rockets"], 50), (hud["cells"], 300))):
            y = ST_AMMO_TABLE_Y + row * 6
            draws += self._number(have, ST_AMMO_TABLE_X, y, "STYSNUM", 4)
            draws += self._number(maximum, ST_AMMO_TABLE_MAX_X, y, "STYSNUM", 4)
        if hud["message"]:
            draws += self._text(hud["message"][:60], 1, 1)
        if hud["dead"]:
            draws += self._text("You died. Press FIRE to restart.", 60, 80)
        return draws

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None:
            return
        hud = view["hud"]
        draws = self._weapon_draws(hud) + self._status_bar_draws(hud)
        rows = numpy.zeros((1, 256, 4), dtype=numpy.float32)
        for n, (entry, x, y, lit) in enumerate(draws[:200]):
            rows[0, n] = (entry, x, y, lit)
        _write_floats(self._hud_texture, rows)
        lightnum = min(15, (int(view["sector_light"]) >> 4) + int(view["extralight"]))
        weapon_map = max(0, min(31, (15 - lightnum) * 4 - 20))
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, VIEW_W, VIEW_H)
        effect_index = float((view.get("state") or {}).get("effect", 0))
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as upstream:
            self._kernel.dispatch(
                bindings={"view_from_renderer": upstream, "atlas": self._atlas_texture, "level": self._level_texture,
                          "hud": self._hud_texture, "colormap": self._colormap_texture, "effect_remap": self._effect_remap, "frame_image": slot},
                group_count=(VIEW_W // 8, VIEW_H // 8, 1),
                push_constants=struct.pack("<4f", float(min(len(draws), 200)), float(weapon_map), effect_index, float(view.get("tick", 0))))
        ctx.outputs.write("frame_to_downstream", _video_bag(slot, VIEW_W, VIEW_H, palette=view["palette"], event_log=view["event_log"], tick=view["tick"],
                                                            stage_ns={**(view.get("stage_ns") or {}), "hud": clock.monotonic_now_ns()}, pid=os.getpid(),
                                                            state=view["state"], hud={k: hud[k] for k in ("bullets", "shells", "health", "armor", "ready", "face", "message", "dead", "kills")}))
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            _log("COMPOSITOR_FRAME", frames=self.frames, draws=len(draws))


# ---------------------------------------------------------------------------
@processor
class BrowserFrameSender:
    """Serves the page, the palettes, the music and the effects over HTTP, and
    every composited frame over a WebSocket as the 8-bit buffer it is."""

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    def __init__(self) -> None:
        self.frames = 0
        self.last_event_tick = -1
        self.broadcaster = wsserver.Broadcaster()
        self.latest_indices: bytes = bytes(VIEW_W * VIEW_H)
        self.latest_palette = 0
        self.latest_state: dict = {}
        self.palette_table = None

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        wad = Wad()
        page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html"), "rb").read()
        palettes = wad.palettes().tobytes()
        self.palette_table = wad.palettes()
        bridge = self
        score = synth.render_score(wad, "D_E1M1", 96.0)
        mono = ((score[:, 0] + score[:, 1]) * 0.5)
        mono = mono[::2][: len(mono) // 2]  # 24 kHz is plenty for an FM score, and a quarter of the bytes
        music = _wav_bytes(numpy.clip(mono * 1.6, -1, 1), 24000)
        effects = {}
        for name in wad.lumps:
            pass
        for lump_name, _o, _s in wad.lumps:
            if lump_name.startswith("DS"):
                try:
                    rate, pcm = wad.sound(lump_name)
                    effects[lump_name] = _wav_bytes(pcm, rate)
                except (AssertionError, ValueError):
                    continue
        broadcaster = self.broadcaster

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/frames":
                    key = self.headers.get("Sec-WebSocket-Key")
                    if not key:
                        self.send_error(400)
                        return
                    self.connection.sendall(wsserver.handshake_response(key))
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    broadcaster.add(self.connection)
                    try:
                        while True:
                            opcode, payload = wsserver.read_frame(self.connection)
                            if opcode == 0x8:
                                break
                            if opcode == 0x9:
                                self.connection.sendall(wsserver.encode_frame(payload, 0xA))
                    except (OSError, ConnectionError):
                        pass
                    finally:
                        broadcaster.remove(self.connection)
                    self.close_connection = True
                    return
                if self.path in ("/", "/index.html"):
                    body, content_type = page, "text/html; charset=utf-8"
                elif self.path == "/config.json":
                    body, content_type = json.dumps({"whep_url": os.environ.get("STREAMLIB_WHEP_URL", "")}).encode(), "application/json"
                elif self.path == "/state.json":
                    body, content_type = json.dumps(bridge.latest_state).encode(), "application/json"
                elif self.path == "/snapshot.png":
                    body, content_type = bridge.snapshot_png(), "image/png"
                elif self.path == "/palettes.bin":
                    body, content_type = palettes, "application/octet-stream"
                elif self.path == "/music.wav":
                    body, content_type = music, "audio/wav"
                elif self.path.startswith("/sfx/") and self.path.endswith(".wav") and self.path[5:-4] in effects:
                    body, content_type = effects[self.path[5:-4]], "audio/wav"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("0.0.0.0", PAGE_PORT), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _log("BRIDGE_SETUP", page_port=PAGE_PORT, effects=len(effects), music_bytes=len(music))

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        latest = ctx.inputs.read("frame_from_upstream")
        if latest is None:
            return
        events = [name for tick, name in latest.get("event_log") or [] if tick > self.last_event_tick]
        if latest.get("event_log"):
            self.last_event_tick = max(self.last_event_tick, max(tick for tick, _ in latest["event_log"]))
        self.frames += 1
        if self.broadcaster.count() == 0 and self.frames % 7 != 0:
            return
        with ctx.gpu_limited_access.resolve_surface(latest["surface_id"]) as surface:
            surface.lock()
            indices = surface.as_numpy()[:VIEW_H, :VIEW_W, 0].tobytes()
            surface.unlock()
        self.latest_indices, self.latest_palette = indices, int(latest.get("palette", 0))
        self.latest_state = {**(latest.get("state") or {}), "hud": latest.get("hud") or {}, "tick": latest.get("tick")}
        if self.broadcaster.count() == 0:
            return
        compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
        payload = bytes([int(latest.get("palette", 0)) & 0xFF, 1]) + compressor.compress(indices) + compressor.flush()
        self.broadcaster.send(wsserver.binary_frame(payload))
        if events:
            self.broadcaster.send(wsserver.text_frame(json.dumps({"events": events})))
        if self.frames in (1, 35, 35 * 60):
            _log("BRIDGE_FRAME", frames=self.frames, bytes=len(payload), clients=self.broadcaster.count())


def _png_bytes(rgb: numpy.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(height))
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _snapshot_png(bridge) -> bytes:
    indices = numpy.frombuffer(bridge.latest_indices, dtype=numpy.uint8).reshape(VIEW_H, VIEW_W)
    rgb = bridge.palette_table[bridge.latest_palette][indices]
    rgb = numpy.repeat(numpy.repeat(rgb, 2, axis=0), 2, axis=1)  # 640x400, readable at a glance
    return _png_bytes(numpy.ascontiguousarray(rgb))


BrowserFrameSender.snapshot_png = _snapshot_png


def _wav_bytes(samples: numpy.ndarray, rate: int) -> bytes:
    pcm = (numpy.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# ---------------------------------------------------------------------------
AUDIO_LEAD_NS = 30_000_000
GRAPH_URL = os.environ.get("STREAMLIB_DOOM_CONTROL_URL", "http://127.0.0.1:9200") + "/api/graph"
EFFECT_GAIN = {"DSSHOTGN": 0.55, "DSPISTOL": 0.5, "DSTELEPT": 0.7, "DSITEMUP": 0.6, "DSWPNUP": 0.6, "DSDOROPN": 0.5, "DSDORCLS": 0.5, "DSPLPAIN": 0.5}


def _chime(frequencies: tuple, note_seconds: float) -> numpy.ndarray:
    """Two FM-ish sine notes in a row with a quick decay: the sound of the graph changing."""
    n = int(synth.SAMPLE_RATE * note_seconds)
    t = numpy.arange(n) / synth.SAMPLE_RATE
    notes = [numpy.sin(2 * math.pi * f * t) * numpy.exp(-t * 14.0) * 0.5 for f in frequencies]
    return numpy.concatenate(notes).astype(numpy.float32)
AUDIO_BLOCK_FRAMES = synth.SAMPLE_RATE // 100


@processor(execution="continuous", interval_ms=4)
class DoomAudioMixer:
    """The score and every sound effect the game fires, mixed live into 10 ms
    blocks for an Opus encoder — what a WHIP session or a recording hears."""

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def audio(self) -> None: ...

    def __init__(self) -> None:
        self.anchor_ns = 0
        self.next_block = 0
        self.last_event_tick = -1
        self.pending: list[tuple[int, str]] = []

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        wad = Wad()
        self.music = synth.render_score(wad, "D_E1M1", 96.0) * 0.7
        self.effects = {}
        for lump_name, _o, _s in wad.lumps:
            if lump_name.startswith("DS"):
                try:
                    self.effects[lump_name] = synth.resample_effect(*wad.sound(lump_name)) * EFFECT_GAIN.get(lump_name, 0.6)
                except (AssertionError, ValueError):
                    continue
        self.effects["GRAPH_NODE_ADDED"] = _chime((880.0, 1318.5), 0.11)
        self.effects["GRAPH_LINK_CUT"] = _chime((110.0, 82.4), 0.16) * 1.4
        self.graph_events: "_queue.Queue[str]" = _queue.Queue()
        threading.Thread(target=self._watch_graph, daemon=True).start()
        _log("MIXER_SETUP", effects=len(self.effects))

    def _watch_graph(self) -> None:
        """Polls the node's own graph off the audio thread and turns growth into a sound cue."""
        import urllib.request
        known_nodes, known_links = None, None
        while True:
            try:
                with urllib.request.urlopen(GRAPH_URL, timeout=2) as reply:
                    graph = json.load(reply)
                nodes = {n["id"] for n in graph.get("nodes", [])}
                links = {(l.get("from") or l.get("source") or str(l), l.get("to") or l.get("target") or "") for l in graph.get("links", graph.get("edges", []))}
                if known_nodes is not None:
                    for _ in nodes - known_nodes:
                        self.graph_events.put("GRAPH_NODE_ADDED")
                    if links - known_links == set() and known_links - links:
                        self.graph_events.put("GRAPH_LINK_CUT")
                known_nodes, known_links = nodes, links
            except Exception:
                pass
            time.sleep(0.25)

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        now = clock.monotonic_now_ns()
        if self.anchor_ns == 0:
            self.anchor_ns = now
        while not self.graph_events.empty():
            self.pending.append((self.next_block, self.graph_events.get_nowait()))
        world = ctx.inputs.read("world_from_upstream")
        if world is not None:
            for tick, name in world.get("event_log") or []:
                if tick > self.last_event_tick and name in self.effects:
                    self.pending.append((self.next_block, name))
                    _log("MIXER_EFFECT", name=name, tick=tick, lag_ms=(self.anchor_ns + self.next_block * 10_000_000 - now) // 1_000_000)
            if world.get("event_log"):
                self.last_event_tick = max(self.last_event_tick, max(tick for tick, _ in world["event_log"]))
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 500 == 0:
            _log("MIXER_CADENCE", calls=self.calls, elapsed_ms=(now - self.anchor_ns) // 1_000_000)
        while True:
            block_start_ns = self.anchor_ns + self.next_block * 10_000_000
            if block_start_ns - now > AUDIO_LEAD_NS:
                return
            start = (self.next_block * AUDIO_BLOCK_FRAMES) % len(self.music)
            block = self.music[start : start + AUDIO_BLOCK_FRAMES].copy()
            if len(block) < AUDIO_BLOCK_FRAMES:
                block = numpy.concatenate([block, self.music[: AUDIO_BLOCK_FRAMES - len(block)]])
            if self.pending:
                block *= 0.4
            still_pending = []
            for started_block, name in self.pending:
                effect = self.effects[name]
                offset = (self.next_block - started_block) * AUDIO_BLOCK_FRAMES
                if offset < len(effect):
                    chunk = effect[offset : offset + AUDIO_BLOCK_FRAMES]
                    block[: len(chunk), 0] += chunk
                    block[: len(chunk), 1] += chunk
                    still_pending.append((started_block, name))
            self.pending = still_pending
            block = numpy.tanh(block * 1.3) * 0.95  # a soft knee instead of a hard clip when shots overlap
            ctx.outputs.write("audio", {
                "samples": block.astype(numpy.float32).tobytes(), "sample_rate": synth.SAMPLE_RATE,
                "channels": 2, "sample_count": AUDIO_BLOCK_FRAMES, "dtype": "f32", "first_sample_timestamp_ns": block_start_ns,
            })
            self.next_block += 1
