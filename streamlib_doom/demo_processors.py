"""E1M1 as a StreamLib graph: five processors, each in its own helper process.

DoomWorldState   ticks the scripted demo and animates the level's lights and things.
E1M1Renderer     casts every screen column through the real E1M1 linedefs on the GPU.
StatusBarCompositor  lays the weapon and the 1993 status bar over the view.
PaletteUpscaler  applies PLAYPAL and the 4:3 stretch, 320x200 -> 1280x960.
AtDoomsGate      streams the FM-synthesized score with the demo's sound effects.

Native H264Encoder, OpusEncoder and Mp4Sink complete the graph; the assembly
happens over MCP in `assemble.py`, never here.
"""
from __future__ import annotations

import math
import random
import struct

import numpy

from streamlib import (
    ProcessorOutputTextureRing,
    RuntimeContextFullAccess,
    RuntimeContextLimitedAccess,
    clock,
    input,
    log,
    monotonic_now_ns,
    output,
    processor,
)

from . import script
from . import shaders
from . import synth
from .atlas import Atlas
from .wad import Level, Wad

LEVEL_NAME = "E1M1"
VIEW_W, VIEW_H = 320, 200
OUT_W, OUT_H = 1280, 960
RING_USAGE = ["texture_binding", "storage_binding"]
DEMO_START_DELAY_NS = 6_000_000_000

THING_SPRITES: dict[int, tuple[str, str, bool]] = {  # type: (prefix, frames cycled, fullbright)
    3004: ("POSS", "A", False), 3001: ("TROO", "A", False), 9: ("SPOS", "A", False),
    2018: ("ARM1", "AB", False), 2014: ("BON1", "ABCDCB", True), 2015: ("BON2", "ABCDCB", True),
    2011: ("STIM", "A", False), 2012: ("MEDI", "A", False), 2007: ("CLIP", "A", False),
    2008: ("SHOT", "A", False), 2048: ("AMMO", "A", False), 2049: ("SBOX", "A", False),
    2046: ("BROK", "A", False), 2035: ("BAR1", "AB", False), 2028: ("COLU", "A", True),
    48: ("ELEC", "A", False), 24: ("POL5", "A", False), 10: ("PLAY", "W", False),
    12: ("PLAY", "W", False), 15: ("PLAY", "N", False), 18: ("POSS", "L", False),
    2001: ("SHOT", "A", False),
}
SPRITE_PREFIXES = sorted({v[0] for v in THING_SPRITES.values()})
MEDIUM_SKILL = 2

NUKAGE_CYCLE = ("NUKAGE1", "NUKAGE2", "NUKAGE3")
NUKAGE_TICS_PER_FRAME = 8


def _log(tag: str, **fields: object) -> None:
    log.info(f"MARKER:{tag} " + " ".join(f"{k}={v}" for k, v in fields.items()))


def _video_bag(handle, width: int, height: int, **extra) -> dict:
    return {"surface_id": handle.surface_id, "width": width, "height": height,
            "timestamp_ns": clock.monotonic_now_ns(), "fps": script.FPS, **extra}


def _write_floats(handle, rows: numpy.ndarray) -> None:
    """Write (h, w, 4) float32 rows into an rgba32_float texture through its staging."""
    handle.lock(read_only=False)
    view = handle.as_numpy()
    view[: rows.shape[0], : rows.shape[1], :] = rows
    handle.unlock()


class _Shared:
    """What every GPU stage builds for itself: the WAD, the level and the atlas."""

    def __init__(self) -> None:
        self.wad = Wad()
        self.level: Level = self.wad.level(LEVEL_NAME)
        textures, flats = set(), set()
        for sd in self.level.sidedefs:
            for name in sd[2:5]:
                if name != "-":
                    textures.add(name)
        for sector in self.level.sectors:
            flats.add(sector[2])
            flats.add(sector[3])
        self.atlas = Atlas(self.wad, textures, flats, SPRITE_PREFIXES)

    def flat_entry(self, name: str, tick: int) -> int:
        if name in NUKAGE_CYCLE:
            name = NUKAGE_CYCLE[(tick // NUKAGE_TICS_PER_FRAME) % 3]
        return self.atlas.ids.get("F:" + name, -1)

    def level_rows(self) -> numpy.ndarray:
        """(4, 4096, 4) float32: linedefs, sidedefs, sectors, atlas entries."""
        rows = numpy.zeros((4, 4096, 4), dtype=numpy.float32)
        lv, a = self.level, self.atlas
        for i, (v1, v2, flags, special, _tag, right, left) in enumerate(lv.linedefs):
            rows[0, 2 * i] = (lv.vertexes[v1][0], lv.vertexes[v1][1], lv.vertexes[v2][0], lv.vertexes[v2][1])
            rows[0, 2 * i + 1] = (flags, right, left, special)
        for j, (xoff, yoff, upper, lower, middle, sector) in enumerate(lv.sidedefs):
            rows[1, 2 * j] = (xoff, yoff, sector, 0)
            rows[1, 2 * j + 1] = (a.ids.get("T:" + upper, -1), a.ids.get("T:" + lower, -1), a.ids.get("T:" + middle, -1), 0)
        for k, (floor, ceiling, floor_flat, ceiling_flat, light, special, tag) in enumerate(lv.sectors):
            rows[2, 2 * k] = (floor, ceiling, light, special)
            rows[2, 2 * k + 1] = (self.flat_entry(floor_flat, 0), self.flat_entry(ceiling_flat, 0), tag, 1.0 if ceiling_flat == "F_SKY1" else 0.0)
        rows[3, : 2 * len(a.entries)] = a.table().reshape(-1, 4)
        return rows


# ---------------------------------------------------------------------------
@processor(execution="continuous", interval_ms=1000 // script.FPS)
class DoomWorldState:
    """The demo's clock, camera, lights and things, one bag per frame."""

    @output()
    def world_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self.epoch_ns = 0
        self.frames = 0
        self.random = random.Random(1993)
        self.flicker: dict[int, tuple[float, float]] = {}

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self.wad = Wad()
        self.level = self.wad.level(LEVEL_NAME)
        self.epoch_ns = monotonic_now_ns() + DEMO_START_DELAY_NS
        neighbours: dict[int, set[int]] = {}
        for v1, v2, flags, _s, _t, right, left in self.level.linedefs:
            if right >= 0 and left >= 0:
                a, b = self.level.sidedefs[right][5], self.level.sidedefs[left][5]
                neighbours.setdefault(a, set()).add(b)
                neighbours.setdefault(b, set()).add(a)
        self.min_light = [min([self.level.sectors[n][4] for n in neighbours.get(k, ())] + [s[4]]) for k, s in enumerate(self.level.sectors)]
        self.things = [tuple(int(v) for v in t) for t in self.level.things if int(t[3]) in THING_SPRITES and int(t[4]) & MEDIUM_SKILL]
        # Point-in-sector is a Python loop over every linedef; things that never move pay it once.
        self.thing_sector = [self.level.sector_at(x, y) for x, y, _a, _k, _f in self.things]
        _log("WORLD_SETUP", things=len(self.things), epoch_ns=self.epoch_ns)

    def _lights(self, tick: int, t: float) -> list[float]:
        lights = []
        for k, (floor, ceiling, ff, cf, light, special, tag) in enumerate(self.level.sectors):
            low = self.min_light[k]
            if special == 1:  # random flicker
                until, level = self.flicker.get(k, (0.0, light))
                if t >= until:
                    level = low if level == light else light
                    until = t + (self.random.randint(1, 7) if level == light else self.random.randint(1, 4)) * 4 / 35
                    self.flicker[k] = (until, level)
                lights.append(level)
            elif special in (2, 12):  # fast strobe
                lights.append(light if (tick % 20) < 5 else low)
            elif special in (3, 13):  # slow strobe
                lights.append(light if (tick % 40) < 5 else low)
            elif special == 8:  # glow
                span = max(light - low, 0)
                lights.append(low + span * (0.5 + 0.5 * math.sin(tick * 8 / max(span, 1) * 0.5)))
            elif special == 17:  # fire flicker
                lights.append(max(low, light - self.random.randint(0, 3) * 16))
            else:
                lights.append(light)
        return lights

    def _zombieman_one(self, t: float, px: float, py: float) -> tuple[float, float, str, float]:
        x, y = script.ZOMBIEMAN_ONE
        angle = 90.0
        frame = "A"
        if t >= script.ZOMBIEMAN_ONE_SIGHT_TIME:
            # It walks toward the marine until the shot lands, and dies where it stood.
            alive_until = min(t, script.ZOMBIEMAN_ONE_DEATH_TIME)
            advance = min(120.0, (alive_until - script.ZOMBIEMAN_ONE_SIGHT_TIME) * 45.0)
            heading = math.atan2(py - y, px - x)
            x += math.cos(heading) * advance
            y += math.sin(heading) * advance
            angle = math.degrees(heading)
            if t >= script.ZOMBIEMAN_ONE_DEATH_TIME:
                death_tics = (t - script.ZOMBIEMAN_ONE_DEATH_TIME) * 35
                frame = "HIJKL"[min(4, int(death_tics / 5))]
            elif any(0.0 <= t - f < 0.35 for f in script.ZOMBIEMAN_ONE_FIRE_TIMES):
                frame = "EF"[int(((t * 35) % 10) >= 5)]
            else:
                frame = "ABCD"[int((t * 35 / 4) % 4)]
        return x, y, frame, angle

    def _imp(self, t: float, px: float, py: float, x: float, y: float) -> tuple[float, float, str, float]:
        angle, frame = 135.0, "A"
        if t >= script.IMP_SIGHT_TIME:
            advance = min(300.0, (t - script.IMP_SIGHT_TIME) * 70.0)
            heading = math.atan2(py - y, px - x)
            x += math.cos(heading) * advance
            y += math.sin(heading) * advance
            angle = math.degrees(heading)
            frame = "ABCD"[int((t * 35 / 4) % 4)]
            if any(0.0 <= t - f < 0.2 for f in (16.25, 16.65)):
                frame = "G"
        return x, y, frame, angle

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        now = monotonic_now_ns()
        t = max(0.0, (now - self.epoch_ns) / 1e9)
        if t > script.DEMO_SECONDS + 1.0:
            return
        tick = int(t * 35)
        px, py, angle = script.pose_at(t)
        sector = self.level.sector_at(px, py)
        floor = self.level.sectors[sector][0]
        moving = script.is_moving(t)
        bob = math.sin(t * 2 * math.pi * 1.8) * (5.0 if moving else 0.0)
        things = []
        for index, (x, y, thing_angle, kind, _flags) in enumerate(self.things):
            prefix, frames, fullbright = THING_SPRITES[kind]
            frame = frames[(tick // 6) % len(frames)]
            fx, fy, fangle = float(x), float(y), float(thing_angle)
            thing_sector = self.thing_sector[index]
            if kind == 2018 and t >= script.ARMOUR_PICKUP_TIME:
                continue
            if (x, y) == script.ZOMBIEMAN_ONE:
                fx, fy, frame, fangle = self._zombieman_one(t, px, py)
                thing_sector = self.level.sector_at(fx, fy)
            elif kind == 3001 and (x, y) == (3440, -3472):
                fx, fy, frame, fangle = self._imp(t, px, py, fx, fy)
                thing_sector = self.level.sector_at(fx, fy)
            things.append([fx, fy, float(self.level.sectors[thing_sector][0]), prefix, frame, fangle, 1 if fullbright else 0, float(self.level.sectors[thing_sector][4])])
        extralight = 2 if script.pistol_flash_active(t) else 0
        ctx.outputs.write("world_to_downstream", {
            "t": t, "tick": tick, "epoch_ns": self.epoch_ns,
            "x": px, "y": py, "z": floor + script.VIEW_HEIGHT + bob * 0.5, "angle": angle,
            "sector": sector, "extralight": extralight, "moving": moving, "bob": bob,
            "bullets": script.bullets_at(t), "health": 100, "armour": script.armour_at(t),
            "palette": script.palette_at(t), "flash": script.pistol_flash_active(t),
            "lights": self._lights(tick, t), "things": things,
        })
        self.frames += 1
        if self.frames in (1, 35, 350):
            _log("WORLD_FRAME", frames=self.frames, t=round(t, 2), sector=sector)


# ---------------------------------------------------------------------------
@processor
class E1M1Renderer:
    """The 320x200 view: every column cast through the real level on the GPU."""

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def view_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        self.shared = _Shared()
        atlas = self.shared.atlas
        self._atlas_texture = gpu.acquire_texture(atlas.image.shape[1], atlas.image.shape[0], "rgba8_unorm", ["texture_binding"])
        self._atlas_texture.lock(read_only=False)
        self._atlas_texture.as_numpy()[:, :, :] = atlas.image
        self._atlas_texture.unlock()
        self._level_texture = gpu.acquire_texture(4096, 4, "rgba32_float", ["texture_binding"])
        _write_floats(self._level_texture, self.shared.level_rows())
        self._dynamic_texture = gpu.acquire_texture(512, 2, "rgba32_float", ["texture_binding"])
        colormap = self.shared.wad.colormap()
        self._colormap_texture = gpu.acquire_texture(256, 34, "rgba8_unorm", ["texture_binding"])
        self._colormap_texture.lock(read_only=False)
        cm = self._colormap_texture.as_numpy()
        cm[:, :, 0] = colormap
        cm[:, :, 3] = 255
        self._colormap_texture.unlock()
        self._kernel = gpu.create_compute_kernel(
            source=shaders.RENDERER_GLSL, push_constant_size=48,
            bindings={"atlas": "sampled_texture", "level": "sampled_texture", "dynamic": "sampled_texture",
                      "colormap": "sampled_texture", "view_image": "storage_image"})
        self.sky_entry = atlas.ids["T:SKY1"]
        _log("RENDERER_SETUP", atlas_entries=len(atlas.entries), linedefs=len(self.shared.level.linedefs))

    def _dynamic_rows(self, world: dict) -> tuple[numpy.ndarray, int]:
        rows = numpy.zeros((2, 512, 4), dtype=numpy.float32)
        px, py, view_angle = world["x"], world["y"], world["angle"]
        f = (math.cos(view_angle), math.sin(view_angle))
        visible = []
        for x, y, z, prefix, frame, thing_angle, fullbright, light in world["things"]:
            depth = (x - px) * f[0] + (y - py) * f[1]
            if depth < 4.0 or depth > 4000.0:
                continue
            to_viewer = math.degrees(math.atan2(py - y, px - x))
            rotation = int(((to_viewer - thing_angle + 202.5) % 360.0) / 45.0) & 7
            frames = self.shared.atlas.sprite_frames.get(prefix, {})
            hit = frames.get((frame, rotation)) or frames.get((frame, 0))
            if hit is None:
                continue
            entry, mirrored = hit
            visible.append((depth, x, y, z, entry, mirrored, light, fullbright))
        visible.sort(key=lambda v: -v[0])
        for n, (depth, x, y, z, entry, mirrored, light, fullbright) in enumerate(visible[:256]):
            rows[0, 2 * n] = (x, y, z, entry)
            rows[0, 2 * n + 1] = (1.0 if mirrored else 0.0, light, float(fullbright), 0.0)
        tick = world["tick"]
        for k, (floor, ceiling, floor_flat, ceiling_flat, light, special, tag) in enumerate(self.shared.level.sectors):
            rows[1, k] = (world["lights"][k], self.shared.flat_entry(floor_flat, tick), self.shared.flat_entry(ceiling_flat, tick), 1.0 if ceiling_flat == "F_SKY1" else 0.0)
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
        ctx.outputs.write("view_to_downstream", _video_bag(slot, VIEW_W, VIEW_H, world={
            k: world[k] for k in ("t", "sector", "extralight", "bob", "moving", "bullets", "health", "armour", "palette", "flash", "tick")
        }, sector_light=world["lights"][world["sector"]]))
        self.frames += 1
        if self.frames in (1, 35, 350):
            _log("RENDERER_FRAME", frames=self.frames, things=thing_count, t=round(world["t"], 2))


# ---------------------------------------------------------------------------
ST_AMMO_X, ST_AMMO_Y = 44, 171
ST_HEALTH_X, ST_ARMOR_X = 90, 221
ST_FACE_X, ST_FACE_Y = 143, 168
ST_ARMS_X, ST_ARMS_Y = 111, 172
ST_AMMO_TABLE_X, ST_AMMO_TABLE_MAX_X, ST_AMMO_TABLE_Y = 288, 314, 173
VIEWPORT_H = 168


@processor
class StatusBarCompositor:
    """The weapon in the room's light, then the status bar as it shipped in the WAD."""

    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def frame_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        self.shared = _Shared()
        atlas = self.shared.atlas
        self._atlas_texture = gpu.acquire_texture(atlas.image.shape[1], atlas.image.shape[0], "rgba8_unorm", ["texture_binding"])
        self._atlas_texture.lock(read_only=False)
        self._atlas_texture.as_numpy()[:, :, :] = atlas.image
        self._atlas_texture.unlock()
        self._level_texture = gpu.acquire_texture(4096, 4, "rgba32_float", ["texture_binding"])
        _write_floats(self._level_texture, self.shared.level_rows())
        self._hud_texture = gpu.acquire_texture(256, 1, "rgba32_float", ["texture_binding"])
        self._colormap_texture = gpu.acquire_texture(256, 34, "rgba8_unorm", ["texture_binding"])
        self._colormap_texture.lock(read_only=False)
        cm = self._colormap_texture.as_numpy()
        cm[:, :, 0] = self.shared.wad.colormap()
        cm[:, :, 3] = 255
        self._colormap_texture.unlock()
        self._kernel = gpu.create_compute_kernel(
            source=shaders.COMPOSITOR_GLSL, push_constant_size=16,
            bindings={"view_from_renderer": "sampled_texture", "atlas": "sampled_texture", "level": "sampled_texture",
                      "hud": "sampled_texture", "colormap": "sampled_texture", "frame_image": "storage_image"})
        self.pistol = self.shared.wad.patch("PISGA0")
        _log("COMPOSITOR_SETUP")

    def _entry(self, name: str) -> int:
        return self.shared.atlas.ids["P:" + name]

    def _weapon_draws(self, world: dict) -> list[tuple[int, int, int, int]]:
        bob = world["bob"]
        bob_x, bob_y = bob * 1.2, abs(bob) * 0.8
        gun_x0 = int(160 - self.pistol.width / 2 + bob_x)
        gun_top = int(VIEWPORT_H - self.pistol.height + 4 + bob_y)
        draws = []
        gun_name = "PISGB0" if world["flash"] else "PISGA0"
        gun = self.shared.wad.patch(gun_name)
        draws.append((self._entry(gun_name), gun_x0 + (self.pistol.left_offset - gun.left_offset), gun_top + (self.pistol.top_offset - gun.top_offset), 1))
        if world["flash"]:
            flash = self.shared.wad.patch("PISFA0")
            draws.append((self._entry("PISFA0"), gun_x0 + (self.pistol.left_offset - flash.left_offset), gun_top + (self.pistol.top_offset - flash.top_offset), 1))
        return draws

    def _number(self, value: int, right_x: int, y: int, font: str, width: int) -> list[tuple[int, int, int, int]]:
        draws = []
        x = right_x
        for digit in reversed(str(max(0, value))):
            x -= width
            draws.append((self._entry(f"{font}{digit}"), x, y, 0))
        return draws

    def _status_bar_draws(self, world: dict) -> list[tuple[int, int, int, int]]:
        draws = [(self._entry("STBAR"), 0, 168, 0), (self._entry("STARMS"), 104, 168, 0)]
        draws += self._number(world["bullets"], ST_AMMO_X, ST_AMMO_Y, "STTNUM", 14)
        draws += self._number(world["health"], ST_HEALTH_X, ST_AMMO_Y, "STTNUM", 14)
        draws.append((self._entry("STTPRCNT"), ST_HEALTH_X, ST_AMMO_Y, 0))
        draws += self._number(world["armour"], ST_ARMOR_X, ST_AMMO_Y, "STTNUM", 14)
        draws.append((self._entry("STTPRCNT"), ST_ARMOR_X, ST_AMMO_Y, 0))
        for slot in range(6):
            weapon = slot + 2
            font = "STYSNUM" if weapon == 2 else "STGNUM"
            draws.append((self._entry(f"{font}{weapon}"), ST_ARMS_X + (slot % 3) * 12, ST_ARMS_Y + (slot // 3) * 10, 0))
        face = "STFST01" if world["flash"] else ("STFST00", "STFST01", "STFST02", "STFST01")[(world["tick"] // 35) % 4]
        draws.append((self._entry(face), ST_FACE_X, ST_FACE_Y, 0))
        for row, (have, maximum) in enumerate(((world["bullets"], 200), (0, 50), (0, 50), (0, 300))):
            y = ST_AMMO_TABLE_Y + row * 6
            draws += self._number(have, ST_AMMO_TABLE_X, y, "STYSNUM", 4)
            draws += self._number(maximum, ST_AMMO_TABLE_MAX_X, y, "STYSNUM", 4)
        return draws

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None:
            return
        world = view["world"]
        draws = self._weapon_draws(world) + self._status_bar_draws(world)
        rows = numpy.zeros((1, 256, 4), dtype=numpy.float32)
        for n, (entry, x, y, lit) in enumerate(draws[:128]):
            rows[0, n] = (entry, x, y, lit)
        _write_floats(self._hud_texture, rows)
        lightnum = min(15, (int(view["sector_light"]) >> 4) + int(world["extralight"]))
        weapon_map = max(0, min(31, (15 - lightnum) * 4 - 20))
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, VIEW_W, VIEW_H)
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as upstream:
            self._kernel.dispatch(
                bindings={"view_from_renderer": upstream, "atlas": self._atlas_texture, "level": self._level_texture,
                          "hud": self._hud_texture, "colormap": self._colormap_texture, "frame_image": slot},
                group_count=(VIEW_W // 8, VIEW_H // 8, 1),
                push_constants=struct.pack("<4f", float(min(len(draws), 128)), float(weapon_map), 0.0, 0.0))
        ctx.outputs.write("frame_to_downstream", _video_bag(slot, VIEW_W, VIEW_H, palette=world["palette"], t=world["t"]))
        self.frames += 1
        if self.frames in (1, 35, 350):
            _log("COMPOSITOR_FRAME", frames=self.frames, draws=len(draws))


# ---------------------------------------------------------------------------
@processor
class PaletteUpscaler:
    """PLAYPAL and the 4:3 stretch: the 8-bit frame becomes 1280x960 RGB."""

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    @output()
    def video(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        palettes = Wad().palettes()
        self._palette_texture = gpu.acquire_texture(256, 14, "rgba8_unorm", ["texture_binding"])
        self._palette_texture.lock(read_only=False)
        pal = self._palette_texture.as_numpy()
        pal[:, :, :3] = palettes
        pal[:, :, 3] = 255
        self._palette_texture.unlock()
        self._kernel = gpu.create_compute_kernel(
            source=shaders.UPSCALER_GLSL, push_constant_size=16,
            bindings={"frame_from_compositor": "sampled_texture", "palettes": "sampled_texture", "output_image": "storage_image"})
        self._scratch = gpu.acquire_texture(8, 8, "rgba8_unorm", RING_USAGE)
        self._settle_kernel = gpu.create_compute_kernel(
            source=shaders.SETTLE_GLSL,
            bindings={"settled_source": "sampled_texture", "scratch_image": "storage_image"})
        _log("UPSCALER_SETUP")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        frame = ctx.inputs.read("frame_from_upstream")
        if frame is None:
            return
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, OUT_W, OUT_H)
        with ctx.gpu_limited_access.resolve_surface(frame["surface_id"]) as upstream:
            self._kernel.dispatch(
                bindings={"frame_from_compositor": upstream, "palettes": self._palette_texture, "output_image": slot},
                group_count=(OUT_W // 8, OUT_H // 8, 1),
                push_constants=struct.pack("<4f", float(frame.get("palette", 0)), 0.0, 0.0, 0.0))
        # The encoder samples its source in SHADER_READ_ONLY_OPTIMAL and checks the
        # registry's tracked layout; a storage write leaves GENERAL, a sampled read leaves this.
        self._settle_kernel.dispatch(bindings={"settled_source": slot, "scratch_image": self._scratch}, group_count=(1, 1, 1))
        ctx.outputs.write("video", _video_bag(slot, OUT_W, OUT_H, t=frame.get("t", 0.0), texture_layout=5))
        self.frames += 1
        if self.frames in (1, 35, 350):
            _log("UPSCALER_FRAME", frames=self.frames)


# ---------------------------------------------------------------------------
AUDIO_LEAD_NS = 120_000_000
AUDIO_BLOCK_FRAMES = synth.SAMPLE_RATE // 100


@processor(execution="continuous", interval_ms=4)
class AtDoomsGate:
    """The score and the demo's sound effects, streamed as 10 ms audio blocks."""

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def audio(self) -> None: ...

    def __init__(self) -> None:
        self.epoch_ns = 0
        self.next_block = 0
        self.blocks = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        wad = Wad()
        seconds = script.DEMO_SECONDS + 6.0
        track = synth.render_score(wad, "D_E1M1", seconds) * 0.8
        effects = {name: synth.resample_effect(*wad.sound(name)) for name in
                   ("DSPISTOL", "DSITEMUP", "DSPOSIT1", "DSPODTH1", "DSBGSIT1", "DSPOPAIN")}
        for fire_time in script.PISTOL_FIRE_TIMES:
            synth.mix_effect(track, effects["DSPISTOL"], fire_time, gain=0.9)
        synth.mix_effect(track, effects["DSITEMUP"], script.ARMOUR_PICKUP_TIME, gain=0.8)
        synth.mix_effect(track, effects["DSPOSIT1"], script.ZOMBIEMAN_ONE_SIGHT_TIME, gain=0.6, pan=0.7)
        synth.mix_effect(track, effects["DSPISTOL"], script.ZOMBIEMAN_ONE_FIRE_TIMES[0], gain=0.45, pan=0.7)
        synth.mix_effect(track, effects["DSPODTH1"], script.ZOMBIEMAN_ONE_DEATH_TIME, gain=0.7, pan=0.7)
        synth.mix_effect(track, effects["DSBGSIT1"], script.IMP_SIGHT_TIME, gain=0.6, pan=0.85)
        synth.mix_effect(track, effects["DSPOPAIN"], 16.25, gain=0.55, pan=0.8)
        synth.mix_effect(track, effects["DSPOPAIN"], 16.65, gain=0.55, pan=0.8)
        self.track = numpy.clip(track, -1.0, 1.0).astype(numpy.float32)
        self.total_blocks = len(self.track) // AUDIO_BLOCK_FRAMES
        _log("AUDIO_SETUP", seconds=seconds, blocks=self.total_blocks)

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        if self.epoch_ns == 0:
            world = ctx.inputs.read("world_from_upstream")
            if world is None:
                return
            self.epoch_ns = int(world["epoch_ns"])
            now = monotonic_now_ns()
            self.next_block = max(0, (now - self.epoch_ns) // 10_000_000)
            _log("AUDIO_ANCHORED", first_block=self.next_block)
        now = monotonic_now_ns()
        while self.next_block < self.total_blocks:
            block_start_ns = self.epoch_ns + self.next_block * 10_000_000
            if block_start_ns - now > AUDIO_LEAD_NS:
                return
            samples = self.track[self.next_block * AUDIO_BLOCK_FRAMES : (self.next_block + 1) * AUDIO_BLOCK_FRAMES]
            ctx.outputs.write("audio", {
                "samples": samples.tobytes(), "sample_rate": synth.SAMPLE_RATE, "channels": 2,
                "sample_count": AUDIO_BLOCK_FRAMES, "dtype": "f32", "first_sample_timestamp_ns": block_start_ns,
            })
            self.next_block += 1
            self.blocks += 1
            if self.blocks in (1, 100, 1000):
                _log("AUDIO_BLOCK", blocks=self.blocks)
