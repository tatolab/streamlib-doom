"""Reads the shareware DOOM1.WAD: the level, the art, the sounds and the score.

Byte layouts follow id Software's released source (doomdata.h, r_defs.h) and
Chocolate Doom's mus2mid.c. Nothing here touches the engine; every processor in
the Doom graph parses the WAD for itself in its own helper process.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy

from .assets import ensure_wad

ML_TWOSIDED = 4
ML_DONTPEGTOP = 8
ML_DONTPEGBOTTOM = 16

MUS_TICKS_PER_SECOND = 140


@dataclass
class Patch:
    width: int
    height: int
    left_offset: int
    top_offset: int
    index: numpy.ndarray  # (height, width) uint8 palette index
    alpha: numpy.ndarray  # (height, width) uint8, 255 where a post covered it


class Wad:
    def __init__(self, path: "str | None" = None) -> None:
        self.data = open(path if path is not None else ensure_wad(), "rb").read()
        count, directory = struct.unpack("<II", self.data[4:12])
        self.lumps: list[tuple[str, int, int]] = []
        self.index: dict[str, int] = {}
        for i in range(count):
            offset, size, raw_name = struct.unpack("<II8s", self.data[directory + 16 * i : directory + 16 * i + 16])
            name = raw_name.rstrip(b"\0").decode("ascii", "replace").upper()
            self.lumps.append((name, offset, size))
            self.index.setdefault(name, i)
        self._patch_cache: dict[str, Patch] = {}
        self._texture_cache: dict[str, Patch] = {}
        self._flat_names = self._names_between("F_START", "F_END")
        self._sprite_names = self._names_between("S_START", "S_END")
        self._read_texture_directory()

    # -- raw access -----------------------------------------------------------
    def lump(self, name: str) -> bytes:
        _, offset, size = self.lumps[self.index[name]]
        return self.data[offset : offset + size]

    def lump_at(self, i: int) -> bytes:
        _, offset, size = self.lumps[i]
        return self.data[offset : offset + size]

    def _names_between(self, start: str, end: str) -> list[str]:
        return [self.lumps[i][0] for i in range(self.index[start] + 1, self.index[end])]

    # -- palette and light ----------------------------------------------------
    def palettes(self) -> numpy.ndarray:
        """(14, 256, 3) uint8: palette 0 is normal, 1-8 pain, 9-12 item pickup, 13 radiation suit."""
        return numpy.frombuffer(self.lump("PLAYPAL"), dtype=numpy.uint8).reshape(14, 256, 3)

    def colormap(self) -> numpy.ndarray:
        """(34, 256) uint8: rows 0-31 are light levels (0 brightest), 32 invulnerability, 33 unused."""
        return numpy.frombuffer(self.lump("COLORMAP"), dtype=numpy.uint8).reshape(34, 256)

    # -- pictures ---------------------------------------------------------------
    def patch(self, name: str) -> Patch:
        cached = self._patch_cache.get(name)
        if cached is not None:
            return cached
        b = self.lump(name)
        width, height, left, top = struct.unpack("<hhhh", b[:8])
        columns = struct.unpack(f"<{width}I", b[8 : 8 + 4 * width])
        index = numpy.zeros((height, width), dtype=numpy.uint8)
        alpha = numpy.zeros((height, width), dtype=numpy.uint8)
        for x, column_offset in enumerate(columns):
            p = column_offset
            while b[p] != 0xFF:
                top_delta, length = b[p], b[p + 1]
                pixels = b[p + 3 : p + 3 + length]
                y0 = top_delta
                y1 = min(height, y0 + length)
                if y1 > y0:
                    index[y0:y1, x] = numpy.frombuffer(pixels[: y1 - y0], dtype=numpy.uint8)
                    alpha[y0:y1, x] = 255
                p += 4 + length
        patch = Patch(width, height, left, top, index, alpha)
        self._patch_cache[name] = patch
        return patch

    def _read_texture_directory(self) -> None:
        pnames_lump = self.lump("PNAMES")
        pname_count = struct.unpack("<I", pnames_lump[:4])[0]
        self.pnames = [pnames_lump[4 + 8 * i : 12 + 8 * i].rstrip(b"\0").decode("ascii", "replace").upper() for i in range(pname_count)]
        self.textures: dict[str, tuple[int, int, list[tuple[int, int, int]]]] = {}
        for lump_name in ("TEXTURE1", "TEXTURE2"):
            if lump_name not in self.index:
                continue
            t = self.lump(lump_name)
            count = struct.unpack("<I", t[:4])[0]
            for offset in struct.unpack(f"<{count}I", t[4 : 4 + 4 * count]):
                name = t[offset : offset + 8].rstrip(b"\0").decode("ascii", "replace").upper()
                _masked, width, height, _column_directory, patch_count = struct.unpack("<IHHIH", t[offset + 8 : offset + 22])
                patches = []
                for k in range(patch_count):
                    ox, oy, pnum, _step, _cmap = struct.unpack("<hhhhh", t[offset + 22 + 10 * k : offset + 32 + 10 * k])
                    patches.append((ox, oy, pnum))
                self.textures[name] = (width, height, patches)

    def texture(self, name: str) -> Patch:
        """A wall texture composed from its patches; alpha 0 where nothing covered it."""
        cached = self._texture_cache.get(name)
        if cached is not None:
            return cached
        width, height, patches = self.textures[name]
        index = numpy.zeros((height, width), dtype=numpy.uint8)
        alpha = numpy.zeros((height, width), dtype=numpy.uint8)
        for ox, oy, pnum in patches:
            p = self.patch(self.pnames[pnum])
            x0, y0 = max(0, ox), max(0, oy)
            x1, y1 = min(width, ox + p.width), min(height, oy + p.height)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0, sy0 = x0 - ox, y0 - oy
            src_alpha = p.alpha[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
            src_index = p.index[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
            covered = src_alpha > 0
            index[y0:y1, x0:x1][covered] = src_index[covered]
            alpha[y0:y1, x0:x1][covered] = 255
        texture = Patch(width, height, 0, 0, index, alpha)
        self._texture_cache[name] = texture
        return texture

    def flat(self, name: str) -> numpy.ndarray:
        """(64, 64) uint8 palette indices."""
        return numpy.frombuffer(self.lump(name), dtype=numpy.uint8)[: 64 * 64].reshape(64, 64)

    def sprite_frames(self, prefix: str) -> dict[tuple[str, int], tuple[str, bool]]:
        """{(frame_letter, rotation 0-7): (lump_name, mirrored)} for one sprite prefix."""
        frames: dict[tuple[str, int], tuple[str, bool]] = {}
        for name in self._sprite_names:
            if not name.startswith(prefix) or len(name) < 6:
                continue
            frame, rotation = name[4], int(name[5])
            if rotation == 0:
                for r in range(8):
                    frames[(frame, r)] = (name, False)
            else:
                frames[(frame, rotation - 1)] = (name, False)
                if len(name) == 8:
                    frame2, rotation2 = name[6], int(name[7])
                    frames[(frame2, rotation2 - 1)] = (name, True)
        return frames

    # -- the map ---------------------------------------------------------------
    def level(self, name: str) -> "Level":
        base = self.index[name]
        things = numpy.frombuffer(self.lump_at(base + 1), dtype="<i2").reshape(-1, 5)
        linedefs = numpy.frombuffer(self.lump_at(base + 2), dtype="<i2").reshape(-1, 7)
        raw_sidedefs = self.lump_at(base + 3)
        vertexes = numpy.frombuffer(self.lump_at(base + 4), dtype="<i2").reshape(-1, 2)
        raw_sectors = self.lump_at(base + 8)
        sidedefs = []
        for i in range(len(raw_sidedefs) // 30):
            xo, yo, upper, lower, middle, sector = struct.unpack("<hh8s8s8sh", raw_sidedefs[30 * i : 30 * i + 30])
            sidedefs.append((xo, yo, _name(upper), _name(lower), _name(middle), sector))
        sectors = []
        for i in range(len(raw_sectors) // 26):
            floor, ceiling, floor_flat, ceiling_flat, light, special, tag = struct.unpack("<hh8s8shhh", raw_sectors[26 * i : 26 * i + 26])
            sectors.append((floor, ceiling, _name(floor_flat), _name(ceiling_flat), light, special, tag))
        return Level(things, linedefs, sidedefs, vertexes, sectors)

    # -- sound -------------------------------------------------------------------
    def sound(self, name: str) -> tuple[int, numpy.ndarray]:
        """(sample_rate, float32 samples in -1..1) from a DMX lump: 8-bit unsigned, 16 guard bytes each side."""
        b = self.lump(name)
        fmt, rate, count = struct.unpack("<HHI", b[:8])
        assert fmt == 3, name
        pcm = numpy.frombuffer(b[8 + 16 : 8 + count - 16], dtype=numpy.uint8).astype(numpy.float32)
        return rate, (pcm - 128.0) / 128.0

    # -- music -------------------------------------------------------------------
    def mus_events(self, name: str) -> list[tuple[int, int, int, int, int]]:
        """[(tick, channel, kind, a, b)] — kind 0 release, 1 press (a=note, b=velocity or -1),
        2 pitch wheel, 3 system, 4 controller (a=controller, b=value), 6 end."""
        b = self.lump(name)
        assert b[:4] == b"MUS\x1a", name
        score_length, score_start, _primary, _secondary, _instrument_count = struct.unpack("<HHHHH", b[4:14])
        events = []
        p, tick = score_start, 0
        end = score_start + score_length
        while p < end:
            descriptor = b[p]
            p += 1
            channel, kind, last = descriptor & 0x0F, (descriptor >> 4) & 0x07, descriptor & 0x80
            a = bb = 0
            if kind == 0:
                a = b[p] & 0x7F
                p += 1
            elif kind == 1:
                a = b[p] & 0x7F
                if b[p] & 0x80:
                    bb = b[p + 1] & 0x7F
                    p += 2
                else:
                    bb = -1
                    p += 1
            elif kind == 2:
                a = b[p]
                p += 1
            elif kind == 3:
                a = b[p]
                p += 1
            elif kind == 4:
                a, bb = b[p], b[p + 1]
                p += 2
            elif kind == 6:
                events.append((tick, channel, kind, 0, 0))
                break
            events.append((tick, channel, kind, a, bb))
            if last:
                delay = 0
                while True:
                    working = b[p]
                    p += 1
                    delay = delay * 128 + (working & 0x7F)
                    if not working & 0x80:
                        break
                tick += delay
        return events

    def genmidi(self) -> list[dict]:
        """The 175+ OPL2 instrument patches Doom's Sound Blaster driver used, from GENMIDI."""
        b = self.lump("GENMIDI")
        assert b[:8] == b"#OPL_II#"
        instruments = []
        count = (len(b) - 8) // 36
        for i in range(count):
            r = b[8 + 36 * i : 8 + 36 * i + 36]
            flags, fine_tune, fixed_note = struct.unpack("<HBB", r[:4])
            voices = []
            for v in range(2):
                o = 4 + 16 * v
                (m_tremolo, m_attack_decay, m_sustain_release, m_wave, m_scale, m_level, feedback,
                 c_tremolo, c_attack_decay, c_sustain_release, c_wave, c_scale, c_level) = struct.unpack("<13B", r[o : o + 13])
                base_note_offset = struct.unpack("<h", r[o + 14 : o + 16])[0]
                voices.append({
                    "modulator": _opl_operator(m_tremolo, m_attack_decay, m_sustain_release, m_wave, m_scale, m_level),
                    "carrier": _opl_operator(c_tremolo, c_attack_decay, c_sustain_release, c_wave, c_scale, c_level),
                    "feedback": (feedback >> 1) & 7,
                    "additive": bool(feedback & 1),
                    "note_offset": base_note_offset,
                })
            instruments.append({"flags": flags, "fine_tune": fine_tune, "fixed_note": fixed_note, "voices": voices})
        return instruments


def _opl_operator(tremolo, attack_decay, sustain_release, wave, scale, level) -> dict:
    return {
        "multiplier": tremolo & 0x0F,
        "vibrato": bool(tremolo & 0x40),
        "tremolo": bool(tremolo & 0x80),
        "sustaining": bool(tremolo & 0x20),
        "ksr": bool(tremolo & 0x10),
        "attack": attack_decay >> 4,
        "decay": attack_decay & 0x0F,
        "sustain": sustain_release >> 4,
        "release": sustain_release & 0x0F,
        "wave": wave & 3,
        "level": level & 0x3F,
        "key_scale_level": scale >> 6,
    }


def _name(raw: bytes) -> str:
    return raw.rstrip(b"\0").decode("ascii", "replace").upper()


@dataclass
class Level:
    things: numpy.ndarray  # (n, 5) int16: x, y, angle, type, flags
    linedefs: numpy.ndarray  # (n, 7) int16: v1, v2, flags, special, tag, right sidedef, left sidedef
    sidedefs: list[tuple[int, int, str, str, str, int]]  # xoff, yoff, upper, lower, middle, sector
    vertexes: numpy.ndarray  # (n, 2) int16
    sectors: list[tuple[int, int, str, str, int, int, int]]  # floor, ceiling, floor flat, ceiling flat, light, special, tag

    def player_start(self) -> tuple[int, int, int]:
        x, y, angle, _type, _flags = next(t for t in self.things if t[3] == 1)
        return int(x), int(y), int(angle)

    def sector_at(self, x: float, y: float) -> int:
        """The sector containing (x, y): the front sector of the nearest linedef whose
        facing side we are on. Good enough for a scripted camera path that never
        stands on a line."""
        best, best_distance = 0, 1e18
        for v1, v2, flags, _special, _tag, right, left in self.linedefs:
            ax, ay = self.vertexes[v1]
            bx, by = self.vertexes[v2]
            dx, dy = float(bx - ax), float(by - ay)
            length_squared = dx * dx + dy * dy
            if length_squared == 0:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_squared))
            px, py = ax + t * dx, ay + t * dy
            distance = (x - px) ** 2 + (y - py) ** 2
            if distance < best_distance:
                side_sign = (x - ax) * dy - (y - ay) * dx  # > 0 means on the right of the line
                sidedef = right if side_sign > 0 else left
                if sidedef < 0:
                    sidedef = right if right >= 0 else left
                best_distance, best = distance, self.sidedefs[sidedef][5]
        return best
