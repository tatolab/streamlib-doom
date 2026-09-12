"""Packs every picture the Doom graph draws into one rgba8 texture.

Red is the palette index, green is 255 where the picture covered the texel and
0 where a post left it open, blue and alpha are unused. A companion float
table carries each entry's rectangle and the picture's own draw offsets, so a
shader finds a wall, a flat, a sprite or a status-bar digit by one entry id.
"""
from __future__ import annotations

import numpy

from .wad import Patch, Wad

ATLAS_SIZE = 2048

SKY_TEXTURE = "SKY1"

STATUS_BAR_PATCHES = (
    ["STBAR", "STARMS"]
    + [f"STTNUM{i}" for i in range(10)]
    + ["STTPRCNT", "STTMINUS"]
    + [f"STYSNUM{i}" for i in range(10)]
    + [f"STGNUM{i}" for i in range(2, 8)]
    + [f"STFST{row}{look}" for row in range(5) for look in range(3)]
    + [f"STF{kind}{row}0" for row in range(5) for kind in ("TL", "TR")]
    + [f"STF{kind}{row}" for row in range(5) for kind in ("OUCH", "EVL", "KILL")]
    + ["STFDEAD0", "STFGOD0"]
    + [f"STCFN{code:03d}" for code in range(33, 96)]
)

WEAPON_PATCHES = ["PISGA0", "PISGB0", "PISGC0", "PISGD0", "PISGE0", "PISFA0",
                  "SHTGA0", "SHTGB0", "SHTGC0", "SHTGD0", "SHTFA0", "SHTFB0"]


class Atlas:
    def __init__(self, wad: Wad, level_texture_names: set[str], flat_names: set[str], sprite_prefixes: list[str]) -> None:
        self.wad = wad
        self.image = numpy.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=numpy.uint8)
        self.image[:, :, 3] = 255
        self.entries: list[tuple[int, int, int, int, int, int]] = []  # x, y, w, h, left, top
        self.ids: dict[str, int] = {}
        self._shelf_x = 0
        self._shelf_y = 0
        self._shelf_height = 0

        for name in sorted(level_texture_names):
            if name in wad.textures:
                self._add("T:" + name, wad.texture(name))
        self._add("T:" + SKY_TEXTURE, wad.texture(SKY_TEXTURE))
        for name in sorted(flat_names):
            if name == "F_SKY1" or name not in wad.index:
                continue
            self._add_flat("F:" + name, wad.flat(name))
        # Animated nukage: NUKAGE1-3 cycle, 8 tics each, whatever the sector named.
        for name in ("NUKAGE1", "NUKAGE2", "NUKAGE3"):
            if "F:" + name not in self.ids and name in wad.index:
                self._add_flat("F:" + name, wad.flat(name))
        self.sprite_frames: dict[str, dict[tuple[str, int], tuple[int, bool]]] = {}
        for prefix in sprite_prefixes:
            frames = wad.sprite_frames(prefix)
            resolved: dict[tuple[str, int], tuple[int, bool]] = {}
            for key, (lump_name, mirrored) in frames.items():
                if "S:" + lump_name not in self.ids:
                    self._add("S:" + lump_name, wad.patch(lump_name))
                resolved[key] = (self.ids["S:" + lump_name], mirrored)
            self.sprite_frames[prefix] = resolved
        for name in STATUS_BAR_PATCHES + WEAPON_PATCHES:
            if name in wad.index:
                self._add("P:" + name, wad.patch(name))

    def _place(self, width: int, height: int) -> tuple[int, int]:
        if self._shelf_x + width > ATLAS_SIZE:
            self._shelf_x = 0
            self._shelf_y += self._shelf_height
            self._shelf_height = 0
        if self._shelf_y + height > ATLAS_SIZE:
            raise RuntimeError("atlas full")
        x, y = self._shelf_x, self._shelf_y
        self._shelf_x += width
        self._shelf_height = max(self._shelf_height, height)
        return x, y

    def _add(self, key: str, patch: Patch) -> int:
        x, y = self._place(patch.width, patch.height)
        self.image[y : y + patch.height, x : x + patch.width, 0] = patch.index
        self.image[y : y + patch.height, x : x + patch.width, 1] = patch.alpha
        self.entries.append((x, y, patch.width, patch.height, patch.left_offset, patch.top_offset))
        self.ids[key] = len(self.entries) - 1
        return self.ids[key]

    def _add_flat(self, key: str, flat: numpy.ndarray) -> int:
        return self._add(key, Patch(64, 64, 0, 0, flat, numpy.full((64, 64), 255, dtype=numpy.uint8)))

    def entry_id(self, key: str) -> int:
        return self.ids[key]

    def table(self) -> numpy.ndarray:
        """(n, 8) float32 rows: x, y, w, h, left, top, 0, 0 — two rgba32_float texels per entry."""
        rows = numpy.zeros((len(self.entries), 8), dtype=numpy.float32)
        for i, (x, y, w, h, left, top) in enumerate(self.entries):
            rows[i, :6] = (x, y, w, h, left, top)
        return rows
