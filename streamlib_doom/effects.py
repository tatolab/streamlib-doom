"""A screen effect an agent splices into the live frame path between the
status-bar compositor and the browser.

It works on the 8-bit framebuffer the way 1993 hardware would have: a 256-entry
lookup table built from PLAYPAL and COLORMAP remaps every palette index, and
the CRT effect darkens alternate rows through COLORMAP's own light levels.
"""
from __future__ import annotations

import dataclasses
import struct
from typing import Annotated, Literal

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .wad import Wad

VIEW_W, VIEW_H = 320, 200
RING_USAGE = ["texture_binding", "storage_binding"]

EFFECT_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D frame_from_upstream;
layout(set = 0, binding = 1) uniform sampler2D remap;      // 256 x 1: r = the index each index becomes
layout(set = 0, binding = 2) uniform sampler2D colormap;   // 256 x 34
layout(set = 0, binding = 3, rgba8) uniform writeonly image2D frame_image;
layout(push_constant) uniform PC { float mode; float tick; float unused0; float unused1; } pc;

int index_at(ivec2 at) { return int(texelFetch(frame_from_upstream, at, 0).r * 255.0 + 0.5); }
int shade(int map_index, int palette_index) { return int(texelFetch(colormap, ivec2(palette_index, map_index), 0).r * 255.0 + 0.5); }

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 320 || at.y >= 200) return;
    int mode = int(pc.mode);
    ivec2 source = at;
    if (mode == 4) source = (at / 4) * 4;                                   // pixelate
    if (mode == 5) { float wobble = sin(float(at.y) * 0.18 + pc.tick * 0.25) * 3.0; source.x = clamp(at.x + int(wobble), 0, 319); }
    int index = index_at(source);
    index = int(texelFetch(remap, ivec2(index, 0), 0).r * 255.0 + 0.5);
    if (mode == 1 && at.y < 168) {                                          // crt: scanlines and a vignette, view only
        float dx = (float(at.x) - 160.0) / 160.0, dy = (float(at.y) - 84.0) / 84.0;
        int darken = int(clamp((dx * dx + dy * dy) * 10.0, 0.0, 12.0));
        if ((at.y & 1) == 1) darken += 5;
        index = shade(darken, index);
    }
    imageStore(frame_image, at, vec4(float(index) / 255.0, 0.0, 0.0, 1.0));
}
"""

MODES = {"none": 0, "crt": 1, "night_vision": 2, "invulnerable": 3, "pixelate": 4, "wobble": 5, "thermal": 6}


@dataclasses.dataclass
class ScreenEffectConfig:
    effect: Annotated[
        Literal["crt", "night_vision", "invulnerable", "thermal", "pixelate", "wobble", "none"],
        "crt: scanlines and a curved-glass vignette. night_vision: the green ramp. invulnerable: Doom's own inverted white map. thermal: a heat ramp. pixelate: 4x4 blocks. wobble: the screen swims. none: passthrough.",
    ] = "crt"


def _nearest_palette_index(palette: numpy.ndarray, colors: numpy.ndarray) -> numpy.ndarray:
    """For each colour, the palette index closest to it."""
    distances = ((colors[:, None, :].astype(numpy.int32) - palette[None, :, :].astype(numpy.int32)) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype(numpy.uint8)


def build_all_remaps(wad: Wad) -> numpy.ndarray:
    """(len(MODES), 256) uint8: row `MODES[name]` is the index remap for that effect."""
    table = numpy.zeros((max(MODES.values()) + 1, 256), dtype=numpy.uint8)
    for name, index in MODES.items():
        table[index] = build_remap(wad, name)
    return table


def build_remap(wad: Wad, effect: str) -> numpy.ndarray:
    palette = wad.palettes()[0]
    luminance = (0.299 * palette[:, 0] + 0.587 * palette[:, 1] + 0.114 * palette[:, 2])
    if effect == "night_vision":
        target = numpy.stack([luminance * 0.15, numpy.clip(luminance * 1.2 + 16, 0, 255), luminance * 0.15], axis=1)
        return _nearest_palette_index(palette, target)
    if effect == "thermal":
        t = luminance / 255.0
        target = numpy.stack([numpy.clip(t * 3.0, 0, 1) * 255, numpy.clip(t * 3.0 - 1.0, 0, 1) * 255, numpy.clip((1.0 - t * 2.0), 0, 1) * 200 + numpy.clip(t * 3.0 - 2.0, 0, 1) * 255], axis=1)
        return _nearest_palette_index(palette, numpy.clip(target, 0, 255))
    if effect == "invulnerable":
        return wad.colormap()[32].copy()
    return numpy.arange(256, dtype=numpy.uint8)


@processor(description="A screen effect on the game's 8-bit frame: crt, night_vision, invulnerable, thermal, pixelate, wobble. Splice it between the status bar's `frame_to_downstream` and the browser's `frame_from_upstream`.")
class ScreenEffect:
    def __init__(self, config: ScreenEffectConfig) -> None:
        self.effect = config.effect
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    @output()
    def frame_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        wad = Wad()
        self._remap = gpu.acquire_texture(256, 1, "rgba8_unorm", ["texture_binding"])
        self._remap.lock(read_only=False)
        table = self._remap.as_numpy()
        table[0, :, 0] = build_remap(wad, self.effect)
        table[0, :, 3] = 255
        self._remap.unlock()
        self._colormap = gpu.acquire_texture(256, 34, "rgba8_unorm", ["texture_binding"])
        self._colormap.lock(read_only=False)
        cm = self._colormap.as_numpy()
        cm[:, :, 0] = wad.colormap()
        cm[:, :, 3] = 255
        self._colormap.unlock()
        self._kernel = gpu.create_compute_kernel(
            source=EFFECT_GLSL, push_constant_size=16,
            bindings={"frame_from_upstream": "sampled_texture", "remap": "sampled_texture", "colormap": "sampled_texture", "frame_image": "storage_image"})
        log.info(f"MARKER:EFFECT_SETUP effect={self.effect}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        frame = ctx.inputs.read("frame_from_upstream")
        if frame is None:
            return
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, VIEW_W, VIEW_H)
        with ctx.gpu_limited_access.resolve_surface(frame["surface_id"]) as upstream:
            self._kernel.dispatch(
                bindings={"frame_from_upstream": upstream, "remap": self._remap, "colormap": self._colormap, "frame_image": slot},
                group_count=(VIEW_W // 8, VIEW_H // 8, 1),
                push_constants=struct.pack("<4f", float(MODES.get(self.effect, 0)), float(frame.get("tick", 0)), 0.0, 0.0))
        forwarded = {k: v for k, v in frame.items() if k != "surface_id"}
        forwarded.update(surface_id=slot.surface_id, timestamp_ns=clock.monotonic_now_ns(), effect=self.effect)
        ctx.outputs.write("frame_to_downstream", forwarded)
        self.frames += 1
