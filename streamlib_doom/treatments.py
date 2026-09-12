"""Picture treatments an agent stacks onto the rendered output, one processor at a time.

Each one is the same shape: RGB in, RGB out, same size, one compute kernel. That is the whole
point — they chain. Adding a second is `add_processor` plus two `connect` calls, the graph grows
a box, and the picture changes on the next frame with nothing restarted. Remove it and the
picture goes back.

    add_processor streamlib_doom.treatments:PictureTreatment {"treatment": "bloom"}
    connect Substitute.substitute_to_downstream -> Bloom.frame_from_upstream
    connect Bloom.frame_to_downstream           -> Console.neural_from_upstream
"""
from __future__ import annotations

import dataclasses
import os
import struct
from typing import Annotated, Literal

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

RING_USAGE = ["texture_binding", "storage_binding"]

TREATMENTS = {"none": 0, "bloom": 1, "scanlines": 2, "outline": 3, "grade": 4, "chromatic": 5,
              "vignette": 6, "sharpen": 7, "posterize": 8, "underwater": 9, "neon": 10, "glitch": 11}

TREATMENT_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D frame_from_upstream;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D frame_image;
layout(push_constant) uniform PC { float mode; float amount; float tick; float unused; } pc;

vec3 at_offset(ivec2 at, ivec2 d, ivec2 size) {
    return texelFetch(frame_from_upstream, clamp(at + d, ivec2(0), size - 1), 0).rgb;
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(frame_image);
    if (at.x >= size.x || at.y >= size.y) return;
    int mode = int(pc.mode);
    float k = pc.amount;
    vec3 c = texelFetch(frame_from_upstream, at, 0).rgb;
    vec3 out_c = c;

    if (mode == 1) {                                   // bloom: the bright parts bleed
        vec3 sum = vec3(0.0);
        float weight = 0.0;
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                vec3 s = at_offset(at, ivec2(dx * 2, dy * 2), size);
                float bright = max(luma(s) - 0.62, 0.0);
                float w = 1.0 / (1.0 + float(dx * dx + dy * dy));
                sum += s * bright * w;
                weight += w;
            }
        }
        out_c = c + sum / max(weight, 0.001) * k * 2.6;
    } else if (mode == 2) {                            // scanlines and a phosphor triad
        float line = (at.y % 3 == 0) ? 1.0 - 0.45 * k : 1.0;
        float triad = 1.0 + 0.18 * k * ((at.x % 3 == 0) ? 1.0 : ((at.x % 3 == 1) ? -0.5 : -0.5));
        out_c = c * line * triad;
    } else if (mode == 3) {                            // outline: a Sobel on luma, drawn in ink
        float gx = luma(at_offset(at, ivec2(-1, 0), size)) - luma(at_offset(at, ivec2(1, 0), size));
        float gy = luma(at_offset(at, ivec2(0, -1), size)) - luma(at_offset(at, ivec2(0, 1), size));
        float edge = clamp(sqrt(gx * gx + gy * gy) * 4.0 * k, 0.0, 1.0);
        out_c = mix(c, vec3(0.02, 0.02, 0.05), edge);
    } else if (mode == 4) {                            // grade: teal shadows, warm highlights
        float l = luma(c);
        vec3 shadow = vec3(0.10, 0.36, 0.44);
        vec3 highlight = vec3(1.06, 0.92, 0.68);
        out_c = mix(c * mix(shadow, vec3(1.0), l), c * highlight, smoothstep(0.45, 1.0, l));
        out_c = mix(c, out_c, k);
    } else if (mode == 5) {                            // chromatic aberration, strongest at the edge
        vec2 uv = (vec2(at) + 0.5) / vec2(size);
        vec2 pull = (uv - 0.5) * k * 9.0;
        out_c = vec3(at_offset(at, ivec2(pull), size).r, c.g, at_offset(at, -ivec2(pull), size).b);
    } else if (mode == 6) {                            // vignette
        vec2 uv = (vec2(at) + 0.5) / vec2(size) - 0.5;
        out_c = c * (1.0 - clamp(dot(uv, uv) * 1.8 * k, 0.0, 0.9));
    } else if (mode == 7) {                            // unsharp mask
        vec3 blur = vec3(0.0);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                blur += at_offset(at, ivec2(dx, dy), size);
        out_c = clamp(c + (c - blur / 9.0) * k * 2.2, 0.0, 1.0);
    } else if (mode == 8) {                            // posterize, for a printed look
        float steps = max(2.0, 10.0 - k * 7.0);
        out_c = floor(c * steps + 0.5) / steps;
    } else if (mode == 9) {                            // underwater: a slow swim plus a cold cast
        float wobble = sin(float(at.y) * 0.10 + pc.tick * 0.08) * k * 5.0;
        out_c = at_offset(at, ivec2(int(wobble), 0), size) * vec3(0.72, 0.95, 1.08);
    } else if (mode == 10) {                           // neon: every edge a light tube, cyan to magenta across the frame
        float gx = luma(at_offset(at, ivec2(-1, 0), size)) - luma(at_offset(at, ivec2(1, 0), size));
        float gy = luma(at_offset(at, ivec2(0, -1), size)) - luma(at_offset(at, ivec2(0, 1), size));
        float edge = clamp(sqrt(gx * gx + gy * gy) * 5.5 * k, 0.0, 1.0);
        float across = (float(at.x) + 0.5) / float(size.x);
        vec3 tube = mix(vec3(0.15, 0.95, 1.0), vec3(1.0, 0.25, 0.9), across);
        out_c = c * 0.16 + tube * edge * 1.7;
    } else if (mode == 11) {                           // glitch: bands tear sideways and the channels split
        float band = floor(float(at.y) / 14.0);
        float r = fract(sin(band * 12.9898 + floor(pc.tick / 4.0) * 78.233) * 43758.5453);
        int shift = (r > 0.70) ? int((r - 0.70) * 110.0 * k) : 0;
        int split = int(5.0 * k);
        vec3 s = at_offset(at, ivec2(shift, 0), size);
        out_c = vec3(at_offset(at, ivec2(shift + split, 0), size).r, s.g, at_offset(at, ivec2(shift - split, 0), size).b);
        if (r > 0.94) out_c = mix(out_c, vec3(0.9, 1.0, 1.0), 0.18);
    }

    imageStore(frame_image, at, vec4(clamp(out_c, 0.0, 1.0), 1.0));
}
"""


@dataclasses.dataclass
class PictureTreatmentConfig:
    treatment: Annotated[
        Literal["bloom", "scanlines", "outline", "grade", "chromatic", "vignette", "sharpen", "posterize", "underwater", "neon", "glitch", "none"],
        "bloom: bright parts bleed. scanlines: CRT lines and a phosphor triad. outline: ink on the edges. grade: teal shadows and warm highlights. chromatic: colour fringing at the edges. vignette: darkened corners. sharpen: an unsharp mask. posterize: banded colour, a printed look. underwater: a slow swim and a cold cast. neon: every edge a light tube, cyan to magenta. glitch: bands tear sideways and the channels split.",
    ] = "bloom"
    amount: Annotated[float, "How strong, 0.0 to 2.0. 1.0 is the intended look."] = 1.0


@processor(description="One picture treatment on an RGB frame — bloom, scanlines, outline, grade, chromatic, vignette, sharpen, posterize, underwater, neon, glitch. RGB in, RGB out, same size, one compute kernel, so they stack: splice as many as you like between whatever produces the picture and whatever shows it. Adding one grows the graph a box and changes the next frame with nothing restarted.")
class PictureTreatment:
    def __init__(self, config: PictureTreatmentConfig) -> None:
        self.treatment = config.treatment
        self.amount = float(config.amount)
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0
        self.ms = 0.0

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    @output()
    def frame_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._kernel = ctx.gpu_full_access.create_compute_kernel(
            source=TREATMENT_GLSL, push_constant_size=16,
            bindings={"frame_from_upstream": "sampled_texture", "frame_image": "storage_image"})
        log.info(f"MARKER:TREATMENT_SETUP treatment={self.treatment} amount={self.amount} pid={os.getpid()}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        frame = ctx.inputs.read("frame_from_upstream")
        if frame is None:
            return
        started = clock.monotonic_now_ns()
        width, height = int(frame.get("width", 512)), int(frame.get("height", 320))
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, width, height)
        with ctx.gpu_limited_access.resolve_surface(frame["surface_id"]) as upstream:
            self._kernel.dispatch(
                bindings={"frame_from_upstream": upstream, "frame_image": slot},
                group_count=(max(width // 8, 1), max(height // 8, 1), 1),
                push_constants=struct.pack("<4f", float(TREATMENTS.get(self.treatment, 0)), self.amount,
                                           float(frame.get("tick", 0) or 0), 0.0))
        self.ms = (clock.monotonic_now_ns() - started) / 1e6
        forwarded = {k: v for k, v in frame.items() if k != "surface_id"}
        chain = list(frame.get("treatments") or [])
        chain.append(self.treatment)
        forwarded.update(surface_id=slot.surface_id, width=width, height=height,
                         timestamp_ns=clock.monotonic_now_ns(), pid=os.getpid(),
                         treatments=chain, model=" + ".join(chain),
                         ms=round(self.ms, 2))
        ctx.outputs.write("frame_to_downstream", forwarded)
        self.frames += 1
        if self.frames in (1, 300):
            log.info(f"MARKER:TREATMENT_FRAME treatment={self.treatment} frames={self.frames} ms={self.ms:.2f}")
