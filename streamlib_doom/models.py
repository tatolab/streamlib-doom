"""The monsters, replaced by a raymarched robot the graph tracks in real time.

The renderer already writes what every pixel is and how far away it is, and the game already
publishes where every monster stands. That is everything a compositor needs to put something
else there: the robot is a signed distance field marched in a compute kernel at the monster's
own world position, scaled by range, and depth-tested against the game's own depth channel so
a wall in front of it still hides it.

No model is loaded and no network runs. It is one GPU kernel on data the graph was already
carrying, which is the point.
"""
from __future__ import annotations

import math
import os
import struct

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .wad import Wad

VIEW_W, VIEW_H = 320, 200
# The console samples this pane as 512x320; publishing anything else shears it.
OUT_W, OUT_H = 512, 320
MAX_TRACKED = 16
RING_USAGE = ["texture_binding", "storage_binding"]
SUBSTITUTE_FPS = float(os.environ.get("STREAMLIB_DOOM_SUBSTITUTE_FPS", "30"))
PLAYER_EYE_HEIGHT = 41.0

# Doom's projection for a 320x200 view, the same numbers the renderer draws with.
VIEW_FOCAL, VIEW_CENTER_Y = 160.0, 100.0

SUBSTITUTE_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D frame_from_upstream;   // r index, g depth code, b class
layout(set = 0, binding = 1) uniform sampler2D palettes;              // 256 x 14
layout(set = 0, binding = 2) uniform sampler2D tracked;               // MAX_TRACKED x 1: camera-space x,y,range,awake
layout(set = 0, binding = 3, rgba8) uniform writeonly image2D out_image;
layout(push_constant) uniform PC {
    float palette; float count; float tick; float focal;
    float eye_z; float unused1; float unused2; float unused3;
} pc;

const int MONSTER_CLASS = 4;
const int HUD_CLASS = 8;
const float MODEL_HEIGHT = 56.0;

float sd_box(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sd_sphere(vec3 p, float r) { return length(p) - r; }
float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// One robot, standing at the origin, facing -z. Units are map units; it is about 56 tall.
float robot(vec3 p, float phase, out int part) {
    float sway = sin(phase) * 3.0;
    float torso = sd_box(p - vec3(0.0, 30.0, 0.0), vec3(9.0, 12.0, 6.0)) - 1.5;
    float head = sd_sphere(p - vec3(0.0, 48.0, 0.0), 7.0);
    float eye = sd_sphere(p - vec3(0.0, 49.0, -6.0), 2.6);
    float arms = min(sd_capsule(p, vec3(-10.0, 40.0, 0.0), vec3(-13.0, 22.0, sway * 0.4), 3.0),
                     sd_capsule(p, vec3(10.0, 40.0, 0.0), vec3(13.0, 22.0, -sway * 0.4), 3.0));
    float legs = min(sd_capsule(p, vec3(-5.0, 18.0, 0.0), vec3(-5.0, 0.0, sway), 4.0),
                     sd_capsule(p, vec3(5.0, 18.0, 0.0), vec3(5.0, 0.0, -sway), 4.0));
    float body = min(min(torso, head), min(arms, legs));
    part = 0;
    if (eye < body) { part = 1; return eye; }
    if (head <= body + 0.001 && head <= torso) part = 2;
    return body;
}

float scene(vec3 p, float phase, out int part) {
    return robot(p, phase, part);
}

vec3 scene_normal(vec3 p, float phase) {
    int ignored;
    vec2 e = vec2(0.35, 0.0);
    return normalize(vec3(scene(p + e.xyy, phase, ignored) - scene(p - e.xyy, phase, ignored),
                          scene(p + e.yxy, phase, ignored) - scene(p - e.yxy, phase, ignored),
                          scene(p + e.yyx, phase, ignored) - scene(p - e.yyx, phase, ignored)));
}

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(out_image);
    if (at.x >= size.x || at.y >= size.y) return;

    // The game's own pixel under this one, and what the renderer says it is.
    vec2 uv = (vec2(at) + 0.5) / vec2(size);
    ivec2 src = ivec2(uv * vec2(320.0, 200.0));
    vec4 under = texelFetch(frame_from_upstream, src, 0);
    int index = int(under.r * 255.0 + 0.5);
    int surface = int(under.b * 255.0 + 0.5);
    float game_range = 4.0 * exp2(under.g * 8.0);
    vec3 color = texelFetch(palettes, ivec2(index, int(pc.palette)), 0).rgb;

    // A monster pixel with nothing drawn over it would show the sprite through the robot's
    // silhouette, so it is filled with the wall behind it before anything is marched.
    if (surface == MONSTER_CLASS) {
        vec3 behind = vec3(0.0);
        float found = 0.0;
        for (int r = 2; r <= 26 && found < 0.5; r += 2) {
            for (int s = -1; s <= 1; s += 2) {
                ivec2 probe = ivec2(clamp(src.x + r * s, 0, 319), src.y);
                vec4 near_ = texelFetch(frame_from_upstream, probe, 0);
                if (int(near_.b * 255.0 + 0.5) != MONSTER_CLASS) {
                    behind = texelFetch(palettes, ivec2(int(near_.r * 255.0 + 0.5), int(pc.palette)), 0).rgb;
                    found = 1.0;
                    break;
                }
            }
        }
        if (found > 0.5) color = behind;
        game_range = 1.0e9;  // nothing of the game occludes here
    }

    // The ray for this pixel, in the player's camera frame: +z forward, +x right, +y up.
    float rx = (uv.x * 320.0 - 160.0) / pc.focal;
    float ry = (100.0 - uv.y * 200.0) / pc.focal;
    vec3 ray = normalize(vec3(rx, ry, 1.0));

    float best_t = 1.0e9;
    int best_part = 0;
    vec3 best_p = vec3(0.0);
    float best_phase = 0.0;

    int count = int(pc.count);
    for (int i = 0; i < count; i++) {
        vec4 m = texelFetch(tracked, ivec2(i, 0), 0);
        // camera-space stand point: x right, z forward, both in map units, scaled out of the texture
        vec3 base = vec3((m.r - 0.5) * 4096.0, -pc.eye_z, (m.g - 0.5) * 4096.0);
        float phase = m.a * 6.2831 + pc.tick * 0.12;
        if (base.z < 1.0 || length(base) < 90.0) continue;  // point-blank it would be a flat wall; leave the sprite

        float t = max(1.0, length(base) - MODEL_HEIGHT);
        for (int step = 0; step < 48; step++) {
            vec3 p = ray * t - base;
            int part;
            float d = scene(p, phase, part);
            if (d < 0.35) {
                if (t < best_t) { best_t = t; best_part = part; best_p = p; best_phase = phase; }
                break;
            }
            t += max(d, 0.4);
            if (t > best_t || t > 3000.0) break;
        }
    }

    // The status bar and the weapon are drawn over the world, not in it; nothing marched belongs there.
    if (surface != HUD_CLASS && best_t < 1.0e9 && best_t < game_range) {
        vec3 n = scene_normal(best_p, best_phase);
        vec3 light = normalize(vec3(0.4, 0.9, -0.5));
        float diffuse = max(dot(n, light), 0.0);
        float rim = pow(1.0 - max(dot(n, -normalize(ray)), 0.0), 2.5);
        vec3 body = mix(vec3(0.18, 0.20, 0.26), vec3(0.72, 0.78, 0.92), diffuse);
        body += rim * vec3(0.35, 0.55, 0.95);
        if (best_part == 1) body = vec3(1.0, 0.35, 0.12) * (1.2 + 0.5 * sin(pc.tick * 0.4));
        if (best_part == 2) body = mix(body, vec3(0.95, 0.75, 0.25), 0.35);
        float fog = clamp(1.0 - best_t / 2200.0, 0.25, 1.0);
        color = body * fog;
    }

    imageStore(out_image, at, vec4(color, 1.0));
}
"""


@processor(description="Replaces the monsters with a raymarched robot, tracked in real time from the game's own monster positions and depth-tested against the renderer's depth channel so walls still hide it. One GPU compute kernel, no model and no network. Connect the HUD's `frame_to_downstream` to `view_from_upstream` and the game's `world_to_downstream` to `world_from_upstream`.")
class MonsterModelSubstitution:
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def substitute_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0
        self.next_ns = 0
        self.tracked = 0
        self.ms = 0.0
        self.world: dict = {}

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        wad = Wad()
        self._palettes = gpu.acquire_texture(256, 14, "rgba8_unorm", ["texture_binding"])
        self._palettes.lock(read_only=False)
        table = self._palettes.as_numpy()
        for p, palette in enumerate(wad.palettes()):
            table[p, :, :3] = palette
        table[:, :, 3] = 255
        self._palettes.unlock()
        self._tracked = gpu.acquire_texture(MAX_TRACKED, 1, "rgba8_unorm", ["texture_binding"])
        self._kernel = gpu.create_compute_kernel(
            source=SUBSTITUTE_GLSL, push_constant_size=32,
            bindings={"frame_from_upstream": "sampled_texture", "palettes": "sampled_texture",
                      "tracked": "sampled_texture", "out_image": "storage_image"})
        log.info(f"MARKER:SUBSTITUTE_SETUP pid={os.getpid()} out={OUT_W}x{OUT_H}")

    def _camera_space(self, world: dict) -> numpy.ndarray:
        """Each living monster as the kernel wants it: right and forward of the player, in map units."""
        px, py, angle = float(world.get("x", 0.0)), float(world.get("y", 0.0)), float(world.get("angle", 0.0))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rows = numpy.zeros((MAX_TRACKED, 4), dtype=numpy.uint8)
        n = 0
        for monster in (world.get("monster_positions") or []):
            if n >= MAX_TRACKED:
                break
            dx, dy = float(monster[0]) - px, float(monster[1]) - py
            forward = dx * cos_a + dy * sin_a
            right = dx * sin_a - dy * cos_a
            if forward <= 1.0 or abs(right) > forward * 2.2 + 96.0:
                continue  # behind the player or far outside the frustum
            rows[n] = (int(numpy.clip(right / 4096.0 + 0.5, 0, 1) * 255),
                       int(numpy.clip(forward / 4096.0 + 0.5, 0, 1) * 255),
                       int(numpy.clip(math.hypot(dx, dy) / 4096.0, 0, 1) * 255),
                       int((hash((round(monster[0]), round(monster[1]))) % 255)))
            n += 1
        return rows, n

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        world = ctx.inputs.read("world_from_upstream")
        if world is not None:
            self.world = world
        view = ctx.inputs.read("view_from_upstream")
        now = clock.monotonic_now_ns()
        if view is None or now < self.next_ns:
            return
        self.next_ns = max(now, self.next_ns + int(1e9 / SUBSTITUTE_FPS))
        started = clock.monotonic_now_ns()
        rows, count = self._camera_space(self.world)
        self._tracked.lock(read_only=False)
        self._tracked.as_numpy()[0, :MAX_TRACKED, :] = rows
        self._tracked.unlock()
        self.tracked = count
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, OUT_W, OUT_H)
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as frame:
            self._kernel.dispatch(
                bindings={"frame_from_upstream": frame, "palettes": self._palettes, "tracked": self._tracked, "out_image": slot},
                group_count=(OUT_W // 8, OUT_H // 8, 1),
                push_constants=struct.pack("<8f", 0.0, float(count), float(view.get("tick", 0)),  # base palette: the red flash is a HUD effect
                                           VIEW_FOCAL, PLAYER_EYE_HEIGHT, 0.0, 0.0, 0.0))
        self.ms = (clock.monotonic_now_ns() - started) / 1e6
        self.frames += 1
        ctx.outputs.write("substitute_to_downstream", {
            "surface_id": slot.surface_id, "width": OUT_W, "height": OUT_H,
            "timestamp_ns": clock.monotonic_now_ns(), "pid": os.getpid(),
            "model": f"raymarched robot · {count} tracked", "style": "robot",
            "tick": view.get("tick"), "pose": view.get("pose"), "tracked": count,
            "ms": round(self.ms, 2), "fps": round(SUBSTITUTE_FPS, 1)})
        if self.frames in (1, 60, 600):
            log.info(f"MARKER:SUBSTITUTE_FRAME frames={self.frames} tracked={count} ms={self.ms:.2f}")
