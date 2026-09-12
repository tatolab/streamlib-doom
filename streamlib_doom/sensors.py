"""The robot's sensors, every one a processor in its own process, fanned out from the
same simulation and the same rendered frame with no copies between them.

The renderer's view is rgba8 with the palette index in red, depth in green and a
surface class in blue; the depth and segmentation sensors are lookup tables over
those channels, run as compute kernels on the surface the renderer published.
Perception reads the same frame back and reports monsters by bearing and range.
The lidar casts 360 rays through the level's own linedefs; the mapper folds the
scans into an occupancy grid the way a robot's SLAM stack would.
"""
from __future__ import annotations

import math
import os
import struct

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

VIEW_W, VIEW_H = 320, 200
PANE_W, PANE_H = 360, 225
RING_USAGE = ["texture_binding", "storage_binding"]
LIDAR_RAYS = 360
LIDAR_RANGE = 1600.0
MAP_CELL = 20.0
FOCAL = 160.0

LUT_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D view_from_upstream;
layout(set = 0, binding = 1) uniform sampler2D lut;   // 256 x 1: the colour each byte of the chosen channel becomes
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D sensor_image;
layout(push_constant) uniform PC { float channel; float unused0; float unused1; float unused2; } pc;
void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 320 || at.y >= 200) return;
    vec4 texel = texelFetch(view_from_upstream, at, 0);
    float value = pc.channel < 1.5 ? texel.g : texel.b;
    int code = int(value * 255.0 + 0.5);
    imageStore(sensor_image, at, vec4(texelFetch(lut, ivec2(code, 0), 0).rgb, 1.0));
}
"""

CLASS_COLORS = {0: (12, 12, 30), 1: (72, 96, 210), 2: (44, 170, 96), 3: (150, 84, 190), 4: (235, 44, 44), 5: (250, 222, 40), 6: (240, 140, 40), 7: (255, 255, 255)}
CLASS_NAMES = {0: "sky", 1: "wall", 2: "floor", 3: "ceiling", 4: "monster", 5: "pickup", 6: "decoration", 7: "projectile"}


def depth_lut() -> numpy.ndarray:
    """A 256x4 heat ramp from near (warm) to far (cold); the sky code at 255 is black."""
    anchors = [(0.0, (255, 235, 90)), (0.18, (250, 120, 40)), (0.36, (200, 40, 80)), (0.55, (90, 30, 160)), (0.78, (20, 60, 200)), (1.0, (10, 20, 70))]
    lut = numpy.zeros((256, 4), dtype=numpy.uint8)
    for code in range(256):
        # Codes span 4..1024 units on a log scale; scenes live between 32 and 1024, so stretch that.
        t = min(1.0, max(0.0, (code / 255.0 * 8.0 - 3.0) / 5.0))
        for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                lut[code, :3] = [round(a + (b - a) * f) for a, b in zip(c0, c1)]
                break
    lut[255, :3] = (0, 0, 0)
    lut[:, 3] = 255
    return lut


def class_lut() -> numpy.ndarray:
    lut = numpy.zeros((256, 4), dtype=numpy.uint8)
    lut[:, :3] = (60, 60, 60)
    for code, color in CLASS_COLORS.items():
        lut[code, :3] = color
    lut[:, 3] = 255
    return lut


def decode_depth(codes: numpy.ndarray) -> numpy.ndarray:
    """Map units from the green channel: 4 at 0, 1024 at 255."""
    return 4.0 * numpy.power(2.0, codes.astype(numpy.float32) / 255.0 * 8.0)


class _ViewSensorMixin:
    """A kernel that turns one channel of the renderer's view into a picture through a LUT."""

    CHANNEL = 1.0
    NAME = "sensor"

    def _setup_sensor(self, ctx: RuntimeContextFullAccess, lut: numpy.ndarray) -> None:
        gpu = ctx.gpu_full_access
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self._lut = gpu.acquire_texture(256, 1, "rgba8_unorm", ["texture_binding"])
        self._lut.lock(read_only=False)
        self._lut.as_numpy()[0, :, :] = lut
        self._lut.unlock()
        self._kernel = gpu.create_compute_kernel(source=LUT_GLSL, push_constant_size=16, bindings={"view_from_upstream": "sampled_texture", "lut": "sampled_texture", "sensor_image": "storage_image"})
        self.frames = 0
        log.info(f"MARKER:{self.NAME.upper()}_SETUP pid={os.getpid()}")

    def _process_sensor(self, ctx: RuntimeContextLimitedAccess, port: str) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None:
            return
        gpu = ctx.gpu_limited_access
        slot = self._ring.next_texture_for_this_frame(gpu, VIEW_W, VIEW_H)
        with gpu.resolve_surface(view["surface_id"]) as upstream:
            self._kernel.dispatch(bindings={"view_from_upstream": upstream, "lut": self._lut, "sensor_image": slot},
                                  group_count=(VIEW_W // 8, VIEW_H // 8, 1), push_constants=struct.pack("<4f", self.CHANNEL, 0.0, 0.0, 0.0))
        ctx.outputs.write(port, {"surface_id": slot.surface_id, "width": VIEW_W, "height": VIEW_H, "timestamp_ns": clock.monotonic_now_ns(),
                                 "tick": view.get("tick"), "source_surface_id": view["surface_id"], "pid": os.getpid(), "sensor": self.NAME,
                                 "stage_ns": {**(view.get("stage_ns") or {}), self.NAME: clock.monotonic_now_ns()}})
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            log.info(f"MARKER:{self.NAME.upper()}_FRAME frames={self.frames}")


@processor(description="Depth sensor: the renderer's per-pixel depth (green channel of the view) as a heat map, a kernel over the same GPU surface the status bar composites. Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class DepthSensor(_ViewSensorMixin):
    CHANNEL, NAME = 1.0, "depth"

    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def depth_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_sensor(ctx, depth_lut())

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        self._process_sensor(ctx, "depth_to_downstream")


@processor(description="Segmentation sensor: every pixel of the view classed sky, wall, floor, ceiling, monster, pickup, decoration or projectile (blue channel of the view), coloured by class. Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class SegmentationSensor(_ViewSensorMixin):
    CHANNEL, NAME = 2.0, "segmentation"

    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def segmentation_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_sensor(ctx, class_lut())

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        self._process_sensor(ctx, "segmentation_to_downstream")


@processor(description="Perception: reads the rendered frame back and reports every monster in it by bearing, range and size, from the segmentation and depth channels alone — no access to the simulation. Feeds the planner.")
class PerceptionNode:
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def detections_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        log.info(f"MARKER:PERCEPTION_SETUP pid={os.getpid()}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None:
            return
        started = clock.monotonic_now_ns()
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as surface:
            surface.lock()
            pixels = surface.as_numpy()[:VIEW_H, :VIEW_W, 1:3].copy()
            surface.unlock()
        detections = detect_monsters(pixels[:, :, 1], pixels[:, :, 0])
        ctx.outputs.write("detections_to_downstream", {"tick": view.get("tick"), "detections": detections, "pid": os.getpid(),
                                                       "source_surface_id": view["surface_id"], "timestamp_ns": clock.monotonic_now_ns(),
                                                       "inference_ms": round((clock.monotonic_now_ns() - started) / 1e6, 2)})
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            log.info(f"MARKER:PERCEPTION_FRAME frames={self.frames} detections={len(detections)}")


def detect_monsters(classes: numpy.ndarray, depth_codes: numpy.ndarray) -> list[dict]:
    """Runs of columns holding monster pixels, each reported as bearing (degrees, left positive),
    range (median depth of its pixels) and width in columns."""
    mask = classes == 4
    columns = mask.any(axis=0)
    out = []
    start = None
    for col in range(VIEW_W + 1):
        on = col < VIEW_W and columns[col]
        if on and start is None:
            start = col
        elif not on and start is not None:
            if col - start >= 3:
                center = (start + col - 1) / 2.0
                region = mask[:, start:col]
                codes = depth_codes[:, start:col][region]
                distance = float(numpy.median(decode_depth(codes))) if len(codes) else 0.0
                rows = numpy.nonzero(region.any(axis=1))[0]
                out.append({"bearing_degrees": -math.degrees(math.atan((center - 160.0) / FOCAL)), "range": round(distance, 1),
                            "width_px": int(col - start), "column": int(center), "top": int(rows[0]), "bottom": int(rows[-1])})
            start = None
    return out


def _publish_pane(self, ctx: RuntimeContextLimitedAccess, canvas: numpy.ndarray, port: str, extra: dict) -> None:
    slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, PANE_W, PANE_H)
    slot.lock(read_only=False)
    slot.as_numpy()[:, :, :] = canvas
    slot.unlock()
    ctx.outputs.write(port, {"surface_id": slot.surface_id, "width": PANE_W, "height": PANE_H, "timestamp_ns": clock.monotonic_now_ns(), "pid": os.getpid(), **extra})


def _scatter(canvas: numpy.ndarray, xs: numpy.ndarray, ys: numpy.ndarray, color: tuple, size: int = 2) -> None:
    xs, ys = xs.astype(numpy.int32), ys.astype(numpy.int32)
    for dx in range(size):
        for dy in range(size):
            px, py = xs + dx, ys + dy
            keep = (px >= 0) & (px < canvas.shape[1]) & (py >= 0) & (py < canvas.shape[0])
            canvas[py[keep], px[keep], :3] = color


def _polyline(canvas: numpy.ndarray, points: list, color: tuple, size: int = 2) -> None:
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0)))
        t = numpy.linspace(0.0, 1.0, n)
        _scatter(canvas, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, color, size)


@processor(description="Lidar: 360 rays a tic through the level's own linedefs at the robot's eye height, drawn as a polar scan. Connect the game's `world_to_downstream` to `world_from_upstream`.")
class LidarScanner:
    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def scan_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.scans = 0
        self.last_tick = -1

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        from .game import Game
        from .wad import Wad
        wad = Wad()
        game = Game(wad, wad.level("E1M1"))
        self.l1, self.ld = game.l1.astype(numpy.float64), game.ld.astype(numpy.float64)
        self.two_sided = game.two_sided.astype(bool)
        self.right_sector, self.left_sector = game.right_sector.astype(int), game.left_sector.astype(int)
        self.relative = numpy.linspace(-math.pi, math.pi, LIDAR_RAYS, endpoint=False)
        log.info(f"MARKER:LIDAR_SETUP pid={os.getpid()} lines={len(self.l1)}")

    def scan(self, x: float, y: float, angle: float, floors: numpy.ndarray, ceilings: numpy.ndarray, sector: int) -> numpy.ndarray:
        """Range per ray, in map units, for a beam at eye height."""
        z = floors[sector] + 41.0
        directions = numpy.stack([numpy.cos(angle + self.relative), numpy.sin(angle + self.relative)], axis=1) * LIDAR_RANGE  # (R, 2)
        a = numpy.array((x, y))
        w = self.l1 - a  # (L, 2)
        denom = directions[:, 0:1] * self.ld[None, :, 1] - directions[:, 1:2] * self.ld[None, :, 0]  # (R, L)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * self.ld[None, :, 1] - w[None, :, 1] * self.ld[None, :, 0]) / denom
            u = (w[None, :, 0] * directions[:, 1:2] - w[None, :, 1] * directions[:, 0:1]) / denom
        hit = (numpy.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        lower = numpy.maximum(floors[self.right_sector], floors[self.left_sector])
        upper = numpy.minimum(ceilings[self.right_sector], ceilings[self.left_sector])
        solid = ~self.two_sided | (z < lower) | (z > upper)  # the beam meets a wall face at its height
        t = numpy.where(hit & solid[None, :], t, numpy.inf).min(axis=1)
        return numpy.where(numpy.isfinite(t), t * LIDAR_RANGE, LIDAR_RANGE)

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        world = ctx.inputs.read("world_from_upstream")
        if world is None or world["tick"] == self.last_tick:
            return
        self.last_tick = world["tick"]
        floors, ceilings = numpy.asarray(world["floors"], dtype=numpy.float64), numpy.asarray(world["ceilings"], dtype=numpy.float64)
        ranges = self.scan(world["x"], world["y"], world["angle"], floors, ceilings, int(world["sector"]))
        canvas = numpy.zeros((PANE_H, PANE_W, 4), dtype=numpy.uint8)
        canvas[:, :, :3] = (12, 14, 20)
        canvas[:, :, 3] = 255
        cx, cy, scale = PANE_W / 2, PANE_H / 2 + 14, 0.16
        theta = numpy.arange(0, 2 * math.pi, 0.02)
        for ring in (200, 400, 600):
            _scatter(canvas, cx + numpy.cos(theta) * ring * scale, cy - numpy.sin(theta) * ring * scale, (34, 40, 56), 1)
        hits = ranges < LIDAR_RANGE - 1
        px = cx + numpy.sin(self.relative) * ranges * scale * -1.0  # heading up, left of the robot on the left
        py = cy - numpy.cos(self.relative) * ranges * scale
        for k in range(0, LIDAR_RAYS, 6):
            _polyline(canvas, [(cx, cy), (float(px[k]), float(py[k]))], (26, 60, 70), 1)
        _scatter(canvas, px[hits], py[hits], (80, 230, 255), 3)
        _scatter(canvas, px[~hits], py[~hits], (40, 90, 110), 1)
        _polyline(canvas, [(cx - 5, cy + 6), (cx, cy - 8), (cx + 5, cy + 6), (cx - 5, cy + 6)], (255, 200, 60), 2)
        _publish_pane(self, ctx, canvas, "scan_to_downstream", {
            "tick": world["tick"], "x": world["x"], "y": world["y"], "angle": world["angle"], "ranges": [int(r) for r in ranges],
            "route": world.get("route") or [], "mission": world.get("mission"), "sensor": "lidar",
            "stage_ns": {**(world.get("stage_ns") or {}), "lidar": clock.monotonic_now_ns()}})
        self.scans += 1
        if self.scans in (1, 35, 35 * 60):
            log.info(f"MARKER:LIDAR_SCAN scans={self.scans} nearest={int(ranges.min())}")


@processor(description="Occupancy mapper: folds every lidar scan into a grid of free and occupied cells with the robot's trajectory and the planner's route — the level as the robot has discovered it, never the map file. Connect the lidar's `scan_to_downstream` to `scan_from_upstream`.")
class OccupancyMapper:
    @input(delivery_profile="newest")
    def scan_from_upstream(self) -> None: ...

    @output()
    def map_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0
        self.trajectory: list[tuple[float, float]] = []
        self.last_tick = -1

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        from .game import Game
        from .wad import Wad
        wad = Wad()
        lines = Game(wad, wad.level("E1M1")).l1.astype(numpy.float64)  # only the extents; the map is built from scans
        xs, ys = lines[:, 0], lines[:, 1]
        self.x0, self.y0 = float(xs.min()) - MAP_CELL, float(ys.min()) - MAP_CELL
        self.cols, self.rows = int((xs.max() - self.x0) / MAP_CELL) + 2, int((ys.max() - self.y0) / MAP_CELL) + 2
        self.grid = numpy.zeros((self.rows, self.cols), dtype=numpy.float32)  # -1 free .. +1 occupied, 0 unknown
        # The pane samples the grid; a wide level is letterboxed.
        scale = min(PANE_W / self.cols, PANE_H / self.rows)
        self.px_per_cell = scale
        self.ox = int((PANE_W - self.cols * scale) / 2)
        self.oy = int((PANE_H - self.rows * scale) / 2)
        self.sample_c = numpy.clip(((numpy.arange(PANE_W) - self.ox) / scale).astype(int), 0, self.cols - 1)
        self.sample_r = numpy.clip(((numpy.arange(PANE_H) - self.oy) / scale).astype(int), 0, self.rows - 1)
        self.relative = numpy.linspace(-math.pi, math.pi, LIDAR_RAYS, endpoint=False)
        log.info(f"MARKER:MAPPER_SETUP pid={os.getpid()} grid={self.cols}x{self.rows}")

    def to_pane(self, x: float, y: float) -> tuple[float, float]:
        # Map y grows north; the pane's y grows down.
        return (self.ox + (x - self.x0) / MAP_CELL * self.px_per_cell, self.oy + (self.rows - (y - self.y0) / MAP_CELL) * self.px_per_cell)

    def integrate(self, x: float, y: float, angle: float, ranges: numpy.ndarray) -> None:
        directions = numpy.stack([numpy.cos(angle + self.relative), numpy.sin(angle + self.relative)], axis=1)
        steps = numpy.arange(0.0, LIDAR_RANGE, MAP_CELL)
        along = steps[None, :]  # (1, S)
        keep = along < ranges[:, None] - MAP_CELL * 0.5
        px = x + directions[:, 0:1] * along
        py = y + directions[:, 1:2] * along
        c = ((px - self.x0) / MAP_CELL).astype(int)
        r = ((py - self.y0) / MAP_CELL).astype(int)
        inside = keep & (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows)
        self.grid[r[inside], c[inside]] = numpy.maximum(self.grid[r[inside], c[inside]] - 0.12, -1.0)
        hit = ranges < LIDAR_RANGE - 1
        hx, hy = x + directions[hit, 0] * ranges[hit], y + directions[hit, 1] * ranges[hit]
        hc, hr = ((hx - self.x0) / MAP_CELL).astype(int), ((hy - self.y0) / MAP_CELL).astype(int)
        ok = (hc >= 0) & (hc < self.cols) & (hr >= 0) & (hr < self.rows)
        self.grid[hr[ok], hc[ok]] = numpy.minimum(self.grid[hr[ok], hc[ok]] + 0.5, 1.0)

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        scan = ctx.inputs.read("scan_from_upstream")
        if scan is None or scan["tick"] == self.last_tick:
            return
        self.last_tick = scan["tick"]
        x, y, angle = scan["x"], scan["y"], scan["angle"]
        self.integrate(x, y, angle, numpy.asarray(scan["ranges"], dtype=numpy.float64))
        if not self.trajectory or math.hypot(x - self.trajectory[-1][0], y - self.trajectory[-1][1]) > 12:
            self.trajectory.append((x, y))
            self.trajectory = self.trajectory[-3000:]
        canvas = numpy.zeros((PANE_H, PANE_W, 4), dtype=numpy.uint8)
        canvas[:, :, 3] = 255
        sampled = self.grid[self.sample_r[:, None], self.sample_c[None, :]]
        sampled = sampled[::-1, :]  # north up
        canvas[:, :, :3] = (22, 24, 32)
        free = sampled < -0.05
        occupied = sampled > 0.25
        canvas[free, :3] = (48, 58, 84)
        canvas[occupied, :3] = (222, 226, 236)
        _polyline(canvas, [self.to_pane(px, py) for px, py in self.trajectory[-600:]], (60, 200, 255), 2)
        route = [(x, y)] + [(float(rx), float(ry)) for rx, ry in (scan.get("route") or [])]
        if len(route) > 1:
            _polyline(canvas, [self.to_pane(px, py) for px, py in route], (255, 210, 60), 2)
            gx, gy = self.to_pane(*route[-1])
            _polyline(canvas, [(gx - 5, gy - 5), (gx + 5, gy + 5)], (255, 210, 60), 2)
            _polyline(canvas, [(gx - 5, gy + 5), (gx + 5, gy - 5)], (255, 210, 60), 2)
        rx, ry = self.to_pane(x, y)
        heading = (math.cos(angle), -math.sin(angle))
        tip = (rx + heading[0] * 9, ry + heading[1] * 9)
        left = (rx - heading[1] * 5 - heading[0] * 4, ry + heading[0] * 5 - heading[1] * 4)
        right = (rx + heading[1] * 5 - heading[0] * 4, ry - heading[0] * 5 - heading[1] * 4)
        _polyline(canvas, [left, tip, right, left], (255, 90, 60), 2)
        known = int((self.grid != 0).sum())
        _publish_pane(self, ctx, canvas, "map_to_downstream", {"tick": scan["tick"], "cells_known": known, "cells_total": int(self.grid.size), "sensor": "map",
                                                             "stage_ns": {**(scan.get("stage_ns") or {}), "map": clock.monotonic_now_ns()}})
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            log.info(f"MARKER:MAPPER_FRAME frames={self.frames} known={known}")
