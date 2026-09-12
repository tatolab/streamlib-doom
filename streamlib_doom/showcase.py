"""The ops console: a 1920x1080 picture the runtime records of itself.

The node's live graph is the hero pane, drawn from `/api/graph` by a processor
that watches nodes appear and links change. The operator's view, the sensors
Claude adds, the telemetry and the caption bar are each a texture another
processor publishes; the console kernel samples them all into one frame for
the H.264 encoder.
"""
from __future__ import annotations

import collections
import json
import os
import struct
import subprocess
import time
import urllib.request

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .wad import Wad

OUT_W, OUT_H = 1920, 1080
VIEW_W, VIEW_H = 320, 200
PANE_W, PANE_H = 360, 225
GRAPH_W, GRAPH_H = 1128, 552
CAPTION_W, CAPTION_H = 1888, 168
TELEMETRY_W, TELEMETRY_H = PANE_W, PANE_H
RING_USAGE = ["texture_binding", "storage_binding"]
CONTROL_URL = os.environ.get("STREAMLIB_DOOM_CONTROL_URL", "http://127.0.0.1:9200")
TRANSCRIPT_PATH = os.environ.get("STREAMLIB_DOOM_TRANSCRIPT", "/tmp/streamlib-doom-transcript.log")
CAPTION_PATH = os.environ.get("STREAMLIB_DOOM_CAPTION", "/tmp/streamlib-doom-caption.txt")
FONT = "DejaVu-Sans-Mono"
FONT_BOLD = "DejaVu-Sans-Mono-Bold"
SENSOR_PANES = ("depth", "segmentation", "lidar", "map", "telemetry")
SENSOR_TITLES = {"depth": "DEPTH", "segmentation": "SEGMENTATION", "lidar": "LIDAR", "map": "MAP · occupancy + plan", "telemetry": "TELEMETRY"}

# Regions of the 1080p frame, mirrored by the kernel below.
GRAPH_AT = (16, 64)
OPERATOR_AT, OPERATOR_SIZE = (1184, 64), (720, 540)
SENSOR_Y, SENSOR_LABEL_H = 640, 26
SENSOR_X = [16 + i * 382 for i in range(5)]
CAPTION_AT = (16, 904)


def render_text(text: str, width: int, height: int, pointsize: int = 20, color: str = "white", font: str = FONT, wrap: bool = False) -> numpy.ndarray:
    """(height, width, 4) uint8 RGBA of `text` on a transparent canvas, through ImageMagick."""
    if not text:
        return numpy.zeros((height, width, 4), dtype=numpy.uint8)
    if wrap:
        command = ["convert", "-background", "none", "-fill", color, "-font", font, "-pointsize", str(pointsize), "-size", f"{width}x{height}", "-gravity", "NorthWest", f"caption:{text}", "-depth", "8", "rgba:-"]
    else:
        command = ["convert", "-size", f"{width}x{height}", "xc:none", "-fill", color, "-font", font, "-pointsize", str(pointsize), "-gravity", "NorthWest", "-annotate", "+0+0", text, "-depth", "8", "rgba:-"]
    try:
        raw = subprocess.run(command, capture_output=True, timeout=10, check=True).stdout
        pixels = numpy.frombuffer(raw, dtype=numpy.uint8)
        if len(pixels) == width * height * 4:
            return pixels.reshape(height, width, 4).copy()
    except (subprocess.SubprocessError, OSError):
        pass
    return numpy.zeros((height, width, 4), dtype=numpy.uint8)


_TEXT_CACHE: dict = {}
_FONTS: dict = {}
GLYPHS = "".join(chr(c) for c in range(32, 127)) + "·›θ—…"


class BitmapFont:
    """Printable ASCII (and a few glyphs the panels use) rendered once by ImageMagick, in white;
    drawing text is then a row of tinted blits, cheap enough for every frame."""

    def __init__(self, pointsize: int, bold: bool = False) -> None:
        self.size = pointsize
        self.advance = max(1, int(round(pointsize * 0.602)))
        self.height = int(pointsize * 1.3)
        # One ImageMagick call draws every glyph at its own cell, so slicing cannot drift.
        cell = pointsize + 2
        width, height = cell * len(GLYPHS), self.height
        command = ["convert", "-size", f"{width}x{height}", "xc:none", "-font", FONT_BOLD if bold else FONT, "-pointsize", str(pointsize), "-fill", "white"]
        baseline = int(pointsize * 0.95)
        for i, ch in enumerate(GLYPHS):
            escaped = ch.replace("\\", "\\\\").replace("'", "\\'")
            command += ["-draw", f"text {i * cell},{baseline} '{escaped}'"]
        command += ["-depth", "8", "rgba:-"]
        strip = numpy.zeros((height, width, 4), dtype=numpy.uint8)
        try:
            raw = subprocess.run(command, capture_output=True, timeout=20, check=True).stdout
            pixels = numpy.frombuffer(raw, dtype=numpy.uint8)
            if len(pixels) == width * height * 4:
                strip = pixels.reshape(height, width, 4)
        except (subprocess.SubprocessError, OSError):
            pass
        self.glyphs = {ch: strip[:, i * cell : i * cell + self.advance, 3].copy() for i, ch in enumerate(GLYPHS)}

    def draw(self, canvas: numpy.ndarray, text: str, x: int, y: int, color: tuple, max_width: int | None = None) -> int:
        if max_width is not None:
            text = text[: max(0, max_width // self.advance)]
        h = self.height
        for ch in text:
            alpha = self.glyphs[ch] if ch in self.glyphs else self.glyphs["?"]
            x0, y0 = x, y
            x1, y1 = min(canvas.shape[1], x + self.advance), min(canvas.shape[0], y + h)
            if x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0:
                a = alpha[: y1 - y0, : x1 - x0].astype(numpy.float32)[:, :, None] / 255.0
                dst = canvas[y0:y1, x0:x1, :3].astype(numpy.float32)
                canvas[y0:y1, x0:x1, :3] = (numpy.array(color, dtype=numpy.float32) * a + dst * (1.0 - a)).astype(numpy.uint8)
                canvas[y0:y1, x0:x1, 3] = 255
            x += self.advance
        return x


def font(pointsize: int, bold: bool = False) -> BitmapFont:
    key = (pointsize, bold)
    if key not in _FONTS:
        _FONTS[key] = BitmapFont(pointsize, bold)
    return _FONTS[key]


def text_sprite(text: str, pointsize: int, color: str, font: str = FONT, width: int | None = None, height: int | None = None) -> numpy.ndarray:
    key = (text, pointsize, color, font, width, height)
    if key not in _TEXT_CACHE:
        w = width or int(len(text) * pointsize * 0.62) + 8
        h = height or int(pointsize * 1.35)
        _TEXT_CACHE[key] = render_text(text, w, h, pointsize, color, font)
        if len(_TEXT_CACHE) > 600:
            _TEXT_CACHE.pop(next(iter(_TEXT_CACHE)))
    return _TEXT_CACHE[key]


def blit(canvas: numpy.ndarray, sprite: numpy.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Alpha-composite `sprite` onto `canvas` at (x, y), clipped; the canvas's alpha grows with it."""
    h, w = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = sprite[y0 - y : y1 - y, x0 - x : x1 - x].astype(numpy.float32)
    alpha = src[:, :, 3:4] / 255.0 * opacity
    dst = canvas[y0:y1, x0:x1].astype(numpy.float32)
    dst[:, :, :3] = src[:, :, :3] * alpha + dst[:, :, :3] * (1.0 - alpha)
    dst[:, :, 3:4] = numpy.minimum(255.0, dst[:, :, 3:4] + alpha * 255.0)
    canvas[y0:y1, x0:x1] = dst.astype(numpy.uint8)


def rect(canvas: numpy.ndarray, x: int, y: int, w: int, h: int, color: tuple, border: tuple | None = None, thickness: int = 2) -> None:
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1, :3] = color
    canvas[y0:y1, x0:x1, 3] = 255
    if border is not None:
        t = thickness
        canvas[y0 : min(y1, y0 + t), x0:x1, :3] = border
        canvas[max(y0, y1 - t) : y1, x0:x1, :3] = border
        canvas[y0:y1, x0 : min(x1, x0 + t), :3] = border
        canvas[y0:y1, max(x0, x1 - t) : x1, :3] = border


def line(canvas: numpy.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple, thickness: int = 2) -> None:
    n = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
    xs = numpy.linspace(x0, x1, n).astype(int)
    ys = numpy.linspace(y0, y1, n).astype(int)
    for dx in range(thickness):
        for dy in range(thickness):
            px, py = xs + dx, ys + dy
            keep = (px >= 0) & (px < canvas.shape[1]) & (py >= 0) & (py < canvas.shape[0])
            canvas[py[keep], px[keep], :3] = color
            canvas[py[keep], px[keep], 3] = 255


class _PanelPublisher:
    """A CPU-drawn panel published as a texture another processor samples."""

    WIDTH, HEIGHT = PANE_W, PANE_H

    def _publish(self, ctx, canvas: numpy.ndarray, port: str, extra: dict | None = None) -> None:
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, self.WIDTH, self.HEIGHT)
        slot.lock(read_only=False)
        slot.as_numpy()[:, :, :] = canvas
        slot.unlock()
        ctx.outputs.write(port, {"surface_id": slot.surface_id, "width": self.WIDTH, "height": self.HEIGHT, "timestamp_ns": clock.monotonic_now_ns(), "pid": os.getpid(), **(extra or {})})


# ---------------------------------------------------------------------------------------------
# The live graph, the hero pane.

NODE_W, NODE_H = 176, 60
NATIVE_TYPES = ("H264", "Opus", "Mp4", "Jpeg", "Aac", "Camera", "Display", "Audio", "Virtual")
KIND_COLORS = {
    "core": ((38, 84, 160), (120, 170, 240)),
    "native": ((70, 72, 84), (150, 152, 166)),
    "live": ((190, 140, 20), (255, 214, 90)),
    "claude": ((196, 120, 20), (255, 200, 80)),
    "api": ((36, 38, 46), (70, 74, 88)),
}
SHORT_NAMES = {"StatusBar": "HUD", "Browser": "Phone", "Renderer": "Render", "Recorder": "MP4", "GraphPanel": "Graph", "Transcript": "Caption", "Perception": "Perceive", "Segmentation": "Segment", "Telemetry": "Telemetry", "Console": "Console"}


def _layer(nodes: list, links: list) -> dict:
    """Column per node: longest path from a source once back edges are cut, so a loop through
    the game still reads left to right."""
    ids = [n["id"] for n in nodes]
    outgoing = collections.defaultdict(list)
    incoming = collections.defaultdict(int)
    for a, b in links:
        if a in ids and b in ids and a != b:
            outgoing[a].append(b)
            incoming[b] += 1
    by_id = {n["id"]: n for n in nodes}
    seeds = [i for i in ids if incoming[i] == 0]
    seeds.sort(key=lambda i: -len(outgoing[i]))
    state, order, kept = {}, [], []
    def visit(node):
        state[node] = 1
        for nxt in outgoing[node]:
            if state.get(nxt) == 1:
                continue  # a back edge: the loop's return trip
            kept.append((node, nxt))
            if nxt not in state:
                visit(nxt)
        state[node] = 2
        order.append(node)
    remaining = [i for i in ids]
    while remaining:
        seed = next((s for s in seeds if s not in state), None) or max(remaining, key=lambda i: (len(outgoing[i]), by_id[i].get("display_name") == "Game"))
        visit(seed)
        remaining = [i for i in ids if i not in state]
    depth = {i: 0 for i in ids}
    for node in reversed(order):
        for a, b in kept:
            if a == node:
                depth[b] = max(depth[b], depth[a] + 1)
    return depth


@processor(execution="continuous", interval_ms=80)
class GraphPanel(_PanelPublisher):
    """Draws the node's own graph, big enough to read on a phone: every box a process, links
    pulsing with flow, a node that appears live scaling in gold with a flash."""

    WIDTH, HEIGHT = GRAPH_W, GRAPH_H

    @output()
    def panel_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.initial: set | None = None
        self.first_seen: dict = {}
        self.ghosts: dict = {}
        self.link_seen: dict = {}
        self.cut_links: dict = {}
        self.last_graph: dict | None = None
        self.last_poll = 0.0
        self.frames = 0
        self.positions: dict = {}

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        log.info(f"MARKER:GRAPH_PANEL_SETUP pid={os.getpid()}")

    def _poll(self) -> dict | None:
        try:
            with urllib.request.urlopen(CONTROL_URL + "/api/graph", timeout=1.5) as reply:
                return json.load(reply)
        except Exception:
            return None

    def _kind(self, node: dict) -> str:
        name, type_name = node.get("display_name") or "", node.get("type") or ""
        if "ApiServer" in type_name or "ApiServer" in name:
            return "api"
        if name.startswith("Claude") or "DirectorCommand" in type_name:
            return "claude"
        if self.initial is not None and node["id"] not in self.initial:
            return "live"
        if any(tag in type_name.split(":")[-1] for tag in NATIVE_TYPES) and "streamlib_doom" not in type_name:
            return "native"
        return "core"

    def _draw(self, graph: dict, now: float) -> numpy.ndarray:
        canvas = numpy.zeros((GRAPH_H, GRAPH_W, 4), dtype=numpy.uint8)
        canvas[:, :, :3] = (16, 18, 24)
        canvas[:, :, 3] = 255
        rect(canvas, 0, 0, GRAPH_W, GRAPH_H, (16, 18, 24), (40, 44, 58), 2)
        nodes = [n for n in graph.get("nodes", []) if self._kind(n) != "api"]
        raw_links = graph.get("links", graph.get("edges", []))
        links = []
        for l in raw_links:
            a = l.get("from_processor_id") or l.get("from") or l.get("source") or (l.get("from_") or {}).get("processor_id")
            b = l.get("to_processor_id") or l.get("to") or l.get("target") or (l.get("to_") or {}).get("processor_id")
            if isinstance(a, dict):
                a = a.get("processor_id") or a.get("id")
            if isinstance(b, dict):
                b = b.get("processor_id") or b.get("id")
            if a and b:
                links.append((a, b))
        depth = _layer(nodes, links)
        columns = max(depth.values(), default=0) + 1
        node_w = NODE_W if columns <= 6 else max(132, (GRAPH_W - 32 - 10 * (columns - 1)) // columns)
        pitch_x = min(224, (GRAPH_W - 32 - node_w) // max(columns - 1, 1)) if columns > 1 else 0
        by_col = collections.defaultdict(list)
        for n in nodes:
            by_col[depth[n["id"]]].append(n)
        for col_nodes in by_col.values():
            col_nodes.sort(key=lambda n: (self.first_seen.get(n["id"], 0), n.get("display_name") or ""))
        positions = {}
        for col, col_nodes in by_col.items():
            pitch_y = min(92, (GRAPH_H - 40) // max(len(col_nodes), 1))
            top = (GRAPH_H - 24 - pitch_y * len(col_nodes)) // 2 + 8
            for row, n in enumerate(col_nodes):
                positions[n["id"]] = (16 + col * pitch_x, top + row * pitch_y)
        self.positions = positions
        # Links first, so boxes sit on top.
        for a, b in links:
            if a not in positions or b not in positions:
                continue
            ax, ay = positions[a][0] + node_w, positions[a][1] + NODE_H // 2
            bx, by = positions[b][0], positions[b][1] + NODE_H // 2
            back = bx < ax
            color = (70, 160, 110) if not back else (90, 70, 110)
            if back:
                ax, bx = positions[a][0] + node_w // 2, positions[b][0] + node_w // 2
                ay, by = positions[a][1] + NODE_H, positions[b][1] + NODE_H
                mid_y = min(GRAPH_H - 30, max(ay, by) + 14)
                line(canvas, ax, ay, ax, mid_y, color, 2)
                line(canvas, ax, mid_y, bx, mid_y, color, 2)
                line(canvas, bx, mid_y, bx, by, color, 2)
                continue
            line(canvas, ax, ay, bx, by, color, 3)
            # Flow: dots travelling along the link.
            span = max(1.0, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
            for k in range(int(span // 46) + 1):
                phase = ((now * 90.0) % 46.0 + k * 46.0) / span
                if phase <= 1.0:
                    dx, dy = int(ax + (bx - ax) * phase), int(ay + (by - ay) * phase)
                    rect(canvas, dx - 3, dy - 3, 7, 7, (170, 255, 200))
        for (a, b), cut_at in list(self.cut_links.items()):
            if now - cut_at > 1.2 or a not in positions or b not in positions:
                self.cut_links.pop((a, b), None)
                continue
            ax, ay = positions[a][0] + node_w, positions[a][1] + NODE_H // 2
            bx, by = positions[b][0], positions[b][1] + NODE_H // 2
            line(canvas, ax, ay, bx, by, (230, 60, 60), 4)
        for n in nodes:
            x, y = positions[n["id"]]
            kind = self._kind(n)
            fill, edge = KIND_COLORS[kind]
            age = now - self.first_seen.get(n["id"], now)
            scale = min(1.0, 0.55 + age / 0.4 * 0.45) if self.initial is not None and n["id"] not in self.initial else 1.0
            w, h = int(node_w * scale), int(NODE_H * scale)
            bx, by = x + (node_w - w) // 2, y + (NODE_H - h) // 2
            if age < 0.7 and n["id"] not in (self.initial or ()):
                glow = int(255 * (1.0 - age / 0.7))
                rect(canvas, bx - 6, by - 6, w + 12, h + 12, (glow, glow, glow), None)
            rect(canvas, bx, by, w, h, fill, edge, 3)
            name = n.get("display_name") or n.get("type", "?").split(":")[-1]
            if name.startswith("Claude"):
                label = name.split(":", 1)[-1].strip() or "Claude"
            else:
                label = SHORT_NAMES.get(name, name)
            if kind == "claude":
                sub = "added by Claude"
            elif kind == "live":
                sub = "added live · MCP"
            elif kind == "native":
                sub = "native rust"
            else:
                sub = "python · own process"
            if scale > 0.9:
                size = next((size for size in (24, 21, 18, 15) if font(size, True).advance * len(label) <= node_w - 14), 13)
                font(size, True).draw(canvas, label, x + 7, y + 4 + (24 - size) // 2, (255, 255, 255), node_w - 12)
                font(13).draw(canvas, sub, x + 7, y + 38, (215, 222, 235) if kind != "native" else (200, 200, 210), node_w - 12)
        for gid, (ghost, gone_at) in list(self.ghosts.items()):
            if now - gone_at > 0.6:
                self.ghosts.pop(gid, None)
                continue
            x, y = ghost
            opacity = 1.0 - (now - gone_at) / 0.6
            rect(canvas, x, y, NODE_W - 20, NODE_H, (int(90 * opacity), int(40 * opacity), int(40 * opacity)), (int(220 * opacity), 60, 60), 2)
        footer = f"{len(nodes)} processors  ·  {len(links)} links  ·  every python box is its own process, every link zero-copy on the GPU"
        font(15).draw(canvas, footer, 14, GRAPH_H - 24, (150, 160, 180), GRAPH_W - 28)
        return canvas

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        now = time.monotonic()
        if now - self.last_poll > 0.25:
            graph = self._poll()
            self.last_poll = now
            if graph is not None:
                ids = {n["id"] for n in graph.get("nodes", [])}
                if self.initial is None:
                    self.initial = set(ids)
                    for i in ids:
                        self.first_seen[i] = 0.0
                for i in ids:
                    if i not in self.first_seen:
                        self.first_seen[i] = now
                        log.info(f"MARKER:GRAPH_NODE_APPEARED id={i}")
                if self.last_graph is not None:
                    for old in {n["id"] for n in self.last_graph.get("nodes", [])} - ids:
                        if old in self.positions:
                            self.ghosts[old] = (self.positions[old], now)
                    def pairs(g):
                        out = set()
                        for l in g.get("links", g.get("edges", [])):
                            a = l.get("from_processor_id") or l.get("from") or l.get("source")
                            b = l.get("to_processor_id") or l.get("to") or l.get("target")
                            if isinstance(a, dict): a = a.get("processor_id") or a.get("id")
                            if isinstance(b, dict): b = b.get("processor_id") or b.get("id")
                            if a and b: out.add((a, b))
                        return out
                    for cut in pairs(self.last_graph) - pairs(graph):
                        self.cut_links[cut] = now
                self.last_graph = graph
        if self.last_graph is None:
            return
        canvas = self._draw(self.last_graph, now)
        self._publish(ctx, canvas, "panel_to_downstream", {"nodes": len(self.last_graph.get("nodes", []))})
        self.frames += 1
        if self.frames in (1, 50):
            log.info(f"MARKER:GRAPH_PANEL frames={self.frames}")


# ---------------------------------------------------------------------------------------------
# Captions and what Claude is saying.

def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


@processor(execution="continuous", interval_ms=200)
class CaptionPanel(_PanelPublisher):
    """The caption bar: a few big words for a muted viewer, and Claude's latest prompt and answer."""

    WIDTH, HEIGHT = CAPTION_W, CAPTION_H

    @output()
    def panel_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.last_key = None
        self.canvas = None
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        log.info(f"MARKER:CAPTION_PANEL_SETUP pid={os.getpid()}")

    def _read_caption(self) -> tuple[str, str]:
        try:
            lines = [l.rstrip() for l in open(CAPTION_PATH, encoding="utf-8").read().splitlines()]
        except OSError:
            return "", ""
        big = lines[0] if lines else ""
        small = lines[1] if len(lines) > 1 else ""
        return big, small

    def _read_transcript(self) -> tuple[str, str]:
        try:
            text = open(TRANSCRIPT_PATH, encoding="utf-8").read()
        except OSError:
            return "", ""
        prompt, answer = "", ""
        for raw in text.splitlines():
            if raw.startswith("> "):
                prompt, answer = raw[2:].strip(), ""
            elif raw.strip() and prompt:
                answer = (answer + " " + raw.strip()).strip() if answer else raw.strip()
        answer = answer.replace("**", "")
        return prompt, answer

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        big, small = self._read_caption()
        prompt, answer = self._read_transcript()
        key = (big, small, prompt, answer[:400])
        if key != self.last_key or self.canvas is None:
            canvas = numpy.zeros((CAPTION_H, CAPTION_W, 4), dtype=numpy.uint8)
            canvas[:, :, :3] = (12, 13, 18)
            canvas[:, :, 3] = 255
            rect(canvas, 0, 0, CAPTION_W, CAPTION_H, (12, 13, 18), (40, 44, 58), 2)
            if big:
                size = next((size for size in (58, 52, 46, 40, 34) if len(big) * size * 0.62 <= 1110), 30)
                blit(canvas, render_text(big, 1120, 76, size, "white", FONT_BOLD), 20, 18 + (58 - size) // 2)
            if small:
                size = next((size for size in (26, 23, 20, 18) if len(small) * size * 0.62 <= 1110), 16)
                blit(canvas, render_text(small, 1120, 40, size, "rgb(180,190,210)", FONT), 22, 104 + (26 - size) // 2)
            x = 1180
            line(canvas, x - 22, 14, x - 22, CAPTION_H - 14, (40, 44, 58), 2)
            if prompt:
                blit(canvas, render_text("CLAUDE, DIRECTING  ·  claude -p on the desktop, MCP into this node", 700, 24, 15, "rgb(255,200,80)", FONT_BOLD), x, 12)
                y = 38
                for l in _wrap("› " + prompt, 52)[:2]:
                    blit(canvas, render_text(l, 700, 30, 22, "rgb(255,214,90)", FONT_BOLD), x, y)
                    y += 28
                for l in _wrap(answer, 62)[:3]:
                    blit(canvas, render_text(l, 700, 26, 18, "rgb(230,234,240)", FONT), x, y)
                    y += 23
            else:
                blit(canvas, render_text("CLAUDE, DIRECTING  ·  claude -p on the desktop, MCP into this node", 700, 24, 15, "rgb(255,200,80)", FONT_BOLD), x, 12)
                blit(canvas, render_text("waiting for a prompt…", 700, 30, 20, "rgb(120,130,150)", FONT), x, 44)
            self.canvas, self.last_key = canvas, key
        self._publish(ctx, self.canvas, "panel_to_downstream", {"caption": big})
        self.frames += 1
        if self.frames == 1:
            log.info("MARKER:CAPTION_PANEL frames=1")


# ---------------------------------------------------------------------------------------------
# Telemetry: the numbers a robotics engineer looks at first.

@processor(execution="continuous", interval_ms=150)
class TelemetryPanel(_PanelPublisher):
    """Pose, health, mission, detections, loop timing and the frame witness — which surface id
    is in which process right now."""

    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def witness_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def detections_from_upstream(self) -> None: ...

    @output()
    def panel_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.world = None
        self.witness = None
        self.detections = None
        self.tick_times: collections.deque = collections.deque(maxlen=70)
        self.last_tick = -1
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        log.info(f"MARKER:TELEMETRY_SETUP pid={os.getpid()}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        world = ctx.inputs.read("world_from_upstream")
        if world is not None:
            self.world = world
            if world["tick"] != self.last_tick:
                self.last_tick = world["tick"]
                self.tick_times.append((time.monotonic(), world["tick"]))
        witness = ctx.inputs.read("witness_from_upstream")
        if witness is not None:
            self.witness = witness
        seen = ctx.inputs.read("detections_from_upstream")
        if seen is not None:
            self.detections = seen
        canvas = numpy.zeros((PANE_H, PANE_W, 4), dtype=numpy.uint8)
        canvas[:, :, :3] = (14, 16, 22)
        canvas[:, :, 3] = 255
        y = 6
        def rgb(spec):
            return tuple(int(v) for v in spec[4:-1].split(",")) if spec.startswith("rgb(") else (225, 230, 240)
        def row(label, value, color="rgb(225,230,240)"):
            nonlocal y
            font(13, True).draw(canvas, label, 8, y, (120, 135, 160), 88)
            font(13).draw(canvas, value, 98, y, rgb(color), PANE_W - 104)
            y += 19
        w = self.world or {}
        state = w.get("state") or {}
        source = state.get("control_source", "—")
        row("control", source.upper(), "rgb(255,214,90)" if source == "autonomy" else ("rgb(80,230,255)" if source == "teleop" else "rgb(200,200,210)"))
        row("pose", f"x {state.get('x', 0):>5}  y {state.get('y', 0):>6}  θ {state.get('angle_degrees', 0):>3}°")
        row("health", f"{state.get('health', 0):>3}%  armor {state.get('armor', 0):>3}%  sh {state.get('shells', 0):>2} bu {state.get('bullets', 0):>3}")
        route = w.get("route") or []
        row("mission", f"{state.get('mission', '—')} · {len(route)} corners left" if route else f"{state.get('mission', '—')}")
        dets = (self.detections or {}).get("detections") or []
        if dets and w and w.get("tick", 0) - (self.detections or {}).get("tick", -99) < 18:
            nearest = min(dets, key=lambda d: d["range"])
            row("perception", f"{len(dets)} monster{'s' if len(dets) != 1 else ''} · {nearest['bearing_degrees']:+.0f}° · {nearest['range']:.0f}u · {(self.detections or {}).get('inference_ms', 0)}ms", "rgb(255,120,120)")
        else:
            row("perception", "clear · from the frame alone", "rgb(150,200,160)")
        if len(self.tick_times) > 10:
            (t0, k0), (t1, k1) = self.tick_times[0], self.tick_times[-1]
            hz = (k1 - k0) / max(t1 - t0, 1e-6)
        else:
            hz = 0.0
        stage = (self.witness or {}).get("stage_ms") or {}
        chain = " › ".join(f"{k} {v:.0f}" for k, v in stage.items()) if stage else "—"
        row("loop", f"sim {hz:4.1f} Hz · console {((self.witness or {}).get('fps') or 0):4.1f} fps")
        row("latency ms", chain)
        panes = dict((self.witness or {}).get("panes") or {})
        if self.detections is not None:
            panes["perception"] = {"surface_id": self.detections.get("source_surface_id"), "pid": self.detections.get("pid"), "source_surface_id": self.detections.get("source_surface_id")}
        frame_id = (panes.get("operator") or {}).get("surface_id") or "—"
        procs = {v.get("pid") for v in panes.values() if v.get("pid")}
        row("frame", f"{len(procs)} procs · 0 copies · {str(frame_id)[:8]}", "rgb(170,255,200)")
        y += 2
        line(canvas, 8, y, PANE_W - 8, y, (40, 44, 58), 1)
        y += 5
        for name in ("operator", "depth", "segmentation", "perception", "lidar", "map"):
            v = panes.get(name)
            if v:
                from_frame = name in ("operator", "depth", "segmentation", "perception")
                row(name[:10], f"pid {v.get('pid', '—')} · {'same GPU frame' if from_frame else 'from the sim'}", "rgb(170,255,200)" if from_frame else "rgb(200,205,215)")
        self._publish(ctx, canvas, "panel_to_downstream", {"hz": hz})
        self.frames += 1
        if self.frames == 1:
            log.info("MARKER:TELEMETRY frames=1")


# ---------------------------------------------------------------------------------------------
# The console kernel: every pane sampled into one 1080p frame.

CONSOLE_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D game_frame;     // 320x200 indexed, what the phone sees
layout(set = 0, binding = 1) uniform sampler2D palettes;       // 256x14
layout(set = 0, binding = 2) uniform sampler2D depth_pane;     // 320x200
layout(set = 0, binding = 3) uniform sampler2D seg_pane;       // 320x200
layout(set = 0, binding = 4) uniform sampler2D lidar_pane;     // 360x225
layout(set = 0, binding = 5) uniform sampler2D map_pane;       // 360x225
layout(set = 0, binding = 6) uniform sampler2D telemetry;      // 360x225
layout(set = 0, binding = 7) uniform sampler2D graph_panel;    // 1128x552
layout(set = 0, binding = 8) uniform sampler2D caption;        // 1888x168
layout(set = 0, binding = 9) uniform sampler2D chrome;         // 1920x1080 overlay: title, pane labels
layout(set = 0, binding = 10) uniform sampler2D badges;        // 720x84: three 28-row badges
layout(set = 0, binding = 11, rgba8) uniform writeonly image2D output_image;
layout(push_constant) uniform PC { float palette; float badge; float unused0; float unused1; } pc;

const int SENSOR_Y = 640; const int LABEL_H = 26; const int PANE_H = 225; const int PANE_W = 360;

vec4 pane(sampler2D tex, int x, int y, int src_w, int src_h) {
    ivec2 src = ivec2((x * src_w) / PANE_W, (y * src_h) / PANE_H);
    return vec4(texelFetch(tex, src, 0).rgb, 1.0);
}

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 1920 || at.y >= 1080) return;
    vec4 color = vec4(0.045, 0.05, 0.065, 1.0);
    if (at.x >= 16 && at.x < 16 + 1128 && at.y >= 64 && at.y < 64 + 552) {
        color = vec4(texelFetch(graph_panel, ivec2(at.x - 16, at.y - 64), 0).rgb, 1.0);
    } else if (at.x >= 1184 && at.x < 1184 + 720 && at.y >= 64 && at.y < 64 + 540) {
        ivec2 src = ivec2(((at.x - 1184) * 320) / 720, ((at.y - 64) * 200) / 540);
        int index = int(texelFetch(game_frame, src, 0).r * 255.0 + 0.5);
        color = vec4(texelFetch(palettes, ivec2(index, int(pc.palette)), 0).rgb, 1.0);
    } else if (at.x >= 1184 && at.x < 1184 + 720 && at.y >= 604 && at.y < 632) {
        color = vec4(texelFetch(badges, ivec2(at.x - 1184, at.y - 604 + int(pc.badge) * 28), 0).rgb, 1.0);
    } else if (at.y >= SENSOR_Y + LABEL_H && at.y < SENSOR_Y + LABEL_H + PANE_H) {
        int y = at.y - SENSOR_Y - LABEL_H;
        for (int i = 0; i < 5; i++) {
            int x0 = 16 + i * 382;
            if (at.x >= x0 && at.x < x0 + PANE_W) {
                int x = at.x - x0;
                if (i == 0) color = pane(depth_pane, x, y, 320, 200);
                else if (i == 1) color = pane(seg_pane, x, y, 320, 200);
                else if (i == 2) color = pane(lidar_pane, x, y, 360, 225);
                else if (i == 3) color = pane(map_pane, x, y, 360, 225);
                else color = pane(telemetry, x, y, 360, 225);
            }
        }
    } else if (at.x >= 16 && at.x < 16 + 1888 && at.y >= 904 && at.y < 904 + 168) {
        color = vec4(texelFetch(caption, ivec2(at.x - 16, at.y - 904), 0).rgb, 1.0);
    }
    vec4 over = texelFetch(chrome, at, 0);
    color = vec4(mix(color.rgb, over.rgb, over.a), 1.0);
    imageStore(output_image, at, color);
}
"""

SETTLE_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D settled_source;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D scratch_image;
void main() { ivec2 at = ivec2(gl_GlobalInvocationID.xy); imageStore(scratch_image, at, texelFetch(settled_source, at, 0)); }
"""

BADGES = ("AUTONOMY  ·  the planner drives, from its own camera", "TELEOP  ·  a phone is driving over WebSocket", "IDLE  ·  waiting for a hand or a plan")


@processor(execution="continuous", interval_ms=5)
class ConsoleCompositor:
    """1920x1080: the live graph, the operator's view with its control badge, five sensor panes,
    and the caption bar — sampled by one kernel from textures other processes published."""

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def depth_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def segmentation_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def lidar_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def map_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def telemetry_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def graph_panel_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def caption_from_upstream(self) -> None: ...

    @output()
    def video(self) -> None: ...

    @output()
    def witness_to_downstream(self) -> None: ...

    PANE_INPUTS = {"depth": "depth_from_upstream", "segmentation": "segmentation_from_upstream", "lidar": "lidar_from_upstream", "map": "map_from_upstream",
                   "telemetry": "telemetry_from_upstream", "graph": "graph_panel_from_upstream", "caption": "caption_from_upstream"}
    # Resolving a surface from another process costs a few milliseconds; a pane's handle is kept
    # until its surface id changes, and the sensor panes take a new one every few frames.
    REFRESH_EVERY = {"depth": 2, "segmentation": 2, "lidar": 3, "map": 3, "telemetry": 1, "graph": 1, "caption": 1}

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=4)
        self.latest: dict = {}
        self.handles: dict = {}  # pane -> (surface_id, handle, frame index it was resolved at)
        self.perception: dict | None = None
        self.frames = 0
        self.frame: dict | None = None
        self.frame_handle = None
        self.next_frame_ns = 0
        self.fps_window: collections.deque = collections.deque(maxlen=70)
        self.last_witness = 0.0

    def _placeholder(self, gpu, w: int, h: int, text: str):
        texture = gpu.acquire_texture(w, h, "rgba8_unorm", ["texture_binding"])
        texture.lock(read_only=False)
        pixels = texture.as_numpy()
        pixels[:, :, :3] = (18, 20, 26)
        pixels[:, :, 3] = 255
        canvas = numpy.zeros((h, w, 4), dtype=numpy.uint8)
        blit(canvas, render_text(text, w - 20, 24, 16, "rgb(90,100,120)", FONT), 12, h // 2 - 12)
        alpha = canvas[:, :, 3:4].astype(numpy.float32) / 255.0
        pixels[:, :, :3] = (canvas[:, :, :3] * alpha + pixels[:, :, :3] * (1 - alpha)).astype(numpy.uint8)
        texture.unlock()
        return texture

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        wad = Wad()
        self._palettes = gpu.acquire_texture(256, 14, "rgba8_unorm", ["texture_binding"])
        self._palettes.lock(read_only=False)
        pal = self._palettes.as_numpy()
        pal[:, :, :3] = wad.palettes()
        pal[:, :, 3] = 255
        self._palettes.unlock()
        self._blank = {
            "view": self._placeholder(gpu, VIEW_W, VIEW_H, "no node yet · add one over MCP"),
            "pane": self._placeholder(gpu, PANE_W, PANE_H, "no node yet · add one over MCP"),
            "graph": self._placeholder(gpu, GRAPH_W, GRAPH_H, "graph panel starting…"),
            "caption": self._placeholder(gpu, CAPTION_W, CAPTION_H, ""),
        }
        chrome = numpy.zeros((OUT_H, OUT_W, 4), dtype=numpy.uint8)
        blit(chrome, render_text("STREAMLIB", 300, 44, 34, "white", FONT_BOLD), 16, 8)
        blit(chrome, render_text("·  DOOM E1M1 as a robot  ·  one runtime, one graph, every box its own process  ·  the runtime is recording this picture of itself", 1500, 30, 19, "rgb(160,170,190)", FONT), 214, 18)
        blit(chrome, render_text("LIVE GRAPH  ·  what this node is running right now, from its own /api/graph", 1100, 22, 15, "rgb(150,160,180)", FONT_BOLD), 16, 46)
        blit(chrome, render_text("OPERATOR VIEW  ·  what the phone sees", 700, 22, 15, "rgb(150,160,180)", FONT_BOLD), 1184, 46)
        for i, name in enumerate(SENSOR_PANES):
            blit(chrome, render_text(SENSOR_TITLES[name], PANE_W, 22, 15, "rgb(150,160,180)", FONT_BOLD), SENSOR_X[i], SENSOR_Y + 4)
            rect_border = numpy.zeros((PANE_H + 2, PANE_W + 2, 4), dtype=numpy.uint8)
            rect(rect_border, 0, 0, PANE_W + 2, PANE_H + 2, (0, 0, 0), (40, 44, 58), 1)
            rect_border[1:-1, 1:-1, 3] = 0
            blit(chrome, rect_border, SENSOR_X[i] - 1, SENSOR_Y + SENSOR_LABEL_H - 1)
        border = numpy.zeros((OPERATOR_SIZE[1] + 2, OPERATOR_SIZE[0] + 2, 4), dtype=numpy.uint8)
        rect(border, 0, 0, OPERATOR_SIZE[0] + 2, OPERATOR_SIZE[1] + 2, (0, 0, 0), (40, 44, 58), 1)
        border[1:-1, 1:-1, 3] = 0
        blit(chrome, border, OPERATOR_AT[0] - 1, OPERATOR_AT[1] - 1)
        self._chrome = gpu.acquire_texture(OUT_W, OUT_H, "rgba8_unorm", ["texture_binding"])
        self._chrome.lock(read_only=False)
        self._chrome.as_numpy()[:, :, :] = chrome
        self._chrome.unlock()
        badges = numpy.zeros((84, 720, 4), dtype=numpy.uint8)
        for i, (text, color) in enumerate(zip(BADGES, ((60, 44, 10), (10, 50, 60), (30, 32, 40)))):
            rect(badges, 0, i * 28, 720, 28, color)
            blit(badges, render_text(text, 700, 24, 16, "rgb(255,214,90)" if i == 0 else ("rgb(80,230,255)" if i == 1 else "rgb(170,175,190)"), FONT_BOLD), 10, i * 28 + 4)
        self._badges = gpu.acquire_texture(720, 84, "rgba8_unorm", ["texture_binding"])
        self._badges.lock(read_only=False)
        self._badges.as_numpy()[:, :, :] = badges
        self._badges.unlock()
        self._kernel = gpu.create_compute_kernel(source=CONSOLE_GLSL, push_constant_size=16, bindings={
            "game_frame": "sampled_texture", "palettes": "sampled_texture", "depth_pane": "sampled_texture", "seg_pane": "sampled_texture",
            "lidar_pane": "sampled_texture", "map_pane": "sampled_texture", "telemetry": "sampled_texture", "graph_panel": "sampled_texture",
            "caption": "sampled_texture", "chrome": "sampled_texture", "badges": "sampled_texture", "output_image": "storage_image"})
        self._scratch = gpu.acquire_texture(8, 8, "rgba8_unorm", RING_USAGE)
        self._settle = gpu.create_compute_kernel(source=SETTLE_GLSL, bindings={"settled_source": "sampled_texture", "scratch_image": "storage_image"})
        log.info(f"MARKER:CONSOLE_SETUP pid={os.getpid()}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        for name, port in self.PANE_INPUTS.items():
            bag = ctx.inputs.read(port)
            if bag is not None:
                self.latest[name] = bag
        frame = ctx.inputs.read("frame_from_upstream")
        gpu = ctx.gpu_limited_access
        if frame is not None and (self.frame is None or frame["surface_id"] != self.frame["surface_id"]):
            if self.frame_handle is not None:
                self.frame_handle.close()
            try:
                self.frame_handle = gpu.resolve_surface(frame["surface_id"])
                self.frame = frame
            except Exception:
                self.frame_handle = None
        if self.frame_handle is None:
            return
        now_ns = clock.monotonic_now_ns()
        if now_ns < self.next_frame_ns:
            return
        self.next_frame_ns = max(self.next_frame_ns + 1_000_000_000 // 35, now_ns - 2 * 1_000_000_000 // 35)
        frame = self.frame
        t_start = time.monotonic()
        slot = self._ring.next_texture_for_this_frame(gpu, OUT_W, OUT_H)
        for name in self.PANE_INPUTS:
            bag = self.latest.get(name)
            if bag is None:
                continue
            cached = self.handles.get(name)
            if cached is not None and (cached[0] == bag["surface_id"] or self.frames - cached[2] < self.REFRESH_EVERY[name]):
                continue
            if cached is not None and cached[1] is not None:
                cached[1].close()
            try:
                handle = gpu.resolve_surface(bag["surface_id"])
            except Exception:
                handle = None
            self.handles[name] = (bag["surface_id"], handle, self.frames)
        handles = {name: cached[1] for name, cached in self.handles.items()}
        try:
            t_resolved = time.monotonic()
            if True:
                game_frame = self.frame_handle
                source = ((frame.get("state") or {}).get("control_source")) or "idle"
                badge = 0.0 if source == "autonomy" else (1.0 if source == "teleop" else 2.0)
                self._kernel.dispatch(bindings={
                    "game_frame": game_frame, "palettes": self._palettes,
                    "depth_pane": handles.get("depth") or self._blank["view"], "seg_pane": handles.get("segmentation") or self._blank["view"],
                    "lidar_pane": handles.get("lidar") or self._blank["pane"], "map_pane": handles.get("map") or self._blank["pane"],
                    "telemetry": handles.get("telemetry") or self._blank["pane"], "graph_panel": handles.get("graph") or self._blank["graph"],
                    "caption": handles.get("caption") or self._blank["caption"], "chrome": self._chrome, "badges": self._badges, "output_image": slot,
                }, group_count=(OUT_W // 8, OUT_H // 8, 1), push_constants=struct.pack("<4f", float(frame.get("palette", 0)), badge, 0.0, 0.0))
        finally:
            pass
        t_dispatched = time.monotonic()
        self._settle.dispatch(bindings={"settled_source": slot, "scratch_image": self._scratch}, group_count=(1, 1, 1))
        t_done = time.monotonic()
        self.timing = getattr(self, "timing", numpy.zeros(4))
        self.timing += (t_resolved - t_start, t_dispatched - t_resolved, t_done - t_dispatched, t_done - t_start)
        if self.frames % 200 == 199:
            ms = self.timing / 200 * 1000
            log.info(f"MARKER:CONSOLE_TIMING resolve={ms[0]:.1f} dispatch={ms[1]:.1f} settle={ms[2]:.1f} total={ms[3]:.1f} panes={len(handles)}")
            self.timing[:] = 0
        now_ns = clock.monotonic_now_ns()
        ctx.outputs.write("video", {"surface_id": slot.surface_id, "width": OUT_W, "height": OUT_H, "timestamp_ns": now_ns, "fps": 35, "texture_layout": 5})
        self.frames += 1
        self.fps_window.append(time.monotonic())
        if time.monotonic() - self.last_witness > 0.2:
            self.last_witness = time.monotonic()
            stage = dict(frame.get("stage_ns") or {})
            stage["console"] = now_ns
            keys = list(stage.keys())
            stage_ms = {keys[i + 1]: (stage[keys[i + 1]] - stage[keys[i]]) / 1e6 for i in range(len(keys) - 1)} if len(keys) > 1 else {}
            panes = {"operator": {"surface_id": frame["surface_id"], "pid": frame.get("pid"), "source_surface_id": frame.get("surface_id")}}
            for name in ("depth", "segmentation", "lidar", "map"):
                bag = self.latest.get(name)
                if bag is not None:
                    panes[name] = {"surface_id": bag.get("surface_id"), "pid": bag.get("pid"), "source_surface_id": bag.get("source_surface_id")}
            fps = 0.0
            if len(self.fps_window) > 5:
                fps = (len(self.fps_window) - 1) / max(self.fps_window[-1] - self.fps_window[0], 1e-6)
            ctx.outputs.write("witness_to_downstream", {"panes": panes, "stage_ms": {k: round(v, 1) for k, v in stage_ms.items()}, "fps": round(fps, 1),
                                                        "console_pid": os.getpid(), "tick": frame.get("tick"), "timestamp_ns": now_ns})
        if self.frames in (1, 35, 35 * 60):
            log.info(f"MARKER:CONSOLE_FRAME frames={self.frames} panes={sorted(self.latest)}")
