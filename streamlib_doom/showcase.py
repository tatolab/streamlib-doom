"""The showcase: the game as the phone sees it, the live graph beside it, and
the director's transcript, composited to 1920x1080 by a processor and recorded
by the engine.

The graph panel reads the node's own `/api/graph`, so a processor an agent
adds over MCP appears in the picture the moment it exists. The transcript
panel tails the file `director.sh` writes, so what was asked and what Claude
answered are on screen with the frame they changed.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import time
import urllib.request

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .wad import Wad

PANEL_W, PANEL_H = 640, 510
OUT_W, OUT_H = 1920, 1080
RING_USAGE = ["texture_binding", "storage_binding"]
CONTROL_URL = os.environ.get("STREAMLIB_DOOM_CONTROL_URL", "http://127.0.0.1:9200")
TRANSCRIPT_PATH = os.environ.get("STREAMLIB_DOOM_TRANSCRIPT", "/tmp/streamlib-doom-transcript.log")
FONT = "DejaVu-Sans-Mono"
FONT_BOLD = "DejaVu-Sans-Mono-Bold"

CORE_LAYOUT = {  # display name -> (column, row) on the panel's grid
    "Game": (0, 0), "Renderer": (1, 0), "StatusBar": (2, 0), "Browser": (3, 0),
    "Mixer": (0, 1), "Opus": (1, 1), "Showcase": (2, 1), "H264": (3, 1), "Recorder": (3, 2),
    "GraphPanel": (0, 2), "Transcript": (1, 2), "Upscaler": (2, 2), "WHIP": (2, 2),
}
COLORS = {"core": (70, 130, 220), "native": (110, 110, 120), "director": (230, 180, 40), "effect": (200, 70, 200), "api": (60, 60, 70)}


def render_text(text: str, width: int, height: int, pointsize: int = 20, color: str = "white", font: str = FONT, wrap: bool = False) -> numpy.ndarray:
    """(height, width, 4) uint8 RGBA of `text` on a transparent canvas, through ImageMagick."""
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


def blit(canvas: numpy.ndarray, sprite: numpy.ndarray, x: int, y: int) -> None:
    """Alpha-composite `sprite` onto `canvas` at (x, y), clipped."""
    h, w = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = sprite[y0 - y : y1 - y, x0 - x : x1 - x].astype(numpy.float32)
    alpha = src[:, :, 3:4] / 255.0
    dst = canvas[y0:y1, x0:x1, :3].astype(numpy.float32)
    canvas[y0:y1, x0:x1, :3] = (src[:, :, :3] * alpha + dst * (1.0 - alpha)).astype(numpy.uint8)


def rect(canvas: numpy.ndarray, x: int, y: int, w: int, h: int, color: tuple, border: tuple | None = None) -> None:
    canvas[y : y + h, x : x + w, :3] = color
    canvas[y : y + h, x : x + w, 3] = 255
    if border:
        canvas[y, x : x + w, :3] = border
        canvas[y + h - 1, x : x + w, :3] = border
        canvas[y : y + h, x, :3] = border
        canvas[y : y + h, x + w - 1, :3] = border


def line(canvas: numpy.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple, thickness: int = 2) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for k in range(steps + 1):
        x = int(x0 + (x1 - x0) * k / steps)
        y = int(y0 + (y1 - y0) * k / steps)
        canvas[max(0, y - thickness // 2) : y + thickness // 2 + 1, max(0, x - thickness // 2) : x + thickness // 2 + 1, :3] = color


class _PanelPublisher:
    """A CPU-drawn panel published as a texture another processor samples."""

    def _publish(self, ctx, canvas: numpy.ndarray, port: str) -> None:
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, PANEL_W, PANEL_H)
        slot.lock(read_only=False)
        slot.as_numpy()[:, :, :] = canvas
        slot.unlock()
        ctx.outputs.write(port, {"surface_id": slot.surface_id, "width": PANEL_W, "height": PANEL_H, "timestamp_ns": clock.monotonic_now_ns()})


@processor(execution="continuous", interval_ms=100)
class GraphPanel(_PanelPublisher):
    """The node's live graph, drawn from its own `/api/graph` twice a second."""

    @output()
    def panel_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=2)
        self.labels: dict[str, numpy.ndarray] = {}
        self.last_fetch = 0.0
        self.frames = 0
        self.flow_phase = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self.title = render_text("LIVE GRAPH  ·  what the runtime is running right now", PANEL_W, 26, 17, "rgb(160,170,190)", FONT_BOLD)
        log.info("MARKER:GRAPH_PANEL_SETUP")

    def _label(self, text: str, color: str = "white", size: int = 15) -> numpy.ndarray:
        key = f"{text}|{color}|{size}"
        if key not in self.labels:
            self.labels[key] = render_text(text, 150, 22, size, color, FONT_BOLD)
        return self.labels[key]

    def _draw(self, graph: dict) -> numpy.ndarray:
        canvas = numpy.zeros((PANEL_H, PANEL_W, 4), dtype=numpy.uint8)
        canvas[:, :, :3] = (18, 20, 26)
        canvas[:, :, 3] = 255
        blit(canvas, self.title, 12, 8)
        nodes = [n for n in graph.get("nodes", []) if n["display_name"] != "ApiServerProcessor"]
        positions: dict[str, tuple[int, int]] = {}
        box_w, box_h = 132, 44
        col_x = [14, 170, 326, 482]
        row_y = [56, 130, 204]
        extra_row = 0
        for n in nodes:
            name = n["display_name"]
            if name in CORE_LAYOUT:
                c, r = CORE_LAYOUT[name]
                positions[n["id"]] = (col_x[c], row_y[r])
            else:
                positions[n["id"]] = (col_x[extra_row % 4], 300 + (extra_row // 4) * 70)
                extra_row += 1
        for link_ in graph.get("links", []):
            a, b = positions.get(link_["source"]["processor_id"]), positions.get(link_["target"]["processor_id"])
            if not a or not b:
                continue
            color = (90, 200, 120) if link_.get("state") == "wired" else (120, 60, 60)
            line(canvas, a[0] + box_w, a[1] + box_h // 2, b[0], b[1] + box_h // 2, color, 2)
            # a moving dot says bags are flowing
            t = ((self.flow_phase * 0.18) % 1.0)
            dx, dy = int(a[0] + box_w + (b[0] - a[0] - box_w) * t), int(a[1] + box_h // 2 + (b[1] - a[1]) * t)
            rect(canvas, max(0, dx - 3), max(0, dy - 3), 6, 6, (230, 240, 255))
        for n in nodes:
            name, state = n["display_name"], n["components"].get("state", "?")
            x, y = positions[n["id"]]
            kind = "director" if name.startswith("Director") else "effect" if "Effect" in name or n["type"].endswith("ScreenEffect") else "native" if "::" in n["type"] else "core"
            fill = COLORS[kind] if state == "Running" else (120, 40, 40)
            rect(canvas, x, y, box_w, box_h, fill, (240, 240, 250))
            blit(canvas, self._label(name[:14]), x + 8, y + 3)
            blit(canvas, self._label(("python helper" if "::" not in n["type"] else "native rust") if kind in ("core", "native") else ("added by Claude" if kind == "director" else "spliced by Claude"), "rgb(230,230,235)", 11), x + 8, y + 24)
        blit(canvas, self._label(f"{len(nodes)} processors · {len(graph.get('links', []))} links · each Python one is its own process", "rgb(150,160,180)", 12), 12, PANEL_H - 24)
        return canvas

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        now = time.monotonic()
        self.flow_phase += 1
        if now - self.last_fetch < 0.45:
            return
        self.last_fetch = now
        try:
            graph = json.load(urllib.request.urlopen(CONTROL_URL + "/api/graph", timeout=2))
        except Exception:
            return
        self._publish(ctx, self._draw(graph), "panel_to_downstream")
        self.frames += 1
        if self.frames in (1, 20):
            log.info(f"MARKER:GRAPH_PANEL frames={self.frames}")


@processor(execution="continuous", interval_ms=250)
class TranscriptPanel(_PanelPublisher):
    """What was asked of Claude and what it answered, tailed from the director's log."""

    @output()
    def panel_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=2)
        self.last_text = None
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self.title = render_text("CLAUDE, DIRECTING  ·  `claude -p` on the desktop, MCP into the node", PANEL_W, 26, 17, "rgb(160,170,190)", FONT_BOLD)
        log.info("MARKER:TRANSCRIPT_PANEL_SETUP")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        try:
            text = open(TRANSCRIPT_PATH, encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        if text == self.last_text and self.frames > 0:
            return
        self.last_text = text
        canvas = numpy.zeros((PANEL_H, PANEL_W, 4), dtype=numpy.uint8)
        canvas[:, :, :3] = (18, 20, 26)
        canvas[:, :, 3] = 255
        blit(canvas, self.title, 12, 8)
        # Word-wrap by hand so each visual line keeps its own colour: a prompt
        # line is gold, Claude's reply is white.
        raw_lines = text.strip().splitlines()
        wrapped: list[tuple[str, bool]] = []
        for raw in raw_lines:
            is_prompt = raw.startswith(">")
            body = ("you \u203a " + raw[1:].strip()) if is_prompt else raw
            words, current = body.split(), ""
            for word in words:
                trial = (current + " " + word).strip()
                if len(trial) > 58 and current:
                    wrapped.append((current, is_prompt))
                    current = word
                else:
                    current = trial
            wrapped.append((current, is_prompt))
        wrapped = wrapped[-24:]
        if not wrapped:
            wrapped = [("waiting for a prompt\u2026", False), ("", False), ("  ./director.sh \"teleport three imps behind the player\"", False)]
        y = 42
        for body, is_prompt in wrapped:
            if body:
                blit(canvas, render_text(body, PANEL_W - 24, 24, 16, "rgb(250,200,80)" if is_prompt else "rgb(225,230,240)", FONT_BOLD if is_prompt else FONT), 12, y)
            y += 19
        self._publish(ctx, canvas, "panel_to_downstream")
        self.frames += 1


SHOWCASE_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D game_frame;     // 320x200 indexed
layout(set = 0, binding = 1) uniform sampler2D palettes;       // 256x14
layout(set = 0, binding = 2) uniform sampler2D graph_panel;    // 640x510
layout(set = 0, binding = 3) uniform sampler2D transcript;     // 640x510
layout(set = 0, binding = 4) uniform sampler2D chrome;         // 1920x1080 title and footer
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D output_image;
layout(push_constant) uniform PC { float palette; float unused0; float unused1; float unused2; } pc;

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 1920 || at.y >= 1080) return;
    vec4 color = vec4(0.05, 0.055, 0.07, 1.0);
    if (at.x < 1280 && at.y >= 60 && at.y < 1020) {
        ivec2 src = ivec2(at.x / 4, ((at.y - 60) * 200) / 960);
        int index = int(texelFetch(game_frame, src, 0).r * 255.0 + 0.5);
        color = vec4(texelFetch(palettes, ivec2(index, int(pc.palette)), 0).rgb, 1.0);
    } else if (at.x >= 1280 && at.y >= 30 && at.y < 540) {
        color = vec4(texelFetch(graph_panel, ivec2(at.x - 1280, at.y - 30), 0).rgb, 1.0);
    } else if (at.x >= 1280 && at.y >= 550 && at.y < 1060) {
        color = vec4(texelFetch(transcript, ivec2(at.x - 1280, at.y - 550), 0).rgb, 1.0);
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


@processor
class ShowcaseCompositor:
    """1920x1080: the game at 4:3 on the left, the live graph and the transcript on the right."""

    @input(delivery_profile="newest")
    def frame_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def graph_panel_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def transcript_from_upstream(self) -> None: ...

    @output()
    def video(self) -> None: ...

    def __init__(self) -> None:
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.graph_surface = None
        self.transcript_surface = None
        self.frames = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        gpu = ctx.gpu_full_access
        wad = Wad()
        self._palettes = gpu.acquire_texture(256, 14, "rgba8_unorm", ["texture_binding"])
        self._palettes.lock(read_only=False)
        pal = self._palettes.as_numpy()
        pal[:, :, :3] = wad.palettes()
        pal[:, :, 3] = 255
        self._palettes.unlock()
        self._blank_panel = gpu.acquire_texture(PANEL_W, PANEL_H, "rgba8_unorm", ["texture_binding"])
        self._blank_panel.lock(read_only=False)
        blank = self._blank_panel.as_numpy()
        blank[:, :, :3] = (18, 20, 26)
        blank[:, :, 3] = 255
        self._blank_panel.unlock()
        chrome = numpy.zeros((OUT_H, OUT_W, 4), dtype=numpy.uint8)
        blit(chrome, render_text("DOOM E1M1  ·  four StreamLib processors, one GPU column-caster  ·  a phone is playing", 1270, 40, 24, "rgb(235,238,245)", FONT_BOLD), 12, 12)
        blit(chrome, render_text("github.com/tatolab/streamlib-doom  ·  the runtime is recording this picture itself", 1270, 34, 18, "rgb(150,160,180)", FONT), 12, 1032)
        self._chrome = gpu.acquire_texture(OUT_W, OUT_H, "rgba8_unorm", ["texture_binding"])
        self._chrome.lock(read_only=False)
        self._chrome.as_numpy()[:, :, :] = chrome
        self._chrome.unlock()
        self._kernel = gpu.create_compute_kernel(source=SHOWCASE_GLSL, push_constant_size=16, bindings={
            "game_frame": "sampled_texture", "palettes": "sampled_texture", "graph_panel": "sampled_texture",
            "transcript": "sampled_texture", "chrome": "sampled_texture", "output_image": "storage_image"})
        self._scratch = gpu.acquire_texture(8, 8, "rgba8_unorm", RING_USAGE)
        self._settle = gpu.create_compute_kernel(source=SETTLE_GLSL, bindings={"settled_source": "sampled_texture", "scratch_image": "storage_image"})
        log.info("MARKER:SHOWCASE_SETUP")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        panel = ctx.inputs.read("graph_panel_from_upstream")
        if panel is not None:
            self.graph_surface = panel["surface_id"]
        transcript = ctx.inputs.read("transcript_from_upstream")
        if transcript is not None:
            self.transcript_surface = transcript["surface_id"]
        frame = ctx.inputs.read("frame_from_upstream")
        if frame is None:
            return
        gpu = ctx.gpu_limited_access
        slot = self._ring.next_texture_for_this_frame(gpu, OUT_W, OUT_H)
        with gpu.resolve_surface(frame["surface_id"]) as game_frame:
            graph_handle = gpu.resolve_surface(self.graph_surface) if self.graph_surface else None
            transcript_handle = gpu.resolve_surface(self.transcript_surface) if self.transcript_surface else None
            try:
                self._kernel.dispatch(bindings={
                    "game_frame": game_frame, "palettes": self._palettes,
                    "graph_panel": graph_handle or self._blank_panel, "transcript": transcript_handle or self._blank_panel,
                    "chrome": self._chrome, "output_image": slot,
                }, group_count=(OUT_W // 8, OUT_H // 8, 1), push_constants=struct.pack("<4f", float(frame.get("palette", 0)), 0.0, 0.0, 0.0))
            finally:
                for handle in (graph_handle, transcript_handle):
                    if handle is not None:
                        handle.close()
        self._settle.dispatch(bindings={"settled_source": slot, "scratch_image": self._scratch}, group_count=(1, 1, 1))
        ctx.outputs.write("video", {"surface_id": slot.surface_id, "width": OUT_W, "height": OUT_H, "timestamp_ns": clock.monotonic_now_ns(), "fps": 35, "texture_layout": 5})
        self.frames += 1
        if self.frames in (1, 35, 35 * 60):
            log.info(f"MARKER:SHOWCASE_FRAME frames={self.frames}")
