"""Capture the renderer's view to disk — the frame and the camera pose that drew it.

A scene captured this way replays offline against a model as many times as a parameter
sweep needs, which beats restarting the node once per setting. Add it over MCP, let it
fill its directory, remove it.
"""
from __future__ import annotations

import json
import os

import numpy

from streamlib import RuntimeContextFullAccess, RuntimeContextLimitedAccess, input, log, processor

VIEW_W, VIEW_H = 320, 200
CAPTURE_DIR = os.environ.get("STREAMLIB_DOOM_CAPTURE_DIR", "/tmp/doom/capture")
CAPTURE_FRAMES = int(os.environ.get("STREAMLIB_DOOM_CAPTURE_FRAMES", "48"))
CAPTURE_EVERY = int(os.environ.get("STREAMLIB_DOOM_CAPTURE_EVERY", "5"))


@processor(description="Writes the renderer's view — palette index, depth and class per pixel — and the camera pose that drew it, as .npy and .json, for replaying a scene offline against a model. Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class ViewRecorder:
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    def __init__(self) -> None:
        self.saved = 0
        self.seen = 0
        self.last_tick = -1

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        log.info(f"MARKER:CAPTURE_SETUP dir={CAPTURE_DIR} frames={CAPTURE_FRAMES} every={CAPTURE_EVERY}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None or view.get("tick") == self.last_tick or self.saved >= CAPTURE_FRAMES:
            return
        self.last_tick = view.get("tick")
        self.seen += 1
        if self.seen % CAPTURE_EVERY:
            return
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as surface:
            surface.lock()
            pixels = surface.as_numpy()[:VIEW_H, :VIEW_W].copy()
            surface.unlock()
        numpy.save(os.path.join(CAPTURE_DIR, f"{self.saved:04d}.npy"), pixels)
        with open(os.path.join(CAPTURE_DIR, f"{self.saved:04d}.json"), "w") as out:
            json.dump({"pose": view.get("pose"), "tick": view.get("tick"), "palette": view.get("palette", 0)}, out)
        self.saved += 1
        if self.saved in (1, CAPTURE_FRAMES):
            log.info(f"MARKER:CAPTURE_FRAME saved={self.saved}")
