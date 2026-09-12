"""A causal video-to-video re-render: StreamDiffusionV2 (Wan 2.1 1.3B, DMD-distilled) over the
renderer's frames, in place of the per-frame image model in `neural.DiffusionRerender`.

The model carries its own temporal state — a rolling KV cache over four-frame chunks — so the
look holds across frames without the reprojected feedback loop the image model needs. It reads
the same `view_from_upstream` and publishes the same `neural_to_downstream` bag, so the console
and `scripts/neural-setup.py` swap one for the other unchanged.
"""
from __future__ import annotations

import collections
import os
import threading
import time

import numpy

from streamlib import RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .neural import DEFAULT_STYLE, NEURAL_H, NEURAL_W, STYLES, _NeuralBase, _view_tensor

# The model is trained at 480x832; the DOOM view is 320x200. Height and width must be multiples
# of 16 — the DiT's token grid is height/16 x width/16 and its KV cache is sized from it. 320x544
# measured the most surviving detail per millisecond of the six sizes tried on a 3090.
MODEL_H = int(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_HEIGHT", "320"))
MODEL_W = int(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_WIDTH", "544"))
STEP = int(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_STEP", "1"))
# How much of the input frame survives the noising: 1.0 re-imagines freely, 0 passes the frame
# through. 0.85 measured 45% more surviving detail than 0.75 at no cost in frame time.
NOISE_SCALE = float(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_NOISE", "0.85"))
USE_TAEHV = os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_TAEHV", "1") == "1"
# The model sustained 19.4 fps alone on a 3090; the four networks and a 60 fps renderer share it.
VIDEO_DIFFUSION_FPS = float(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_FPS", "16"))
LIFT = float(os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_LIFT", "0.7"))
CHECKPOINT_FOLDER = os.environ.get("STREAMLIB_DOOM_VIDEO_DIFFUSION_CKPT", "/tmp/doom/sdv2-models/ckpts/wan_causal_dmd_v2v")
MODEL_ROOT = os.environ.get("STREAMDIFFUSIONV2_ROOT", "/tmp/doom/sdv2-models")
CHUNK_FRAMES = 4


def cpu_prompt_embedding_cache(encoder, torch):
    """The umt5-xxl text encoder moved to the CPU, its embedding cached and served to the DiT.

    `CausalStreamInferencePipeline.to(device)` moves this 11 GB encoder onto the GPU with
    everything else, which on a 24 GB card shared with the game leaves no room; a prompt only
    changes when someone asks for a new style, so it lives on the CPU and is encoded off the
    frame thread. The pipeline is an `nn.Module` and only accepts one in that slot, and this
    module must import without torch so the parent can read the catalog — hence the class here.
    """

    class PromptEmbeddingCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder.to("cpu", torch.float32)
            self._cached: dict[str, dict] = {}
            self._lock = threading.Lock()

        def forward(self, text_prompts):
            prompt = text_prompts[0]
            with self._lock:
                hit = self._cached.get(prompt)
            if hit is None:
                out = self.encoder(text_prompts=[prompt])
                hit = {key: (value.to("cuda", torch.bfloat16) if torch.is_tensor(value) else value) for key, value in out.items()}
                with self._lock:
                    self._cached[prompt] = hit
                    if len(self._cached) > 8:
                        self._cached.pop(next(iter(self._cached)))
            return dict(hit)  # `prepare` mutates the dict it is handed

        def warm(self, prompt: str) -> None:
            self([prompt])

        def holds(self, prompt: str) -> bool:
            with self._lock:
                return prompt in self._cached

    return PromptEmbeddingCache()


@processor(description="Causal video-to-video re-render: StreamDiffusionV2 (Wan 2.1 1.3B, DMD-distilled) holds a rolling KV cache over four-frame chunks, so the style is temporally consistent by construction rather than by reprojection. The style comes from the game's state (`style` director verb). Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class VideoDiffusionRerender(_NeuralBase):
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def neural_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_common(ctx, "video_diffusion")
        os.environ.setdefault("STREAMDIFFUSIONV2_ROOT", MODEL_ROOT)
        self.style = DEFAULT_STYLE
        self.prompt = STYLES.get(DEFAULT_STYLE, DEFAULT_STYLE)
        self.manager = None
        self.embeddings = None
        self.rate = 0.0
        self.session = None
        self.path = None

        self._intake: collections.deque = collections.deque(maxlen=2)
        self._finished: collections.deque = collections.deque(maxlen=12)
        self._pending_style: tuple[str, str] | None = None
        self._wake = threading.Condition()
        self._pending_frames: list = []
        self._pending_takes: list[dict] = []
        self._pending_meta: list[dict] = []
        self._next_emit_ns = 0
        self._loaded = threading.Event()

        # The engine gives setup 60 s; this checkpoint plus a CPU pass of the text encoder needs
        # more, so the load runs on the worker thread and the node publishes nothing until it ends.
        self._worker = threading.Thread(target=self._load_then_denoise_forever, name="video-diffusion", daemon=True)
        self._worker.start()
        log.info(f"MARKER:VIDEO_DIFFUSION_SPAWNED pid={os.getpid()} loading in the background")

    def _load(self) -> None:
        """Build the pipeline and encode the default style — tens of seconds, off the frame thread."""
        torch = self.torch
        started = time.monotonic()
        log.info(f"MARKER:VIDEO_DIFFUSION_LOADING size={MODEL_H}x{MODEL_W} step={STEP} taehv={USE_TAEHV} noise={NOISE_SCALE} ckpt={CHECKPOINT_FOLDER}")
        from streamdiffusionv2 import StreamDiffusionV2Pipeline

        self.stream = StreamDiffusionV2Pipeline(
            checkpoint_folder=CHECKPOINT_FOLDER, mode="single", height=MODEL_H, width=MODEL_W,
            step=STEP, noise_scale=NOISE_SCALE, use_taehv=USE_TAEHV,
        )
        manager = self.stream.pipeline_manager
        self.embeddings = cpu_prompt_embedding_cache(manager.pipeline.text_encoder, torch)
        manager.pipeline.text_encoder = self.embeddings
        torch.cuda.empty_cache()
        self.embeddings.warm(self.prompt)
        self.manager = manager
        self._loaded.set()
        log.info(f"MARKER:VIDEO_DIFFUSION_READY load_s={time.monotonic() - started:.0f} pid={os.getpid()} vram_mib={round(torch.cuda.memory_allocated() / 2**20)}")

    def _load_then_denoise_forever(self) -> None:
        try:
            self._load()
        except Exception as failure:
            log.error(f"MARKER:VIDEO_DIFFUSION_LOAD_FAILED {failure}")
            return
        self._denoise_forever()

    def _model_input(self, view_tensor):
        """One view as the (3, H, W) bfloat16 tensor in [-1, 1] the Wan stream encoder wants."""
        torch = self.torch
        rgb = self._rgb(view_tensor, 0).permute(2, 0, 1)[None].float() / 255.0
        rgb = rgb.pow(LIFT)  # a re-imagined room is lit; Doom's corridors are not
        rgb = torch.nn.functional.interpolate(rgb, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False, antialias=True)
        return (rgb * 2.0 - 1.0)[0].to(torch.bfloat16)

    def _to_pane(self, frames_thwc: numpy.ndarray) -> list[numpy.ndarray]:
        """Decoded model frames as 512x320 uint8 panes — the size the console's kernel samples."""
        torch = self.torch
        tensor = torch.from_numpy(frames_thwc).cuda().permute(0, 3, 1, 2).float().clamp(0, 1)
        tensor = torch.nn.functional.interpolate(tensor, size=(NEURAL_H, NEURAL_W), mode="bilinear", align_corners=False, antialias=True)
        pixels = (tensor * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [pixels[i] for i in range(pixels.shape[0])]

    def _restart_session(self, prompt: str, chunk):
        """Open a new stream session on the first 1 + 4 frames; resets the rolling KV cache."""
        torch = self.torch
        frames = torch.stack(chunk["frames"], dim=1)[None]
        first = torch.cat([frames[:, :, :1], frames], dim=2)  # the session wants 1 + chunk frames
        self.session, initial = self.manager.start_stream_session(prompt, first, NOISE_SCALE)
        panes = self._to_pane(initial)
        return panes[1:] if len(panes) > len(chunk["meta"]) else panes

    def _denoise_forever(self) -> None:
        """The model loop: one four-frame chunk at a time, off the process's own frame thread."""
        while True:
            with self._wake:
                while not self._intake:
                    self._wake.wait()
                chunk = self._intake.popleft()
            try:
                started = time.monotonic()
                switching = self._pending_style
                if switching is not None and self.embeddings.holds(switching[1]):
                    self._pending_style = None
                    self.style, self.prompt = switching
                    self.session = None
                    log.info(f"MARKER:VIDEO_DIFFUSION_STYLE style={self.style}")
                if self.session is None:
                    self._pending_meta.clear()  # a restart drops the frames still in flight; their poses go with them
                    panes = self._restart_session(self.prompt, chunk)
                else:
                    frames = self.torch.stack(chunk["frames"], dim=1)[None]
                    panes = [pane for produced in self.manager.run_stream_batch(self.session, frames) for pane in self._to_pane(produced)]
                self.ms = (time.monotonic() - started) * 1000
                self.rate = 0.9 * self.rate + 0.1 * (1000.0 * len(chunk["meta"]) / max(self.ms, 1.0))
                self._pending_meta.extend(chunk["meta"])
                for pane in panes:
                    if self._pending_meta:
                        self._finished.append((pane, self._pending_meta.pop(0)))
            except Exception as failure:
                log.error(f"MARKER:VIDEO_DIFFUSION_FAILED {failure}")
                self.session = None
                self._pending_meta.clear()

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is not None:
            self._take(ctx, view)
        self._emit(ctx)

    def _take(self, ctx: RuntimeContextLimitedAccess, view: dict) -> None:
        """Sample the view at the configured rate and hand whole chunks to the worker."""
        now = clock.monotonic_now_ns()
        period = int(1e9 / VIDEO_DIFFUSION_FPS)
        if now < getattr(self, "_next_take_ns", 0):
            return
        self._next_take_ns = max(getattr(self, "_next_take_ns", 0) + period, now - period)
        if not self._loaded.is_set():
            return
        if len(self._intake) >= self._intake.maxlen:
            return  # the worker is behind: skip this view rather than let the deque drop an older one out of order

        style = str(((view.get("state") or {}).get("style")) or DEFAULT_STYLE)
        prompt = STYLES.get(style, style)
        if not prompt:
            return
        if style != self.style and (self._pending_style or ("", ""))[0] != style:
            self._pending_style = (style, prompt)
            threading.Thread(target=self.embeddings.warm, args=(prompt,), daemon=True).start()

        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as handle:
            tensor, self.path = _view_tensor(handle, self.torch)
        self._pending_frames.append(self._model_input(tensor))
        self._pending_takes.append({"pose": view.get("pose"), "tick": view.get("tick"), "style": style, "prompt": prompt})
        if len(self._pending_frames) < CHUNK_FRAMES:
            return
        with self._wake:
            self._intake.append({"frames": self._pending_frames, "meta": self._pending_takes})
            self._wake.notify()
        self._pending_frames, self._pending_takes = [], []

    def _emit(self, ctx: RuntimeContextLimitedAccess) -> None:
        """Publish at most one finished frame per output period, so four never land at once."""
        now = clock.monotonic_now_ns()
        if now < self._next_emit_ns or not self._finished:
            return
        period = int(1e9 / VIDEO_DIFFUSION_FPS)
        self._next_emit_ns = max(self._next_emit_ns + period, now - period)
        # Draining faster than the period when the queue backs up keeps latency from growing.
        if len(self._finished) > 2 * CHUNK_FRAMES:
            self._next_emit_ns = now
        pane, meta = self._finished.popleft()
        self._publish(ctx, pane, "neural_to_downstream", {
            "style": meta["style"], "prompt": meta["prompt"][:80],
            "model": f"streamdiffusionv2 wan1.3b {MODEL_H}x{MODEL_W} s{STEP}",
            "tick": meta["tick"], "pose": meta["pose"],
            "fps": round(min(self.rate, VIDEO_DIFFUSION_FPS), 1), "queued": len(self._finished),
        })
        self.frames += 1
        if self.frames in (1, 30, 300, 3000):
            log.info(f"MARKER:VIDEO_DIFFUSION_FRAME frames={self.frames} chunk_ms={self.ms:.0f} queued={len(self._finished)} path={self.path} style={self.style}")
