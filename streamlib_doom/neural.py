"""Neural processors over the same rendered frame: a diffusion re-render conditioned on the
renderer's own depth, a monocular depth network graded against the renderer's true depth,
an open-vocabulary detector graded against the renderer's own labels, and a repainter that
generates textures and patches them into the running renderer's atlas.

Each is its own process holding its own model on the GPU. The frame reaches them as a
CUDA tensor through the surface's DLPack door — one engine-side blit, no CPU hop.
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy

from streamlib import ProcessorOutputTextureRing, RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

from .sensors import decode_depth, depth_lut, detect_monsters
from .wad import Wad

VIEW_W, VIEW_H = 320, 200
NEURAL_W, NEURAL_H = 512, 320
RING_USAGE = ["texture_binding", "storage_binding"]
# The four models share one GPU with a 60 fps game, so each takes a slice of one second per
# second: the re-render is the picture people watch, so it gets most of it.
DIFFUSION_FPS = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_FPS", "12"))
DEPTH_FPS = float(os.environ.get("STREAMLIB_DOOM_NEURAL_DEPTH_FPS", "4"))
DETECTOR_FPS = float(os.environ.get("STREAMLIB_DOOM_DETECTOR_FPS", "1.5"))
COMPILE = os.environ.get("STREAMLIB_DOOM_COMPILE", "1") == "1"
HF_MODELS = {
    "diffusion": "stabilityai/sd-turbo",
    "controlnet": "thibaud/controlnet-sd21-depth-diffusers",
    "controlnet_seg": "thibaud/controlnet-sd21-ade20k-diffusers",
    "vae": "madebyollin/taesd",
    "depth": "depth-anything/Depth-Anything-V2-Small-hf",
    "detector": "IDEA-Research/grounding-dino-base",
}
# Composed scenes rather than keyword lists: a subject, its materials, the light and the lens.
# The subject stays the room itself, because the rendered frame already fixes what is in shot.
STYLES = {
    "lego": "the walls, floor and ceiling of this room rebuilt out of LEGO bricks, stacked studded plastic bricks with visible stud tops and seams between them, each brick keeping the exact colour of the surface it replaces, glossy moulded plastic catching the light, macro toy photograph, crisp focus",
    "cyberpunk": "a rain-soaked cyberpunk corridor deep inside a megatower, hot pink and electric cyan signage burning through the haze, wet concrete reflecting the neon, chrome panelling and exposed cabling along the walls, holographic adverts flickering in the distance, anamorphic flare, cinematic night photography, richly detailed",
    "bladerunner": "a Blade Runner 2049 film still of a vast brutalist corridor, dense volumetric fog lit from above, towering cyan and magenta signage fading into the murk, wet reflective floor, weathered concrete and cold chrome, anamorphic lens flare, cinematic teal and orange grade, moody and enormous",
    "night_city": "a Night City interior from Cyberpunk 2077, saturated neon signage crowding the walls, holographic advertisements drifting in the air, chrome and carbon fibre panels bolted over grimy industrial plating, magenta and cyan rim light raking across every surface, high detail game screenshot",
    "photoreal": "a photograph inside an abandoned military base, corroded steel walls streaked with rust, harsh fluorescent light from overhead strips, dust hanging in the beam, shot on 35mm film, highly detailed",
    "anime": "a hand-painted anime film still of a sci-fi corridor, cel shaded with bold ink outlines, warm Studio Ghibli light falling through the doorway, vivid saturated colours, painted background art",
    "claymation": "a stop-motion claymation diorama of a spaceship corridor, every surface hand-moulded plasticine with visible thumbprints, soft warm studio lighting, shallow macro photograph of a miniature set",
    "watercolor": "a loose watercolour painting of a dark space station corridor, wet pigment blooming into rough paper, ink outlines drawn over the wash, muted blues and ochres, visible brush strokes",
    "alien": "the inside of a biomechanical alien hive, wet organic walls of ribbed chitin and cabling, bioluminescent veins glowing green through the dark, H.R. Giger, cinematic and claustrophobic",
    "none": "",
}
# The renderer's surface classes painted in ADE20K's own colours, so a ControlNet trained on
# ADE20K reads sky, wall, floor, ceiling and monster as the things they are.
ADE_CLASS_COLORS = {0: (6, 230, 230), 1: (120, 120, 120), 2: (80, 50, 50), 3: (120, 120, 80),
                    4: (150, 5, 61), 5: (255, 6, 82), 6: (204, 255, 4), 7: (224, 5, 255)}
# Pushing off the game's own look is what lets the model replace the flat textures rather than
# tint them. It needs guidance above 1 to apply at all, which costs a second pass per step.
NEGATIVE_PROMPT = os.environ.get("STREAMLIB_DOOM_NEGATIVE_PROMPT",
                                 "3d render, video game graphics, low-poly, cell-shaded, flat shading, "
                                 "smooth shading, blurry, washed out, grainy, muddy, low detail")
GUIDANCE = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_GUIDANCE", "1.5"))
DEFAULT_STYLE = os.environ.get("STREAMLIB_DOOM_STYLE", "lego")
# The renderer's camera: FOCAL and CENTER_Y for a 320x200 view, scaled with the output.
VIEW_FOCAL, VIEW_CENTER_Y = 160.0, 100.0


def reproject(previous, previous_pose, current_pose, depth_z, torch):
    """`previous` (1,3,H,W) seen from `previous_pose`, re-drawn from `current_pose` using the
    current frame's per-pixel z depth (H,W) — the same reprojection a game's TAA does, with the
    depth the renderer already wrote. Returns the resampled image and a validity mask (1,1,H,W)."""
    _, _, H, W = previous.shape
    scale = W / 320.0
    focal, center_y = VIEW_FOCAL * scale, VIEW_CENTER_Y * scale
    cx, cy, cz, ca = current_pose
    px, py, pz, pa = previous_pose
    device = previous.device
    cols = torch.arange(W, device=device, dtype=torch.float32)[None, :].expand(H, W) + 0.5
    rows = torch.arange(H, device=device, dtype=torch.float32)[:, None].expand(H, W) + 0.5
    z = depth_z
    lateral = z * (cols - W / 2.0) / focal
    height = cz + (center_y - rows) * z / focal
    fx, fy = math.cos(ca), math.sin(ca)
    rx, ry = fy, -fx
    wx = cx + fx * z + rx * lateral
    wy = cy + fy * z + ry * lateral
    pfx, pfy = math.cos(pa), math.sin(pa)
    prx, pry = pfy, -pfx
    relx, rely = wx - px, wy - py
    zp = relx * pfx + rely * pfy
    xp = relx * prx + rely * pry
    safe = zp.clamp(min=1.0)
    cp = W / 2.0 + xp / safe * focal
    rp = center_y - (height - pz) * focal / safe
    grid = torch.stack([cp / W * 2.0 - 1.0, rp / H * 2.0 - 1.0], dim=-1)[None]
    sampled = torch.nn.functional.grid_sample(previous.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    valid = ((zp > 4.0) & (cp >= 0) & (cp < W) & (rp >= 0) & (rp < H)).float()[None, None]
    return sampled, valid


def _due(self, period_ns: int) -> bool:
    now = clock.monotonic_now_ns()
    if now < getattr(self, "_next_due_ns", 0):
        return False
    self._next_due_ns = max(getattr(self, "_next_due_ns", 0) + period_ns, now - period_ns)
    return True


def _view_tensor(handle, torch):
    """The view's rgba8 pixels as a CUDA uint8 tensor (H, W, 4), through DLPack when the
    surface offers its device side, else through the host mapping."""
    handle.lock()
    try:
        try:
            tensor = torch.from_dlpack(handle)
            path = "dlpack"
            if tensor.device.type != "cuda":
                tensor = tensor.cuda()
                path = "dlpack-host"
            tensor = tensor.clone()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # a refused device export surfaces as a pyo3 panic, not an Exception
            tensor = torch.from_numpy(handle.as_numpy()[:VIEW_H, :VIEW_W].copy()).cuda()
            path = "numpy"
    finally:
        handle.unlock()
    if tensor.ndim == 3 and tensor.shape[0] >= VIEW_H and tensor.shape[1] >= VIEW_W:
        tensor = tensor[:VIEW_H, :VIEW_W]
    return tensor, path


class _NeuralBase:
    def _setup_common(self, ctx: RuntimeContextFullAccess, name: str) -> None:
        import torch
        self.torch = torch
        wad = Wad()
        self.palettes = torch.from_numpy(numpy.asarray(wad.palettes(), dtype=numpy.uint8)).cuda()  # (14, 256, 3)
        self._ring = ProcessorOutputTextureRing("rgba8_unorm", RING_USAGE, depth=3)
        self.frames = 0
        self.path = None
        self.ms = 0.0
        self.name = name
        log.info(f"MARKER:{name.upper()}_SETUP pid={os.getpid()} gpu={torch.cuda.get_device_name(0)}")

    def _rgb(self, view_tensor, palette_index: int):
        """(H, W, 3) uint8 CUDA: the palette applied on the GPU, flash palette and all."""
        palette = self.palettes[min(max(int(palette_index), 0), self.palettes.shape[0] - 1)]
        return palette[view_tensor[:, :, 0].long()]

    def _publish(self, ctx, rgb_uint8_hwc: numpy.ndarray, port: str, extra: dict) -> None:
        h, w = rgb_uint8_hwc.shape[:2]
        slot = self._ring.next_texture_for_this_frame(ctx.gpu_limited_access, w, h)
        slot.lock(read_only=False)
        pixels = slot.as_numpy()
        pixels[:h, :w, :3] = rgb_uint8_hwc
        pixels[:h, :w, 3] = 255
        slot.unlock()
        ctx.outputs.write(port, {"surface_id": slot.surface_id, "width": w, "height": h, "timestamp_ns": clock.monotonic_now_ns(),
                                 "pid": os.getpid(), "path": self.path, "ms": round(self.ms, 1), **extra})


@processor(description="Diffusion re-render: a one-step image-to-image model (sd-turbo) with a depth ControlNet fed the renderer's own depth channel, so the picture is re-imagined in any style while its geometry stays the game's. The style comes from the game's state (`style` director verb). Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class DiffusionRerender(_NeuralBase):
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def neural_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_common(ctx, "diffusion")
        # Loading two ControlNets and compiling them past 60 s, and the engine caps setup at 60 s.
        # The work moves to a worker; process() publishes nothing until it reports ready.
        self.ready = False
        threading.Thread(target=self._load, name="diffusion-load", daemon=True).start()

    def _load(self) -> None:
            torch = self.torch
            from diffusers import AutoencoderTiny, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
            controlnets = [ControlNetModel.from_pretrained(HF_MODELS["controlnet"], torch_dtype=torch.float16),
                           ControlNetModel.from_pretrained(HF_MODELS["controlnet_seg"], torch_dtype=torch.float16)]
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(HF_MODELS["diffusion"], controlnet=controlnets, torch_dtype=torch.float16, safety_checker=None).to("cuda")
            lut = numpy.zeros((256, 3), dtype=numpy.uint8)
            for code, colour in ADE_CLASS_COLORS.items():
                lut[code] = colour
            self._class_lut = torch.from_numpy(lut).cuda()
            self.pipe.vae = AutoencoderTiny.from_pretrained(HF_MODELS["vae"], torch_dtype=torch.float16).to("cuda")
            self.pipe.set_progress_bar_config(disable=True)
            self.generator = torch.Generator("cuda").manual_seed(7)
            # These models are launch-bound at this size — the UNet costs the same on 64x40 latents
            # as on 44x28 — so the wins are not resolution: encode each prompt once, and let inductor
            # replay the step as a CUDA graph.
            self.embeds: dict = {}
            self.compiled = False
            self.rate = 0.0
            self.style = DEFAULT_STYLE
            # Temporal coherence: every frame after the first starts from its own previous output,
            # reprojected through the game's depth into the new camera pose, blended with the new game
            # frame — so the bricks stay where they were instead of being reinvented each frame.
            self.previous = None
            self.previous_pose = None
            self.previous_style = None
            # The rendered frame is the camera, and it drives every frame: a loop that mostly re-denoises
            # its own last output holds still but drains its colour and its studs within seconds. The carry
            # is a minority partner now, enough to damp the shimmer, and the two ControlNets — the game's
            # depth and its own per-pixel surface classes — hold the shape steady instead.
            self.carry = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_CARRY", "0.0"))
            self.unsharp = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_UNSHARP", "0.8"))
            self.depth_scale = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_DEPTH_SCALE", "1.0"))
            self.seg_scale = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_SEG_SCALE", "0.8"))
            self.refine_steps = int(os.environ.get("STREAMLIB_DOOM_DIFFUSION_REFINE_STEPS", "3"))
            self.lift = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_LIFT", "0.7"))
            self.keyframe_steps = int(os.environ.get("STREAMLIB_DOOM_DIFFUSION_KEY_STEPS", "4"))
            self.keyframe_strength = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_KEY_STRENGTH", "0.85"))
            self.refine_strength = float(os.environ.get("STREAMLIB_DOOM_DIFFUSION_REFINE_STRENGTH", "0.85"))
            self.strength = self.refine_strength
            self.steps = self.refine_steps
            assert int(self.steps * self.strength) >= 1, "refine steps x strength must round to at least one denoising step"
            if COMPILE:
                import torch._inductor.config as inductor_config
                inductor_config.triton.cudagraph_trees_generation_cloning = "user_visible"
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
                self.pipe.controlnet.nets = torch.nn.ModuleList([torch.compile(net, mode="reduce-overhead", fullgraph=True) for net in self.pipe.controlnet.nets])
                self.pipe.vae.decoder = torch.compile(self.pipe.vae.decoder, mode="reduce-overhead", fullgraph=True)
                self.compiled = True
            # Warm the pipeline so the first live frame is not the slow one; compiling happens here too.
            started = time.monotonic()
            blank = torch.zeros((1, 3, NEURAL_H, NEURAL_W), dtype=torch.float16, device="cuda")
            for _ in range(3 if COMPILE else 1):
                self._run(blank, [blank, blank], STYLES[DEFAULT_STYLE])
            self._run(blank, [blank, blank], STYLES[DEFAULT_STYLE], steps=self.keyframe_steps, strength=self.keyframe_strength)
            log.info(f"MARKER:DIFFUSION_READY compiled={self.compiled} warmup_s={time.monotonic() - started:.0f}")
            self.ready = True

    def _prompt_embeds(self, prompt: str):
        """The text encoder costs 7 ms a frame and the prompt only changes when Claude does."""
        if prompt not in self.embeds:
            with self.torch.inference_mode():
                positive, negative = self.pipe.encode_prompt(prompt, "cuda", 1, GUIDANCE > 1.0, negative_prompt=NEGATIVE_PROMPT if GUIDANCE > 1.0 else None)
                self.embeds[prompt] = (positive, negative)
            if len(self.embeds) > 24:
                self.embeds.pop(next(iter(self.embeds)))
        return self.embeds[prompt]

    def _run(self, init, control, prompt: str, steps: int | None = None, strength: float | None = None):
        positive, negative = self._prompt_embeds(prompt)
        if self.compiled:
            self.torch.compiler.cudagraph_mark_step_begin()
        with self.torch.inference_mode():
            out = self.pipe(prompt_embeds=positive, negative_prompt_embeds=negative, image=init, control_image=control,
                            num_inference_steps=steps or self.steps, strength=strength or self.strength,
                            guidance_scale=GUIDANCE, controlnet_conditioning_scale=[self.depth_scale, self.seg_scale],
                            generator=self.generator, output_type="pt").images[0]
        return out.clone() if self.compiled else out

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if not self.ready or view is None or not _due(self, int(1e9 / DIFFUSION_FPS)):
            return
        torch = self.torch
        style = str(((view.get("state") or {}).get("style")) or DEFAULT_STYLE)
        prompt = STYLES.get(style, style)
        if not prompt:
            return
        started = time.monotonic()
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as handle:
            tensor, self.path = _view_tensor(handle, torch)
        rgb = self._rgb(tensor, 0).permute(2, 0, 1)[None].half() / 255.0  # (1,3,H,W), base palette: the flash is a HUD effect
        rgb = rgb.float().pow(self.lift).half()  # a toy photograph is lit; Doom's corridors are not
        game = torch.nn.functional.interpolate(rgb, size=(NEURAL_H, NEURAL_W), mode="bilinear", align_corners=False)
        codes = tensor[:, :, 1].float()[None, None]
        near = 1.0 - codes.half() / 255.0  # depth code: 0 near .. 255 far/sky
        control = torch.nn.functional.interpolate(near, size=(NEURAL_H, NEURAL_W), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        segment = self._class_lut[tensor[:, :, 2].long()].permute(2, 0, 1)[None].half() / 255.0
        segment = torch.nn.functional.interpolate(segment, size=(NEURAL_H, NEURAL_W), mode="nearest")
        control = [control, segment]
        pose = view.get("pose")
        keyframe = self.carry <= 0.0 or self.previous is None or style != self.previous_style or pose is None or self.previous_pose is None
        if not keyframe:
            depth_z = torch.nn.functional.interpolate(codes, size=(NEURAL_H, NEURAL_W), mode="nearest")[0, 0]
            depth_z = 4.0 * torch.pow(2.0, depth_z / 255.0 * 8.0)
            carried, valid = reproject(self.previous, self.previous_pose, pose, depth_z, torch)
            blurred = torch.nn.functional.avg_pool2d(carried, 3, stride=1, padding=1)
            carried = (carried + self.unsharp * (carried - blurred)).clamp(0, 1)  # resampling softens; give the edge back
            weight = valid * self.carry
            init = (carried.half() * weight.half() + game * (1.0 - weight.half())).clamp(0, 1)
            self.generator.manual_seed(7)
            out = self._run(init, control, prompt)
        else:
            self.generator.manual_seed(7)
            out = self._run(game, control, prompt, steps=self.refine_steps, strength=self.refine_strength)
        self.previous = out.detach()[None] if out.ndim == 3 else out.detach()
        self.previous_pose, self.previous_style = pose, style
        image = (out.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        self.ms = (time.monotonic() - started) * 1000
        self.rate = 0.9 * self.rate + 0.1 * (1000.0 / max(self.ms, 1.0))
        self._publish(ctx, image, "neural_to_downstream", {"style": style, "prompt": prompt[:80], "model": "sd-turbo + controlnet depth/class, cfg", "tick": view.get("tick"),
                                                            "fps": round(min(self.rate, DIFFUSION_FPS), 1), "pose": pose, "keyframe": keyframe})
        self.frames += 1
        if self.frames in (1, 30, 300):
            log.info(f"MARKER:DIFFUSION_FRAME frames={self.frames} ms={self.ms:.0f} path={self.path} style={style}")


def _colorize_codes(codes: numpy.ndarray, lut: numpy.ndarray) -> numpy.ndarray:
    return lut[numpy.clip(codes, 0, 255).astype(numpy.uint8)][:, :, :3]


@processor(description="Monocular depth network (Depth Anything V2 small) on the rendered RGB, aligned to the renderer's true depth in log space and graded live: publishes one 960x200 strip — true depth, neural depth, error heat — and the abs-rel error. Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class NeuralDepth(_NeuralBase):
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def depth_trio_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_common(ctx, "neural_depth")
        torch = self.torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(HF_MODELS["depth"])
        self.model = AutoModelForDepthEstimation.from_pretrained(HF_MODELS["depth"], torch_dtype=torch.float16).cuda().eval()
        self.lut = depth_lut()
        self.error_history: list[float] = []
        from .showcase import font
        self.font = font
        log.info("MARKER:NEURAL_DEPTH_READY")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None or not _due(self, int(1e9 / DEPTH_FPS)):
            return
        torch = self.torch
        started = time.monotonic()
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as handle:
            tensor, self.path = _view_tensor(handle, torch)
        rgb = self._rgb(tensor, view.get("palette", 0)).float().permute(2, 0, 1)[None] / 255.0
        codes = tensor[:, :, 1].cpu().numpy()
        classes = tensor[:, :, 2].cpu().numpy()
        with torch.inference_mode():
            pixel = torch.nn.functional.interpolate(rgb, size=(266, 434), mode="bilinear", align_corners=False)  # multiples of 14, 1.6 aspect
            mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
            predicted = self.model(pixel_values=((pixel - mean) / std).half()).predicted_depth  # (1, h, w) relative inverse depth
            predicted = torch.nn.functional.interpolate(predicted[None].float(), size=(VIEW_H, VIEW_W), mode="bilinear", align_corners=False)[0, 0]
        inverse = predicted.clamp(min=1e-3).cpu().numpy()
        true_dist = decode_depth(codes)
        valid = (classes != 0) & (codes < 250)
        # Relative inverse depth to metric-ish: fit log(dist) = a * log(1/inverse) + b on the valid pixels.
        x = numpy.log(1.0 / inverse[valid]); y = numpy.log(true_dist[valid])
        if len(x) > 100:
            a, b = numpy.polyfit(x, y, 1)
        else:
            a, b = 1.0, 0.0
        pred_dist = numpy.exp(a * numpy.log(1.0 / inverse) + b)
        abs_rel = float(numpy.mean(numpy.abs(pred_dist[valid] - true_dist[valid]) / true_dist[valid])) if valid.any() else 0.0
        self.error_history = (self.error_history + [abs_rel])[-60:]
        pred_codes = numpy.clip(numpy.log2(numpy.maximum(pred_dist, 4.0) / 4.0) / 8.0 * 255.0, 0, 255)
        error = numpy.abs(numpy.log(numpy.maximum(pred_dist, 1.0)) - numpy.log(numpy.maximum(true_dist, 1.0)))
        heat = numpy.clip(error / 0.7, 0.0, 1.0)
        error_rgb = numpy.stack([heat * 255, (heat ** 2) * 200, (1.0 - heat) * 40], axis=2).astype(numpy.uint8)
        error_rgb[~valid] = (12, 12, 30)
        strip = numpy.zeros((VIEW_H, VIEW_W * 3, 4), dtype=numpy.uint8)
        strip[:, :VIEW_W, :3] = _colorize_codes(codes, self.lut)
        strip[:, VIEW_W : 2 * VIEW_W, :3] = _colorize_codes(pred_codes, self.lut)
        strip[:, 2 * VIEW_W :, :3] = error_rgb
        strip[:, :, 3] = 255
        mean_err = sum(self.error_history) / len(self.error_history)
        self.font(14, True).draw(strip, f"abs rel {abs_rel * 100:4.1f}%  (60-frame mean {mean_err * 100:4.1f}%)", 2 * VIEW_W + 6, 4, (255, 255, 255))
        self.font(14, True).draw(strip, "TRUE · renderer", 6, 4, (255, 255, 255))
        self.font(14, True).draw(strip, "NEURAL · Depth Anything V2", VIEW_W + 6, 4, (255, 255, 255))
        self.ms = (time.monotonic() - started) * 1000
        self._publish(ctx, strip[:, :, :3], "depth_trio_to_downstream", {"abs_rel": round(abs_rel, 4), "abs_rel_mean": round(mean_err, 4), "model": "depth-anything-v2-small", "tick": view.get("tick")})
        self.frames += 1
        if self.frames in (1, 15, 150):
            log.info(f"MARKER:NEURAL_DEPTH_FRAME frames={self.frames} ms={self.ms:.0f} abs_rel={abs_rel:.3f} path={self.path}")


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@processor(description="Open-vocabulary detector (Grounding DINO) asked for monsters on the rendered RGB, graded live against the renderer's own labels: green boxes are the detector's, red are labelled monsters it missed. Publishes detections in the planner's schema, so the robot can fight from a learned detector instead of the oracle. Connect the renderer's `view_to_downstream` to `view_from_upstream`.")
class MonsterDetector(_NeuralBase):
    @input(delivery_profile="newest")
    def view_from_upstream(self) -> None: ...

    @output()
    def detector_pane_to_downstream(self) -> None: ...

    @output()
    def detections_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_common(ctx, "detector")
        torch = self.torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.processor = AutoProcessor.from_pretrained(HF_MODELS["detector"])
        self.model = GroundingDinoForObjectDetection.from_pretrained(HF_MODELS["detector"], torch_dtype=torch.float16).cuda().eval()
        # Decoys in the prompt absorb what the level is full of; only the monster labels count.
        self.text = os.environ.get("STREAMLIB_DOOM_DETECTOR_PROMPT", "a monster. a brown demon. a zombie soldier. a barrel. a lamp. a pillar. a door.")
        self.keep = ("monster", "demon", "soldier", "zombie")
        self.hits = 0
        self.labelled = 0
        self.false_alarms = 0
        from .showcase import font
        self.font = font
        log.info("MARKER:DETECTOR_READY")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        view = ctx.inputs.read("view_from_upstream")
        if view is None or not _due(self, int(1e9 / DETECTOR_FPS)):
            return
        torch = self.torch
        started = time.monotonic()
        with ctx.gpu_limited_access.resolve_surface(view["surface_id"]) as handle:
            tensor, self.path = _view_tensor(handle, torch)
        rgb = self._rgb(tensor, view.get("palette", 0))
        rgb_np = rgb.cpu().numpy()
        codes = tensor[:, :, 1].cpu().numpy()
        classes = tensor[:, :, 2].cpu().numpy()
        from PIL import Image
        image = Image.fromarray(rgb_np).resize((VIEW_W * 2, VIEW_H * 2), Image.NEAREST)
        with torch.inference_mode():
            inputs = self.processor(images=image, text=self.text, return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].half()
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.36, text_threshold=0.3, target_sizes=[(VIEW_H * 2, VIEW_W * 2)])[0]
        labels = results.get("text_labels") or results.get("labels") or [""] * len(results["boxes"])
        boxes, scores = [], []
        for box, score, label in zip(results["boxes"].tolist(), results["scores"].tolist(), labels):
            x0, y0, x1, y1 = [float(v) / 2.0 for v in box]
            if not any(word in str(label) for word in self.keep) or (x1 - x0) * (y1 - y0) > 0.35 * VIEW_W * VIEW_H:
                continue
            boxes.append([x0, y0, x1, y1])
            scores.append(float(score))
        oracle = [[d["column"] - d["width_px"] / 2, d["top"], d["column"] + d["width_px"] / 2, d["bottom"]] for d in detect_monsters(classes, codes) if d["width_px"] >= 6]
        matched = set()
        for o in oracle:
            self.labelled += 1
            hit = next((i for i, b in enumerate(boxes) if i not in matched and _iou(o, b) >= 0.3), None)
            if hit is not None:
                matched.add(hit)
                self.hits += 1
        self.false_alarms += len(boxes) - len(matched)
        pane = numpy.zeros((VIEW_H, VIEW_W, 4), dtype=numpy.uint8)
        pane[:, :, :3] = rgb_np
        pane[:, :, 3] = 255
        from .sensors import _polyline
        for o in oracle:
            x0, y0, x1, y1 = [int(v) for v in o]
            _polyline(pane, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], (255, 60, 60), 1)
        detections = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = [int(v) for v in box]
            _polyline(pane, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], (80, 255, 120), 2)
            center = (box[0] + box[2]) / 2.0
            region = codes[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            distance = float(numpy.median(decode_depth(region))) if region.size else 0.0
            detections.append({"bearing_degrees": -math.degrees(math.atan((center - 160.0) / 160.0)), "range": round(distance, 1), "width_px": int(box[2] - box[0]),
                               "column": int(center), "top": y0, "bottom": y1, "score": round(score, 2)})
        rate = 100.0 * self.hits / max(1, self.labelled)
        self.ms = (time.monotonic() - started) * 1000
        self.font(14, True).draw(pane, f"GROUNDING DINO · hit {rate:.0f}% of {self.labelled} · fp {self.false_alarms} · {self.ms:.0f} ms", 6, 4, (255, 255, 255))
        self._publish(ctx, pane[:, :, :3], "detector_pane_to_downstream", {"hit_rate": round(rate, 1), "labelled": self.labelled, "false_alarms": self.false_alarms, "model": "grounding-dino-base", "tick": view.get("tick")})
        ctx.outputs.write("detections_to_downstream", {"tick": view.get("tick"), "detections": detections, "pid": os.getpid(), "source_surface_id": view["surface_id"],
                                                       "timestamp_ns": clock.monotonic_now_ns(), "inference_ms": round(self.ms, 1), "model": "grounding-dino-base"})
        self.frames += 1
        if self.frames in (1, 10, 100):
            log.info(f"MARKER:DETECTOR_FRAME frames={self.frames} ms={self.ms:.0f} boxes={len(boxes)} oracle={len(oracle)} hit_rate={rate:.0f} path={self.path}")


REPAINT_TARGETS = {"walls": ["STARTAN3", "BROWNGRN", "BROWN1", "STARG3", "STARGR1", "SUPPORT2", "BROWN144", "STEP6"], "flats": ["FLOOR4_8", "CEIL3_5", "FLOOR5_2", "CEIL5_2", "FLOOR5_1", "FLAT20"]}


@processor(description="Repainter: generates new wall and floor textures with the diffusion model for a style named by the game's `repaint` director verb, quantizes them to Doom's palette, and publishes atlas patches the renderer writes into its running texture atlas. Connect the game's `world_to_downstream` to `world_from_upstream` and `patches_to_downstream` to the renderer's `atlas_patch_from_upstream`.")
class Repainter(_NeuralBase):
    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @output()
    def patches_to_downstream(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        self._setup_common(ctx, "repainter")
        torch = self.torch
        from diffusers import AutoencoderTiny, AutoPipelineForText2Image
        self.pipe = AutoPipelineForText2Image.from_pretrained(HF_MODELS["diffusion"], torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipe.vae = AutoencoderTiny.from_pretrained(HF_MODELS["vae"], torch_dtype=torch.float16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        wad = Wad()
        self.palette = numpy.asarray(wad.palettes()[0], dtype=numpy.int32)  # (256, 3)
        self.sizes = {}
        for name in REPAINT_TARGETS["walls"]:
            try:
                patch = wad.texture(name)
                self.sizes["T:" + name] = (patch.width, patch.height)
            except Exception:
                continue
        for name in REPAINT_TARGETS["flats"]:
            self.sizes["F:" + name] = (64, 64)
        self.done_key = None
        log.info(f"MARKER:REPAINTER_READY targets={len(self.sizes)}")

    def _quantize(self, rgb: numpy.ndarray) -> numpy.ndarray:
        """Doom's palette is 256 colours in a few ramps; a pastel texture would collapse to one of
        them. Stretch the contrast first, then Floyd–Steinberg dither so the detail survives."""
        pixels = rgb.astype(numpy.float32)
        low, high = numpy.percentile(pixels, 2), numpy.percentile(pixels, 98)
        pixels = numpy.clip((pixels - low) / max(high - low, 1.0) * 255.0, 0, 255)
        h, w = pixels.shape[:2]
        palette = self.palette.astype(numpy.float32)
        out = numpy.zeros((h, w), dtype=numpy.uint8)
        for y in range(h):
            for x in range(w):
                wanted = pixels[y, x]
                index = int(((palette - wanted) ** 2).sum(axis=1).argmin())
                out[y, x] = index
                error = wanted - palette[index]
                if x + 1 < w:
                    pixels[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        pixels[y + 1, x - 1] += error * 3 / 16
                    pixels[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        pixels[y + 1, x + 1] += error * 1 / 16
        return out

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        world = ctx.inputs.read("world_from_upstream")
        if world is None:
            return
        key = str(((world.get("state") or {}).get("repaint")) or "")
        if not key or key == self.done_key:
            return
        self.done_key = key
        style = key.split("#", 1)[0]
        started = time.monotonic()
        torch = self.torch
        patches = []
        generator = torch.Generator("cuda").manual_seed(11)
        for atlas_key, (w, h) in self.sizes.items():
            kind = "wall" if atlas_key.startswith("T:") else "floor"
            prompt = f"seamless tileable {style} {kind} texture, video game texture, high contrast, sharp detail, flat even lighting, top-down, no shadows"
            with torch.inference_mode():
                image = self.pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, height=512, width=512, generator=generator, output_type="pt").images[0]
            resized = torch.nn.functional.interpolate(image[None].float(), size=(h, w), mode="area")[0]
            rgb = (resized.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            patches.append({"key": atlas_key, "w": w, "h": h, "indices": self._quantize(rgb).tobytes()})
        self.ms = (time.monotonic() - started) * 1000
        ctx.outputs.write("patches_to_downstream", {"style": style, "patches": patches, "pid": os.getpid(), "ms": round(self.ms), "timestamp_ns": clock.monotonic_now_ns()})
        self.frames += 1
        log.info(f"MARKER:REPAINTER_PATCHES style={style} patches={len(patches)} ms={self.ms:.0f}")
