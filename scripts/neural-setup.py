#!/usr/bin/env python3
"""Adds the neural nodes and the lidar map to a running console node over MCP, then proves each
one is flowing before it returns — the same calls an agent makes, run ahead of a recording.

A link wired into a helper that is still loading its model can fail to open its port; the
engine reports the helper as running before its setup has finished, so state is no signal.
`tap` on the node's output is: no bags after the model has had time to load means the input
link never opened, and re-wiring it is the repair.
"""
import importlib.util, json, os, sys, time, urllib.request
spec = importlib.util.spec_from_file_location("reel", os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-reel.py"))
reel = importlib.util.module_from_spec(spec); spec.loader.exec_module(reel)

# Which re-render fills the neural pane. `video` is the causal video model (StreamDiffusionV2);
# `sdturbo` is the per-frame image model with the reprojected feedback loop.
RERENDER = os.environ.get("STREAMLIB_DOOM_RERENDER", "off")
RERENDER_TYPES = {"sdturbo": "streamlib_doom.neural:DiffusionRerender",
                  "video": "streamlib_doom.streamdiffusion:VideoDiffusionRerender"}
if RERENDER not in ({"off"} | set(RERENDER_TYPES)):
    raise SystemExit(f"STREAMLIB_DOOM_RERENDER must be off, {' or '.join(sorted(RERENDER_TYPES))}, not {RERENDER!r}")

# The re-render is off by default. Stable diffusion has very little to work with at 320x200 and
# the result reads as mush; set STREAMLIB_DOOM_RERENDER=sdturbo or =video to put it back in.
RERENDER_ENABLED = RERENDER in ("sdturbo", "video")

NODES = [  # name, type, (from, from_port) input, (to, to_port) output
    ("Lidar", "streamlib_doom.sensors:LidarScanner", ("Game", "world_to_downstream", "world_from_upstream"), ("scan_to_downstream", None, None)),
    ("Map", "streamlib_doom.sensors:OccupancyMapper", ("Lidar", "scan_to_downstream", "scan_from_upstream"), ("map_to_downstream", "Console", "map_from_upstream")),
    # The finished frame, not the bare view: it carries the weapon and the status bar, and the HUD
    # compositor marks those pixels with their own class so the re-render leaves them alone.
    ("Substitute", "streamlib_doom.models:MonsterModelSubstitution", ("HUD", "frame_to_downstream", "view_from_upstream"), ("substitute_to_downstream", "Console", "neural_from_upstream")),
    ("DepthNet", "streamlib_doom.neural:NeuralDepth", ("Render", "view_to_downstream", "view_from_upstream"), ("depth_trio_to_downstream", "Console", "depth_trio_from_upstream")),
    ("Detector", "streamlib_doom.neural:MonsterDetector", ("Render", "view_to_downstream", "view_from_upstream"), ("detector_pane_to_downstream", "Console", "detector_from_upstream")),
    ("Transcript", "streamlib_doom.transcript:LiveEventTranscript", ("Game", "world_to_downstream", "world_from_upstream"), ("transcript_to_downstream", None, None)),
    ("Repainter", "streamlib_doom.neural:Repainter", ("Game", "world_to_downstream", "world_from_upstream"), ("patches_to_downstream", "Render", "atlas_patch_from_upstream")),
]


if RERENDER_ENABLED:
    NODES.append(("Diffusion", RERENDER_TYPES[RERENDER], ("HUD", "frame_to_downstream", "view_from_upstream"),
                  ("neural_to_downstream", "Console", "neural_from_upstream")))
EXTRA_LINKS = [("Game", "world_to_downstream", "Substitute", "world_from_upstream"),
               ("Perceive", "detections_to_downstream", "Transcript", "detections_from_upstream")]


def has(name):
    return any(n["display_name"] == name for n in reel.graph()["nodes"])


def flowing(name, port) -> bool:
    out = reel.tool("tap", channel=f"{reel.node_id(name).lower()}/{port}", count=1)
    return int(out.get("received", 0)) >= 1 if isinstance(out, dict) else False


def wire(name, source, sink):
    src, src_port, in_port = source
    if reel.link_id(src, name, in_port) is None:
        reel.connect(src, src_port, name, in_port)
    out_port, to, to_port = sink
    if to and reel.link_id(name, to, to_port) is None:
        reel.connect(name, out_port, to, to_port)
    for a, a_port, b, b_port in EXTRA_LINKS:
        if b == name and reel.link_id(a, b, b_port) is None:
            reel.connect(a, a_port, b, b_port)


PANES_URL = os.environ.get("STREAMLIB_DOOM_PANES_URL", "http://127.0.0.1:8669/panes")
CONSOLE_PANE = {"Map": "map", "Substitute": "neural", "Diffusion": "neural", "DepthNet": "depth_trio", "Detector": "detector"}


def pane_age(name) -> float:
    try:
        with urllib.request.urlopen(PANES_URL, timeout=3) as reply:
            return float(json.load(reply)["ages_s"].get(CONSOLE_PANE[name], 1e9))
    except Exception:
        return 1e9


def rewire(a, a_port, b, b_port) -> None:
    cut = reel.link_id(a, b, b_port)
    if cut:
        reel.tool("disconnect", link_id=cut)
        time.sleep(0.5)
    reel.connect(a, a_port, b, b_port)


def main() -> None:
    t0 = time.monotonic()
    for name, type_path, source, sink in NODES:
        if not has(name):
            reel.add(type_path, name)
            time.sleep(0.3)
        wire(name, source, sink)
    print(f"nodes added and wired in {time.monotonic() - t0:.0f}s; proving flow…", flush=True)

    for name, _type, source, sink in NODES:
        out_port, to, to_port = sink
        if name == "Repainter":  # publishes only on a repaint; nothing to prove yet
            continue
        # 1. The node publishes: else its input link never opened — re-wire it.
        started = time.monotonic(); rewired = 0
        while not flowing(name, out_port):
            waited = time.monotonic() - started
            if waited > 240:
                print(f"{name}: NOT publishing after {waited:.0f}s", flush=True); break
            if waited > 45 * (rewired + 1) and rewired < 4:
                rewire(source[0], source[1], name, source[2]); rewired += 1
                print(f"{name}: silent at {waited:.0f}s, re-wired its input ({rewired})", flush=True)
            time.sleep(1.5)
        else:
            print(f"{name}: publishing at {time.monotonic() - t0:.0f}s", flush=True)
        # 2. The console receives it: else the output link never opened — re-wire that.
        if to == "Console":
            started = time.monotonic(); rewired = 0
            while pane_age(name) > 3.0:
                waited = time.monotonic() - started
                if waited > 120:
                    print(f"{name}: console never received its pane", flush=True); break
                if waited > 12 * (rewired + 1) and rewired < 5:
                    rewire(name, out_port, to, to_port); rewired += 1
                    print(f"{name}: pane missing at {waited:.0f}s, re-wired its output ({rewired})", flush=True)
                time.sleep(1.0)
            else:
                print(f"{name}: on the console at {time.monotonic() - t0:.0f}s", flush=True)
    print(f"neural graph live in {time.monotonic() - t0:.0f}s", flush=True)




def watch() -> None:
    """Keep them there. They are dynamic processors, so a node restart leaves them behind;
    this notices the graph came back without them and adds them again."""
    while True:
        time.sleep(10)
        try:
            names = {n["display_name"] for n in reel.graph()["nodes"]}
        except Exception:
            continue  # the node is down or restarting; try again shortly
        missing = [name for name, *_ in NODES if name not in names]
        if not missing:
            continue
        print(f"missing {', '.join(missing)} — the node restarted; adding them back", flush=True)
        try:
            main()
        except Exception as failure:
            print(f"could not restore: {failure}", flush=True)


main()
if "--watch" in sys.argv:
    watch()
