#!/usr/bin/env python3
"""Adds the neural nodes and the lidar map to a running console node over MCP and waits
for their models to load — the same calls an agent makes, run ahead of a recording."""
import importlib.util, sys, time, json, urllib.request, os
spec = importlib.util.spec_from_file_location("reel", os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-reel.py"))
reel = importlib.util.module_from_spec(spec); spec.loader.exec_module(reel)

def has(name):
    return any(n["display_name"] == name for n in reel.graph()["nodes"])

def running(name):
    for n in reel.graph()["nodes"]:
        if n["display_name"] == name:
            return str((n.get("components") or {}).get("state", "")).lower() == "running"
    return False

t0 = time.monotonic()
# Add every node first, let each finish its setup (the models load there), then wire — a link
# wired into a helper still loading is a link that can fail to open its port.
wanted = [("Lidar", "streamlib_doom.sensors:LidarScanner"), ("Map", "streamlib_doom.sensors:OccupancyMapper"), ("Diffusion", "streamlib_doom.neural:DiffusionRerender"),
          ("DepthNet", "streamlib_doom.neural:NeuralDepth"), ("Detector", "streamlib_doom.neural:MonsterDetector"), ("Repainter", "streamlib_doom.neural:Repainter")]
for name, type_path in wanted:
    if not has(name):
        reel.add(type_path, name)
for name, _ in wanted:
    deadline = time.monotonic() + 180
    while not running(name) and time.monotonic() < deadline:
        time.sleep(1.0)
    print(f"{name}: {'running' if running(name) else 'NOT running'} at {time.monotonic()-t0:.0f}s", flush=True)
time.sleep(2.0)
links = [("Game", "world_to_downstream", "Lidar", "world_from_upstream"), ("Lidar", "scan_to_downstream", "Map", "scan_from_upstream"), ("Map", "map_to_downstream", "Console", "map_from_upstream"),
         ("Render", "view_to_downstream", "Diffusion", "view_from_upstream"), ("Diffusion", "neural_to_downstream", "Console", "neural_from_upstream"),
         ("Render", "view_to_downstream", "DepthNet", "view_from_upstream"), ("DepthNet", "depth_trio_to_downstream", "Console", "depth_trio_from_upstream"),
         ("Render", "view_to_downstream", "Detector", "view_from_upstream"), ("Detector", "detector_pane_to_downstream", "Console", "detector_from_upstream"),
         ("Game", "world_to_downstream", "Repainter", "world_from_upstream"), ("Repainter", "patches_to_downstream", "Render", "atlas_patch_from_upstream")]
for a, ap, b, bp in links:
    if reel.link_id(a, b, bp) is None:
        reel.connect(a, ap, b, bp)
        time.sleep(0.6)
print(f"neural graph wired in {time.monotonic()-t0:.0f}s", flush=True)
sys.exit(0)
