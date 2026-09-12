#!/usr/bin/env python3
"""The neural reel: the game re-rendered by a diffusion model, a depth network and a detector
graded against the renderer's own truth, the level repainted, with Claude choosing the looks.
Run scripts/neural-setup.py first (or let this script do it) so the models are loaded."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("reel", os.path.join(HERE, "demo-reel.py"))
reel = importlib.util.module_from_spec(spec); spec.loader.exec_module(reel)
caption, director, state, wait_for, graph, tool, node_id, link_id, connect = reel.caption, reel.director, reel.state, reel.wait_for, reel.graph, reel.tool, reel.node_id, reel.link_id, reel.connect


def claude(prompt: str) -> subprocess.Popen:
    return subprocess.Popen([os.path.join(reel.REPO, "director.sh"), prompt], cwd=reel.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def claude_node_name(before_ids: set) -> str:
    """The display name of the newest node Claude added, minus its prefix."""
    names = [n["display_name"] for n in graph()["nodes"] if n["id"] not in before_ids and str(n["display_name"]).startswith("Claude")]
    return names[-1].split(":", 1)[-1].strip() if names else ""


def main() -> int:
    caption("", "")
    director(command="mission", goal="patrol"); director(command="style", style="lego"); director(command="give", item="everything")
    time.sleep(1.0)
    caption("DOOM, REBUILT IN LEGO — WHILE YOU PLAY", "a diffusion model in its own process re-imagines every frame · 1993 on the left, bricks on the right")
    time.sleep(8)
    caption("ONE FRAME · FOUR NEURAL NETS · FOUR PROCESSES · ZERO COPIES", "the renderer publishes one GPU surface; diffusion, depth, detection and generation each read it in place")
    time.sleep(7)
    caption("THE BRICKS STAY PUT", "each frame starts from the last, reprojected through the game's own depth — the model refines, it doesn't reinvent")
    time.sleep(7)
    caption("12 HZ MODEL, 60 FPS PICTURE", "the console reprojects the newest neural frame to the current camera every frame, with the depth the renderer wrote anyway")
    time.sleep(7)

    # Claude thinks about a material while the perception beats play.
    ids_before = {n["id"] for n in graph()["nodes"]}
    repaint_before = state().get("repaint")
    proc = claude("Pick one material or look for the level's walls and floors — something surprising and vivid — and repaint the level with it: add_processor a DirectorCommand named "
                  "'Claude: <material>' with command repaint and the material in the style field. After the tool call, end with two short plain-text sentences: what the level looked like, and what you turned it into.")
    caption("NEURAL DEPTH, GRADED AGAINST THE TRUTH — LIVE", "Depth Anything V2 on the RGB · the renderer knows the real depth · abs-rel error on screen every frame")
    time.sleep(8)
    caption("A DETECTOR ASKED FOR MONSTERS, GRADED BY THE RENDERER", "Grounding DINO · green: what it found · red: labelled monsters it missed · hit rate live")
    director(command="spawn", kind="zombieman", count=2, where="ahead")
    time.sleep(8)
    try:
        for target, port in (("Planner", "detections_from_upstream"), ("Telemetry", "detections_from_upstream")):
            cut = link_id("Perceive", target, port)
            if cut:
                tool("disconnect", link_id=cut)
            connect("Detector", "detections_to_downstream", target, port)
        caption("SWAPPED LIVE: THE ROBOT NOW FIGHTS FROM THE LEARNED DETECTOR", "one link cut, one link made, no restart · the planner never noticed")
    except Exception as failure:
        print("swap skipped:", failure, flush=True)
    time.sleep(7)
    director(command="heal")
    caption("CLAUDE IS REPAINTING THE LEVEL", "claude -p on the desktop, MCP into this node · it picks a material, the model generates it, the renderer's atlas takes it live")
    if wait_for(lambda: state().get("repaint") != repaint_before, 60):
        wait_for(lambda: False, 4)
        name = claude_node_name(ids_before) or str(state().get("repaint")).split("#")[0]
        caption(f"REPAINTED BY CLAUDE: {name.upper()[:30]}", "generated, dithered into the 1993 palette, written into the running atlas · the lego view follows")
        time.sleep(8)
    # Claude's written reply lands when its run ends; give it time to reach the caption panel.
    if wait_for(lambda: proc.poll() is not None, 30):
        time.sleep(5)

    caption("ONE GPU · FOUR NEURAL NETS · A 60 FPS GAME · ONE GRAPH", "every model its own process on the same frame · the runtime recorded this picture of itself")
    time.sleep(5)
    caption("github.com/tatolab/streamlib-doom", "StreamLib · every box its own process · every link a shared GPU frame")
    time.sleep(4)
    caption("", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
