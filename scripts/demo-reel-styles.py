#!/usr/bin/env python3
"""The styles reel: DOOM in LEGO, then in neon, then handed to Claude.

Run scripts/neural-setup.py first so the models are loaded and proven; this only tells the
running graph what to look like and lets the director take a turn.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("reel", os.path.join(HERE, "demo-reel.py"))
reel = importlib.util.module_from_spec(spec); spec.loader.exec_module(reel)
caption, director, state, wait_for, graph = reel.caption, reel.director, reel.state, reel.wait_for, reel.graph


def claude(prompt: str) -> subprocess.Popen:
    return subprocess.Popen([os.path.join(reel.REPO, "director.sh"), prompt], cwd=reel.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def claude_node(before_ids: set) -> str:
    """The first node Claude added, not the last — it adds the style, then the monsters, and it is
    the style the caption is about."""
    names = [n["display_name"] for n in graph()["nodes"] if n["id"] not in before_ids and str(n["display_name"]).startswith("Claude")]
    return names[0].split(":", 1)[-1].strip() if names else ""


def main() -> int:
    caption("", "")
    director(command="control", mode="auto"); director(command="mission", goal="patrol")
    director(command="give", item="everything"); director(command="style", style="lego")
    time.sleep(6)

    caption("DOOM, REBUILT IN LEGO — WHILE IT PLAYS", "a diffusion model in its own process re-imagines every frame · 1993 on the left, bricks on the right")
    time.sleep(9)
    caption("EVERY GLOWING BORDER IS A LIVE CHANNEL", "each pane lights when a frame lands in it — fast ones hold, slow ones pulse, one runtime feeding all of them")
    time.sleep(9)

    # Claude starts thinking here so its answer lands as the reel reaches it, not thirty seconds later.
    ids_before = {n["id"] for n in graph()["nodes"]}
    style_before = state().get("style")
    monsters_before = state().get("monsters_alive", 0)
    proc = claude("Look at the game, then pick one striking visual style of your own for the diffusion re-render — not lego and not cyberpunk — "
                  "and apply it with add_processor: a DirectorCommand named 'Claude: <two-word name>' with command style and your prompt in the style field. "
                  "Then spawn three imps ahead of the robot with a second DirectorCommand. Reply in two short sentences: what you saw, and what you chose.")
    caption("GEOMETRY LOCKED BY THE GAME'S OWN DEPTH", "the renderer writes depth into every frame anyway; the model takes it as conditioning, free")
    time.sleep(9)

    caption("ONE COMMAND CHANGES THE WORLD", "the same graph, the same frames — only the prompt moved")
    time.sleep(4)
    director(command="style", style="cyberpunk")
    time.sleep(3)
    caption("…AND NOW IT IS BLADE RUNNER", "neon, chrome and wet concrete on the same corridors · nothing restarted, nothing re-wired")
    time.sleep(11)
    caption("THE SENSORS NEVER NOTICED", "depth, detection and the map read the game's frame, not the dream · they are unchanged")
    time.sleep(9)

    caption("NOW CLAUDE PICKS ONE", 'claude -p on the desktop · MCP into this node · ./director.sh "pick a look"')
    if wait_for(lambda: state().get("style") not in (style_before, "cyberpunk"), 60):
        time.sleep(2)
        caption(f"CLAUDE: {(claude_node(ids_before) or 'a look of its own').upper()[:34]}", "it chose the words, added the processor, and the next frame wore them")
        time.sleep(10)
    if wait_for(lambda: state().get("monsters_alive", 0) > monsters_before, 40):
        caption("AND SENT SOMETHING TO MEET IT", "a second processor · the robot fights from what its own camera sees, in whatever the world now looks like")
        time.sleep(10)
    wait_for(lambda: proc.poll() is not None, 20)
    time.sleep(3)

    caption("ONE GPU · ONE GRAPH · EVERY BOX ITS OWN PROCESS", "a 35 Hz game, a 60 fps picture, four neural nets and an agent — the runtime recorded this of itself")
    time.sleep(8)
    caption("github.com/tatolab/streamlib-doom", "StreamLib")
    time.sleep(6)
    caption("", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
