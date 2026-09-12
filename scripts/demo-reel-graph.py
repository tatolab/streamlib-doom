#!/usr/bin/env python3
"""The graph reel: the picture on the right built up one processor at a time, live, and the
events the graph wrote about it, read by an agent that never looked at a frame.

Run scripts/neural-setup.py first so the sensors, the substitution and the transcript are in;
this only adds and removes treatments over MCP, spawns things, and sets captions.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("reel", os.path.join(HERE, "demo-reel.py"))
reel = importlib.util.module_from_spec(spec); spec.loader.exec_module(reel)
caption, director, state, wait_for, graph = reel.caption, reel.director, reel.state, reel.wait_for, reel.graph

TRANSCRIPT = os.environ.get("STREAMLIB_DOOM_TRANSCRIPT_URL", "http://127.0.0.1:8670/transcript")
TREATMENT = "streamlib_doom.treatments:PictureTreatment"
CHAIN_ROOT = ("Substitute", "substitute_to_downstream")


def transcript(tail: int = 0) -> dict:
    with urllib.request.urlopen(TRANSCRIPT + (f"?tail={tail}" if tail else ""), timeout=3) as answer:
        return json.loads(answer.read())


def claude(prompt: str) -> subprocess.Popen:
    return subprocess.Popen([os.path.join(reel.REPO, "director.sh"), prompt], cwd=reel.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def splice(name: str, treatment: str, tail: tuple) -> tuple:
    """Add one treatment on the end of the chain and hand back the new end."""
    reel.add(TREATMENT, name, config={"treatment": treatment, "amount": 1.0})
    time.sleep(1.2)
    reel.connect(tail[0], tail[1], name, "frame_from_upstream")
    old = reel.link_id(tail[0], "Console", "neural_from_upstream")
    if old:
        reel.tool("disconnect", link_id=old)
    reel.connect(name, "frame_to_downstream", "Console", "neural_from_upstream")
    return (name, "frame_to_downstream")


def unsplice_all(names: list) -> None:
    """Take every treatment out and put the root back on the console."""
    for name in names:
        try:
            reel.tool("remove_processor", processor_id=reel.node_id(name))
        except Exception:
            pass
    time.sleep(0.8)
    if reel.link_id(CHAIN_ROOT[0], "Console", "neural_from_upstream") is None:
        reel.connect(CHAIN_ROOT[0], CHAIN_ROOT[1], "Console", "neural_from_upstream")


def claude_added(before_ids: set) -> str:
    names = [n["display_name"] for n in graph()["nodes"] if n["id"] not in before_ids and str(n["display_name"]).startswith("Claude")]
    return names[0].split(":", 1)[-1].strip() if names else ""


def main() -> int:
    caption("", "")
    director(command="control", mode="auto"); director(command="mission", goal="patrol")
    director(command="heal"); director(command="god", on=False); director(command="give", item="everything")
    time.sleep(4)

    nodes = len(graph()["nodes"])
    caption("DOOM, RUNNING AS A STREAMLIB GRAPH", f"{nodes} processes on one GPU · every box on the left is one of them · the picture on the right is the graph's own")
    time.sleep(8)

    # The robots: hold the marine still so the planner does not shoot them before they are seen.
    director(command="control", mode="stop"); director(command="god", on=True)
    for where in ("ahead", "left", "right"):  # across the room, not in your face: no fireballs filling the frame
        director(command="spawn", kind="zombieman", count=3, where=where, distance=420)
    time.sleep(3)
    director(command="face")  # whichever way the marine was standing, the nearest one is now in shot
    caption("THE OUTPUT TRACKS THE MONSTERS AND SWAPS THEM", "the renderer's own depth and class channels · one compute kernel · no model, no network")
    time.sleep(5)
    director(command="face")
    time.sleep(6)

    # Treatments, one processor at a time, each a new box and a visibly different picture.
    tail = CHAIN_ROOT
    added = []
    for name, treatment, blurb in (("Grade", "grade", "teal shadows, warm highlights"),
                                   ("Neon", "neon", "every edge a light tube"),
                                   ("Glitch", "glitch", "bands tear and the channels split")):
        tail = splice(name, treatment, tail)
        added.append(name)
        caption(f"PROCESSOR ADDED · {name.upper()}", f"{blurb} · spliced into the running graph over MCP · nothing restarted")
        time.sleep(8)
    unsplice_all(added)
    caption("PROCESSORS REMOVED", "the picture goes back · the graph is a live object, not a config file")
    time.sleep(6)

    # The events: let the planner fight a wave, and watch the graph write it down on the left.
    ids_before = {n["id"] for n in graph()["nodes"]}
    director(command="clear")  # the robots beat's grunts must not follow the marine into this
    director(command="god", on=False); director(command="control", mode="auto"); director(command="heal"); director(command="give", item="everything")
    director(command="spawn", kind="zombieman", count=4, where="ahead", distance=300)  # a wave it survives
    director(command="spawn", kind="imp", count=2, where="ahead", distance=520)  # fireballs at range, not in the lens
    caption("EVERY EVENT WRITTEN BY THE GRAPH", "contact · damage · kills · pickups — one line per change, under the graph on the left")
    proc = claude("Read the live transcript with: curl -s 'http://127.0.0.1:8670/transcript?tail=14' and say in one sentence what just happened to the marine. "
                  "Then add ONE PictureTreatment of your choice to the rendered output exactly as your instructions describe, naming the processor 'Claude: <treatment>'. "
                  "Reply in two short sentences: what happened, and what look you chose.")
    time.sleep(14)
    if state().get("dead"):
        director(command="heal")
    try:
        t = transcript()
        caption(f"{t['observations']} FRAMES WATCHED · {t['latest'] + 1} LINES WRITTEN", "an agent reads the lines, never the frames · that is the whole token bill")
    except Exception:
        caption("FRAMES WATCHED, LINES WRITTEN", "an agent reads the lines, never the frames · that is the whole token bill")
    time.sleep(7)

    caption("DIRECTOR · READS THE TRANSCRIPT, NOT THE SCREEN", 'claude -p on the desktop, MCP into this node · ./director.sh "what just happened? then add a look"')
    time.sleep(6)
    if wait_for(lambda: bool(claude_added(ids_before)), 60):
        time.sleep(2)
        caption(f"PROCESSOR ADDED · {claude_added(ids_before).upper()[:28]}", "chosen after reading the transcript · spliced over MCP by the agent")
        time.sleep(9)
    wait_for(lambda: proc.poll() is not None, 8)
    time.sleep(2)

    caption("ONE GRAPH · ONE GPU · EVERY BOX ITS OWN PROCESS", "a 35 Hz game, a 60 fps picture, sensors, a tracked model and an agent — the runtime recorded this of itself")
    time.sleep(7)
    caption("github.com/tatolab/streamlib-doom", "StreamLib")
    time.sleep(6)
    caption("", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
