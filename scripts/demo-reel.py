#!/usr/bin/env python3
"""The reel: drives the running console node through its story, over the same MCP
tools an agent uses, and writes the captions the caption panel shows.

Every graph change here is real — add_processor and connect on the live node — and
the beats that say "Claude" run `claude -p` for real, with its reply on screen.
"""
from __future__ import annotations

import base64
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request

CONTROL = os.environ.get("STREAMLIB_DOOM_CONTROL_URL", "http://127.0.0.1:9200")
DIRECTOR = "http://127.0.0.1:8668/director"
STATE = "http://127.0.0.1:8666/state.json"
CAPTION_PATH = os.environ.get("STREAMLIB_DOOM_CAPTION", "/tmp/streamlib-doom-caption.txt")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mcp(method: str, params: dict | None = None) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    request = urllib.request.Request(CONTROL + "/mcp", data=body, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    reply = json.load(urllib.request.urlopen(request, timeout=60))
    result = reply.get("result", reply)
    try:
        return json.loads(result["content"][0]["text"])
    except (KeyError, TypeError, ValueError):
        return result


def tool(name: str, **arguments) -> dict:
    return mcp("tools/call", {"name": name, "arguments": arguments})


def graph() -> dict:
    return json.load(urllib.request.urlopen(CONTROL + "/api/graph", timeout=10))


def node_id(display_name: str) -> str:
    for n in graph()["nodes"]:
        if n["display_name"] == display_name:
            return n["id"]
    raise KeyError(display_name)


def link_id(from_name: str, to_name: str, to_port: str) -> str | None:
    g = graph()
    ids = {n["id"]: n["display_name"] for n in g["nodes"]}
    for l in g["links"]:
        if ids.get(l["source"]["processor_id"]) == from_name and ids.get(l["target"]["processor_id"]) == to_name and l["target"]["port_name"] == to_port:
            return l["id"]
    return None


def add(type_path: str, display_name: str, config: dict | None = None) -> str:
    out = tool("add_processor", type=type_path, display_name=display_name, **({"config": config} if config else {}))
    return out["processor_id"]


def connect(from_name: str, from_port: str, to_name: str, to_port: str) -> None:
    out = tool("connect", from_processor_id=node_id(from_name), from_port=from_port, to_processor_id=node_id(to_name), to_port=to_port)
    if "link_id" not in out:
        raise RuntimeError(f"connect {from_name}.{from_port} -> {to_name}.{to_port}: {out}")


def caption(big: str, small: str = "") -> None:
    with open(CAPTION_PATH, "w", encoding="utf-8") as f:
        f.write(big + "\n" + small + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {big}  —  {small}", flush=True)


def director(**command) -> None:
    urllib.request.urlopen(urllib.request.Request(DIRECTOR, data=json.dumps(command).encode(), headers={"Content-Type": "application/json"}), timeout=5).read()


def state() -> dict:
    return json.load(urllib.request.urlopen(STATE, timeout=5))


def wait_for(predicate, timeout: float, interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


class PhoneHand:
    """Sends the controls a phone would, over the same WebSocket, so the badge flips to TELEOP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8667) -> None:
        self.sock = socket.create_connection((host, port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += self.sock.recv(1024)
        if b" 101 " not in reply.split(b"\r\n")[0]:
            raise RuntimeError(f"controls websocket refused: {reply[:80]!r}")

    def send(self, **controls) -> None:
        payload = json.dumps(controls).encode()
        mask = os.urandom(4)
        head = bytes([0x81])
        if len(payload) < 126:
            head += bytes([0x80 | len(payload)])
        else:
            head += bytes([0x80 | 126]) + len(payload).to_bytes(2, "big")
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def close(self) -> None:
        try:
            self.send(f=0, s=0, t=0, fire=0, use=0)
            self.sock.close()
        except OSError:
            pass


def claude_beat(prompt: str) -> subprocess.Popen:
    """Runs the real director in the background; its transcript lands on the caption panel."""
    return subprocess.Popen([os.path.join(REPO, "director.sh"), prompt], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    random.seed(7)
    caption("", "")
    time.sleep(1.0)
    # 1. The runtime
    caption("THIS IS A RUNTIME, NOT A GAME", "every box on the left is its own process · DOOM is one node's output")
    director(command="mission", goal="patrol")
    time.sleep(5)

    # 2. Sensors, added live
    caption("ADDING SENSORS — LIVE", "add_processor + connect over MCP, while frames keep flowing")
    time.sleep(2.5)
    add("streamlib_doom.sensors:DepthSensor", "Depth")
    connect("Render", "view_to_downstream", "Depth", "view_from_upstream")
    connect("Depth", "depth_to_downstream", "Console", "depth_from_upstream")
    time.sleep(1.2)
    caption("DEPTH · SAME FRAME · ZERO COPIES", "a kernel in a new process, over the surface the renderer already published")
    time.sleep(4)
    add("streamlib_doom.sensors:SegmentationSensor", "Segment")
    connect("Render", "view_to_downstream", "Segment", "view_from_upstream")
    connect("Segment", "segmentation_to_downstream", "Console", "segmentation_from_upstream")
    time.sleep(1.2)
    caption("SEGMENTATION · ANOTHER PROCESS", "wall · floor · monster · pickup, per pixel, from the same frame")
    time.sleep(4)
    add("streamlib_doom.sensors:LidarScanner", "Lidar")
    connect("Game", "world_to_downstream", "Lidar", "world_from_upstream")
    connect("Lidar", "scan_to_downstream", "Console", "lidar_from_upstream")
    time.sleep(1.2)
    caption("LIDAR · 360 RAYS A TIC", "cast through the level's own walls at eye height, in its own process")
    time.sleep(4)
    add("streamlib_doom.sensors:OccupancyMapper", "Map")
    connect("Lidar", "scan_to_downstream", "Map", "scan_from_upstream")
    connect("Map", "map_to_downstream", "Console", "map_from_upstream")
    time.sleep(1.2)
    caption("MAP · BUILT FROM THE SCANS", "never the map file · the yellow line is the planner's route")
    time.sleep(6)

    # 3. The robot fights from its camera
    caption("IT AIMS FROM ITS OWN CAMERA", "perception reads the frame → bearing, range → the planner turns and fires")
    director(command="give", item="everything")
    director(command="mission", goal="courtyard")
    time.sleep(2)
    director(command="spawn", kind="imp", count=3, where="ahead")
    time.sleep(8)

    # 4. Claude directs
    caption("CLAUDE DIRECTS THE MISSION", "claude -p on the desktop · sees the game over MCP · adds nodes live")
    before = {n["id"] for n in graph()["nodes"]}
    proc = claude_beat(
        "Send the robot to the courtyard, taunt it on the HUD, ambush it with four imps, then go night vision. "
        "Mechanics: first look at the snapshot and read the state. Then, using add_processor with DirectorCommand nodes named 'Claude: ...' "
        "(one per action, in this order, a second apart): mission with goal courtyard; message with text 'CLAUDE IS WATCHING'; "
        "spawn kind imp count 4 where ahead; effect night_vision. Do not remove the nodes. "
        "Reply in exactly two short sentences: what you saw, and what you did.")
    appeared = wait_for(lambda: len({n["id"] for n in graph()["nodes"]} - before) >= 1, 60)
    if appeared:
        caption("CLAUDE ADDED A NODE", "…and the world changed: a processor added over MCP fired into the game")
    monsters_before = state().get("monsters_alive", 0)
    if wait_for(lambda: state().get("monsters_alive", 0) >= monsters_before + 3, 30):
        caption("AMBUSH: FOUR IMPS FROM CLAUDE", "perception finds them in the frame · the planner turns to fight")
    if wait_for(lambda: state().get("effect", 0) != 0, 20):
        caption("NIGHT VISION, EVERY FRAME", "the phone and this recording see the same picture · the sensors are untouched")
    wait_for(lambda: proc.poll() is not None, 25)
    director(command="mission", goal="patrol")  # back on the move for the rest of the reel
    time.sleep(5)

    # 5. Rewired live
    try:
        caption("REWIRED LIVE · NO RESTART", "cut the HUD → Console link, splice a thermal node in, reconnect")
        cut = link_id("HUD", "Console", "frame_from_upstream")
        if cut:
            tool("disconnect", link_id=cut)
        add("streamlib_doom.effects:ScreenEffect", "Thermal", {"effect": "thermal"})
        connect("HUD", "frame_to_downstream", "Thermal", "frame_from_upstream")
        connect("Thermal", "frame_to_downstream", "Console", "frame_from_upstream")
        time.sleep(7)
        caption("THE GRAPH IS THE PROGRAM", "remove the node, reconnect the link — and back")
        tool("remove_processor", processor_id=node_id("Thermal"))
        time.sleep(0.5)
        connect("HUD", "frame_to_downstream", "Console", "frame_from_upstream")
        time.sleep(4)
    except Exception as failure:
        print("splice beat skipped:", failure, flush=True)
        try:
            if not link_id("HUD", "Console", "frame_from_upstream"):
                connect("HUD", "frame_to_downstream", "Console", "frame_from_upstream")
        except Exception:
            pass

    # 6. Teleop
    caption("TELEOP: A HAND TAKES OVER", "controls arrive over WebSocket · the badge flips · autonomy yields")
    try:
        hand = PhoneHand()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            phase = time.monotonic() - t0
            hand.send(f=1.0 if phase % 2.0 < 1.4 else 0.0, s=0, t=(2.5 if phase < 2.5 else (-2.5 if phase < 5 else 0)), fire=1 if 3.0 < phase < 3.4 else 0, use=0)
            time.sleep(0.05)
        hand.close()
    except Exception as failure:
        print("teleop beat skipped:", failure, flush=True)
    caption("…AND LETS GO. AUTONOMY RESUMES.", "the planner picks the mission back up")
    time.sleep(5)

    caption("github.com/tatolab/streamlib-doom", "one runtime · one graph · a game, a robot, an agent — all processors")
    time.sleep(6)
    caption("", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
