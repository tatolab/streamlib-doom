"""Builds the playable graph on an already-running node over its MCP endpoint.

The same four processors and three links `app.py` declares in code, added and
connected live — the way an agent holding only the node's URL would do it.
Start a bare node first (`uv run streamlib run -f empty.py`, or any app that
imports `streamlib_doom.processors` so the classes are in the catalog), then:

    uv run python assemble.py http://127.0.0.1:9200

A link wired after its consumer's setup can, once in a while, report Wired
while the helper never opened its port; the node's own log names it, and a
fresh connect heals it. Pass the node's log path as the second argument to let
this script do that.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

CONTROL_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9200"
NODE_LOG = sys.argv[2] if len(sys.argv) > 2 else None
WIRING_FAILURE = "could not open its port for a link wired after setup"


def mcp(method: str, params: dict | None = None) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    request = urllib.request.Request(
        CONTROL_URL + "/mcp", data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    reply = json.load(urllib.request.urlopen(request, timeout=60))
    if "error" in reply:
        raise RuntimeError(reply["error"])
    result = reply["result"]
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return json.loads(result["content"][0]["text"])


def tool(name: str, **arguments) -> dict:
    return mcp("tools/call", {"name": name, "arguments": arguments})


def add(type_path: str, display_name: str) -> str:
    processor_id = tool("add_processor", type=type_path, display_name=display_name)["processor_id"]
    print(f"added {display_name:10s} {processor_id}  ({type_path})", flush=True)
    return processor_id


def link(from_id: str, from_port: str, to_id: str, to_port: str) -> tuple:
    link_id = tool("connect", from_processor_id=from_id, from_port=from_port, to_processor_id=to_id, to_port=to_port)["link_id"]
    print(f"linked {from_id}/{from_port} -> {to_id}/{to_port}  {link_id}", flush=True)
    return (from_id, from_port, to_id, to_port, link_id)


def graph() -> dict:
    return json.load(urllib.request.urlopen(CONTROL_URL + "/api/graph", timeout=30))


def failed_wirings(since: int) -> tuple[list[tuple[str, str]], int]:
    if not NODE_LOG:
        return [], 0
    lines = open(NODE_LOG, encoding="utf-8", errors="replace").read().splitlines()
    found = []
    for line in lines[since:]:
        if WIRING_FAILURE in line:
            processor = line.split("processor_id=")[1].split()[0]
            port = line.split('port \\"')[1].split('\\"')[0] if 'port \\"' in line else line.split('port "')[1].split('"')[0]
            found.append((processor, port))
    return found, len(lines)


def main() -> None:
    catalog = json.load(urllib.request.urlopen(CONTROL_URL + "/api/registry", timeout=30))
    ours = sorted(e["processor_class_import_path"] for e in catalog["processors"] if e["processor_class_import_path"].startswith("streamlib_doom."))
    print("catalog holds:", ours, flush=True)

    game = add("streamlib_doom.processors:DoomGame", "Game")
    renderer = add("streamlib_doom.processors:E1M1GameRenderer", "Renderer")
    status_bar = add("streamlib_doom.processors:GameStatusBarCompositor", "StatusBar")
    browser = add("streamlib_doom.processors:BrowserFrameSender", "Browser")
    links = [
        link(game, "world_to_downstream", renderer, "world_from_upstream"),
        link(renderer, "view_to_downstream", status_bar, "view_from_upstream"),
        link(status_bar, "frame_to_downstream", browser, "frame_from_upstream"),
    ]

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        states = {n["display_name"]: n["components"]["state"] for n in graph()["nodes"]}
        if all(states.get(n) == "Running" for n in ("Game", "Renderer", "StatusBar", "Browser")):
            break
        if any(v == "Error" for v in states.values()):
            print("processor in error:", states, flush=True)
            sys.exit(2)
        time.sleep(1)
    else:
        print("timed out waiting for Running:", states, flush=True)
        sys.exit(2)
    print("every processor running", flush=True)

    time.sleep(3)
    seen = 0
    for _attempt in range(4):
        failures, seen = failed_wirings(seen)
        dead = [entry for entry in links if (entry[2], entry[3]) in failures]
        if not dead:
            break
        for from_id, from_port, to_id, to_port, link_id in dead:
            print(f"healing {link_id}", flush=True)
            tool("disconnect", link_id=link_id)
            links[links.index((from_id, from_port, to_id, to_port, link_id))] = link(from_id, from_port, to_id, to_port)
        time.sleep(3)
    print("READY — open http://<this machine>:8666/", flush=True)


if __name__ == "__main__":
    main()
