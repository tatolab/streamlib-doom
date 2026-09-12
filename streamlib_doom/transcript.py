"""What happened, in sentences, so an agent never has to look at a frame.

An agent watching this game directly would have to pull frames, and a frame costs orders of
magnitude more tokens than a line of text and arrives thirty-five times a second. This processor
watches instead. It reads the world and the perception node's detections at full rate, keeps the
state a narrator would keep, and emits a line only when something actually changes: contact,
damage, a kill, a pickup, a new room, the controls changing hands.

The transcript is served over HTTP with a cursor, so an agent asks "what happened since line 40"
and gets back the handful of lines it has not seen. That is the whole idea: the graph watches at
sensor rate, the agent reads at reading rate.
"""
from __future__ import annotations

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streamlib import RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

TRANSCRIPT_PORT = int(os.environ.get("STREAMLIB_DOOM_TRANSCRIPT_PORT", "8670"))
KEEP_LINES = int(os.environ.get("STREAMLIB_DOOM_TRANSCRIPT_LINES", "400"))
TICRATE = 35.0


def _bearing_words(degrees: float) -> str:
    if abs(degrees) < 8:
        return "dead ahead"
    side = "left" if degrees > 0 else "right"
    return f"{abs(degrees):.0f}° to the {side}"


class LiveEventTranscriptHandler(BaseHTTPRequestHandler):
    """GET /transcript?since=N for the lines after N, or /transcript?tail=20 for the last 20."""

    transcript = None

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path not in ("/transcript", "/"):
            self.send_response(404)
            self.end_headers()
            return
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        body = self.transcript.read(since=int(params.get("since", -1)), tail=int(params.get("tail", 0)))
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:
        pass


@processor(description="Turns the running game into a transcript an agent can read: it watches the world and the perception node's detections at full rate and writes one line only when something changes — contact, damage, a kill, a pickup, a new area, the controls changing hands. Served on http://127.0.0.1:8670/transcript?since=N so an agent reads only what it has not seen. Connect the game's `world_to_downstream` and the perception node's `detections_to_downstream`.")
class LiveEventTranscript:
    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def detections_from_upstream(self) -> None: ...

    @output()
    def transcript_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.lock = threading.Lock()
        self.next_index = 0
        self.seen: dict = {}
        self.contact = 0
        self.frames = 0
        self.observations = 0
        self.reported_at = (0.0, 0.0)
        self.reported_tick = -10 ** 9
        self.contact_since = -10 ** 9

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        LiveEventTranscriptHandler.transcript = self
        self.server = ThreadingHTTPServer(("0.0.0.0", TRANSCRIPT_PORT), LiveEventTranscriptHandler)
        threading.Thread(target=self.server.serve_forever, name="transcript-http", daemon=True).start()
        log.info(f"MARKER:TRANSCRIPT_SETUP port={TRANSCRIPT_PORT} pid={os.getpid()}")

    def teardown(self, ctx: RuntimeContextFullAccess) -> None:
        self.server.shutdown()

    def read(self, since: int = -1, tail: int = 0) -> dict:
        with self.lock:
            lines = [l for l in self.lines if l["n"] > since] if since >= 0 else list(self.lines)
            if tail:
                lines = lines[-tail:]
            return {"lines": lines, "latest": self.next_index - 1, "observations": self.observations,
                    "note": "one line per change; the graph watched every frame so you did not have to"}

    def _say(self, tick: int, kind: str, text: str) -> None:
        with self.lock:
            self.lines.append({"n": self.next_index, "t": round(tick / TICRATE, 1), "kind": kind, "text": text})
            self.next_index += 1
            self.lines = self.lines[-KEEP_LINES:]

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        detections = ctx.inputs.read("detections_from_upstream")
        world = ctx.inputs.read("world_from_upstream")
        if detections is not None:
            self.contact = len(detections.get("detections") or [])
            self.latest_detections = detections.get("detections") or []
        if world is None:
            return
        self.observations += 1
        state = world.get("state") or {}
        tick = int(world.get("tick") or 0)
        was, now = self.seen, state
        first = not was

        if first:
            self.reported_at = (float(state.get("x", 0)), float(state.get("y", 0)))
            self.reported_tick = tick
            self._say(tick, "start", f"watching: health {state.get('health')}, {state.get('monsters_alive', 0)} hostiles on the level")
        else:
            health = int(now.get("health", 0))
            before = int(was.get("health", health))
            if health < before - 1:
                self._say(tick, "damage", f"took {before - health} damage, health {health}")
            elif health >= before + 10 and not now.get("dead"):
                self._say(tick, "pickup", f"healed to {health}")  # autonomy regenerates 2 a tic; only a real pickup is news
            if int(now.get("kills", 0)) > int(was.get("kills", 0)):
                killed = int(now.get("kills", 0)) - int(was.get("kills", 0))
                self._say(tick, "kill", f"{killed} hostile down, {now.get('monsters_alive', 0)} still up")
            if now.get("dead") and not was.get("dead"):
                self._say(tick, "death", "the marine is down")
            gains = [(label, int(now.get(item, 0)) - int(was.get(item, 0))) for item, label in
                     (("bullets", "bullets"), ("shells", "shells"), ("armor", "armour"))]
            gains = [f"{n} {label}" for label, n in gains if n > 0]
            if gains:
                self._say(tick, "pickup", "picked up " + " and ".join(gains))
            # Sector changes fire every few steps in a corridor, which drowns everything worth
            # reading. Distance travelled since the last report is what a reader actually wants.
            moved = math.hypot(float(now.get("x", 0)) - self.reported_at[0], float(now.get("y", 0)) - self.reported_at[1])
            if moved > 640 and tick - self.reported_tick > 8 * TICRATE:
                self._say(tick, "move", f"moved about {moved:.0f} units, now in sector {now.get('sector')}")
                self.reported_at = (float(now.get("x", 0)), float(now.get("y", 0)))
                self.reported_tick = tick
            if now.get("control_source") != was.get("control_source"):
                self._say(tick, "control", f"the controls are now {now.get('control_source')}")
            if now.get("mission") != was.get("mission"):
                self._say(tick, "mission", f"mission set to {now.get('mission')}")
            if now.get("style") != was.get("style"):
                self._say(tick, "style", f"the re-render style changed to {str(now.get('style'))[:40]}")
            log_now = now.get("director_log") or []
            log_was = was.get("director_log") or []
            for entry in log_now:
                if entry not in log_was:
                    self._say(tick, "director", f"the director {entry}")

        # Contact is debounced on the count, so a target walking behind a pillar does not chatter.
        contacts = getattr(self, "latest_detections", [])
        previous = int(was.get("_contacts", 0)) if was else 0
        # A target stepping behind a pillar changes the count twice a second; only a count that
        # holds for half a second is worth a line.
        if len(contacts) != previous and tick - self.contact_since > 0.5 * TICRATE:
            self.contact_since = tick
            if contacts:
                nearest = min(contacts, key=lambda d: d.get("range", 1e9))
                self._say(tick, "contact", f"{len(contacts)} in view, nearest {nearest.get('range', 0):.0f} units {_bearing_words(nearest.get('bearing_degrees', 0.0))}")
            elif previous:
                self._say(tick, "contact", "view is clear")
        self.seen = dict(state)
        self.seen["_contacts"] = len(contacts)

        self.frames += 1
        with self.lock:
            recent = [l["text"] for l in self.lines[-4:]]
            written = self.next_index
        ctx.outputs.write("transcript_to_downstream", {
            "timestamp_ns": clock.monotonic_now_ns(), "pid": os.getpid(), "tick": tick,
            "lines_written": written, "observations": self.observations, "recent": recent,
            "url": f"http://127.0.0.1:{TRANSCRIPT_PORT}/transcript"})
        if self.frames in (1, 300, 3000):
            log.info(f"MARKER:TRANSCRIPT_FRAME observations={self.observations} lines={written}")
