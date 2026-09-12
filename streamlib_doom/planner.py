"""The robot's autonomy: a mission planner that follows waypoints, opens doors, and
fights what perception reports, as a processor whose controls enter the game
over a link — the same port a phone's controls would.

`plan()` is pure so the tests drive the real simulation with it, tic by tic.
"""
from __future__ import annotations

import math
import os

from streamlib import RuntimeContextFullAccess, RuntimeContextLimitedAccess, clock, input, log, output, processor

import heapq

import numpy

# Mission goals in map units; the planner finds its own way there over the costmap.
HANGAR_START = (1056, -3616)
GOALS: dict[str, list[tuple[float, float]]] = {
    "patrol": [(-200, -3232), (2860, -3480), (1056, -3500)],
    "courtyard": [(2860, -3480)],
    "hangar": [HANGAR_START],
    "hold": [],
    "circle": [],  # built around wherever the robot stood when the mission was set
}
CIRCLE_RADIUS = 224.0
CIRCLE_POINTS = 8
MISSIONS = GOALS
CELL = 16.0
CLEARANCE = 18.0
ARRIVE_RADIUS = 56.0


class NavGrid:
    """A costmap of the level: 32-unit cells kept 20 units off every wall, 8-connected where the
    step between cells crosses no blocking line. Doors count as passable; the follower opens them."""

    def __init__(self, game) -> None:
        self.game = game
        blocking, drops = [], []
        door_lines = {i for i, _mx, _my, _s in game.door_lines}
        for i in range(len(game.level.linedefs)):
            if i in door_lines:
                continue
            if not game.two_sided[i] or (int(game.flags[i]) & 1):
                blocking.append(i)
                continue
            a, b = int(game.right_sector[i]), int(game.left_sector[i])
            floor, ceiling = max(game.base_floors[a], game.base_floors[b]), min(game.base_ceilings[a], game.base_ceilings[b])
            if ceiling - floor < 56:
                blocking.append(i)
            elif abs(game.base_floors[a] - game.base_floors[b]) > 24:
                drops.append(i)  # a ledge: walkable down, a wall going up
        self.l1 = game.l1[blocking].astype(numpy.float64)
        self.ld = game.ld[blocking].astype(numpy.float64)
        self.drop_l1 = game.l1[drops].astype(numpy.float64)
        self.drop_ld = game.ld[drops].astype(numpy.float64)
        # Floor height on the right side of each ledge line (positive cross product side), and on the left.
        self.drop_right_floor = numpy.array([game.base_floors[int(game.right_sector[i])] for i in drops], dtype=numpy.float64)
        self.drop_left_floor = numpy.array([game.base_floors[int(game.left_sector[i])] for i in drops], dtype=numpy.float64)
        xs, ys = self.l1[:, 0], self.l1[:, 1]
        self.x0, self.y0 = float(xs.min()) - CELL, float(ys.min()) - CELL
        self.cols = int((xs.max() - self.x0) / CELL) + 2
        self.rows = int((ys.max() - self.y0) / CELL) + 2
        cx = self.x0 + (numpy.arange(self.cols) + 0.5) * CELL
        cy = self.y0 + (numpy.arange(self.rows) + 0.5) * CELL
        centers = numpy.stack(numpy.meshgrid(cx, cy), axis=-1).reshape(-1, 2)  # row-major: index = r * cols + c
        self.walkable = self._distance_to_lines(centers) > CLEARANCE
        self.centers = centers
        self._edge_cache: dict[tuple, bool] = {}

    def _distance_to_lines(self, points: numpy.ndarray) -> numpy.ndarray:
        out = numpy.full(len(points), numpy.inf)
        length_sq = (self.ld ** 2).sum(axis=1)
        for start in range(0, len(points), 2048):
            chunk = points[start : start + 2048]
            w = chunk[:, None, :] - self.l1[None, :, :]
            t = numpy.clip((w * self.ld[None, :, :]).sum(axis=2) / numpy.maximum(length_sq, 1e-9), 0.0, 1.0)
            nearest = self.l1[None, :, :] + t[:, :, None] * self.ld[None, :, :]
            out[start : start + 2048] = numpy.sqrt(((chunk[:, None, :] - nearest) ** 2).sum(axis=2)).min(axis=1)
        return out

    def crosses(self, a, b, margin: float = CLEARANCE * 0.6) -> bool:
        """Whether the segment a->b, widened by `margin` either side, crosses a blocking line."""
        a, b = numpy.asarray(a, dtype=numpy.float64), numpy.asarray(b, dtype=numpy.float64)
        d = b - a
        n = numpy.array((-d[1], d[0])) / max(numpy.hypot(*d), 1e-9) * margin
        for offset in (0.0, 1.0, -1.0):
            aa, bb = a + n * offset, b + n * offset
            dd = bb - aa
            w = self.l1 - aa
            denom = dd[0] * self.ld[:, 1] - dd[1] * self.ld[:, 0]
            with numpy.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, 0] * self.ld[:, 1] - w[:, 1] * self.ld[:, 0]) / denom
                u = (w[:, 0] * dd[1] - w[:, 1] * dd[0]) / denom
            if numpy.any((numpy.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)):
                return True
        if len(self.drop_l1):
            w = self.drop_l1 - a
            denom = d[0] * self.drop_ld[:, 1] - d[1] * self.drop_ld[:, 0]
            with numpy.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, 0] * self.drop_ld[:, 1] - w[:, 1] * self.drop_ld[:, 0]) / denom
                u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / denom
            crossed = (numpy.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
            if numpy.any(crossed):
                # Which side `a` starts on decides the floor it leaves and the one it lands on.
                side = (self.drop_ld[:, 0] * (a[1] - self.drop_l1[:, 1]) - self.drop_ld[:, 1] * (a[0] - self.drop_l1[:, 0]))
                from_floor = numpy.where(side < 0, self.drop_right_floor, self.drop_left_floor)
                to_floor = numpy.where(side < 0, self.drop_left_floor, self.drop_right_floor)
                if numpy.any(crossed & (to_floor - from_floor > 24.0)):
                    return True
        return False

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (int((y - self.y0) / CELL), int((x - self.x0) / CELL))

    def center(self, rc: tuple[int, int]) -> tuple[float, float]:
        return (self.x0 + (rc[1] + 0.5) * CELL, self.y0 + (rc[0] + 0.5) * CELL)

    def nearest_walkable(self, rc):
        if self.walkable[rc[0] * self.cols + rc[1]]:
            return rc
        best, best_d = rc, 1e9
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                r, c = rc[0] + dr, rc[1] + dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.walkable[r * self.cols + c] and dr * dr + dc * dc < best_d:
                    best, best_d = (r, c), dr * dr + dc * dc
        return best

    def path(self, start_xy, goal_xy) -> list[tuple[float, float]]:
        """A* over the costmap, then string-pulled to the few corners that matter."""
        start, goal = self.nearest_walkable(self.cell(*start_xy)), self.nearest_walkable(self.cell(*goal_xy))
        if start == goal:
            return [tuple(goal_xy)]
        came, cost = {start: None}, {start: 0.0}
        frontier = [(0.0, start)]
        steps = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41))
        while frontier:
            _f, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dr, dc, step in steps:
                r, c = current[0] + dr, current[1] + dc
                if not (0 <= r < self.rows and 0 <= c < self.cols) or not self.walkable[r * self.cols + c]:
                    continue
                if dr and dc and not (self.walkable[current[0] * self.cols + c] and self.walkable[r * self.cols + current[1]]):
                    continue
                new_cost = cost[current] + step
                if new_cost < cost.get((r, c), 1e18):
                    edge = (current, (r, c))
                    blocked = self._edge_cache.get(edge)
                    if blocked is None:
                        blocked = self._edge_cache[edge] = self.crosses(self.center(current), self.center((r, c)))
                    if blocked:
                        continue
                    cost[(r, c)], came[(r, c)] = new_cost, current
                    heapq.heappush(frontier, (new_cost + math.hypot(goal[0] - r, goal[1] - c), (r, c)))
        if goal not in came:
            return []
        cells = []
        node = goal
        while node is not None:
            cells.append(node)
            node = came[node]
        cells.reverse()
        points = [self.center(rc) for rc in cells[1:]] + [tuple(goal_xy)]
        pulled, anchor = [], tuple(start_xy)
        i = 0
        while i < len(points):
            j = i
            while j + 1 < len(points) and not self.crosses(anchor, points[j + 1], CLEARANCE):
                j += 1
            pulled.append(points[j])
            anchor = points[j]
            i = j + 1
        return pulled
TURN_RATE_DEGREES = 5.0
STILL_CONTROLS = {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "fire": 0, "use": 0, "weapon": 0}


class PlannerMemory:
    """What the follower carries between tics. Routes are computed off the tic thread: a
    replan that takes a second under load must never stop the controls flowing, because a
    robot that stops reads as stuck and would replan forever."""

    def __init__(self, grid: "NavGrid | None" = None, synchronous: bool = False) -> None:
        import threading
        self.grid = grid
        self.synchronous = synchronous
        self._lock = threading.Lock()
        self._pending: threading.Thread | None = None
        self._result: list | None = None
        self.route_requested_at = -10_000
        self.mission = None
        self.goal_index = 0
        self.route: list[tuple[float, float]] = []
        self.index = 0
        self.replan_at = 0
        self.best_remaining = 1e9
        self.progress_tick = 0
        self.circle: list[tuple[float, float]] = []
        self.circle_anchor: tuple | None = None
        self.stuck_ticks = 0
        self.escape_until = 0
        self.escape_turn = 0.0
        self.use_cooldown = 0
        self.last_x = None
        self.last_y = None
        self.tick = 0


def _request_route(memory: "PlannerMemory", start, goal) -> None:
    """Start computing a route unless one is already computing or was asked for very recently."""
    if memory.grid is None:
        memory.route, memory.index = [goal], 0
        return
    if memory.synchronous:
        memory.route = memory.grid.path(start, goal) or [goal]
        memory.index, memory.best_remaining, memory.progress_tick = 0, 1e9, memory.tick
        return
    if memory._pending is not None and memory._pending.is_alive():
        return
    if memory.tick - memory.route_requested_at < 70:
        return
    import threading
    memory.route_requested_at = memory.tick

    def work():
        found = memory.grid.path(start, goal) or [goal]
        with memory._lock:
            memory._result = found

    memory._pending = threading.Thread(target=work, daemon=True)
    memory._pending.start()


def _take_route(memory: "PlannerMemory") -> None:
    with memory._lock:
        found, memory._result = memory._result, None
    if found is not None:
        memory.route, memory.index, memory.best_remaining, memory.progress_tick = found, 0, 1e9, memory.tick


def _turn_toward(delta_degrees: float, rate: float = TURN_RATE_DEGREES) -> float:
    """A turn control (degrees per tic, positive is right) that closes `delta` without overshooting."""
    return -max(-rate, min(rate, delta_degrees * 0.5))


def plan(world: dict, detections: list[dict], memory: PlannerMemory) -> dict:
    """One tic of controls for the marine described by `world`, given what perception saw."""
    controls = dict(STILL_CONTROLS)
    memory.tick += 1
    x, y, angle = world["x"], world["y"], world["angle"]
    mission = world.get("mission", "patrol")
    goals = GOALS.get(mission, [])
    if mission == "circle":
        anchor = world.get("mission_anchor") or (x, y)
        if memory.circle_anchor != tuple(anchor):
            memory.circle_anchor = tuple(anchor)
            memory.circle = [(anchor[0] + CIRCLE_RADIUS * math.cos(2 * math.pi * i / CIRCLE_POINTS),
                              anchor[1] + CIRCLE_RADIUS * math.sin(2 * math.pi * i / CIRCLE_POINTS)) for i in range(CIRCLE_POINTS)]
        goals = memory.circle
    if mission != memory.mission:
        memory.mission, memory.goal_index, memory.route, memory.index = mission, 0, [], 0
        memory.replan_at = 0
    if memory.last_x is not None:
        moved = math.hypot(x - memory.last_x, y - memory.last_y)
        memory.stuck_ticks = memory.stuck_ticks + 1 if moved < 1.0 else 0
    memory.last_x, memory.last_y = x, y
    memory.use_cooldown = max(0, memory.use_cooldown - 1)

    state = world.get("state") or {}
    # The shotgun is the better tool the moment it's in hand.
    if state.get("weapon") == 2 and 3 in state.get("weapons", []) and state.get("shells", 0) > 0:
        controls["weapon"] = 3

    ammo = state.get("shells", 0) if state.get("weapon") == 3 else state.get("bullets", 0)
    threat = min((d for d in detections if d["range"] < 1100), key=lambda d: d["range"], default=None)
    if threat is not None and ammo > 0:
        controls["turn"] = _turn_toward(threat["bearing_degrees"], 6.0)
        controls["fire"] = 1 if abs(threat["bearing_degrees"]) < 4.0 else 0
        controls["forward"] = 0.4 if threat["range"] > 320 else (-0.3 if threat["range"] < 120 else 0.0)
        return controls

    if memory.tick < memory.escape_until:
        controls["turn"] = memory.escape_turn
        controls["forward"] = -0.6 if memory.tick < memory.escape_until - 10 else 0.8
        return controls
    if memory.stuck_ticks > 25 and goals:
        memory.escape_until = memory.tick + 22
        memory.escape_turn = 5.0 if (memory.tick // 22) % 2 else -5.0
        memory.stuck_ticks = 0
        _request_route(memory, (x, y), goals[memory.goal_index] if memory.goal_index < len(goals) else goals[0])
        return controls

    if not goals:
        return controls
    goal = goals[memory.goal_index]
    if math.hypot(goal[0] - x, goal[1] - y) < ARRIVE_RADIUS:
        if memory.goal_index + 1 < len(goals):
            memory.goal_index += 1
        elif mission in ("patrol", "circle"):
            memory.goal_index = 0
        else:
            return controls  # arrived; hold
        goal = goals[memory.goal_index]
        memory.route, memory.replan_at, memory.best_remaining = [], 0, 1e9
    _take_route(memory)
    if not memory.route or memory.tick >= memory.replan_at:
        _request_route(memory, (x, y), goal)
        memory.replan_at = memory.tick + 140
    if not memory.route:
        # No route yet: face the goal and creep, so the first second is never a dead stop.
        wanted = math.degrees(math.atan2(goal[1] - y, goal[0] - x))
        delta = (wanted - math.degrees(angle) + 180.0) % 360.0 - 180.0
        controls["turn"] = _turn_toward(delta, 7.0)
        controls["forward"] = 0.4 if abs(delta) < 30.0 else 0.0
        return controls
    while memory.index + 1 < len(memory.route) and math.hypot(memory.route[memory.index][0] - x, memory.route[memory.index][1] - y) < 28:
        memory.index += 1
    # Sliding along a wall counts as moving; no progress toward the corner for a second means replan from here.
    target = memory.route[min(memory.index, len(memory.route) - 1)]
    remaining = math.hypot(target[0] - x, target[1] - y)
    if remaining < memory.best_remaining - 4.0:
        memory.best_remaining, memory.progress_tick = remaining, memory.tick
    elif memory.tick - memory.progress_tick > 35:
        # Sliding along a wall for a second: ask for a fresh route but keep following this one.
        memory.best_remaining, memory.progress_tick = 1e9, memory.tick
        _request_route(memory, (x, y), goal)
    target = memory.route[min(memory.index, len(memory.route) - 1)]
    distance = math.hypot(target[0] - x, target[1] - y)
    wanted = math.degrees(math.atan2(target[1] - y, target[0] - x))
    delta = (wanted - math.degrees(angle) + 180.0) % 360.0 - 180.0

    # A closed door on the way gets faced and USEd, the way a robot's arm would hit the button;
    # the game's USE is a short ray along the heading, so facing it is not optional.
    for mx, my, _sector, is_open in world.get("doors", []):
        door_distance = math.hypot(mx - x, my - y)
        if is_open or door_distance > 110:
            continue
        door_heading = math.degrees(math.atan2(my - y, mx - x))
        if abs((door_heading - wanted + 180.0) % 360.0 - 180.0) > 100:
            continue  # behind us relative to where we're going
        door_delta = (door_heading - math.degrees(angle) + 180.0) % 360.0 - 180.0
        controls["turn"] = _turn_toward(door_delta, 7.0)
        controls["forward"] = 0.5 if abs(door_delta) < 20.0 and door_distance > 48 else 0.0
        if abs(door_delta) < 12.0 and memory.use_cooldown == 0:
            controls["use"] = 1
            memory.use_cooldown = 12
        memory.progress_tick = memory.tick  # waiting on a door is not being stuck
        return controls

    controls["turn"] = _turn_toward(delta, 7.0)
    controls["forward"] = 1.0 if abs(delta) < 18.0 else (0.4 if abs(delta) < 45.0 else 0.0)
    return controls


def ground_truth_detections(game) -> list[dict]:
    """What a perfect perception stack would report: every awake monster in line of sight.
    The tests use this in place of the camera-driven perception node."""
    from .game import VIEW_HEIGHT
    p = game.player
    out = []
    for m in game.monsters:
        if not m["alive"] or m["state"] == "idle":
            continue
        d = math.hypot(m["x"] - p["x"], m["y"] - p["y"])
        mz = m.get("z", game.floors[m["sector"]])
        if d < 1400 and game.line_of_sight(p["x"], p["y"], p["z"] + VIEW_HEIGHT, m["x"], m["y"], mz + 32):
            bearing = (math.degrees(math.atan2(m["y"] - p["y"], m["x"] - p["x"])) - math.degrees(p["angle"]) + 180.0) % 360.0 - 180.0
            out.append({"bearing_degrees": bearing, "range": d, "width_px": 0, "column": 160})
    return out


@processor(execution="continuous", interval_ms=10, description="The robot's mission planner: follows the mission's waypoints, opens doors, and turns to fight what the perception node reports. Its controls enter the game over a link, exactly as a phone's do; a phone touching the screen overrides it for 700 ms.")
class MissionPlanner:
    @input(delivery_profile="newest")
    def world_from_upstream(self) -> None: ...

    @input(delivery_profile="newest")
    def detections_from_upstream(self) -> None: ...

    @output()
    def controls_to_downstream(self) -> None: ...

    def __init__(self) -> None:
        self.memory = PlannerMemory()
        self.detections: list[dict] = []
        self.detections_tick = -1
        self.last_tick = -1
        self.plans = 0

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        from .game import Game
        from .wad import Wad
        wad = Wad()
        self.memory = PlannerMemory(NavGrid(Game(wad, wad.level("E1M1"))))
        # Warm the edge cache off the tic thread so the first real route is quick.
        import threading
        threading.Thread(target=lambda: self.memory.grid.path(HANGAR_START, GOALS["courtyard"][0]), daemon=True).start()
        log.info(f"MARKER:PLANNER_SETUP pid={os.getpid()} cells={int(self.memory.grid.walkable.sum())}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        seen = ctx.inputs.read("detections_from_upstream")
        if seen is not None:
            self.detections, self.detections_tick = seen.get("detections", []), seen.get("tick", -1)
        world = ctx.inputs.read("world_from_upstream")
        if world is None or world["tick"] == self.last_tick:
            return
        self.last_tick = world["tick"]
        # Detections older than half a second describe a scene that is gone.
        detections = self.detections if world["tick"] - self.detections_tick < 40 else []  # a 2 Hz learned detector still counts
        controls = plan(world, detections, self.memory)
        controls.update(tick=world["tick"], timestamp_ns=clock.monotonic_now_ns(), pid=os.getpid(),
                        route=[[round(px), round(py)] for px, py in self.memory.route[self.memory.index:]])
        ctx.outputs.write("controls_to_downstream", controls)
        self.plans += 1
        if self.plans in (1, 35, 35 * 60):
            log.info(f"MARKER:PLANNER_CONTROLS plans={self.plans} mission={self.memory.mission} index={self.memory.index}")
