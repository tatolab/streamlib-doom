"""The planner drives the real simulation, tic by tic, with no engine in the loop."""
import math

import pytest

from streamlib_doom.assets import ensure_wad
from streamlib_doom.game import Game
from streamlib_doom.planner import GOALS, NavGrid, PlannerMemory, ground_truth_detections, plan
from streamlib_doom.wad import Wad


@pytest.fixture
def game():
    try:
        ensure_wad()
    except Exception as failure:
        pytest.skip(f"shareware WAD unavailable: {failure}")
    wad = Wad()
    g = Game(wad, wad.level("E1M1"))
    g.control_source = "autonomy"
    return g


def drive(game, mission: str, seconds: float):
    game.mission = mission
    memory = PlannerMemory(NavGrid(game), synchronous=True)
    positions, fired = [], []
    for _ in range(int(seconds * 35)):
        world = game.snapshot()
        controls = plan(world, ground_truth_detections(game), memory)
        game.tick(controls)
        positions.append((game.player["x"], game.player["y"]))
        fired.append(bool(controls["fire"]))
    return memory, positions, fired


def test_the_courtyard_mission_arrives_without_getting_stuck(game):
    memory, positions, fired = drive(game, "courtyard", 45)
    goal = GOALS["courtyard"][-1]
    closest = min(math.hypot(x - goal[0], y - goal[1]) for x, y in positions)
    assert closest < 80, f"never reached the courtyard; closest {closest:.0f} units"
    assert not game.player["dead"]
    # Never parked: every three-second window before arrival either moves the marine or has it shooting.
    arrived = next(i for i, (x, y) in enumerate(positions) if math.hypot(x - goal[0], y - goal[1]) < 80)
    for start in range(0, arrived - 105, 35):
        a, b = positions[start], positions[start + 105]
        assert math.hypot(a[0] - b[0], a[1] - b[1]) > 8 or any(fired[start : start + 105]), f"stuck around tic {start}"


def test_patrol_keeps_moving_for_a_minute_and_survives(game):
    memory, positions, _fired = drive(game, "patrol", 60)
    travelled = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(positions, positions[1:]))
    assert travelled > 3000, f"barely moved: {travelled:.0f} units"
    assert not game.player["dead"] and game.player["health"] > 0


def test_a_threat_in_view_is_turned_toward_and_shot(game):
    game.director({"command": "spawn", "kind": "imp", "count": 1, "where": "ahead", "nonce": "t"})
    for _ in range(5):
        game.tick({})
    for m in game.monsters[-1:]:
        m["state"] = "chase"
    memory = PlannerMemory(NavGrid(game), synchronous=True)
    fired = 0
    for _ in range(35 * 6):
        controls = plan(game.snapshot(), ground_truth_detections(game), memory)
        fired += controls["fire"]
        game.tick(controls)
    assert fired > 0, "never fired at a visible monster"
