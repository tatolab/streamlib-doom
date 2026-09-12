"""The simulation, tic by tic, with no engine in the loop."""
import math

import pytest

from streamlib_doom.assets import ensure_wad
from streamlib_doom.game import Game, TICRATE
from streamlib_doom.wad import Wad


@pytest.fixture
def game():
    try:
        ensure_wad()
    except Exception as failure:
        pytest.skip(f"shareware WAD unavailable: {failure}")
    wad = Wad()
    return Game(wad, wad.level("E1M1"))


def test_walls_stop_the_player(game):
    for _ in range(3 * TICRATE):
        game.tick({"forward": 1.0})
    p = game.player
    assert p["y"] > -3616 and p["y"] < -2860, "the north wall of the hangar holds"
    assert game.sector_at(p["x"], p["y"]) == p["sector"]


def test_the_pistol_fires_every_fourteen_tics(game):
    shots = []
    for tic in range(60):
        game.tick({"fire": 1})
        if "DSPISTOL" in game.events:
            shots.append(tic)
    assert shots[:3] == [0, 14, 28]
    assert game.player["bullets"] == 50 - len(shots)


def test_a_door_opens_when_used_and_closes_on_its_own(game):
    p = game.player
    p["x"], p["y"], p["angle"] = 1500.0, -2496.0, 0.0
    p["sector"] = game.sector_at(p["x"], p["y"])
    game.tick({"use": 1})
    assert "DSDOROPN" in game.events and 4 in game.doors
    for _ in range(60):
        game.tick({})
    assert game.ceilings[4] == 68.0
    for _ in range(220):
        game.tick({})
    assert game.ceilings[4] == 0.0 and 4 not in game.doors


def test_monsters_wake_on_sight_and_the_level_restarts_after_death(game):
    p = game.player
    p["x"], p["y"], p["angle"] = 2860.0, -3480.0, math.radians(298)
    p["sector"] = game.sector_at(p["x"], p["y"])
    for _ in range(TICRATE):
        game.tick({})
    assert any(m["state"] != "idle" for m in game.monsters)
    game._damage_player(500, None)
    assert p["dead"] and p["face"] == "STFDEAD0"
    for _ in range(TICRATE + 2):
        game.tick({"fire": 1})
    assert not game.player["dead"] and game.player["health"] == 100
