"""The director's verbs, applied to the game with no engine in the loop."""
import pytest

from streamlib_doom.assets import ensure_wad
from streamlib_doom.effects import MODES, build_all_remaps
from streamlib_doom.game import Game
from streamlib_doom.wad import Wad


@pytest.fixture
def game():
    try:
        ensure_wad()
    except Exception as failure:
        pytest.skip(f"shareware WAD unavailable: {failure}")
    wad = Wad()
    return Game(wad, wad.level("E1M1"))


def test_spawn_teleports_monsters_in_and_never_into_a_wall(game):
    before = len(game.game_monsters()) if hasattr(game, "game_monsters") else len(game.monsters)
    did = game.director({"command": "spawn", "kind": "imp", "count": 4, "where": "behind", "nonce": "s"})
    assert "4 imp" in did
    new = game.monsters[before:]
    assert len(new) == 4
    for monster in new:
        # a placed monster stands where its own sector says the floor is, i.e. not inside geometry
        assert game.sector_at(monster["x"], monster["y"]) == monster["sector"]


def test_a_command_is_applied_once_despite_repeats(game):
    assert game.director({"command": "give", "item": "everything", "nonce": "g"}) is not None
    game.player["bullets"] = 5
    assert game.director({"command": "give", "item": "everything", "nonce": "g"}) is None
    assert game.player["bullets"] == 5, "the deduplicated repeat did not re-give ammo"


def test_lights_and_effect_reach_the_snapshot(game):
    game.director({"command": "lights", "factor": 0.25, "nonce": "l"})
    game.director({"command": "effect", "effect": "thermal", "nonce": "e"})
    game.tick({})
    state = game.snapshot()["state"]
    assert state["lights"] == pytest.approx(0.25)
    assert state["effect"] == MODES["thermal"]


def test_god_mode_refuses_damage(game):
    game.director({"command": "god", "on": True, "nonce": "d"})
    game._damage_player(1000, None)
    assert game.player["health"] == 100 and not game.player["dead"]


def test_every_effect_builds_a_full_256_entry_remap():
    ensure_wad()
    table = build_all_remaps(Wad())
    assert table.shape == (max(MODES.values()) + 1, 256)
    assert set(int(v) for v in table[MODES["night_vision"]]) != {0}, "night vision remaps to real indices"
