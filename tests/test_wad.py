"""The parser against the shareware WAD itself; skipped when it cannot be fetched."""
import pytest

from streamlib_doom.assets import ensure_wad
from streamlib_doom.wad import Wad


@pytest.fixture(scope="module")
def wad():
    try:
        ensure_wad()
    except Exception as failure:  # no network on this machine
        pytest.skip(f"shareware WAD unavailable: {failure}")
    return Wad()


def test_the_shareware_wad_is_the_one_id_shipped(wad):
    assert len(wad.lumps) == 1264
    assert "E1M1" in wad.index and "E1M9" in wad.index and "E2M1" not in wad.index


def test_e1m1_has_its_geometry(wad):
    level = wad.level("E1M1")
    assert len(level.linedefs) == 475 and len(level.sectors) == 85 and len(level.vertexes) == 467
    assert level.player_start() == (1056, -3616, 90)


def test_wall_textures_compose_from_patches(wad):
    startan = wad.texture("STARTAN3")
    assert (startan.width, startan.height) == (128, 128)
    assert int(startan.alpha.min()) == 255, "a wall texture covers every texel"


def test_the_score_parses_to_the_riff(wad):
    events = wad.mus_events("D_E1M1")
    first_guitar = [a for _t, ch, kind, a, _b in events if ch == 1 and kind == 1][:6]
    assert first_guitar == [40, 40, 52, 40, 40, 50]  # E2 E2 E3 E2 E2 D3


def test_sound_effects_decode(wad):
    rate, pcm = wad.sound("DSPISTOL")
    assert rate == 11025 and 0.4 < len(pcm) / rate < 0.6
