"""The showcase node: the playable game, plus the live graph panel, the
director's transcript and a 1920x1080 compositor recorded by the engine.

    uv run streamlib run -f showcase.py
    ./director.sh "teleport three imps behind the player"      # in another terminal

Set STREAMLIB_DOOM_RECORDING to choose the MP4 path; STREAMLIB_DOOM_AUTOPILOT=1
hands the marine to the autopilot until a phone takes over.
"""

import os

from streamlib import H264Encoder, Mp4Sink, OpusEncoder, Runtime

import streamlib_doom.director  # noqa: F401 — DirectorCommand joins the catalog
import streamlib_doom.effects  # noqa: F401 — ScreenEffect joins the catalog
from streamlib_doom.director import DirectorCommand
from streamlib_doom.processors import BrowserFrameSender, DoomAudioMixer, DoomGame, E1M1GameRenderer, GameStatusBarCompositor
from streamlib_doom.showcase import GraphPanel, ShowcaseCompositor, TranscriptPanel

RECORDING_PATH = os.environ.get("STREAMLIB_DOOM_RECORDING", "showcase.mp4")


def setup(rt: Runtime) -> None:
    game = rt.add(DoomGame, display_name="Game")
    renderer = rt.add(E1M1GameRenderer, display_name="Renderer")
    status_bar = rt.add(GameStatusBarCompositor, display_name="StatusBar")
    browser = rt.add(BrowserFrameSender, display_name="Browser")
    rt.connect(game.output("world_to_downstream"), renderer.input("world_from_upstream"))
    rt.connect(renderer.output("view_to_downstream"), status_bar.input("view_from_upstream"))
    rt.connect(status_bar.output("frame_to_downstream"), browser.input("frame_from_upstream"))

    graph_panel = rt.add(GraphPanel, display_name="GraphPanel")
    transcript = rt.add(TranscriptPanel, display_name="Transcript")
    showcase = rt.add(ShowcaseCompositor, display_name="Showcase")
    rt.connect(status_bar.output("frame_to_downstream"), showcase.input("frame_from_upstream"))
    rt.connect(graph_panel.output("panel_to_downstream"), showcase.input("graph_panel_from_upstream"))
    rt.connect(transcript.output("panel_to_downstream"), showcase.input("transcript_from_upstream"))

    # The 1920x1080 composite is recorded only when a path is given, so the live
    # service can run the showcase panels without ever filling a disk.
    if os.environ.get("STREAMLIB_DOOM_RECORDING"):
        mixer = rt.add(DoomAudioMixer, display_name="Mixer")
        h264 = rt.add(H264Encoder, config={"fps": 35, "keyframe_interval_seconds": 1}, display_name="H264")
        opus = rt.add(OpusEncoder, config={"bitrate_bps": 128000}, display_name="Opus")
        recorder = rt.add(Mp4Sink, config={"path": RECORDING_PATH}, display_name="Recorder")
        rt.connect(showcase.output("video"), h264.input("video"))
        rt.connect(h264.output("encoded_video"), recorder.input("tracks"))
        rt.connect(game.output("world_to_downstream"), mixer.input("world_from_upstream"))
        rt.connect(mixer.output("audio"), opus.input("audio"))
        rt.connect(opus.output("encoded_audio"), recorder.input("tracks"))

    if os.environ.get("STREAMLIB_DOOM_AUTOPILOT"):
        # Delivers over the game's HTTP endpoint from its own helper; needs no link.
        rt.add(DirectorCommand, config={"command": "autopilot", "on": True}, display_name="DirectorAutopilot")
