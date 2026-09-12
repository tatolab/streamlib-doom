"""The ops console: the playable game as a robot, its autonomy stack, the live graph
as the hero pane, and a 1920x1080 picture the engine records of itself.

    uv run streamlib run -f showcase.py                       # the console, phone-playable, nothing recorded
    STREAMLIB_DOOM_RECORDING=console.mp4 uv run streamlib run -f showcase.py
    scripts/neural-setup.py                                   # adds the models over MCP and proves each is delivering

Set STREAMLIB_DOOM_CONSOLE_WHIP_URL to publish the console itself over WebRTC, so the
dashboard can be watched in a browser from anywhere on the network while someone plays.

The sensors — depth, segmentation, lidar, map — are not declared here. The reel (or
Claude, or you) adds them to the running graph over MCP and their panes light up.
"""

import os

from streamlib import H264Encoder, Mp4Sink, OpusEncoder, Runtime

import streamlib_doom.director  # noqa: F401 — DirectorCommand joins the catalog
import streamlib_doom.effects  # noqa: F401 — ScreenEffect joins the catalog
import streamlib_doom.sensors  # noqa: F401 — the sensors join the catalog, for add_processor
from streamlib_doom.planner import MissionPlanner
from streamlib_doom.processors import BrowserFrameSender, DoomAudioMixer, DoomGame, E1M1GameRenderer, GameStatusBarCompositor
from streamlib_doom.sensors import PerceptionNode
from streamlib_doom.showcase import CaptionPanel, ConsoleCompositor, GraphPanel, TelemetryPanel

RECORDING_PATH = os.environ.get("STREAMLIB_DOOM_RECORDING", "")


def setup(rt: Runtime) -> None:
    game = rt.add(DoomGame, display_name="Game")
    renderer = rt.add(E1M1GameRenderer, display_name="Render")
    status_bar = rt.add(GameStatusBarCompositor, display_name="HUD")
    browser = rt.add(BrowserFrameSender, display_name="Phone")
    rt.connect(game.output("world_to_downstream"), renderer.input("world_from_upstream"))
    rt.connect(renderer.output("view_to_downstream"), status_bar.input("view_from_upstream"))
    rt.connect(status_bar.output("frame_to_downstream"), browser.input("frame_from_upstream"))

    # The autonomy stack: perception reads the camera frame, the planner closes the loop into the game.
    perception = rt.add(PerceptionNode, display_name="Perceive")
    planner = rt.add(MissionPlanner, display_name="Planner")
    rt.connect(renderer.output("view_to_downstream"), perception.input("view_from_upstream"))
    rt.connect(perception.output("detections_to_downstream"), planner.input("detections_from_upstream"))
    rt.connect(game.output("world_to_downstream"), planner.input("world_from_upstream"))
    rt.connect(planner.output("controls_to_downstream"), game.input("controls_from_upstream"))

    graph_panel = rt.add(GraphPanel, display_name="Graph")
    caption = rt.add(CaptionPanel, display_name="Caption")
    telemetry = rt.add(TelemetryPanel, display_name="Telemetry")
    console = rt.add(ConsoleCompositor, display_name="Console")
    rt.connect(status_bar.output("frame_to_downstream"), console.input("frame_from_upstream"))
    rt.connect(graph_panel.output("panel_to_downstream"), console.input("graph_panel_from_upstream"))
    rt.connect(caption.output("panel_to_downstream"), console.input("caption_from_upstream"))
    rt.connect(telemetry.output("panel_to_downstream"), console.input("telemetry_from_upstream"))
    rt.connect(game.output("world_to_downstream"), telemetry.input("world_from_upstream"))
    rt.connect(console.output("witness_to_downstream"), telemetry.input("witness_from_upstream"))
    rt.connect(perception.output("detections_to_downstream"), telemetry.input("detections_from_upstream"))

    whip_url = os.environ.get("STREAMLIB_WHIP_URL")
    console_whip_url = os.environ.get("STREAMLIB_DOOM_CONSOLE_WHIP_URL")
    mixer = rt.add(DoomAudioMixer, display_name="Mixer") if (RECORDING_PATH or whip_url or console_whip_url) else None
    if whip_url:
        # The phone's WebRTC option, exactly as app.py offers it: the operator's picture and the mix over WHIP.
        from streamlib_webrtc import WhipPublisher
        from streamlib_doom.demo_processors import PaletteUpscaler
        upscaler = rt.add(PaletteUpscaler, display_name="Upscaler")
        whip_h264 = rt.add(H264Encoder, config={"fps": 35, "keyframe_interval_seconds": 1}, display_name="H264 · WHIP")
        whip_opus = rt.add(OpusEncoder, config={"bitrate_bps": 96000}, display_name="Opus · WHIP")
        publisher_config = {"url": whip_url}
        if os.environ.get("STREAMLIB_WHIP_BEARER_TOKEN"):
            publisher_config["bearer_token"] = os.environ["STREAMLIB_WHIP_BEARER_TOKEN"]
        publisher = rt.add(WhipPublisher, config=publisher_config, display_name="WHIP")
        rt.connect(status_bar.output("frame_to_downstream"), upscaler.input("frame_from_upstream"))
        rt.connect(upscaler.output("video"), whip_h264.input("video"))
        rt.connect(whip_h264.output("encoded_video"), publisher.input("tracks"))
        rt.connect(game.output("world_to_downstream"), mixer.input("world_from_upstream"))
        rt.connect(mixer.output("audio"), whip_opus.input("audio"))
        rt.connect(whip_opus.output("encoded_audio"), publisher.input("tracks"))

    if console_whip_url:
        # The console itself, live: the same 1920x1080 picture the recorder gets, published over WHIP
        # so anyone on the network can watch the dashboard in a browser while someone else plays.
        from streamlib_webrtc import WhipPublisher
        console_h264 = rt.add(H264Encoder, config={"fps": 60, "keyframe_interval_seconds": 1}, display_name="H264 · Console")
        console_opus = rt.add(OpusEncoder, config={"bitrate_bps": 96000}, display_name="Opus · Console")
        console_config = {"url": console_whip_url}
        if os.environ.get("STREAMLIB_WHIP_BEARER_TOKEN"):
            console_config["bearer_token"] = os.environ["STREAMLIB_WHIP_BEARER_TOKEN"]
        console_publisher = rt.add(WhipPublisher, config=console_config, display_name="WHIP · Console")
        rt.connect(console.output("video"), console_h264.input("video"))
        rt.connect(console_h264.output("encoded_video"), console_publisher.input("tracks"))
        if not whip_url:
            rt.connect(game.output("world_to_downstream"), mixer.input("world_from_upstream"))
        rt.connect(mixer.output("audio"), console_opus.input("audio"))
        rt.connect(console_opus.output("encoded_audio"), console_publisher.input("tracks"))

    if RECORDING_PATH:
        h264 = rt.add(H264Encoder, config={"fps": 60, "keyframe_interval_seconds": 1}, display_name="H264")
        opus = rt.add(OpusEncoder, config={"bitrate_bps": 128000}, display_name="Opus")
        recorder = rt.add(Mp4Sink, config={"path": RECORDING_PATH}, display_name="MP4")
        rt.connect(console.output("video"), h264.input("video"))
        rt.connect(h264.output("encoded_video"), recorder.input("tracks"))
        if not (whip_url or console_whip_url):
            rt.connect(game.output("world_to_downstream"), mixer.input("world_from_upstream"))
        rt.connect(mixer.output("audio"), opus.input("audio"))
        rt.connect(opus.output("encoded_audio"), recorder.input("tracks"))
