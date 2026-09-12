"""DOOM E1M1, playable from a phone: the StreamLib app.

`streamlib run` finds `setup(rt)` below by convention. Four processors, each in
its own helper process: the game at 35 tics a second, the column renderer, the
status-bar compositor, and the browser bridge that serves the page on :8666 and
takes controls on :8667. Open http://<this machine>:8666/ on the same network.

Set STREAMLIB_WHIP_URL to also publish 1280x960 H.264 and Opus to a WHIP
endpoint — a MediaMTX on the same box, or Cloudflare Stream — through the
streamlib-webrtc extension; the page then offers WebRTC playback when
STREAMLIB_WHEP_URL names where to play it from.

Processors live in `streamlib_doom/`, never here: each runs in its own child
interpreter, which imports its class by name.
"""

import os

from streamlib import H264Encoder, OpusEncoder, Runtime

from streamlib_doom.demo_processors import PaletteUpscaler
from streamlib_doom.processors import BrowserFrameSender, DoomAudioMixer, DoomGame, E1M1GameRenderer, GameStatusBarCompositor
import streamlib_doom.director  # noqa: F401 — DirectorCommand joins the catalog for an agent to add over MCP
import streamlib_doom.effects  # noqa: F401 — so does ScreenEffect


def setup(rt: Runtime) -> None:
    game = rt.add(DoomGame, display_name="Game")
    renderer = rt.add(E1M1GameRenderer, display_name="Renderer")
    status_bar = rt.add(GameStatusBarCompositor, display_name="StatusBar")
    browser = rt.add(BrowserFrameSender, display_name="Browser")
    rt.connect(game.output("world_to_downstream"), renderer.input("world_from_upstream"))
    rt.connect(renderer.output("view_to_downstream"), status_bar.input("view_from_upstream"))
    rt.connect(status_bar.output("frame_to_downstream"), browser.input("frame_from_upstream"))

    whip_url = os.environ.get("STREAMLIB_WHIP_URL")
    if whip_url:
        from streamlib_webrtc import WhipPublisher

        upscaler = rt.add(PaletteUpscaler, display_name="Upscaler")
        h264 = rt.add(H264Encoder, config={"fps": 35, "keyframe_interval_seconds": 1}, display_name="H264")
        mixer = rt.add(DoomAudioMixer, display_name="Mixer")
        opus = rt.add(OpusEncoder, config={"bitrate_bps": 96000}, display_name="Opus")
        publisher_config = {"url": whip_url}
        if os.environ.get("STREAMLIB_WHIP_BEARER_TOKEN"):
            publisher_config["bearer_token"] = os.environ["STREAMLIB_WHIP_BEARER_TOKEN"]
        publisher = rt.add(WhipPublisher, config=publisher_config, display_name="WHIP")
        rt.connect(status_bar.output("frame_to_downstream"), upscaler.input("frame_from_upstream"))
        rt.connect(upscaler.output("video"), h264.input("video"))
        rt.connect(h264.output("encoded_video"), publisher.input("tracks"))
        rt.connect(game.output("world_to_downstream"), mixer.input("world_from_upstream"))
        rt.connect(mixer.output("audio"), opus.input("audio"))
        rt.connect(opus.output("encoded_audio"), publisher.input("tracks"))
