"""The scripted seventeen seconds of E1M1, recorded to an MP4 with its audio.

`streamlib run -f demo.py` builds the graph in code: the scripted world, the
same renderer and status bar the game uses, the palette upscaler, then the
engine's own H.264 and Opus encoders into one fragmented MP4. Stop the node
(Ctrl-C) once the demo has run its course; the file is complete to its last
closed fragment either way.
"""

import os

from streamlib import H264Encoder, Mp4Sink, OpusEncoder, Runtime

from streamlib_doom.demo_processors import AtDoomsGate, DoomWorldState, E1M1Renderer, PaletteUpscaler, StatusBarCompositor

RECORDING_PATH = os.environ.get("STREAMLIB_DOOM_RECORDING", "e1m1-demo.mp4")


def setup(rt: Runtime) -> None:
    world = rt.add(DoomWorldState, display_name="World")
    renderer = rt.add(E1M1Renderer, display_name="Renderer")
    status_bar = rt.add(StatusBarCompositor, display_name="StatusBar")
    upscaler = rt.add(PaletteUpscaler, display_name="Upscaler")
    music = rt.add(AtDoomsGate, display_name="AtDoomsGate")
    h264 = rt.add(H264Encoder, config={"fps": 35, "keyframe_interval_seconds": 1}, display_name="H264")
    opus = rt.add(OpusEncoder, config={"bitrate_bps": 128000}, display_name="Opus")
    recorder = rt.add(Mp4Sink, config={"path": RECORDING_PATH}, display_name="Recorder")

    rt.connect(world.output("world_to_downstream"), renderer.input("world_from_upstream"))
    rt.connect(renderer.output("view_to_downstream"), status_bar.input("view_from_upstream"))
    rt.connect(status_bar.output("frame_to_downstream"), upscaler.input("frame_from_upstream"))
    rt.connect(upscaler.output("video"), h264.input("video"))
    rt.connect(h264.output("encoded_video"), recorder.input("tracks"))
    rt.connect(world.output("world_to_downstream"), music.input("world_from_upstream"))
    rt.connect(music.output("audio"), opus.input("audio"))
    rt.connect(opus.output("encoded_audio"), recorder.input("tracks"))
