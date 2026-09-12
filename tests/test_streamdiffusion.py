"""The video re-render's contract, without the GPU or the 23 GB of weights it loads at setup."""
import sys

from streamlib_doom import neural, streamdiffusion


def test_the_module_imports_without_pulling_torch_into_the_parent():
    # The parent imports this module to read the processor catalog; torch belongs to the helper.
    assert "torch" not in sys.modules


def test_it_publishes_the_same_wire_contract_as_the_image_re_render():
    video = streamdiffusion.VideoDiffusionRerender
    image = neural.DiffusionRerender
    for port in ("view_from_upstream", "neural_to_downstream"):
        assert hasattr(video, port), port
        assert hasattr(image, port), port


def test_the_model_size_is_a_multiple_of_the_dit_token_grid():
    # The DiT's token grid is height/16 x width/16 and its KV cache is sized from it.
    assert streamdiffusion.MODEL_H % 16 == 0
    assert streamdiffusion.MODEL_W % 16 == 0


def test_it_publishes_at_the_size_the_console_kernel_samples():
    assert (streamdiffusion.NEURAL_W, streamdiffusion.NEURAL_H) == (neural.NEURAL_W, neural.NEURAL_H)


def test_every_style_preset_is_reachable_from_the_video_re_render():
    assert streamdiffusion.STYLES is neural.STYLES
    assert streamdiffusion.DEFAULT_STYLE in neural.STYLES
