import pytest


def test_engines_share_exclusion_and_public_unload_cannot_interrupt():
    from fish_speech_mlx.pipeline import unload
    from wee_todd_mlx.speech_lifecycle import serialized

    @serialized
    def inner():
        return "ok"

    @serialized
    def outer():
        with pytest.raises(RuntimeError, match="active"):
            inner()
        with pytest.raises(RuntimeError, match="active"):
            unload()

    outer()
    assert inner() == "ok"


def test_failed_inference_releases_ownership_and_traceback_locals():
    from wee_todd_mlx.speech_lifecycle import serialized

    @serialized
    def bad():
        retained_tensor = object()
        assert retained_tensor
        raise ValueError("decode failed")

    with pytest.raises(ValueError) as error:
        bad()
    tb = error.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name == "bad":
            assert not tb.tb_frame.f_locals
        tb = tb.tb_next
    assert serialized(lambda: 7)() == 7
