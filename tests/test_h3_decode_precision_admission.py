from dataclasses import asdict
from unittest.mock import patch

import pytest

from wee_todd_mlx.generation_selection import resolve_generation_selection
from wee_todd_nodes.runtime import H3GenerationConfig


def test_python_fp32_recipe_admission_preserves_config_and_dataclass_identity():
    fields = {"memory_mode": "low_memory_bf16", "steps": 5}
    old = H3GenerationConfig(**fields)
    assert H3GenerationConfig.from_recipe_fields(fields) == old
    explicit = {**fields, "video_decode_precision": "float32"}
    assert H3GenerationConfig.from_recipe_fields(explicit) == old
    assert explicit["video_decode_precision"] == "float32"
    assert "video_decode_precision" not in asdict(old)


@pytest.mark.parametrize("mode", ["normal", "low_memory_bf16"])
def test_python_rejects_swift_fp16_before_any_checkpoint_work(mode):
    fields = {"memory_mode": mode, "video_decode_precision": "float16"}
    with pytest.raises(ValueError, match="Swift MLX runtime"):
        H3GenerationConfig.from_recipe_fields(fields)
    with patch(
        "wee_todd_mlx.generation_selection.h3_checkpoint_sampling",
        side_effect=AssertionError("No checkpoint inspection expected"),
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            resolve_generation_selection(
                {"h3VideoDecodePrecision": "float16"}, {"engine": "h3"}, [], {}
            )


@pytest.mark.parametrize("value", ["bfloat16", True, 1, None, ["float16"]])
def test_python_recipe_rejects_invalid_precision_without_silent_default(value):
    with pytest.raises(ValueError, match="video_decode_precision"):
        H3GenerationConfig.from_recipe_fields({"video_decode_precision": value})


def test_python_headless_fp16_rejects_before_preflight_components(tmp_path):
    from wee_todd_mlx.headless_preflight import preflight_recipe
    recipe = {"engine": "h3", "components": {"checkpoint": str(tmp_path), "task": "t2va"},
              "config": {"video_decode_precision": "float16"}}
    with patch(
        "wee_todd_nodes.preflight.preflight_components",
        side_effect=AssertionError("Weights/component work must not start"),
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            preflight_recipe(recipe)
