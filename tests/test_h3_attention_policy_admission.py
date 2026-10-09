from dataclasses import asdict
from unittest.mock import patch

import pytest

from wee_todd_mlx.generation_selection import resolve_generation_selection
from wee_todd_nodes.runtime import H3GenerationConfig


def test_dense_recipe_preserves_old_config_and_input_identity():
    fields = {"steps": 5, "memory_mode": "low_memory_bf16"}
    old = H3GenerationConfig(**fields)
    explicit = {**fields, "attention_policy": "dense"}
    assert H3GenerationConfig.from_recipe_fields(fields) == old
    assert H3GenerationConfig.from_recipe_fields(explicit) == old
    assert explicit["attention_policy"] == "dense"
    assert "attention_policy" not in asdict(old)


@pytest.mark.parametrize("value", ["sol", True, 1, None, ["dense"]])
def test_invalid_policy_rejects_instead_of_silent_dense(value):
    with pytest.raises(ValueError, match="attention_policy"):
        H3GenerationConfig.from_recipe_fields({"attention_policy": value})


def test_sol_rejects_before_checkpoint_or_model_work():
    with pytest.raises(ValueError, match="Swift MLX runtime"):
        H3GenerationConfig.from_recipe_fields({"attention_policy": "sol_experimental"})
    with patch(
        "wee_todd_mlx.generation_selection.h3_checkpoint_sampling",
        side_effect=AssertionError("No checkpoint inspection"),
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            resolve_generation_selection(
                {"h3AttentionPolicy": "sol_experimental"}, {"engine": "h3"}, [], {}
            )
    from wee_todd_mlx.headless_preflight import preflight_recipe

    recipe = {
        "engine": "h3",
        "components": {"checkpoint": "/nonexistent", "task": "t2va"},
        "config": {"attention_policy": "sol_experimental"},
    }
    with patch(
        "wee_todd_nodes.preflight.preflight_components", side_effect=AssertionError("No weights")
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            preflight_recipe(recipe)


def test_inherited_sol_rejects_but_explicit_dense_can_override_without_mutating_profile():
    profile = {
        "id": "h3.json",
        "recipe": {
            "engine": "h3",
            "components": {"task": "t2va"},
            "config": {"steps": 20, "attention_policy": "sol_experimental"},
            "conditioning": {"task": "t2v", "inputs": []},
        },
    }
    with patch(
        "wee_todd_mlx.generation_selection.h3_checkpoint_sampling",
        side_effect=AssertionError("No checkpoint inspection"),
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            resolve_generation_selection(None, {"engine": "h3", "attachments": []}, [profile], {})
    result = resolve_generation_selection(
        {"h3AttentionPolicy": "dense"}, {"engine": "h3", "attachments": []}, [profile], {}
    )
    assert result["recipe"]["config"]["attention_policy"] == "dense"
    assert profile["recipe"]["config"]["attention_policy"] == "sol_experimental"


def test_h3_policy_is_not_silently_accepted_for_ltx():
    with pytest.raises(ValueError, match="H3 only"):
        resolve_generation_selection({"h3AttentionPolicy": "dense"}, {"engine": "ltx25"}, [], {})


def test_transformer_weight_cache_requires_swift_before_model_work():
    with pytest.raises(ValueError, match="Swift MLX runtime"):
        H3GenerationConfig.from_recipe_fields({"transformer_weight_cache_gb": 48})
    assert (
        H3GenerationConfig.from_recipe_fields({"transformer_weight_cache_gb": 0})
        == H3GenerationConfig()
    )
    for value in [True, 8.5, "8", -1, None]:
        with pytest.raises(ValueError, match="transformer_weight_cache_gb"):
            H3GenerationConfig.from_recipe_fields({"transformer_weight_cache_gb": value})
    with patch(
        "wee_todd_mlx.generation_selection.h3_checkpoint_sampling",
        side_effect=AssertionError("No checkpoint access"),
    ):
        with pytest.raises(ValueError, match="Swift MLX runtime"):
            resolve_generation_selection(
                {"h3TransformerWeightCacheGB": 48}, {"engine": "h3"}, [], {}
            )
