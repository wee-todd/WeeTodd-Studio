"""Cheap lifecycle and capability regressions; no released checkpoints required."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from wee_todd_nodes import nodes
from wee_todd_nodes.runtime import H3GenerationConfig, H3ModelSpec


@pytest.mark.parametrize("memory_mode", ["normal", "low_memory_bf16"])
@pytest.mark.parametrize("terminal", [None, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("terminal_stage", ["text", "sample", "publish"])
def test_h3_convenience_uses_staged_contract_and_releases_every_terminal_path(
    tmp_path, monkeypatch, memory_mode, terminal, terminal_stage,
):
    config = H3GenerationConfig(
        memory_mode=memory_mode, sampling_method="res_multistep",
        inference_optimization="combined", paging_cache_gb=4,
    )
    stages = []
    resident = set()
    runtimes = ["RUNTIME", "TEXT_ENCODER_RUNTIME", "TRANSFORMER_RUNTIME",
                "VIDEO_VAE_RUNTIME", "AUDIO_VAE_RUNTIME"]
    for name in runtimes:
        monkeypatch.setattr(nodes, name, SimpleNamespace(unload=lambda n=name: resident.discard(n)))
    monkeypatch.setattr(nodes, "preflight_components", lambda *_args: None)
    monkeypatch.setattr(H3GenerationConfig, "validate_paging", lambda *_args: None)

    def encode(_self, components, prompt, unload_after_encode, **kwargs):
        assert not resident
        assert components.transformer == str(tmp_path / "alternate")
        assert kwargs["config"] is config
        assert unload_after_encode
        stages.append("text")
        if terminal and terminal_stage == "text":
            resident.add("TEXT_ENCODER_RUNTIME")
            raise terminal("interrupted")
        return object(), "{}"

    def sample(_self, components, conditioning, forwarded, unload_after_sample):
        assert not resident
        assert forwarded is config
        assert unload_after_sample
        stages.append("sample")
        if terminal and terminal_stage == "sample":
            resident.add("TRANSFORMER_RUNTIME")
            raise terminal("interrupted")
        return SimpleNamespace(generation_config=config), "{}"

    def publish(_self, **kwargs):
        assert not resident
        stages.append("publish")
        if terminal and terminal_stage == "publish":
            resident.update({"VIDEO_VAE_RUNTIME", "AUDIO_VAE_RUNTIME"})
            raise terminal("interrupted")
        return {"result": (str(tmp_path / "output.mp4"), "{}")}

    monkeypatch.setattr(nodes.WeeToddH3TextEncode, "encode", encode)
    monkeypatch.setattr(nodes.WeeToddH3Sample, "sample", sample)
    monkeypatch.setattr(nodes.WeeToddH3DirectPublishLatents, "publish", publish)
    model = H3ModelSpec(str(tmp_path), transformer=str(tmp_path / "alternate"))
    if terminal:
        with pytest.raises(terminal, match="interrupted"):
            nodes.WeeToddH3Generate().generate(model, config, "A scene.", "test")
    else:
        result = nodes.WeeToddH3Generate().generate(model, config, "A scene.", "test")
        assert result[0].endswith("output.mp4")
    assert not resident
    expected = ["text", "sample", "publish"]
    if terminal:
        expected = expected[:expected.index(terminal_stage) + 1]
    assert stages == expected


@pytest.mark.parametrize("changes, message", [
    ({"pipeline_mode": "guided", "stage1_steps": 30}, "guided"),
    ({"pipeline_mode": "guided_hq", "stage1_steps": 15}, "guided"),
    ({"stage1_sampler": "euler_ancestral_cfg_pp", "ic_lora_single_stage": True,
      "stage2_steps": 0}, "CFG"),
    ({"generated_keyframes": 2}, "keyframe"),
    ({"dfr_enabled": True}, "DFR"),
    ({"dfr_temporal_rounds": 1}, "DFR"),
    ({"duration_mode": "automatic"}, "duration"),
])
def test_ltx25_chain_rejects_unimplemented_modes_before_component_inspection(
    tmp_path, monkeypatch, changes, message,
):
    from ltx25_mlx.runtime import LTX25GenerationConfig, LTX25RuntimeCache

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported chain must be rejected before loading or encoding")

    spec = SimpleNamespace(validate=forbidden)
    runtime = LTX25RuntimeCache()
    monkeypatch.setattr(runtime, "get", forbidden)
    with pytest.raises(ValueError, match=message):
        runtime.generate_chain_to_file(
            spec, replace(LTX25GenerationConfig(), **changes), ["one", "two"],
            tmp_path / "chain.mp4", window_count=2, overlap_frames=25,
        )


def test_h3_changed_schedule_reloads_discarded_resident_adaln_before_cache_build(
    tmp_path, monkeypatch,
):
    import weakref

    import mlx.core as mx

    from minimax_h3_mlx.config import PipelineConfig
    from minimax_h3_mlx.dit import MiniMaxH3DiT
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from tests.test_dit_smoke import tiny_config

    loaded = []

    def load_dit(_path):
        if loaded:
            assert loaded[-1]() is None, "Old transformer must release before replacement"
        dit = MiniMaxH3DiT(tiny_config())
        loaded.append(weakref.ref(dit))
        return dit

    monkeypatch.setattr("minimax_h3_mlx.load.load_dit", load_dit)
    monkeypatch.setattr("minimax_h3_mlx.load.load_video_vae", lambda *_: None)
    monkeypatch.setattr("minimax_h3_mlx.load.load_audio_vae", lambda *_: None)
    monkeypatch.setattr("minimax_h3_mlx.text_encoder.MiniMaxH3TextEncoder", lambda *a, **k: None)
    monkeypatch.setattr(PipelineConfig, "from_model_index", lambda *_: PipelineConfig())
    pipeline = MiniMaxH3Pipeline.from_pretrained(tmp_path, verbose=False)
    pipeline._ensure_cache(mx.array([0.0, 0.5, 1.0]), True, False)
    assert not hasattr(pipeline.dit.blocks[0].adaln_proj.linear, "weight")
    pipeline._ensure_cache(mx.array([0.0, 0.5, 1.0]), True, False)
    assert len(loaded) == 1
    pipeline._ensure_cache(mx.array([0.0, 0.25, 0.75, 1.0]), True, False)
    assert len(loaded) == 2
    assert pipeline._cache.num_timesteps == 4
    assert pipeline._cache_builds == 2
    assert pipeline._cache_hits == 1
    assert not hasattr(pipeline.dit.blocks[0].adaln_proj.linear, "weight")


@pytest.mark.parametrize("transition", ["guided", "ic_lora", "spatial_dfr", "temporal_dfr"])
def test_ltx_transformer_swap_releases_weight_owners_before_loader(
    tmp_path, monkeypatch, transition,
):
    import weakref

    from ltx25_mlx.pipeline import LTX25DistilledPipeline

    class WeightedTransformer:
        pass

    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.transformer_path = tmp_path / "base.safetensors"
    pipeline.loras = ()
    pipeline.ic_loras = (("guide.safetensors", 1.0),)
    pipeline.dit = WeightedTransformer()
    original = weakref.ref(pipeline.dit)
    pipeline._loaded_transformer_path = pipeline.transformer_path
    pipeline._loaded_loras = pipeline.ic_loras if transition == "ic_lora" else ()
    pipeline.low_ram_streaming = False
    pipeline.feed_forward_backend = "reference_fp32"
    # Retain the wrapper exactly as generate_two_stage / temporal-round locals do.
    wrapper = pipeline._sampling_model(frozen_audio=transition == "temporal_dfr")
    assert wrapper.model is original()

    def load(*_args, **_kwargs):
        assert original() is None, "Previous sampling wrapper still owns transformer weights"
        return WeightedTransformer()

    monkeypatch.setattr("ltx25_mlx.pipeline.load_ltx25_transformer", load)
    monkeypatch.setattr("ltx25_mlx.sol_attention.configure_ltx25_sol_attention", lambda *a, **k: {})
    if transition == "guided":
        pipeline._load_transformer(extra_loras=(("distilled.safetensors", 1.0),))
    elif transition == "ic_lora":
        pipeline._load_transformer(include_ic_loras=False)
    elif transition == "spatial_dfr":
        pipeline._load_transformer(transformer_path=tmp_path / "detail.safetensors")
    else:
        pipeline._release_transformer()
        assert original() is None
        pipeline._load_transformer()
    assert original() is None
    assert wrapper.model is None


def test_direct_ltx_chain_rejects_cfg_pp_before_encoding():
    from ltx25_mlx.pipeline import LTX25DistilledPipeline

    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.encode_prompt_batch = lambda *a, **k: pytest.fail("Prompt encoding was started")
    with pytest.raises(ValueError, match="CFG"):
        pipeline.generate_chained_and_save(
            prompts=["one", "two"], output_path="unused.mp4", height=512, width=768,
            total_frames=121, window_count=2, overlap_frames=25, frame_rate=24, seed=0,
            stage1_sampler="euler_ancestral_cfg_pp", ic_lora_single_stage=True,
            stage2_steps=0,
        )


def test_h3_convenience_rejects_vision_load_without_multimodal_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "preflight_components", lambda *_: pytest.fail("Preflight started"))
    with pytest.raises(ValueError, match="text-only"):
        nodes.WeeToddH3Generate().generate(
            H3ModelSpec(str(tmp_path), load_vision=True), H3GenerationConfig(), "A scene.", "test",
        )


def test_ltx_chain_node_rejects_guided_before_component_validation():
    from ltx25_mlx.runtime import LTX25GenerationConfig
    from wee_todd_nodes.ltx25_nodes import WeeToddLTX25GenerateChained

    spec = SimpleNamespace(validate=lambda *a, **k: pytest.fail("Component validation started"))
    with pytest.raises(ValueError, match="guided"):
        WeeToddLTX25GenerateChained().generate(
            spec, LTX25GenerationConfig(pipeline_mode="guided"), 2, 25,
            "one", "two", "", "", "unused", True,
        )
