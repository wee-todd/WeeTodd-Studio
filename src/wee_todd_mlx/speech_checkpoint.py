"""Lightweight local speech checkpoint inspection, without tensor allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yue2_mlx.checkpoint import tensor_header


def layout_signature(inventory):
    value = {k: [v["shape"], v["dtype"]] for k, v in inventory.items()}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def architecture(value):
    keys = {
        "model_type",
        "tts_model_type",
        "tts_model_size",
        "vocab_size",
        "text_vocab_size",
        "dim",
        "text_dim",
        "hidden_size",
        "text_hidden_size",
        "intermediate_size",
        "n_layer",
        "n_head",
        "n_local_heads",
        "head_dim",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_codebooks",
        "num_code_groups",
        "norm_eps",
        "rms_norm_eps",
        "rope_base",
        "rope_theta",
        "rope_scaling",
        "enc_dim",
        "attention_qk_norm",
        "attention_qkv_bias",
        "attention_o_bias",
        "attention_bias",
        "codec_language_id",
        "max_seq_len",
        "max_position_embeddings",
    }
    return {
        k: (
            v
            if k in {"spk_id", "spk_is_dialect", "codec_language_id"}
            else architecture(v)
            if isinstance(v, dict)
            else v
        )
        for k, v in value.items()
        if isinstance(v, dict) or k in keys or k.endswith("_token_id") or k.startswith("codec_")
    }


def validate_layout(config, inventory, codec_inventory, codec_config=None):
    models = json.loads(Path(__file__).with_name("speech_layouts.json").read_text())
    model_key, codec_key = layout_signature(inventory), layout_signature(codec_inventory)
    match = next((m for m in models if m["model"] == model_key and m["codec"] == codec_key), None)
    if (
        match is None
        or architecture(config) != architecture(match["config"])
        or codec_config != match.get("codec_config")
    ):
        raise ValueError(
            "Unsupported speech tensor layout or architecture. "
            "Choose a qualified checkpoint from speech setup."
        )


def inspect(model_path, *, engine, precision="auto"):
    if engine not in {"fishS2Pro", "qwen3TTS"}:
        raise ValueError("Choose a supported speech engine")
    root = Path(model_path).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    expected = "fish_qwen3_omni" if engine == "fishS2Pro" else "qwen3_tts"
    if config.get("model_type") != expected:
        raise ValueError(f"Choose a {expected} checkpoint")
    quant = config.get("quantization")
    actual = "8bit" if quant else "bf16"
    if precision not in {"auto", actual}:
        raise ValueError("Requested precision differs from the installed checkpoint")
    if quant and (quant.get("bits"), quant.get("group_size"), quant.get("mode", "affine")) != (
        8,
        64,
        "affine",
    ):
        raise ValueError("Speech requires qualified affine 8-bit/group-64 weights or BF16")
    codec_config = None
    if engine == "qwen3TTS":
        kind, size = config.get("tts_model_type"), config.get("tts_model_size")
        if not (kind == "base" and size in {"1b7", "0b6"}) and not (
            kind == "custom_voice" and size == "1b7"
        ):
            raise ValueError("Choose Qwen3-TTS 12Hz Base or qualified 1.7B CustomVoice")
        codec = root / "speech_tokenizer/model.safetensors"
        codec_config = json.loads((root / "speech_tokenizer/config.json").read_text())
        if codec_config.get("model_type") != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Qwen Base requires its 12Hz encoder and decoder")
        required = ["tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]
        layout = "qwen-custom-voice" if kind == "custom_voice" else "qwen-base"
        rate = 24000
    else:
        if (root / "codec-mlx/model.safetensors").is_file():
            codec = root / "codec-mlx/model.safetensors"
            layout = "fish-bundled"
        else:
            codec = root / "codec.safetensors"
            layout = "fish-flat"
        if not codec.is_file() and (root / "codec.pth").is_file():
            raise ValueError(
                "Convert the original codec.pth to a qualified safetensors codec first"
            )
        required = ["tokenizer.json", "tokenizer_config.json"]
        rate = 44100
    for name in required:
        if not (root / name).is_file():
            raise ValueError(f"Missing speech model file: {name}")
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise ValueError("No speech model weights were found")
    inventory = {}
    owner = {}
    for filename in files:
        entries = tensor_header(filename)
        if set(inventory) & entries.keys():
            raise ValueError("Duplicate speech tensors across checkpoint shards")
        inventory.update(entries)
        owner.update(dict.fromkeys(entries, filename.name))
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text()).get("weight_map", {})
        if index != owner:
            raise ValueError("Checkpoint index does not match its tensor shards")
    codec_inventory = tensor_header(codec, additional_widths={"BOOL": 1, "I64": 8, "I32": 4})
    if engine == "qwen3TTS":
        if kind == "base" and not any(k.startswith("speaker_encoder.") for k in inventory):
            raise ValueError("Qwen Base speaker encoder is missing")
        if kind == "custom_voice" and any(k.startswith("speaker_encoder.") for k in inventory):
            raise ValueError("Qwen CustomVoice must not contain a reference speaker encoder")
        if not all(
            any(k.startswith(prefix) for k in codec_inventory)
            for prefix in ("encoder.", "decoder.")
        ):
            raise ValueError("Qwen reference cloning requires codec encoder and decoder")
    validate_layout(config, inventory, codec_inventory, codec_config)
    all_files = [root / "config.json", *files, codec, *(root / name for name in required)]
    if engine == "qwen3TTS":
        all_files.append(root / "speech_tokenizer/config.json")
    digest = hashlib.sha256()
    for filename in all_files:
        digest.update(str(filename.relative_to(root)).encode())
        with filename.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return dict(
        engine=engine,
        root=str(root),
        config=config,
        precision=actual,
        layout=layout,
        model_files=[str(p) for p in files],
        codec_path=str(codec),
        sample_rate=rate,
        tensor_inventory=inventory,
        codec_inventory=codec_inventory,
        identity=digest.hexdigest(),
    )
