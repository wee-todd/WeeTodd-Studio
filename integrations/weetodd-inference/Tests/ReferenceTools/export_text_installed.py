"""Opt-in FP32 MLX oracle for installed Q8 Gemma and trained LTX connectors.

All checkpoint paths are arguments. This is qualification only, never production
execution. Decode affine weights in FP32 to match the native slab reader.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.gemma4_text import DecoderLayer, ModelArgs
from safetensors import safe_open
from tokenizers import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gemma-root", type=Path, required=True)
    p.add_argument("--connector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--component-output", type=Path)
    p.add_argument("--prompt", default="A red fox.")
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    if a.prompt_file:
        a.prompt = a.prompt_file.read_text()
    sys.path.insert(0, str(a.project / "src"))
    from ltx25_mlx.gemma_encoder import load_gemma4_feature_extractor
    from ltx25_mlx.gemma_pack import gemma4_mlx_model_config

    m = json.loads((a.gemma_root / "paged_manifest.json").read_text())
    cfg = gemma4_mlx_model_config(m["metadata"]["gemma_config"])["text_config"]
    args = ModelArgs.from_dict(cfg)
    fixed = a.gemma_root / m["fixed"]["file"]
    with safe_open(fixed, framework="numpy") as f:
        tokenizer = Tokenizer.from_buffer(f.get_tensor("tokenizer_json").tobytes())
        ids = tokenizer.encode(a.prompt.strip()).ids
        if not ids or ids[0] != 2:
            ids = [2] + ids
    raw = mx.load(str(fixed))
    hidden = (
        raw["model.embed_tokens.weight"][mx.array(ids)].astype(mx.float32) * args.hidden_size**0.5
    )
    mx.eval(hidden)
    del raw
    hidden = hidden[None, :, :]
    states = [hidden]
    T = len(ids)
    rows = mx.arange(T)[:, None]
    cols = mx.arange(T)[None, :]
    for index, page in enumerate(m["layers"]):
        layer = DecoderLayer(args, index)
        raw = mx.load(str(a.gemma_root / page["file"]))
        prefix = f"model.layers.{index}."
        weights = {k.removeprefix(prefix): v for k, v in raw.items()}
        dense = {}
        for name, value in weights.items():
            if name.endswith((".scales", ".biases")):
                continue
            if value.dtype == mx.uint32:
                stem = name.removesuffix(".weight")
                value = mx.dequantize(
                    value,
                    weights[stem + ".scales"].astype(mx.float32),
                    weights[stem + ".biases"].astype(mx.float32),
                    group_size=64,
                    bits=8,
                )
            dense[name] = value.astype(mx.float32)
        layer.load_weights(list(dense.items()), strict=True)
        mx.eval(layer.parameters())
        mask = cols <= rows
        if layer.layer_type == "sliding_attention":
            mask = mask & (cols > rows - args.sliding_window)
        hidden, _, _ = layer(hidden, mask=mask)
        mx.eval(hidden)
        states.append(hidden)
        print(f"ORACLE gemma {index + 1}/48", flush=True)
        del layer, raw, weights, dense
        gc.collect()
        mx.clear_cache()
    pack_weights = mx.load(str(fixed))
    extractor = load_gemma4_feature_extractor(
        a.gemma_root, connector_path=a.connector, _pack_weights=pack_weights
    )
    del pack_weights
    # All loaded dense projections/connectors must match the FP32 native decoder.

    extractor.update(tree_map(lambda x: x.astype(mx.float32), extractor.parameters()))
    if a.component_output:
        from ltx25_mlx.transformer import precompute_rope_freqs_float64

        components = {f"gemma_input_{i}": states[i] for i in (0, 5, 47)}
        components.update({f"gemma_expected_{i}": states[i + 1] for i in (0, 5, 47)})
        stacked = mx.stack(states, axis=-1)
        stacked = stacked * mx.rsqrt(mx.mean(stacked * stacked, axis=2, keepdims=True) + 1e-6)
        stacked = stacked.reshape(1, T, -1)
        projections = extractor.connector.text_embedding_projection(stacked)
        for name, projection, width in zip(
            ("video", "audio"), projections, (4096, 2048), strict=True
        ):
            connector = getattr(extractor.connector, name + "_embeddings_connector")
            registers = mx.tile(connector.learnable_registers, (8, 1))[T:]
            hidden = mx.concatenate([projection, registers[None]], axis=1)
            positions = mx.arange(1024, dtype=mx.float32)[None, :, None]
            frequencies = precompute_rope_freqs_float64(
                positions,
                inner_dim=width,
                num_heads=32,
                theta=10000.0,
                max_pos=[4096],
                rope_type="split",
            )
            expected = connector.transformer_1d_blocks[0](hidden, rope_freqs=frequencies)
            mx.eval(hidden, expected)
            components[name + "_connector_input"] = hidden
            components[name + "_connector_expected"] = expected
        a.component_output.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(a.component_output), components)
    video, audio = extractor(states, attention_mask=mx.ones((1, T), dtype=mx.int32))
    mx.eval(video, audio)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(a.output), {"video": video, "audio": audio, "token_ids": mx.array(ids, dtype=mx.int32)}
    )
    print(
        json.dumps(
            {"ids": ids, "video": video.shape, "audio": audio.shape, "output": str(a.output)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
