"""Independent installed MLX CPU connector oracle with exact saved GPU inputs."""

import argparse
import gc
import sys
from pathlib import Path

import mlx.core as mx
from ltx_core_mlx.text_encoders.gemma.embeddings_connector import ConnectorTransformerBlock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    from ltx25_mlx.transformer import precompute_rope_freqs_float64

    outputs = {}
    with mx.stream(mx.cpu):
        references = mx.load(str(args.components))
        for name, width in [("video", 4096), ("audio", 2048)]:
            layer = ConnectorTransformerBlock(width, 32, width // 32)
            packed = mx.load(str(args.connector))
            prefix = f"model.diffusion_model.{name}_embeddings_connector.transformer_1d_blocks.0."
            weights = [
                (k.removeprefix(prefix), v.astype(mx.float32))
                for k, v in packed.items()
                if k.startswith(prefix)
            ]
            layer.load_weights(weights, strict=True)
            mx.eval(layer.parameters())
            del packed, weights
            positions = mx.arange(1024, dtype=mx.float32)[None, :, None]
            frequencies = precompute_rope_freqs_float64(
                positions,
                inner_dim=width,
                num_heads=32,
                theta=10000.0,
                max_pos=[4096],
                rope_type="split",
            )
            output = layer(references[name + "_connector_input"], rope_freqs=frequencies)
            mx.eval(output)
            outputs[name] = output
            print(f"CPU connector {name} complete", flush=True)
            del layer
            gc.collect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(args.output), outputs)


if __name__ == "__main__":
    main()
