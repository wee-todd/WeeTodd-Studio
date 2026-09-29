"""Developer-only Float64 LoRA merge oracle; never used for native inference."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(size))
        self.offset = size + 8

    def read(self, name):
        record = self.header[name]
        dtype = {"BF16": "<u2", "F32": "<f4", "F16": "<f2", "U32": "<u4"}[record["dtype"]]
        data = np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            shape=tuple(record["shape"]),
            offset=self.offset + record["data_offsets"][0],
        )
        if record["dtype"] == "BF16":
            return (data.astype(np.uint32) << 16).view(np.float32)
        return np.array(data)

    def original_key(self, suffix):
        return next(key for key in self.header if key.endswith(suffix))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--older", type=Path, required=True)
    parser.add_argument("--newer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "paged_manifest.json").read_text())
    fixed = Checkpoint(args.root / manifest["fixed"]["file"])
    block = Checkpoint(args.root / manifest["layers"][0]["file"])
    adapters = [(Checkpoint(args.older), 0.3), (Checkpoint(args.newer), -0.2)]
    results = {}
    for name in [
        "proj_out.weight",
        "audio_proj_out.weight",
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.0.audio_attn1.to_q.weight",
    ]:
        source = block if name.startswith("transformer_blocks.") else fixed
        key = source.original_key("." + name)
        base = source.read(key)
        if source.header[key]["dtype"] == "U32":
            stem = key.removesuffix(".weight")
            scale = source.read(stem + ".scales").astype(np.float64)
            bias = source.read(stem + ".biases").astype(np.float64)
            lanes = base.view(np.uint8).reshape(base.shape[0], -1, 64)
            base = (
                (lanes * scale[..., None] + bias[..., None])
                .astype(np.float32)
                .reshape(base.shape[0], -1)
            )
        else:
            base = base.astype(np.float32)
        for adapter, strength in adapters:
            stem = adapter.original_key("." + name.removesuffix(".weight") + ".lora_A.weight")
            a = adapter.read(stem).astype(np.float64)
            b = adapter.read(stem.replace(".lora_A.", ".lora_B.")).astype(np.float64)
            metadata = adapter.header.get("__metadata__", {})
            # The installed files have standard global rank/alpha metadata, or
            # baked scaling. Unsupported conventions fail instead of guessing.
            alpha = metadata.get("lora_alpha")
            scale = (
                1.0
                if alpha is None or alpha in ["baked", "baked_scale"]
                else float(alpha) / float(metadata.get("lora_rank", a.shape[0]))
            )
            scale = float(np.float32(scale * float(np.float32(strength))))
            base = (base.astype(np.float64) + scale * (b @ a)).astype(np.float32)
        results[name] = base
    header = {}
    offset = 0
    for name, value in results.items():
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + value.nbytes],
        }
        offset += value.nbytes
    data = json.dumps(header).encode()
    with args.output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        for value in results.values():
            stream.write(value.tobytes())
    print(
        json.dumps({"output": str(args.output), "tensors": len(results), "payload_bytes": offset})
    )


if __name__ == "__main__":
    main()
