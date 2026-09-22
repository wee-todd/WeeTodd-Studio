"""Read Comfy INT8 H3 checkpoints in place through the native H3 block executor.

ConvRot uses a normalized regular Hadamard basis, not the Sylvester basis.
Only active blocks are decoded to native BF16/FP32; no converted files are written.
The storage contract is documented by Comfy-Org/comfy-quants and comfy-kitchen.
This module's header inspection does not import MLX or construct weighted models.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from pathlib import Path

from .config import DiTConfig

PREFIX = "model.diffusion_model."
MAX_HEADER = 16 * 1024 * 1024
ROW_CHUNK = 1024


def _header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing H3 safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER:
            raise ValueError("Invalid H3 safetensors header length")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid H3 safetensors header")
    return header, length + 8


def is_comfy_h3_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        header, _ = _header(path)
        return any(k.startswith(PREFIX + "blocks.") and k.endswith(".comfy_quant") for k in header)
    except (OSError, ValueError, struct.error):
        return False


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _expected_shapes(c):
    h, inner = c.hidden_size, c.inner_dim
    shapes = {}

    def linear(name, rows, columns, bias=False):
        shapes[name + ".weight"] = [rows, columns]
        if bias:
            shapes[name + ".bias"] = [rows]

    def block(prefix):
        for name in ("norm1", "norm2"):
            shapes[prefix + name + ".weight"] = [h]
        for name in ("q_norm", "k_norm"):
            shapes[prefix + "attn." + name + ".weight"] = [c.attention_head_dim]
        linear(prefix + "attn.qkv_proj", inner * 3, h)
        linear(prefix + "attn.out_proj", h, inner)
        linear(prefix + "mlp.fc1", c.ffn_hidden_size * 2, h)
        linear(prefix + "mlp.fc2", h, c.ffn_hidden_size)

    linear("video_patch_proj", h, c.video_patch_dim, True)
    linear("audio_patch_proj", h, c.audio_latents_dim, True)
    linear("condition_proj", h, c.text_dim, True)
    linear("time_embedder.proj_in", c.time_embed_hidden_size, c.timestep_input_dim, True)
    linear("time_embedder.proj_out", c.time_embed_dim, c.time_embed_hidden_size, True)
    linear("final_layer.adaln_proj.linear", c.final_adaln_out_features, c.time_embed_dim, True)
    linear("final_layer.video_out", c.video_patch_dim, h, True)
    linear("final_layer.audio_out", c.audio_latents_dim, h, True)
    shapes["final_layer.norm.weight"] = [h]
    shapes["token_refiner.final_norm.weight"] = [h]
    for index in range(c.token_refiner_num_layers):
        block(f"token_refiner.blocks.{index}.")
    for index in range(c.num_layers):
        block(f"blocks.{index}.")
        linear(f"blocks.{index}.adaln_proj.linear", c.adaln_out_features, c.time_embed_dim, True)
    return shapes


def _fp32(key):
    return key.startswith(
        (
            "video_patch_proj.",
            "audio_patch_proj.",
            "time_embedder.",
            "final_layer.video_out.",
            "final_layer.audio_out.",
        )
    )


def _inspect(path):
    from wee_todd_mlx.model_library import inspect_safetensors_header

    path = Path(path).expanduser().resolve(strict=True)
    identity = _identity(path.stat())
    inspect_safetensors_header(path)  # Validates complete payload, types, offsets and shapes.
    header, offset = _header(path)
    config = DiTConfig()
    records = {}
    for key, record in header.items():
        if key == "__metadata__":
            continue
        if not key.startswith(PREFIX):
            raise ValueError(f"Unexpected H3 checkpoint tensor: {key}")
        records[key[len(PREFIX) :]] = record
    quant = {}
    with path.open("rb") as stream:
        for key, record in records.items():
            if not key.endswith(".comfy_quant"):
                continue
            base = key.removesuffix(".comfy_quant")
            if (
                record["dtype"] != "U8"
                or len(record["shape"]) != 1
                or not 1 <= record["shape"][0] <= 4096
            ):
                raise ValueError(f"Invalid quantization marker: {key}")
            stream.seek(offset + record["data_offsets"][0])
            marker = json.loads(stream.read(record["shape"][0]))
            if not isinstance(marker, dict) or marker.get("format") != "int8_tensorwise":
                raise ValueError(f"Unsupported H3 quantization: {key}")
            if set(marker) - {"format", "convrot", "convrot_groupsize"}:
                raise ValueError(f"Unsupported H3 quantization options: {key}")
            rotated = marker.get("convrot", False)
            group = marker.get("convrot_groupsize", 256) if rotated else 0
            if type(rotated) is not bool or (
                rotated and (type(group) is not int or group not in (4, 16, 64, 256, 1024))
            ):
                raise ValueError(f"Unsupported H3 rotation: {key}")
            weight = records.get(base + ".weight", {})
            scale = records.get(base + ".weight_scale", {})
            shape = weight.get("shape", [])
            if weight.get("dtype") != "I8" or len(shape) != 2 or (group and shape[1] % group):
                raise ValueError(f"Invalid INT8 H3 weight shape: {base}")
            if scale.get("dtype") != "F32" or scale.get("shape") != [shape[0], 1]:
                raise ValueError(f"Invalid INT8 H3 row scales: {base}")
            quant[base + ".weight"] = group
    expected = _expected_shapes(config)
    auxiliary = {
        key
        for weight in quant
        for key in (
            weight.removesuffix(".weight") + ".weight_scale",
            weight.removesuffix(".weight") + ".comfy_quant",
        )
    }
    actual = set(records) - auxiliary - {"rope.inv_freq"}
    if not quant or actual != set(expected):
        raise ValueError("H3 tensor inventory does not match the native transformer architecture")
    for key, shape in expected.items():
        record = records[key]
        if record["shape"] != shape or record["dtype"] not in ("F16", "BF16", "F32", "I8"):
            raise ValueError(f"Incompatible H3 tensor: {key}")
        if (record["dtype"] == "I8") != (key in quant):
            raise ValueError(f"Missing H3 quantization metadata: {key}")
    if _identity(path.stat()) != identity:
        raise ValueError("H3 checkpoint changed during inspection")
    sizes = {key: math.prod(shape) * (4 if _fp32(key) else 2) for key, shape in expected.items()}
    report = dict(
        tensor_count=len(records),
        tensor_bytes=sum(sizes.values()),
        fixed_bytes=sum(size for key, size in sizes.items() if not key.startswith("blocks.")),
        window_bytes=max(
            sum(size for key, size in sizes.items() if key.startswith(f"blocks.{i}."))
            for i in range(config.num_layers)
        ),
        adaln_bytes=sum(
            size
            for key, size in sizes.items()
            if key.startswith("blocks.") and ".adaln_proj." in key
        ),
    )
    # Concatenating decoded row chunks can briefly retain the chunks and the final
    # tensor together. Also reserve three FP32 row workspaces for inverse rotation.
    quant_workspace = [
        sizes[key] + min(ROW_CHUNK, expected[key][0]) * expected[key][1] * 12
        for key in quant
    ]
    raw_workspace = [
        2 * (records[key]["data_offsets"][1] - records[key]["data_offsets"][0]) + sizes[key]
        for key in expected
        if key not in quant
    ]
    report["decode_workspace_bytes"] = max(quant_workspace + raw_workspace)
    return path, identity, config, records, offset, quant, expected, report


def describe_comfy_h3(path):
    return _inspect(path)[-1]


def inverse_convrot(values, group):
    """Invert the regular H4 Kronecker basis using its base-four digit signs."""
    import mlx.core as mx
    import numpy as np

    if group not in (4, 16, 64, 256, 1024) or values.shape[-1] % group:
        raise ValueError("Invalid ConvRot group size")
    indices = np.arange(group)
    signs = np.ones((group, group), dtype=np.float32)
    divisor = 1
    while divisor < group:
        digits = (indices // divisor) % 4
        signs *= np.where(digits[:, None] + digits[None, :] == 3, -1.0, 1.0)
        divisor *= 4
    basis = mx.array(signs / math.sqrt(group))
    return (values.reshape(-1, group) @ basis).reshape(values.shape)


class ComfyH3TensorSource:
    """One read-only file descriptor with bounded per-tensor decoding workspaces."""

    def __init__(self, inspection):
        (
            self.path,
            self.identity,
            self.config,
            self.records,
            self.offset,
            self.quant,
            self.expected,
            self.description,
        ) = inspection
        self.fd = os.open(self.path, os.O_RDONLY)
        self.closed = False
        self.payload_bytes_read = 0
        self.peak_read_bytes = 0
        self.quantized_tensors_read = 0
        try:
            self._check()
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.closed:
            raise ValueError("H3 checkpoint is closed")
        if (
            _identity(os.fstat(self.fd)) != self.identity
            or _identity(self.path.stat()) != self.identity
        ):
            raise ValueError("H3 checkpoint changed after inspection")

    def _bytes(self, key, start=0, count=None):
        self._check()
        record = self.records[key]
        low, high = record["data_offsets"]
        count = high - low - start if count is None else count
        if not 0 <= start <= start + count <= high - low:
            raise ValueError("Invalid H3 tensor read span")
        data = os.pread(self.fd, count, self.offset + low + start)
        if len(data) != count:
            raise ValueError("H3 checkpoint is incomplete")
        self._check()
        self.payload_bytes_read += count
        self.peak_read_bytes = max(self.peak_read_bytes, count)
        return data

    def read(self, key):
        import mlx.core as mx
        import numpy as np

        record = self.records[key]
        dtype = mx.float32 if _fp32(key) else mx.bfloat16
        if key in self.quant:
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            scales = mx.array(np.frombuffer(self._bytes(scale_key), dtype="<f4").reshape(-1, 1))
            if not bool(mx.all(mx.isfinite(scales) & (scales > 0)).item()):
                raise ValueError(f"Invalid H3 scales: {key}")
            rows, columns = record["shape"]
            chunks = []
            for start in range(0, rows, ROW_CHUNK):
                count = min(ROW_CHUNK, rows - start)
                data = self._bytes(key, start * columns, count * columns)
                q = mx.array(np.frombuffer(data, dtype=np.int8).reshape(count, columns))
                decoded = q.astype(mx.float32) * scales[start : start + count]
                if self.quant[key]:
                    decoded = inverse_convrot(decoded, self.quant[key])
                chunk = decoded.astype(dtype)
                mx.eval(chunk)
                chunks.append(chunk)
                del data, q, decoded, chunk
            value = mx.concatenate(chunks, axis=0)
            mx.eval(value)
            del chunks
            self.quantized_tensors_read += 1
        else:
            np_dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[record["dtype"]]
            value = mx.array(
                np.frombuffer(self._bytes(key), dtype=np_dtype).reshape(record["shape"])
            )
            if record["dtype"] == "BF16":
                value = value.view(mx.bfloat16)
            value = value.astype(dtype)
            mx.eval(value)
        if key.endswith(".attn.qkv_proj.weight"):
            # Comfy stores contiguous Q, K, V; the native attention projection
            # groups Q, K, V within each head, including the token refiner.
            value = (
                value.reshape(
                    3, self.config.num_attention_heads, self.config.attention_head_dim, -1
                )
                .transpose(1, 0, 2, 3)
                .reshape(value.shape)
            )
            mx.eval(value)
        self._check()
        return value

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def load_comfy_h3_dit(path, *, window_size=1):
    """Attach the existing H3 sampler/block executor to an unchanged INT8 file."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .dit import MiniMaxH3DiT
    from .paged_checkpoint import (
        PagedBlockExecutor,
        PagedCheckpointManifest,
        PagedTensorStore,
        PageRecord,
    )

    inspection = _inspect(path)
    source = ComfyH3TensorSource(inspection)
    config = source.config
    manifest = PagedCheckpointManifest(
        source.path.parent,
        config.num_layers,
        source.path.stat().st_size,
        PageRecord("fixed", 0, 0, ""),
        tuple(PageRecord(str(i), 0, 0, "") for i in range(config.num_layers)),
    )

    class DirectStore(PagedTensorStore):
        def __init__(self):
            super().__init__(manifest)
            self.source = source

        def _load_record(self, record, *, skip_adaln):
            source._check()
            if self._cache_enabled and record.file in self._retained:
                self.raw_cache_hits += 1
                self.adaln_bytes_avoided += self._retained_adaln_bytes[record.file]
                return self._retained[record.file]
            started = time.perf_counter()
            before = source.payload_bytes_read
            keys = [
                key
                for key in source.expected
                if (
                    not key.startswith("blocks.")
                    if record.file == "fixed"
                    else key.startswith(f"blocks.{record.file}.")
                )
            ]
            avoided = sum(
                math.prod(source.records[k]["shape"]) * 2
                for k in keys
                if skip_adaln and ".adaln_proj." in k
            )
            try:
                values = {
                    k: source.read(k) for k in keys if not (skip_adaln and ".adaln_proj." in k)
                }
            finally:
                self.file_tensor_bytes += source.payload_bytes_read - before
                self.disk_load_seconds += time.perf_counter() - started
            self.disk_page_loads += 1
            self.adaln_bytes_avoided += avoided
            size = sum(v.nbytes for v in values.values())
            if self._cache_enabled:
                self.raw_cache_misses += 1
                if record.file != "fixed" and self.retained_bytes + size <= self.cache_budget_bytes:
                    self._retained[record.file] = values
                    self._retained_adaln_bytes[record.file] = avoided
                    self.retained_bytes += size
                    self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
            return values

    class DirectExecutor(PagedBlockExecutor):
        def close(self):
            try:
                super().close()
                self.store.release()
            finally:
                source.close()

        def report(self):
            return {
                **super().report(),
                "weight_source_format": "comfy_int8_tensorwise",
                "weight_decode_backend": "mlx",
                "execution_dtype": "native_bf16_fp32",
                "source_payload_bytes_read": source.payload_bytes_read,
                "source_peak_read_bytes": source.peak_read_bytes,
                "source_quantized_tensors_read": source.quantized_tensors_read,
            }

    executor = None
    try:
        store = DirectStore()
        model = MiniMaxH3DiT(config)
        model.blocks = []
        fixed = store.load_fixed()
        expected = {k: list(v.shape) for k, v in tree_flatten(model.parameters())}
        if {k: list(v.shape) for k, v in fixed.items()} != expected:
            raise ValueError("Comfy H3 fixed tensors do not match the native architecture")
        model.update(tree_unflatten(list(fixed.items())))
        mx.eval(model.parameters())
        fixed.clear()
        store.release()
        executor = DirectExecutor(manifest, config, None, window_size, prefetch=False)
        executor.store = store
        model.paged_blocks = executor
        return model
    except BaseException:
        if executor is not None:
            executor.close()
        else:
            source.close()
        raise
