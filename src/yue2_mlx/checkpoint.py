"""Read-only merged checkpoint inventory validation and explicit MLX loaders.

Headers are checked before loading tensors. Remote Python and inference packages
are never imported. Split AR/NAR releases require a separately qualified mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from .config import ABC_DEFAULTS, SEMANTIC_DEFAULTS, integer, number, sampling


def tensor_header(filename, *, additional_widths=None):
    filename = Path(filename)
    size = filename.stat().st_size
    with filename.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid safetensors header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, size - 8):
            raise ValueError("Invalid safetensors header size")
        raw = stream.read(length)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    widths = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}
    widths.update(additional_widths or {})
    spans = []
    for name, entry in header.items():
        shape, dtype, offsets = entry.get("shape"), entry.get("dtype"), entry.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(d) is not int or d < 1 for d in shape)
            or dtype not in widths
        ):
            raise ValueError(f"Invalid tensor shape or dtype: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(i) is not int for i in offsets)
        ):
            raise ValueError(f"Invalid tensor offsets: {name}")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * widths[dtype] or end > size - 8 - length:
            raise ValueError(f"Invalid or truncated tensor payload: {name}")
        spans.append((start, end))
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise ValueError("Overlapping or incomplete tensor payload")
        previous = end
    if previous != size - 8 - length:
        raise ValueError("Unexpected trailing tensor payload")
    return header


def _model_shapes(c):
    h, i, d, v = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["vocab_size"]
    q, k = c["num_attention_heads"] * d, c["num_key_value_heads"] * d
    shapes = {
        "model.embed_tokens.weight": (v, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (v, h),
        "llm2vae.weight": (c["latent_dim"], h),
        "llm2vae.bias": (c["latent_dim"],),
        "vae2llm.weight": (h, c["latent_dim"]),
        "vae2llm.bias": (h,),
        "latent_pos_embed.pe": (c["max_latent_frames"], h),
        "time_embedder.fc1.weight": (h, 256),
        "time_embedder.fc1.bias": (h,),
        "time_embedder.fc2.weight": (h, h),
        "time_embedder.fc2.bias": (h,),
    }
    linears = {"lm_head", "llm2vae", "vae2llm", "time_embedder.fc1", "time_embedder.fc2"}
    for layer in range(c["num_hidden_layers"]):
        root = f"model.layers.{layer}."
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "nar_input_layernorm",
            "nar_pre_mlp_layernorm",
        ):
            shapes[root + norm + ".weight"] = (h,)
        for name in ("self_attn", "nar_self_attn"):
            for projection, shape in [
                ("q_proj", (q, h)),
                ("k_proj", (k, h)),
                ("v_proj", (k, h)),
                ("o_proj", (h, q)),
            ]:
                module = root + name + "." + projection
                shapes[module + ".weight"] = shape
                linears.add(module)
            for norm in ("q_norm", "k_norm"):
                shapes[root + name + "." + norm + ".weight"] = (d,)
        for name in ("mlp", "nar_mlp"):
            for projection, shape in [
                ("gate_proj", (i, h)),
                ("up_proj", (i, h)),
                ("down_proj", (h, i)),
            ]:
                module = root + name + "." + projection
                shapes[module + ".weight"] = shape
                linears.add(module)
    return shapes, linears


def _vae_shapes(config):
    channels = [config["channels"] * i for i in [1, *config["c_mults"]]]
    shapes = {}

    def conv(name, source, target, kernel, bias=True):
        shapes[name + ".weight"] = (target, kernel, source)
        if bias:
            shapes[name + ".bias"] = (target,)

    def snake(name, width):
        shapes[name + ".alpha"] = shapes[name + ".beta"] = (width,)

    conv("layers.0", config["latent_dim"], channels[-1], 7)
    for block, i in enumerate(reversed(range(len(config["strides"]))), 1):
        root = f"layers.{block}.layers."
        snake(root + "0", channels[i + 1])
        conv(root + "1", channels[i + 1], channels[i], 2 * config["strides"][i])
        for residual in range(2, 5):
            prefix = root + str(residual) + ".layers."
            snake(prefix + "0", channels[i])
            snake(prefix + "2", channels[i])
            conv(prefix + "1", channels[i], channels[i], 7)
            conv(prefix + "3", channels[i], channels[i], 1)
    end = len(config["strides"]) + 1
    snake(f"layers.{end}", channels[0])
    conv(f"layers.{end + 1}", channels[0], config["out_channels"], 7, False)
    return shapes


def _inventory(header, shapes, dtypes, label):
    missing, extra = set(shapes) - set(header), set(header) - set(shapes)
    if missing or extra:
        raise ValueError(
            f"{label} inventory mismatch; missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]}"
        )
    for name, shape in shapes.items():
        if tuple(header[name]["shape"]) != shape or header[name]["dtype"] != dtypes[name]:
            raise ValueError(
                f"{label} incompatible tensor {name}: expected {shape}/{dtypes[name]}, "
                f"got {header[name]}"
            )


def inspect_checkpoint(model_path, vae_path=None, precision="auto"):
    root = Path(model_path).expanduser().resolve()
    vae_root = Path(vae_path).expanduser().resolve() if vae_path else root
    required = {
        "model": root / "model.safetensors",
        "config": root / "config.json",
        "tokenizer": root / "qwen.tiktoken",
        "generation": root / "yue2_generation_config.json",
        "vae": vae_root / "vae.safetensors",
        "vae_config": vae_root / "vae_config.json",
    }
    for role, filename in required.items():
        if not filename.is_file():
            raise ValueError(
                f"Merged YuE2 layout requires {role}: {filename}; "
                "split/foreign checkpoints are unsupported"
            )
    c = json.loads(required["config"].read_text())
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "latent_dim",
        "max_latent_frames",
    ):
        integer(c.get(name), f"config.{name}", 1, 1000000)
    for name in ("rope_theta", "rms_norm_eps", "timestep_shift"):
        number(c.get(name), f"config.{name}", 0, float("inf"), exclusive_low=True)
    if (
        c.get("model_type") != "yue2"
        or c.get("latent_type") != "vae"
        or c.get("tie_word_embeddings") is not False
    ):
        raise ValueError("Unsupported YuE2 architecture")
    if c["vocab_size"] != 184704 or c["latent_dim"] != 64 or c["max_position_embeddings"] != 24576:
        raise ValueError("Unsupported YuE2 vocabulary, latent dimension or context")
    if (
        c["head_dim"] % 2
        or c["num_attention_heads"] % c["num_key_value_heads"]
        or c.get("rope_scaling")
    ):
        raise ValueError("Unsupported YuE2 attention configuration")
    quant = c.get("quantization")
    effective = "bf16"
    if quant:
        if (
            quant.get("mode", "affine") != "affine"
            or quant.get("bits") not in (4, 8)
            or quant.get("group_size") not in (32, 64, 128)
        ):
            raise ValueError("Unsupported YuE2 quantization")
        if quant.get("nar_bits", quant["bits"]) not in (4, 8, 16):
            raise ValueError("Unsupported NAR quantization")
        effective = f"{quant['bits']}bit"
    if precision not in ("auto", effective):
        raise ValueError(f"Requested {precision} precision does not match checkpoint {effective}")
    header = tensor_header(required["model"])
    shapes, linears = _model_shapes(c)
    dtypes = {name: "BF16" for name in shapes}
    quantized = []
    for module in linears:
        weight = module + ".weight"
        if header.get(weight, {}).get("dtype") == "U32":
            if not quant:
                raise ValueError("Packed tensors require quantization configuration")
            bits = quant.get("nar_bits", quant["bits"]) if ".nar_" in module else quant["bits"]
            rows, width = shapes[weight]
            group = quant["group_size"]
            if bits not in (4, 8) or width % group or width % (32 // bits):
                raise ValueError(f"Unsupported packed shape {module}")
            shapes[weight] = (rows, width // (32 // bits))
            dtypes[weight] = "U32"
            for suffix in ("scales", "biases"):
                shapes[module + "." + suffix] = (rows, width // group)
                dtypes[module + "." + suffix] = "BF16"
            quantized.append(module)
    if bool(quant) != bool(quantized):
        raise ValueError("Quantization metadata does not match tensor inventory")
    _inventory(header, shapes, dtypes, "transformer")
    vc = json.loads(required["vae_config"].read_text())
    dc = vc.get("decoder_config", {})
    expected = dict(
        latent_dim=64,
        out_channels=2,
        strides=[2, 2, 4, 4, 5, 6],
        c_mults=[1, 2, 4, 8, 16, 32],
        channels=64,
    )
    if (
        any(dc.get(k) != v for k, v in expected.items())
        or vc.get("sample_rate") != 48000
        or dc.get("final_tanh", False)
        or dc.get("use_filter", False)
        or dc.get("use_snake") is not True
    ):
        raise ValueError("Unsupported VAE decoder configuration")
    vh = tensor_header(required["vae"])
    vs = _vae_shapes(dc)
    _inventory(vh, vs, {name: "F32" for name in vs}, "VAE")
    gen = json.loads(required["generation"].read_text())
    if (
        gen.get("ode_method") != "midpoint"
        or gen.get("context") != 24576
        or gen.get("version") != "yue2-native-v1"
    ):
        raise ValueError("Unsupported generation protocol")
    sampling(gen.get("abc", {}), ABC_DEFAULTS, "checkpoint.abc")
    sampling(gen.get("semantic", {}), SEMANTIC_DEFAULTS, "checkpoint.semantic")
    integer(gen.get("ode_steps"), "checkpoint.ode_steps", 1, 4096)
    identities = {}
    for role, filename in required.items():
        stat = filename.stat()
        identities[role] = dict(path=str(filename), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if role not in ("model", "vae"):
            identities[role]["sha256"] = hashlib.sha256(filename.read_bytes()).hexdigest()
    return dict(
        layout="merged-mlx",
        precision=effective,
        config=c,
        vae_config=vc,
        generation_config=gen,
        files={k: str(v) for k, v in required.items()},
        identities=identities,
        quantized_modules=sorted(quantized),
        model_tensors=len(header),
        vae_tensors=len(vh),
    )


def load_model(inspection):
    import mlx.core as mx
    import mlx.nn as nn

    from .model import YuE2Model

    model = YuE2Model(inspection["config"])
    quant = inspection["config"].get("quantization")
    if quant:
        selected = set(inspection["quantized_modules"])

        def predicate(name, module):
            if name not in selected:
                return False
            bits = quant.get("nar_bits", quant["bits"]) if ".nar_" in name else quant["bits"]
            return dict(bits=bits, group_size=quant["group_size"])

        nn.quantize(model, class_predicate=predicate)
    model.load_weights(inspection["files"]["model"], strict=True)
    mx.eval(model.parameters())
    return model


def load_vae(inspection):
    import mlx.core as mx

    from .vae import Decoder

    model = Decoder(inspection["vae_config"]["decoder_config"])
    model.load_weights(inspection["files"]["vae"], strict=True)
    mx.eval(model.parameters())
    return model
