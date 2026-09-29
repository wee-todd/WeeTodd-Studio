"""Test-only independent dense LoRA fusion oracle for the Swift MLX factor path."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import mlx.core as mx
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from mlx.utils import tree_flatten

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--checkpoint", type=Path)
a = p.parse_args()
vd, ad, heads, vh, ah = (4096, 2048, 32, 128, 64) if a.checkpoint else (32, 16, 2, 16, 8)
nv, na, nt = 5, 3, 4
config = dict(
    videoDimension=vd,
    audioDimension=ad,
    heads=heads,
    videoHeadDimension=vh,
    audioHeadDimension=ah,
    videoTokens=nv,
    audioTokens=na,
    textTokens=nt,
)
block = BasicAVTransformerBlock(
    video_dim=vd,
    audio_dim=ad,
    video_num_heads=heads,
    audio_num_heads=heads,
    video_head_dim=vh,
    audio_head_dim=ah,
    av_cross_num_heads=heads,
    av_cross_head_dim=ah,
    ff_bias=False,
    audio_ff_bias=True,
)


def synthetic(name, shape):
    count = math.prod(shape)
    seed = sum(name.encode())
    base = [((i * 17 + seed) % 31 - 15) / 128 for i in range(count)]
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        base = [x + 1 for x in base]
    return mx.array(base, dtype=mx.float32).reshape(shape)


source = mx.load(str(a.checkpoint)) if a.checkpoint else None
weights = []
for name, old in tree_flatten(block.parameters()):
    if source is None:
        value = synthetic(name, old.shape)
    else:
        key = "model.diffusion_model.transformer_blocks.0." + name
        key = key.replace(".to_out.", ".to_out.0.")
        key = key.replace(".ff.proj_in.", ".ff.net.0.proj.").replace(".ff.proj_out.", ".ff.net.2.")
        key = key.replace(".audio_ff.proj_in.", ".audio_ff.net.0.proj.").replace(
            ".audio_ff.proj_out.", ".audio_ff.net.2."
        )
        value = source[key]
        if value.dtype == mx.uint32:
            stem = key[:-7]
            value = mx.dequantize(
                value,
                source[stem + ".scales"].astype(mx.float32),
                source[stem + ".biases"].astype(mx.float32),
                group_size=64,
                bits=8,
            )
        else:
            value = value.astype(mx.float32)
    assert value.shape == old.shape, (name, value.shape, old.shape)
    mx.eval(value)
    weights.append((name, value))
block.load_weights(weights, strict=True)
# Independent dense fusion oracle for the Swift factorized adapter path.
down = mx.array([(i % 7 - 3) / 32 for i in range(2 * vd * 4)], dtype=mx.float32).reshape(2, vd * 4)
up = mx.array([(i % 5 - 2) / 64 for i in range(vd * 2)], dtype=mx.float32).reshape(vd, 2)
block.ff.proj_out.weight = block.ff.proj_out.weight + 0.25 * (up @ down)

shapes = dict(
    video=[nv, vd],
    audio=[na, ad],
    video_modulation=[1, 9 * vd],
    audio_modulation=[1, 9 * ad],
    video_prompt_modulation=[1, 2 * vd],
    audio_prompt_modulation=[1, 2 * ad],
    video_av_modulation=[1, 4 * vd],
    audio_av_modulation=[1, 4 * ad],
    video_av_gate=[1, vd],
    audio_av_gate=[1, ad],
    video_text=[nt, vd],
    audio_text=[nt, ad],
)
for name, n, h in [
    ("video", nv, vh),
    ("audio", na, ah),
    ("video_cross", nv, ah),
    ("audio_cross", na, ah),
]:
    for suffix in ["cos", "sin"]:
        shapes[name + "_rope_" + suffix] = [n * heads, h // 2]
inputs = {}
for name, shape in shapes.items():
    if name.endswith(("_cos", "_sin")):
        fn = math.cos if name.endswith("cos") else math.sin
        values = [fn((i % 23 + 1) / 16) for i in range(math.prod(shape))]
        inputs[name] = mx.array(values, dtype=mx.float32).reshape(shape)
    else:
        inputs[name] = synthetic(name, shape) * 3


def rope(name, n, h):
    return tuple(
        inputs[name + "_rope_" + s].reshape(1, n, heads, h // 2).transpose(0, 2, 1, 3)
        for s in ["cos", "sin"]
    ) + ("split",)


kwargs = {"video_hidden": inputs["video"][None], "audio_hidden": inputs["audio"][None]}
for key, name in [
    ("video_adaln_params", "video_modulation"),
    ("audio_adaln_params", "audio_modulation"),
    ("video_prompt_adaln_params", "video_prompt_modulation"),
    ("audio_prompt_adaln_params", "audio_prompt_modulation"),
    ("av_ca_video_params", "video_av_modulation"),
    ("av_ca_audio_params", "audio_av_modulation"),
    ("av_ca_a2v_gate_params", "video_av_gate"),
    ("av_ca_v2a_gate_params", "audio_av_gate"),
]:
    kwargs[key] = inputs[name]
kwargs.update(
    video_text_embeds=inputs["video_text"][None],
    audio_text_embeds=inputs["audio_text"][None],
    video_rope_freqs=rope("video", nv, vh),
    audio_rope_freqs=rope("audio", na, ah),
    video_cross_rope_freqs=rope("video_cross", nv, ah),
    audio_cross_rope_freqs=rope("audio_cross", na, ah),
)
video, audio = block(**kwargs)
mx.eval(video, audio)
source_file = Path(inspect.getfile(BasicAVTransformerBlock))
result = dict(
    configuration=config,
    inputs={k: v.flatten().tolist() for k, v in inputs.items()},
    expected={"video": video.flatten().tolist(), "audio": audio.flatten().tolist()},
    provenance=dict(
        reference="ltx_core_mlx.BasicAVTransformerBlock",
        precision="Float32",
        reference_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        weights="installed Q8 with dense synthetic LoRA fusion, Float32 affine dequantization"
        if source
        else "deterministic synthetic plus rank2/scale0.25 ff.proj_out LoRA, no trained weights",
    ),
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
a.output.with_suffix(".config.json").write_text(json.dumps(config) + "\n")
mx.save_safetensors(str(a.output.with_suffix(".inputs.safetensors")), inputs)
expected = {"video": video.reshape(nv, vd), "audio": audio.reshape(na, ad)}
for name, value in weights:
    expected["weight_sample." + name] = value.flatten()[:32]
mx.save_safetensors(str(a.output.with_suffix(".expected.safetensors")), expected)
print(
    json.dumps(
        {
            "output": str(a.output),
            "video_norm": float(mx.linalg.norm(video)),
            "audio_norm": float(mx.linalg.norm(audio)),
        }
    )
)
