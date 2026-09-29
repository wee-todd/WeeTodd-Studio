"""Test-only oracle for 48 installed LTX blocks; no Python Swift runtime dependency."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import mlx.core as mx
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from mlx.utils import tree_flatten

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("configuration", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
config = json.loads(args.configuration.read_text())
manifest = json.loads((args.root / "paged_manifest.json").read_text())
assert manifest["num_layers"] == 48 and manifest["bits"] == 8 and manifest["group_size"] == 64
vd, ad, heads = config["videoDimension"], config["audioDimension"], config["heads"]
vh, ah = config["videoHeadDimension"], config["audioHeadDimension"]
nv, na = config["videoTokens"], config["audioTokens"]
inputs = mx.load(str(args.inputs))
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
shapes = {name: value.shape for name, value in tree_flatten(block.parameters())}


def rope(name, tokens, width):
    return tuple(
        inputs[name + "_rope_" + suffix].reshape(1, tokens, heads, width // 2).transpose(0, 2, 1, 3)
        for suffix in ["cos", "sin"]
    ) + ("split",)


kwargs = {}
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
video, audio = inputs["video"][None], inputs["audio"][None]
expected = {}
for index, page in enumerate(manifest["layers"]):
    source = mx.load(str(args.root / page["file"]))
    weights = []
    for name, shape in shapes.items():
        key = f"model.diffusion_model.transformer_blocks.{index}." + name
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
        assert value.shape == shape, (name, value.shape, shape)
        mx.eval(value)
        weights.append((name, value))
    block.load_weights(weights, strict=True)
    del weights, source
    video, audio = block(video_hidden=video, audio_hidden=audio, **kwargs)
    mx.eval(video, audio)
    assert bool(mx.all(mx.isfinite(video))) and bool(mx.all(mx.isfinite(audio)))
    if index in [0, 7, 15, 23, 31, 39, 47]:
        expected[f"block_{index}.video"] = video.reshape(nv, vd)
        expected[f"block_{index}.audio"] = audio.reshape(na, ad)
    print(
        json.dumps(
            {
                "completed": index + 1,
                "video_norm": float(mx.linalg.norm(video)),
                "audio_norm": float(mx.linalg.norm(audio)),
            }
        ),
        flush=True,
    )
    mx.clear_cache()
expected.update(video=video.reshape(nv, vd), audio=audio.reshape(na, ad))
args.output.parent.mkdir(parents=True, exist_ok=True)
source_file = Path(inspect.getfile(BasicAVTransformerBlock))
mx.save_safetensors(
    str(args.output),
    expected,
    metadata={
        "reference": "ltx_core_mlx.BasicAVTransformerBlock, 48 sequential calls",
        "reference_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "precision": "Float32 weights, Float32 affine Q8 dequantization",
    },
)
