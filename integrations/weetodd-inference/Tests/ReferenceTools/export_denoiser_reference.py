"""Independent test-only MLX model oracle from raw packed latents to velocity."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MethodType

import mlx.core as mx
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from mlx.utils import tree_flatten

from ltx25_mlx.transformer import _compute_rope_freqs_float64

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--root", type=Path)
parser.add_argument("--trace", action="store_true")
parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
parser.add_argument("--sigmas", type=float, nargs="+")
parser.add_argument("--eta", type=float, default=0)
parser.add_argument("--fixture-seed", type=int, default=0)
parser.add_argument("--video-mask", type=float, nargs="+")
parser.add_argument("--bf16-state", action="store_true")
args = parser.parse_args()
if args.trace and args.sigmas:
    parser.error("--trace exports one evaluation; use .steps.safetensors for trajectories")
mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
trace = {}
real = args.root is not None
vd, ad, heads, vh, ah = (4096, 2048, 32, 128, 64) if real else (32, 16, 2, 16, 8)
nv, na, nt, sigma = 5, 3, 4, 0.731
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
cfg = LTXModelConfig(
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
    av_ca_timestep_scale_multiplier=1000,
    num_layers=1,
)
model = LTXModel(cfg)
model._compute_rope_freqs = MethodType(_compute_rope_freqs_float64, model)
block = model.transformer_blocks[0]
shapes = {name: value.shape for name, value in tree_flatten(model.parameters())}


def synthetic(name, shape):
    base = [
        (i * 17 + sum(name.encode()) + args.fixture_seed * 13) % 31 - 15
        for i in range(math.prod(shape))
    ]
    value = mx.array(base, dtype=mx.float32).reshape(shape) / 128
    if name.endswith(("q_norm.weight", "k_norm.weight")):
        value = value + 1
    return value


def original(name):
    name = "model.diffusion_model." + name
    for old, new in [
        (".to_out.", ".to_out.0."),
        (".ff.proj_in.", ".ff.net.0.proj."),
        (".ff.proj_out.", ".ff.net.2."),
        (".audio_ff.proj_in.", ".audio_ff.net.0.proj."),
        (".audio_ff.proj_out.", ".audio_ff.net.2."),
        (".linear1.", ".linear_1."),
        (".linear2.", ".linear_2."),
    ]:
        name = name.replace(old, new)
    return name


def decoded(source, key):
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
    return value.astype(mx.float32)


if real:
    manifest = json.loads((args.root / "paged_manifest.json").read_text())
    source = mx.load(str(args.root / manifest["fixed"]["file"]))
    fixed = [
        (name, decoded(source, original(name)))
        for name in shapes
        if not name.startswith("transformer_blocks.")
    ]
    model.load_weights(fixed, strict=False)
    mx.eval(model.parameters())
    del source, fixed
else:
    model.load_weights(
        [(name, synthetic(name, shape)) for name, shape in shapes.items()], strict=True
    )

shapes_in = {
    "video_latent": [nv, 128],
    "audio_latent": [na, 128],
    "video_text": [nt, vd],
    "audio_text": [nt, ad],
}
inputs = {name: synthetic(name, shape) * 3 for name, shape in shapes_in.items()}
inputs["video_positions"] = mx.array(
    [[0, 0, 0], [0.04, 0, 32], [0.08, 32, 0], [0.12, 32, 32], [0.16, 64, 64]], mx.float32
)
inputs["audio_positions"] = mx.array([[0], [0.02], [0.04]], mx.float32)


def provider(index):
    source = mx.load(str(args.root / manifest["layers"][index]["file"]))
    weights = []
    for name, value in tree_flatten(block.parameters()):
        key = original(f"transformer_blocks.{index}." + name)
        weight = decoded(source, key)
        assert weight.shape == value.shape
        mx.eval(weight)
        weights.append((name, weight))
    block.load_weights(weights, strict=True)

    def eager(*values, **kwargs):
        if args.trace and index == 0:
            for name, value in kwargs.items():
                if isinstance(value, mx.array):
                    trace[name] = value.astype(mx.float32)
                elif isinstance(value, tuple):
                    for suffix, array in zip(("cos", "sin"), value[:2], strict=True):
                        trace[name + "_" + suffix] = array.transpose(0, 2, 1, 3).astype(mx.float32)
        result = block(*values, **kwargs)
        mx.eval(result)
        if args.trace:
            trace[f"block_{index}_video"], trace[f"block_{index}_audio"] = result
        if args.trace and index == 47:
            trace["video_final_hidden"], trace["audio_final_hidden"] = result
        print(json.dumps({"completed": index + 1}), flush=True)
        return result

    return eager


if real:
    model.config.num_layers = 48
if args.bf16_state:
    for name in ("video_latent", "audio_latent"):
        inputs[name] = inputs[name].astype(mx.bfloat16)
kwargs = {k: v[None] for k, v in inputs.items()}
kwargs["video_text_embeds"] = kwargs.pop("video_text")
kwargs["audio_text_embeds"] = kwargs.pop("audio_text")
if args.video_mask:
    assert len(args.video_mask) == nv and all(0 <= v <= 1 for v in args.video_mask)
    if any(v != 1 for v in args.video_mask):
        kwargs["video_timesteps"] = mx.array(args.video_mask, mx.float32)[None] * sigma
trajectory_noise = {}
trajectory_steps = {}
if args.sigmas:
    from ltx25_mlx.sampling import euler_ancestral_step

    assert len(args.sigmas) >= 2 and args.sigmas[-1] == 0
    assert all(0 <= b < a <= 1 for a, b in zip(args.sigmas, args.sigmas[1:], strict=False))
    assert 0 <= args.eta <= 1
    for index, (sigma, next_sigma) in enumerate(zip(args.sigmas, args.sigmas[1:], strict=False)):
        sigma_array = mx.array([sigma], mx.bfloat16)
        if args.video_mask and any(v != 1 for v in args.video_mask):
            kwargs["video_timesteps"] = mx.array(args.video_mask, mx.float32)[None] * sigma
        predictions = X0Model(model)(
            **kwargs, sigma=sigma_array, block_provider=provider if real else None
        )
        for name, prediction in zip(("video", "audio"), predictions, strict=True):
            current = kwargs[name + "_latent"]
            clean = prediction.astype(mx.float32)
            if name == "video" and args.video_mask:
                mask = mx.array(args.video_mask, mx.float32)[None, :, None]
                clean = clean * mask + inputs["video_latent"][None] * (1 - mask)
            noise = None
            if next_sigma > 0 and args.eta > 0:
                noise = synthetic(f"noise_{index}_{name}", current.shape)
                trajectory_noise[f"{index}.{name}"] = noise.reshape(current.shape[1:])
            updated = euler_ancestral_step(
                current, clean, sigma, next_sigma, noise=noise, eta=args.eta
            )
            if name == "video" and args.video_mask and next_sigma > 0 and args.eta > 0:
                # Production euler_ancestral_denoise_loop restores anchors after
                # re-noising as well as masking the clean prediction.
                updated = updated * mask + inputs["video_latent"][None] * (1 - mask)
            updated = updated.astype(current.dtype)
            mx.eval(updated)
            kwargs[name + "_latent"] = updated
            trajectory_steps[f"{index}.{name}"] = updated.reshape(current.shape[1:])
    video, audio = kwargs["video_latent"], kwargs["audio_latent"]
    if args.video_mask:
        # Qualify masked fixtures through the actual working orchestration too.
        # A handwritten step loop alone previously duplicated the port's missing
        # post-noise anchor restoration and gave a false parity result.
        from unittest.mock import patch

        from ltx_core_mlx.conditioning.types.latent_cond import LatentState

        from ltx25_mlx.sampling import euler_ancestral_denoise_loop

        # Use the actual production velocity-to-x0 wrapper as well as its loop.
        # A hand-written wrapper previously hid per-token sigma/dtype differences.
        def predict_clean(**call):
            return X0Model(model)(**call, block_provider=provider if real else None)

        draws = iter(trajectory_noise.values())

        def replay_noise(shape, **_):
            value = next(draws).reshape(shape)
            return value

        states = []
        for name in ("video", "audio"):
            latent = inputs[name + "_latent"][None]
            mask = (
                mx.array(args.video_mask, mx.float32)[None, :, None]
                if name == "video"
                else mx.ones((1, na, 1))
            )
            states.append(
                LatentState(
                    latent=latent,
                    clean_latent=latent,
                    denoise_mask=mask,
                    positions=inputs[name + "_positions"][None],
                )
            )
        with patch.object(mx.random, "normal", side_effect=replay_noise):
            production = euler_ancestral_denoise_loop(
                predict_clean,
                states[0],
                states[1],
                inputs["video_text"][None],
                inputs["audio_text"][None],
                sigmas=args.sigmas,
                noise_seed=0,
                eta=args.eta,
            )
        for actual, expected in (
            (production.video_latent, video),
            (production.audio_latent, audio),
        ):
            assert mx.max(mx.abs(actual - expected)).item() < 1e-6
        video, audio = production.video_latent, production.audio_latent
else:
    video, audio = model(
        **kwargs, timestep=mx.array([sigma], mx.float32), block_provider=provider if real else None
    )
mx.eval(video, audio)
if args.trace:
    time = model._embed_timestep_scalar(mx.array([sigma], mx.bfloat16))
    trace["time"] = time
    _, trace["video_embedded"] = model.adaln_single(time)
    _, trace["audio_embedded"] = model.audio_adaln_single(time)
    trace["video_velocity"], trace["audio_velocity"] = video, audio
    mx.save_safetensors(str(args.output.with_suffix(".trace.safetensors")), trace)
assert bool(mx.all(mx.isfinite(video))) and bool(mx.all(mx.isfinite(audio)))
args.output.parent.mkdir(parents=True, exist_ok=True)
if not real:
    data = dict(
        configuration=config,
        inputs={k: v.flatten().tolist() for k, v in inputs.items()},
        sigma=sigma,
        video_mask=args.video_mask,
        bf16_state=args.bf16_state,
        expected=dict(video=video.flatten().tolist(), audio=audio.flatten().tolist()),
        provenance=dict(
            reference="LTXModel/X0Model and production denoise loop with 2.5 Float64 grid",
            weights="synthetic",
            reference_sha256=hashlib.sha256(
                Path(inspect.getfile(LTXModel)).read_bytes()
            ).hexdigest(),
        ),
        schedule=dict(sigmas=args.sigmas, eta=args.eta),
        noise={k: v.flatten().tolist() for k, v in trajectory_noise.items()},
    )
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
else:
    if args.sigmas:
        args.output.with_suffix(".schedule.json").write_text(
            json.dumps(dict(sigmas=args.sigmas, eta=args.eta)) + "\n"
        )
        mx.save_safetensors(str(args.output.with_suffix(".steps.safetensors")), trajectory_steps)
        if trajectory_noise:
            mx.save_safetensors(
                str(args.output.with_suffix(".noise.safetensors")), trajectory_noise
            )
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "device": args.device,
                "sigma": sigma if not args.sigmas else None,
                "fixture_seed": args.fixture_seed,
                "expected_kind": "sampled_latents" if args.sigmas else "velocity",
                "sigmas": args.sigmas,
                "eta": args.eta,
                "precision": "Float32 weights; BF16 model input boundary",
                "reference_sha256": hashlib.sha256(
                    Path(inspect.getfile(LTXModel)).read_bytes()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    args.output.with_suffix(".config.json").write_text(json.dumps(config) + "\n")
    mx.save_safetensors(str(args.output.with_suffix(".inputs.safetensors")), inputs)
    mx.save_safetensors(
        str(args.output.with_suffix(".expected.safetensors")),
        {"video": video.reshape(nv, 128), "audio": audio.reshape(na, 128)},
        metadata={"sigma": str(sigma)},
    )
print(
    json.dumps(
        {
            "output": str(args.output),
            "video_norm": float(mx.linalg.norm(video)),
            "audio_norm": float(mx.linalg.norm(audio)),
        }
    )
)
