"""One staged image renderer for Studio and headless execution."""

from __future__ import annotations

import copy
import gc
import json
import secrets
import time
from pathlib import Path


class Runtime:
    def __init__(self, manifest, cancel=lambda: False):
        self.manifest = manifest
        self.cancel = cancel

    def load(self, name):
        from .checkpoint import load_component

        component = self.manifest["components"][name]
        if name == "text_encoder":
            from .text_encoder import make_encoder

            module = make_encoder(component["config"])
        elif name == "transformer":
            from .transformer import QwenImage21Transformer

            module = QwenImage21Transformer(component["config"])
        else:
            from .vae import ImageVAE

            module = ImageVAE(component["config"])
        try:
            return load_component(
                module, component, transpose_convolutions=name == "vae", cancel=self.cancel
            )
        except BaseException:
            self.release(name, module)
            raise

    def release(self, name, module):
        import mlx.core as mx

        mx.synchronize()
        module.clear()
        gc.collect()
        mx.clear_cache()

    def encode(self, model, prompt, images, cancel, progress):
        from .text_encoder import encode

        return encode(
            model, self.manifest["processorPath"], prompt, images, cancel=cancel, progress=progress
        )


def render_prepared(request, manifest, output, *, cancel, progress, runtime=None):
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from .preprocessing import prepare_inputs
    from .preview import LivePreview
    from .scheduler import euler_step, make_schedule

    runtime = runtime or Runtime(manifest, cancel=cancel)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = copy.deepcopy(request)
    config = request["configuration"]
    if config["seed"] == -1:
        config["seed"] = secrets.randbits(32)
    started = time.monotonic()
    timings = {}
    module = None
    active = None
    preview = None

    def report(stage, message, fraction, **extra):
        if cancel():
            raise InterruptedError("Image generation cancelled")
        progress(dict(event="progress", stage=stage, message=message, fraction=fraction, **extra))
        if cancel():
            raise InterruptedError("Image generation cancelled")

    def load(name, fraction):
        nonlocal module, active
        report("loading", f"Loading {name.replace('_', ' ')}", fraction)
        active = name
        before = time.monotonic()
        module = runtime.load(name)
        timings[name + "LoadSeconds"] = (
            timings.get(name + "LoadSeconds", 0) + time.monotonic() - before
        )
        if cancel():
            raise InterruptedError("Image generation cancelled")
        return module

    def release():
        nonlocal module, active
        if module is not None:
            runtime.release(active, module)
            module = None
            active = None

    try:
        report("preparing", "Preparing reference images", 0.01)
        images = prepare_inputs(request)
        encoder = load("text_encoder", 0.03)
        before = time.monotonic()
        conditioning = runtime.encode(encoder, request["prompt"], images, cancel, progress)
        mx.eval(conditioning["features"])
        timings["encodingSeconds"] = time.monotonic() - before
        release()
        del encoder
        reference_latents = []
        if images:
            vae = load("vae", 0.16)
            for index, image in enumerate(images):
                report(
                    "references",
                    f"Encoding reference {index + 1}/{len(images)}",
                    0.17 + 0.1 * index / len(images),
                )
                pixels = mx.array(np.asarray(image, dtype=np.float32)[None] / 127.5 - 1)
                latent = vae.encode(pixels, cancel=cancel).astype(mx.bfloat16)
                mx.eval(latent)
                reference_latents.append(latent.reshape(1, -1, 64))
            release()
            del vae
        conditioning["shapes"] = [(image.height // 16, image.width // 16) for image in images]
        conditioning["shapes"].append((config["height"] // 16, config["width"] // 16))
        images.clear()
        tokens = config["height"] // 16 * (config["width"] // 16)
        mx.random.seed(config["seed"])
        latents = mx.random.normal((1, tokens, 64), dtype=mx.float32).astype(mx.bfloat16)
        mx.eval(latents)
        if config["livePreview"]:
            preview = LivePreview(output, manifest["previewPath"], progress)
        transformer = load("transformer", 0.28)
        cache = {} if request["cacheMode"] == "prefix_kv" else None
        sigmas = make_schedule(config["steps"], tokens, manifest["scheduler"])
        before = time.monotonic()
        for step, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True), 1):
            report(
                "prefill" if step == 1 else "sampling",
                "Preparing image attention" if step == 1 else f"Sampling {step}/{config['steps']}",
                0.3 + 0.55 * (step - 1) / config["steps"],
            )
            combined = (
                mx.concatenate([*reference_latents, latents], axis=1)
                if reference_latents
                else latents
            )
            step_began = time.monotonic()
            predicted = transformer(
                combined,
                conditioning,
                float(sigma),
                cache=cache,
                cancel=cancel,
                progress=lambda completed, total, step=step: report(
                    "prefill" if step == 1 else "sampling",
                    f"Sampling {step}/{config['steps']} · layer {completed}/{total}",
                    0.3 + 0.55 * (step - 1 + completed / total) / config["steps"],
                ),
            )
            timings.setdefault("stepSeconds", []).append(time.monotonic() - step_began)
            mx.eval(predicted)
            if preview:
                preview.update(
                    (latents.astype(mx.float32) - float(sigma) * predicted.astype(mx.float32)),
                    width=config["width"],
                    height=config["height"],
                    step=step,
                    total=config["steps"],
                )
            latents = euler_step(
                latents.astype(mx.float32), predicted.astype(mx.float32), sigma, next_sigma
            ).astype(mx.bfloat16)
            mx.eval(latents)
            report(
                "sampling",
                f"Sampling {step}/{config['steps']}",
                0.3 + 0.55 * step / config["steps"],
                step=step,
                total=config["steps"],
            )
        timings["samplingSeconds"] = time.monotonic() - before
        if cache is not None:
            cache.clear()
        del conditioning, reference_latents, combined, predicted
        release()
        del transformer
        vae = load("vae", 0.88)
        report("decoding", "Decoding final RGBA image", 0.9)
        before = time.monotonic()
        pixels = vae.decode(
            latents.astype(mx.float32).reshape(
                1, config["height"] // 16, config["width"] // 16, 64
            ),
            cancel=cancel,
        )
        mx.eval(pixels)
        image = Image.fromarray(
            (np.clip(np.asarray(pixels[0]) / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)
        )
        timings["decodeSeconds"] = time.monotonic() - before
        release()
        del vae
        report("saving", "Saving image and generation settings", 0.98)
        final = output / "image.png"
        partial = output / "image.partial.png"
        image.save(partial, format="PNG")
        if cancel():
            raise InterruptedError("Image generation cancelled")
        timings["totalSeconds"] = time.monotonic() - started
        result = {
            "asset": {"path": str(final), "width": config["width"], "height": config["height"]},
            "normalizedRequest": request,
            "fingerprint": request["fingerprint"],
            "timings": timings,
            "modelRevision": manifest.get("sourceRevision"),
            "precision": "8bit",
            "mlxPeakBytes": mx.get_peak_memory(),
        }
        metadata = output / "generation.partial.json"
        metadata.write_text(json.dumps(result, indent=2) + "\n")
        if cancel():
            raise InterruptedError("Image generation cancelled")
        # Publication is the commit point. A later cancellation cannot revoke this result.
        partial.replace(final)
        metadata.replace(output / "generation.json")
        progress(
            {"event": "progress", "stage": "complete", "message": "Image saved", "fraction": 1}
        )
        return result
    finally:
        release()
        if preview is not None:
            preview.close()
        (output / "image.partial.png").unlink(missing_ok=True)
        (output / "generation.partial.json").unlink(missing_ok=True)
