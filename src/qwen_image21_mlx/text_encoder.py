"""Qwen 2.1's exact raw prompt and pre-normalization Qwen3-VL features."""

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def prompt_template(prompt, image_count):
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(image_count)
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )


def make_encoder(config):
    import mlx.core as mx
    from mlx_vlm.models.qwen3_vl.config import ModelConfig
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    class ImageVisionModel(VisionModel):
        """Image-only traversal with materialized stages and cancellable layer boundaries."""

        def __call__(self, patches, grid_thw, **kwargs):
            grid = mx.array(grid_thw)
            value = self.patch_embed(patches) + self.fast_pos_embed_interpolate(grid)
            angles = self.rot_pos_emb(grid)
            boundaries = vision_boundaries(grid)
            deep = []
            for index, block in enumerate(self.blocks):
                if getattr(self, "cancel", lambda: False)():
                    raise InterruptedError("Vision encoding cancelled")
                value = block(value, cu_seqlens=boundaries, rotary_pos_emb=angles)
                mx.eval(value)
                if index in self.deepstack_visual_indexes:
                    merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)]
                    feature = merger(value)
                    mx.eval(feature)
                    deep.append(feature)
            return self.merger(value), deep

    configuration = ModelConfig.from_dict(config)
    model = Model(configuration)
    model.vision_tower = ImageVisionModel(configuration.vision_config)
    # This feature extractor never computes vocabulary logits.
    if hasattr(model.language_model, "lm_head"):
        del model.language_model.lm_head
    return model


def encode(model, processor_path, prompt, images, *, cancel, progress):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from transformers import AutoTokenizer

    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil as ImageProcessor,
        )
    except ImportError:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor as ImageProcessor,
        )

    if cancel():
        raise InterruptedError("Text encoding cancelled")
    tokenizer = AutoTokenizer.from_pretrained(processor_path, local_files_only=True)
    image_processor = ImageProcessor.from_pretrained(processor_path, local_files_only=True)
    inputs = processor_inputs(tokenizer, image_processor, prompt, images)
    ids = mx.array(inputs["input_ids"])
    mask = mx.array(inputs["attention_mask"])
    kwargs = {key: mx.array(inputs[key]) for key in ("image_grid_thw",) if key in inputs}
    model.vision_tower.cancel = cancel
    pixels = mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
    progress(
        {
            "event": "progress",
            "stage": "encoding",
            "fraction": 0.1,
            "message": f"Encoding prompt and {len(images)} reference images",
        }
    )
    embedded = model.get_input_embeddings(ids, pixels, mask=mask, **kwargs)
    text = model.language_model.model
    original_norm = text.norm
    original_layers = text.layers

    class CheckedLayer(nn.Module):
        def __init__(self, layer, index):
            super().__init__()
            self.inner = layer
            self.index = index

        @property
        def self_attn(self):
            return self.inner.self_attn

        def __call__(self, *args, **kwargs):
            if cancel():
                raise InterruptedError("Text encoding cancelled")
            result = self.inner(*args, **kwargs)
            mx.eval(result)
            progress(
                {
                    "event": "progress",
                    "stage": "encoding",
                    "fraction": 0.1 + 0.05 * (self.index + 1) / len(original_layers),
                    "message": f"Encoding prompt · layer {self.index + 1}/{len(original_layers)}",
                }
            )
            return result

    try:
        text.norm = nn.Identity()
        text.layers = [CheckedLayer(layer, i) for i, layer in enumerate(original_layers)]
        value = text(
            ids,
            inputs_embeds=embedded.inputs_embeds,
            position_ids=embedded.position_ids,
            visual_pos_masks=embedded.visual_pos_masks,
            deepstack_visual_embeds=embedded.deepstack_visual_embeds,
        )
        mx.eval(value)
    finally:
        text.norm = original_norm
        text.layers = original_layers
    if cancel():
        raise InterruptedError("Text encoding cancelled")
    system_ids = tokenizer(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n", add_special_tokens=False
    )["input_ids"]
    drop = len(system_ids)
    active = np.flatnonzero(np.asarray(inputs["attention_mask"])[0])[drop:]
    features = value[:, mx.array(active)]
    image_mask = np.asarray(inputs["input_ids"])[0, active] == model.config.image_token_id
    mx.eval(features)
    return {"features": features, "imageMask": image_mask}


def processor_inputs(tokenizer, image_processor, prompt, images):
    """Use the library's PIL image processor, avoiding its unrelated video/PyTorch dependency."""
    import numpy as np

    from .preprocessing import vision_copy

    text = prompt_template(prompt or " ", len(images))
    vision = {}
    if images:
        vision = dict(
            image_processor(images=[vision_copy(image) for image in images], return_tensors="np")
        )
        # Expand placeholders in one pass; inserted image tokens must never be expanded again.
        pieces = text.split("<|image_pad|>")
        if len(pieces) != len(images) + 1:
            raise ValueError("Prompts cannot contain reserved image placeholder tokens")
        text = pieces[0]
        for grid, tail in zip(vision["image_grid_thw"], pieces[1:], strict=True):
            text += "<|image_pad|>" * (int(np.prod(grid)) // 4) + tail
    return {
        **dict(tokenizer([text], padding=True, padding_side="left", return_tensors="np")),
        **vision,
    }


def vision_boundaries(grid):
    import mlx.core as mx

    lengths = []
    for frames, height, width in grid.tolist():
        lengths.extend([int(height) * int(width)] * int(frames))
    return mx.concatenate(
        [mx.array([0], dtype=mx.int32), mx.cumsum(mx.array(lengths, dtype=mx.int32))]
    )
