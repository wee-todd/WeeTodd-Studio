import hashlib

import pytest
from PIL import Image

from qwen_image21_mlx.preprocessing import prepare_inputs, vision_copy
from qwen_image21_mlx.text_encoder import prompt_template


def test_transparency_only_flattened_for_vision_and_changed_inputs_rejected(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(path)
    request = {
        "inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "processedWidth": 32,
                "processedHeight": 32,
            }
        ]
    }
    images = prepare_inputs(request)
    assert images[0].mode == "RGBA"
    assert images[0].getpixel((0, 0))[3] == 0
    assert vision_copy(images[0]).getpixel((0, 0)) == (255, 255, 255)
    Image.new("RGB", (32, 32)).save(path)
    with pytest.raises(ValueError, match="changed"):
        prepare_inputs(request)


def test_prompt_template_lists_all_ten_images_in_order():
    value = prompt_template("Replace the coat.", 10)
    assert value.count("<|image_pad|>") == 10
    assert value.index("<image1>") < value.index("<image10>") < value.index("Replace")
    assert "<|image_pad|>" not in prompt_template("Generate a coat.", 0)


def test_processor_load_is_image_only_without_torch(monkeypatch):
    from qwen_image21_mlx.text_encoder import processor_inputs

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts[0].count("<|image_pad|>") == 6
            return {"input_ids": [[1]], "attention_mask": [[1]]}

    class ImageProcessor:
        def __call__(self, **kwargs):
            return {"pixel_values": [1], "image_grid_thw": [[1, 4, 6]]}

    values = processor_inputs(
        Tokenizer(), ImageProcessor(), "prompt", [Image.new("RGBA", (96, 64))]
    )
    assert values["pixel_values"] == [1]


def test_vision_grid_uses_host_repeat_count():
    import mlx.core as mx

    from qwen_image21_mlx.text_encoder import vision_boundaries

    assert vision_boundaries(mx.array([[1, 4, 6], [1, 8, 8]])).tolist() == [0, 24, 88]
