import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_local_text_bridge_streams_and_returns_text_without_connection(tmp_path):
    from studio_prompt_assist import assist

    helper = tmp_path / "helper"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        + """import json, sys
r = json.load(sys.stdin)
assert sys.argv[1] == "text"
assert "connection" not in r and "apiKey" not in r
assert r["images"] == [{"path": "/images/first.png", "label": "First frame"}]
events = [("progress", {"stage": "vision", "images": 1}),
          ("progress", {"stage": "writing", "tokens": 4}),
          ("result", {"text": "A fox in moonlight.", "outputTokens": 4, "truncated": False})]
for t, v in events:
 print(json.dumps({"requestID": r["requestID"], "type": t, "value": v}), flush=True)
"""
    )
    helper.chmod(0o755)
    progress = []
    result = assist(
        {
            "runtime": {"drawThingsHelperPath": str(helper)},
            "textRequest": {
                "modelPath": "/models/qwen_3.5_4b_i8x.ckpt",
                "systemPrompt": "Write a prompt.",
                "prompt": "A fox",
                "maxTokens": 128,
                "images": [{"path": "/images/first.png", "label": "First frame"}],
            },
        },
        progress=progress.append,
    )
    assert result["text"] == "A fox in moonlight."
    assert "Reading 1 images" in progress[0]
    assert "4" in progress[1]


def test_text_bridge_rejects_empty_completion(tmp_path):
    from studio_prompt_assist import validate_result

    with pytest.raises(ValueError, match="text"):
        validate_result({"text": "   "})
    with pytest.raises(ValueError):
        validate_result({"text": 5})


def test_local_text_cancellation_terminates_helper(tmp_path):
    from studio_prompt_assist import assist

    helper = tmp_path / "helper"
    helper.write_text(
        f"#!{sys.executable}\nimport json, sys, time\nr = json.load(sys.stdin)\ntime.sleep(30)\n"
    )
    helper.chmod(0o755)
    start = time.monotonic()
    with pytest.raises(InterruptedError):
        assist(
            {"runtime": {"drawThingsHelperPath": str(helper)}, "textRequest": {}},
            cancelled=lambda: time.monotonic() - start > 0.2,
        )
    assert time.monotonic() - start < 3


@pytest.mark.parametrize("code,message", [
    ("text_model_unsupported", "H3 Qwen encoders cannot"),
    ("text_input_bytes_exceeded", "UTF-8"),
])
def test_local_text_errors_explain_model_requirement(tmp_path, code, message):
    from studio_prompt_assist import assist

    helper = tmp_path / "helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        + f"code = {code!r}\n"
        + """import json, sys
r = json.load(sys.stdin)
print(json.dumps({"requestID": r["requestID"], "type": "error", "code": code}))
sys.exit(1)
"""
    )
    helper.chmod(0o755)
    with pytest.raises(ValueError, match=message):
        assist({"runtime": {"drawThingsHelperPath": str(helper)}, "textRequest": {}})
