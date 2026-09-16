import hashlib
import io
import json

import pytest

from wee_todd_mlx import assistant_models as models
from wee_todd_mlx.model_downloads import DownloadFile, partial_path


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


@pytest.fixture
def package(monkeypatch):
    body = b"SQLite format 3\x00" + b"fixture checkpoint"
    item = DownloadFile(
        "drawthingsai/draw-things-community",
        "a" * 40,
        "qwen_3.5_4b_i8x.ckpt",
        len(body),
        hashlib.sha256(body).hexdigest(),
        "qwen_3.5_4b_i8x.ckpt",
        provider="drawthings-static",
    )
    monkeypatch.setattr(models, "CHECKPOINT", item)
    return item, body


def test_catalog_pins_official_format_and_never_downloads():
    item = models.catalog()
    assert item["downloadBytes"] == 4888076288
    assert item["sha256"] == "9e07b480a4e093d1d304ee59a625fc3169a53dc3a597d17f69a2dc167958ee9c"
    assert item["runtime"] == "drawthings-qwen-local"
    assert item["sourceURL"] == "https://static.libnnc.org/qwen_3.5_4b_i8x.ckpt"


def test_static_provider_cannot_be_used_for_arbitrary_source():
    with pytest.raises(ValueError):
        DownloadFile(
            "attacker/repo",
            "a" * 40,
            "evil.ckpt",
            1,
            "b" * 64,
            "evil.ckpt",
            provider="drawthings-static",
        )


def test_install_integrity_receipt_and_reuse_without_overwrite(tmp_path, package):
    item, body = package
    result = models.install(tmp_path, opener=lambda *a, **k: Response(body))
    assert result["status"] == "checksum_verified"
    assert result["inferenceChecked"] is False
    receipt = json.loads((tmp_path / models.RECEIPT_NAME).read_text())
    assert receipt["sha256"] == item.sha256
    assert (tmp_path / item.filename).read_bytes() == body
    assert (
        models.install(tmp_path, opener=lambda *a, **k: pytest.fail("network reuse"))["path"]
        == result["path"]
    )


def test_interruption_resumes_with_range(tmp_path, package):
    item, body = package
    with pytest.raises(ValueError, match="incomplete"):
        models.install(tmp_path, opener=lambda *a, **k: Response(body[:10]))
    assert not (tmp_path / models.RECEIPT_NAME).exists()

    def resume(request, **kwargs):
        assert request.get_header("Range") == "bytes=10-"
        return Response(body[10:], 206, {"Content-Range": f"bytes 10-{len(body) - 1}/{len(body)}"})

    assert models.install(tmp_path, opener=resume)["status"] == "checksum_verified"


def test_corruption_and_cancel_never_report_success(tmp_path, package):
    item, body = package
    with pytest.raises(ValueError, match="SHA-256"):
        models.install(tmp_path, opener=lambda *a, **k: Response(b"x" * len(body)))
    assert not (tmp_path / item.filename).exists()
    assert not partial_path(item, tmp_path / item.filename).exists()
    with pytest.raises(InterruptedError):
        models.install(tmp_path, cancelled=lambda: True)
    assert not (tmp_path / models.RECEIPT_NAME).exists()
    assert not list(tmp_path.glob("*.lock"))


def test_existing_external_file_reused_in_place_and_invalid_not_ready(tmp_path, package):
    item, body = package
    source = tmp_path / item.filename
    source.write_bytes(body)
    result = models.inspect_model(source)
    assert result["path"] == str(source)
    assert result["status"] == "checksum_verified"
    assert list(tmp_path.iterdir()) == [source]
    source.write_bytes(b"SQLite format 3\x00broken")
    with pytest.raises(ValueError, match="incomplete|checksum"):
        models.inspect_model(source)


def test_insufficient_space_never_opens_network(tmp_path, package, monkeypatch):
    monkeypatch.setattr(
        models, "_check_space", lambda *a: (_ for _ in ()).throw(ValueError("space"))
    )
    with pytest.raises(ValueError, match="space"):
        models.install(tmp_path, opener=lambda *a, **k: pytest.fail("network"))


def test_symlink_partial_does_not_touch_other_file(tmp_path, package):
    item, body = package
    other = tmp_path / "private"
    other.write_bytes(b"keep")
    partial_path(item, tmp_path / item.filename).symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        models.install(tmp_path, opener=lambda *a, **k: Response(body))
    assert other.read_bytes() == b"keep"


def test_health_check_executes_text_and_vision_without_certifying_quality(tmp_path, package):
    item, body = package
    source = tmp_path / item.filename
    source.write_bytes(body)
    helper = tmp_path / "helper"
    helper.write_text("fixture")
    helper.chmod(0o700)
    calls = []

    def invoke(command, payload, **kwargs):
        assert command == "text"
        assert payload["maxTokens"] == 64
        if payload["images"]:
            from PIL import Image

            image = Image.open(payload["images"][0]["path"])
            assert image.size == (128, 128)
            assert image.getpixel((0, 0)) == (255, 0, 0)
            assert image.getpixel((127, 127)) == (0, 0, 255)
        calls.append(payload)
        yield {
            "type": "result",
            "value": {"text": "fixture response", "imagesUsed": len(payload["images"])},
        }

    result = models.health_check(source, helper, invoke=invoke)
    assert len(calls) == 2 and calls[0]["requestID"] != calls[1]["requestID"]
    assert result["inferenceChecked"] is True
    assert "not a quality certification" in result["message"]
    assert not __import__("pathlib").Path(calls[1]["images"][0]["path"]).exists()


def test_failed_vision_check_never_returns_healthy(tmp_path, package):
    item, body = package
    source = tmp_path / item.filename
    source.write_bytes(body)
    helper = tmp_path / "helper"
    helper.write_text("fixture")
    helper.chmod(0o700)

    def invoke(command, payload, **kwargs):
        yield {"type": "result", "value": {"text": "no image read", "imagesUsed": 0}}

    with pytest.raises(ValueError, match="test image"):
        models.health_check(source, helper, invoke=invoke)


def test_cancel_during_stream_retains_resumable_partial(tmp_path, package):
    item, body = package
    state = {"cancelled": False}

    class InterruptedResponse(Response):
        def read(self, size):
            if self.tell():
                state["cancelled"] = True
            return super().read(10)

    with pytest.raises(InterruptedError):
        models.install(
            tmp_path,
            opener=lambda *a, **k: InterruptedResponse(body),
            cancelled=lambda: state["cancelled"],
        )
    assert partial_path(item, tmp_path / item.filename).read_bytes() == body[:10]
    assert not (tmp_path / models.RECEIPT_NAME).exists()


def test_bridge_catalog_and_cli_work_without_draw_things(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[1]
    request = tmp_path / "request.json"
    request.write_text("{}")
    command = [
        sys.executable,
        str(root / "scripts/studio_bridge.py"),
        "assistant-model-catalog",
        "--request",
        str(request),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["result"]["filename"] == models.CHECKPOINT.filename
    result = subprocess.run(
        [sys.executable, str(root / "scripts/setup_assistant_model.py"), "catalog"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["sha256"] == models.CHECKPOINT.sha256


def test_real_http_fixture_recovers_disconnect_and_verifies_bytes(tmp_path, package, monkeypatch):
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen

    item, body = package
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            assert self.headers.get("Authorization") is None
            resumed = self.headers.get("Range") == "bytes=10-"
            self.send_response(206 if resumed else 200)
            if resumed:
                self.send_header("Content-Range", f"bytes 10-{len(body) - 1}/{len(body)}")
            self.end_headers()
            self.wfile.write(body[10:] if resumed else body[:10])

        def log_message(self, *args):
            pass

    # HTTPServer otherwise reverse-resolves the local host through machine DNS.
    monkeypatch.setattr(socket, "getfqdn", lambda *_: "localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def local_open(request, timeout):
        assert request.full_url == "https://static.libnnc.org/" + item.filename
        return urlopen(
            Request(
                f"http://127.0.0.1:{server.server_port}/model", headers=dict(request.header_items())
            ),
            timeout=timeout,
        )

    try:
        with pytest.raises(ValueError, match="incomplete"):
            models.install(tmp_path, opener=local_open)
        assert models.install(tmp_path, opener=local_open)["status"] == "checksum_verified"
        assert requests == [None, "bytes=10-"]
        assert (tmp_path / item.filename).read_bytes() == body
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unrelated_receipt_and_existing_checkpoint_preserved(tmp_path, package):
    item, body = package
    receipt = tmp_path / models.RECEIPT_NAME
    receipt.write_text("[]")
    with pytest.raises(ValueError, match="receipt differs"):
        models.install(tmp_path, opener=lambda *a, **k: pytest.fail("network"))
    assert receipt.read_text() == "[]"
    receipt.unlink()
    target = tmp_path / item.filename
    target.write_bytes(b"user file")
    with pytest.raises(ValueError, match="existing download differs"):
        models.install(tmp_path, opener=lambda *a, **k: pytest.fail("network"))
    assert target.read_bytes() == b"user file"


def test_oversized_owned_partial_can_recover_without_manual_deletion(tmp_path, package):
    item, body = package
    partial_path(item, tmp_path / item.filename).write_bytes(body + b"corrupt extra")
    assert (
        models.install(tmp_path, opener=lambda *a, **k: Response(body))["status"]
        == "checksum_verified"
    )


def test_other_installer_lock_is_not_removed(tmp_path, package):
    lock = tmp_path / ".qwen35-4b-setup.lock"
    lock.write_text("another process")
    with pytest.raises(ValueError, match="already active"):
        models.install(tmp_path, opener=lambda *a, **k: pytest.fail("network"))
    assert lock.read_text() == "another process"
