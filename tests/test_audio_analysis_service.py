import hashlib
import json

import numpy as np
import pytest

from wee_todd_mlx.audio_analysis.service import analyze
from wee_todd_mlx.audio_analysis.setup import catalog, model_identity
from wee_todd_mlx.model_downloads import DownloadFile


def test_neural_analysis_requires_model_setup_before_decode(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"audio")
    with pytest.raises(ValueError, match="[Mm]odel|[Ss]et up"):
        analyze(
            song,
            expected_sha256=hashlib.sha256(b"audio").hexdigest(),
            model_directory=tmp_path / "missing",
            cache_directory=tmp_path / "cache",
        )


def test_model_identity_rejects_substituted_checkpoint(tmp_path):
    for model in catalog()["models"]:
        for item in model["files"]:
            target = tmp_path / item["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="[Mm]odel|[Cc]heckpoint"):
        model_identity(tmp_path)


def test_beat_download_provider_has_fixed_publisher_url():
    item = next(f for m in catalog()["models"] for f in m["files"] if f.get("provider") == "cpjku")
    assert DownloadFile(**item).url.startswith("https://cloud.cp.jku.at/")
    with pytest.raises(ValueError):
        DownloadFile(**dict(item, repo="other/repo"))


@pytest.fixture
def analysis_run(tmp_path, monkeypatch, request):
    """Exercise real orchestration/cache/alignment; replace only media/model hardware."""
    from wee_todd_mlx.audio_analysis import beat_model, ctc_model, service

    source = tmp_path / "song.wav"
    source.write_bytes(b"fixture audio")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = tmp_path / "models"
    (root / "wav2vec2-base-960h").mkdir(parents=True)
    vocabulary = dict(
        zip(
            ["<pad>", "<s>", "</s>", "<unk>", "|", *list("ETAONIHSRDLUMWCFGYPBVK'XJQZ")],
            range(32),
            strict=True,
        )
    )
    (root / "wav2vec2-base-960h" / "vocab.json").write_text(json.dumps(vocabulary))
    duration = getattr(request, "param", 2.0)
    samples = np.zeros(round(duration * 22050), np.float32)
    calls = dict(beats=0, ctc=0, decode=0)
    monkeypatch.setattr(service, "model_identity", lambda *_args, **_kwargs: {"fixture": "fixed"})

    def decode(*_args, **kwargs):
        kwargs["check"]()
        calls["decode"] += 1
        return samples.tobytes()

    monkeypatch.setattr(service, "_decode_audio", decode)

    def basic(*_args, **kwargs):
        kwargs["check"]()
        return dict(
            sourceSHA256=digest,
            sampleRate=8000,
            durationSeconds=duration,
            cues=[],
            energyEnvelope=[0.0] * int(duration),
            energyHopSeconds=1.0,
            beatSuggestion=None,
            decode=dict(channels=1, sampleRate=8000, format="float32"),
        )

    monkeypatch.setattr(service, "analyze_audio", basic)

    def beats(*_args, **kwargs):
        kwargs["check"]()
        calls["beats"] += 1
        return dict(beats=[dict(timeSeconds=0.5, confidence=0.8)], downbeats=[], metadata={})

    monkeypatch.setattr(beat_model, "beat_events", beats)

    def ctc(*_args, **kwargs):
        kwargs["check"]()
        calls["ctc"] += 1
        probabilities = np.full(
            ((round(duration * 16000) - 400) // 320 + 1, 32), 0.001 / 31, np.float32
        )
        probabilities[:, 0] = 0.999
        return dict(
            log_probs=np.log(probabilities),
            vocabulary=vocabulary,
            frame_seconds=0.02,
            offset_seconds=0.0125,
        )

    monkeypatch.setattr(ctc_model, "ctc_emissions", ctc)

    def run(**kwargs):
        return analyze(
            source,
            expected_sha256=digest,
            model_directory=root,
            cache_directory=tmp_path / "cache",
            **kwargs,
        )

    return run, calls, tmp_path / "cache"


def test_valid_cache_reuses_acoustics_after_lyrics_change(analysis_run):
    run, calls, _ = analysis_run
    first = run()
    assert run() == first
    assert calls == dict(beats=1, ctc=1, decode=1)
    changed = run(lyrics="HELLO", lyrics_status="supplied")
    assert changed["cacheKey"] != first["cacheKey"]
    assert calls == dict(beats=1, ctc=1, decode=2)


@pytest.mark.parametrize(
    "corrupt", ["truncated", "missing", "nonfinite", "wrong-type", "schema", "vocal-provenance"]
)
def test_full_result_cache_corruption_recomputes_without_reloading_weights(analysis_run, corrupt):
    run, calls, folder = analysis_run
    result = run()
    full = folder / (result["cacheKey"] + ".json")
    if corrupt == "truncated":
        full.write_text("{")
    elif corrupt == "wrong-type":
        full.write_text("[]")
    elif corrupt == "vocal-provenance":
        result["vocalAnalysis"]["mode"] = "isolated"
        full.write_text(json.dumps(result))
    elif corrupt == "schema":
        result["cues"][0]["kind"] = "unsupported cue"
        full.write_text(json.dumps(result))
    elif corrupt == "missing":
        del result["alignment"]
        full.write_text(json.dumps(result))
    else:
        result["beats"][0]["timeSeconds"] = float("nan")
        full.write_text(json.dumps(result))
    repaired = run()
    assert repaired["beats"][0]["timeSeconds"] == 0.5
    assert repaired["vocalAnalysis"]["mode"] == "mixed"
    assert "alignment" in repaired
    assert calls == dict(beats=1, ctc=1, decode=2)


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"beats":[],"downbeats":[{"timeSeconds":0.5,"confidence":NaN}]}']
)
def test_corrupt_beat_cache_recomputes_only_beats(analysis_run, content):
    run, calls, folder = analysis_run
    result = run()
    (folder / (result["cacheKey"] + ".json")).unlink()
    next(folder.glob("*-beats.json")).write_text(content)
    repaired = run()
    assert repaired["beats"][0]["timeSeconds"] == 0.5
    assert calls == dict(beats=2, ctc=1, decode=2)


@pytest.mark.parametrize(
    "corrupt",
    ["truncated", "missing", "nan", "columns", "frames", "vocabulary", "timing", "unnormalized"],
)
def test_corrupt_ctc_cache_recomputes_only_ctc(analysis_run, corrupt):
    run, calls, folder = analysis_run
    result = run()
    (folder / (result["cacheKey"] + ".json")).unlink()
    target = next(folder.glob("*-ctc.npz"))
    if corrupt == "truncated":
        target.write_bytes(b"PK broken")
    else:
        with np.load(target, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if corrupt == "missing":
            del arrays["scores"]
        elif corrupt == "nan":
            arrays["scores"][0, 0] = np.nan
        elif corrupt == "columns":
            arrays["scores"] = arrays["scores"][:, :2]
        elif corrupt == "frames":
            arrays["scores"] = arrays["scores"][:2]
        elif corrupt == "vocabulary":
            arrays["vocabulary"] = json.dumps({"<pad>": 0, "X": 1})
        elif corrupt == "timing":
            arrays["frame_seconds"] = 0.1
        elif corrupt == "unnormalized":
            arrays["scores"][:] = 0.0
        np.savez_compressed(target, **arrays)
    assert run()["alignment"]["status"] == "no_words_detected"
    assert calls == dict(beats=1, ctc=2, decode=2)


def test_cancellation_during_corrupt_cache_recovery_propagates(analysis_run):
    run, calls, folder = analysis_run
    result = run()
    (folder / (result["cacheKey"] + ".json")).unlink()
    next(folder.glob("*-beats.json")).write_text("{")

    class Cancelled(RuntimeError):
        pass

    def check():
        if calls["decode"] >= 2:
            raise Cancelled("stopped")

    with pytest.raises(Cancelled):
        run(check=check)
    assert calls["beats"] == 1
    assert not (folder / (result["cacheKey"] + ".json")).exists()


def test_neural_workflow_resume_checks_model_and_analysis_version(
    analysis_run, monkeypatch, tmp_path
):
    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.audio_analysis import service, setup
    from wee_todd_mlx.workflows.service import builtin, dispatch

    run, _calls, _directory = analysis_run
    baseline = run()
    calls = []
    identity = {"fixture": "fixed"}
    monkeypatch.setattr(setup, "model_identity", lambda *_a, **_k: dict(identity))

    def counted(*_a, **_k):
        calls.append(1)
        return baseline

    monkeypatch.setattr(service, "analyze", counted)
    doc = builtin("music-video-planning")
    doc["steps"] = [s for s in doc["steps"] if s["id"] == "analysis"]
    doc["outputs"] = {"analysis": {"step": "analysis", "output": "analysis"}}
    source = tmp_path / "song.wav"
    request = dict(
        definition=doc,
        runDirectory=str(tmp_path / "workflow"),
        inputs=dict(
            audio_path=str(source),
            audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            analysis_model_directory=str(tmp_path / "models"),
            analysis_mode="neural",
        ),
    )
    for _ in range(2):
        result = dispatch("workflow-run", request, backend=TextBackend([]))
        assert result["status"] == "completed", result
    assert len(calls) == 1
    monkeypatch.setattr(service, "VERSION", service.VERSION + ".test")
    assert dispatch("workflow-run", request, backend=TextBackend([]))["status"] == "completed"
    assert len(calls) == 2
    identity["fixture"] = "new model"
    assert dispatch("workflow-run", request, backend=TextBackend([]))["status"] == "completed"
    assert len(calls) == 3


def test_setup_cancellation_releases_ownership_lock(tmp_path, monkeypatch):
    from wee_todd_mlx.audio_analysis import setup

    def stop(*_args, **_kwargs):
        raise InterruptedError("cancel")

    monkeypatch.setattr(setup, "download_file", stop)
    with pytest.raises(InterruptedError):
        setup.download(tmp_path)
    assert not (tmp_path / "WeeTodd-Analysis/.setup-lock").exists()


def test_hidden_lyrics_are_ignored_when_status_is_unknown(analysis_run):
    run, _calls, _directory = analysis_run
    result = run(lyrics="HIDDEN OLD LYRICS", lyrics_status="unknown")
    assert result["alignment"]["status"] == "no_words_detected"


def test_deeply_nested_result_cache_is_recomputed(analysis_run):
    run, calls, folder = analysis_run
    first = run()
    nested = 0
    for _ in range(70):
        nested = [nested]
    changed = dict(first, alignment=nested)
    (folder / (first["cacheKey"] + ".json")).write_text(json.dumps(changed))
    assert run()["alignment"] == first["alignment"]
    assert calls["ctc"] == 1


@pytest.mark.parametrize("analysis_run", [6.0], indirect=True)
def test_extra_recognized_words_prevent_false_vocal_break(analysis_run, monkeypatch, tmp_path):
    from wee_todd_mlx.audio_analysis import ctc_model

    run, _calls, _folder = analysis_run
    vocabulary = json.loads((tmp_path / "models/wav2vec2-base-960h/vocab.json").read_text())
    probabilities = np.full((299, 32), 0.001 / 31, dtype=np.float32)
    probabilities[:, 0] = 0.999
    for frame, text in [(1, "A"), (61, "B"), (121, "B"), (181, "B"), (241, "C")]:
        for offset, char in [(0, text), (2, "|")]:
            probabilities[frame + offset, :] = 0.001 / 31
            probabilities[frame + offset, vocabulary[char]] = 0.999
    monkeypatch.setattr(
        ctc_model,
        "ctc_emissions",
        lambda *_a, **_k: dict(
            log_probs=np.log(probabilities),
            vocabulary=vocabulary,
            frame_seconds=0.02,
            offset_seconds=0.0125,
        ),
    )
    result = run(lyrics="A C", lyrics_status="supplied")
    assert len(result["alignment"]["extraWords"]) == 3
    assert result["vocalGaps"] == []


def test_vocal_mode_rejects_invalid_or_fast_request(analysis_run):
    run, _calls, _folder = analysis_run
    with pytest.raises(ValueError, match="vocal"):
        run(vocal_mode="unknown")
    with pytest.raises(ValueError, match="[Ll]earned|neural"):
        run(vocal_mode="isolated", mode="fast")


@pytest.fixture
def vocal_run(analysis_run, monkeypatch):
    """Replace weighted stages only; exercise real independent cache persistence."""
    import sys
    from types import SimpleNamespace

    from wee_todd_mlx.audio_analysis import beat_model, ctc_model, service

    run, calls, folder = analysis_run
    stage_calls = []
    stereo = np.column_stack((np.full(88200, 0.2, np.float32), np.full(88200, -0.1, np.float32)))
    old_decode = service._decode_audio
    old_beats, old_ctc = beat_model.beat_events, ctc_model.ctc_emissions
    identities = {
        "beat-this-small0/model": "beat",
        "wav2vec2-base-960h/model": "ctc",
        "umxhq-vocals/model": "vocal",
    }
    monkeypatch.setattr(
        service,
        "model_identity",
        lambda *_a, **kw: {
            k: v
            for k, v in identities.items()
            if kw.get("include_vocals") or not k.startswith("umxhq")
        },
    )

    def decode(command, **kwargs):
        if command[command.index("-ac") + 1] == "2":
            stage_calls.append("stereo_decode")
            kwargs["check"]()
            return stereo.tobytes()
        return old_decode(command, **kwargs)

    def separate(samples, rate, directory, check=None):
        check()
        assert rate == 44100
        np.testing.assert_array_equal(samples, stereo)
        stage_calls.append("separate")
        return dict(vocals=samples * 0.5, sample_rate=rate, metadata={"method": "test native"})

    def beats(samples, *args, **kwargs):
        np.testing.assert_array_equal(samples, np.zeros(44100, np.float32))
        stage_calls.append("beats")
        return old_beats(samples, *args, **kwargs)

    def ctc(samples, *args, **kwargs):
        stage_calls.append(("ctc", float(np.mean(samples))))
        return old_ctc(samples, *args, **kwargs)

    monkeypatch.setattr(service, "_decode_audio", decode)
    monkeypatch.setattr(beat_model, "beat_events", beats)
    monkeypatch.setattr(ctc_model, "ctc_emissions", ctc)
    monkeypatch.setitem(
        sys.modules,
        "wee_todd_mlx.audio_analysis.separation",
        SimpleNamespace(separate_vocals=separate, SETTINGS_ID="test-settings"),
    )
    return run, calls, folder, stage_calls, identities


def test_opt_in_vocals_only_feed_ctc_and_reuse_beats(vocal_run):
    run, calls, _folder, stages, _identities = vocal_run
    mixed = run()
    isolated = run(vocal_mode="isolated")
    assert isolated["cacheKey"] != mixed["cacheKey"]
    assert isolated["sourceSHA256"] == mixed["sourceSHA256"]
    assert isolated["vocalAnalysis"]["mode"] == "isolated"
    assert calls["beats"] == 1
    assert stages.index("separate") < len(stages) - 1
    ctc_inputs = [value for kind, value in (s for s in stages if isinstance(s, tuple))]
    assert ctc_inputs[0] == 0
    assert ctc_inputs[1] == pytest.approx(0.025, abs=1e-5)
    run(vocal_mode="isolated", lyrics_status="supplied", lyrics="HELLO")
    assert stages.count("separate") == 1
    assert calls["ctc"] == 2


def test_stem_survives_ctc_model_change_and_invalidation_is_independent(vocal_run):
    run, calls, _folder, stages, identities = vocal_run
    run(vocal_mode="isolated")
    identities["wav2vec2-base-960h/model"] = "ctc-v2"
    run(vocal_mode="isolated")
    assert stages.count("separate") == 1
    assert calls["beats"] == 1
    assert calls["ctc"] == 2
    identities["umxhq-vocals/model"] = "vocals-v2"
    run(vocal_mode="isolated")
    assert stages.count("separate") == 2
    assert calls["beats"] == 1
    assert calls["ctc"] == 3


def test_instrumental_never_requires_or_runs_vocal_stage(vocal_run):
    run, calls, _folder, stages, _identities = vocal_run
    result = run(vocal_mode="isolated", lyrics_status="instrumental")
    assert result["alignment"] is None
    assert "separate" not in stages
    assert calls["ctc"] == 0
    assert not any(name.startswith("umxhq") for name in result["modelProvenance"])


def test_cancel_after_separation_does_not_start_ctc(vocal_run):
    run, calls, folder, stages, _identities = vocal_run

    def check():
        if "separate" in stages:
            raise InterruptedError("stop after isolation")

    with pytest.raises(InterruptedError):
        run(vocal_mode="isolated", check=check)
    assert calls["ctc"] == 0
    assert not list(folder.glob("*-vocals.npy"))


@pytest.mark.parametrize("corruption", ["truncated", "archive", "shape", "nonfinite"])
def test_corrupt_vocal_cache_recomputes_stem_after_ctc_change(vocal_run, corruption):
    run, calls, folder, stages, identities = vocal_run
    run(vocal_mode="isolated")
    target = next(folder.glob("*-vocals.npy"))
    if corruption == "truncated":
        target.write_bytes(b"broken")
    elif corruption == "archive":
        with target.open("wb") as stream:
            np.savez(stream, audio=np.zeros(10))
    else:
        values = np.zeros((88200, 2 if corruption == "nonfinite" else 1), np.float32)
        if corruption == "nonfinite":
            values[0, 0] = np.nan
        np.save(target, values)
    receipt = next(folder.glob("*-vocals.json"))
    content = json.loads(receipt.read_text())
    content["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(content))
    identities["wav2vec2-base-960h/model"] = "ctc-v2"
    run(vocal_mode="isolated")
    assert stages.count("separate") == 2
    assert calls["beats"] == 1


def test_setup_models_are_optional_and_hashing_follows_selection(tmp_path, monkeypatch):
    from wee_todd_mlx.audio_analysis import setup

    required = dict(target="ctc/weights", size=4, sha256=hashlib.sha256(b"base").hexdigest())
    optional = dict(
        target="umxhq-vocals/weights", size=5, sha256=hashlib.sha256(b"vocal").hexdigest()
    )
    monkeypatch.setattr(
        setup,
        "catalog",
        lambda: {
            "models": [
                dict(id="ctc", files=[required]),
                dict(id="umxhq-vocals", optional=True, files=[optional]),
            ]
        },
    )
    (tmp_path / "ctc").mkdir()
    (tmp_path / required["target"]).write_bytes(b"base")
    assert model_identity(tmp_path) == {required["target"]: required["sha256"]}
    with pytest.raises(ValueError, match="[Ss]et up|model"):
        model_identity(tmp_path, include_vocals=True)
    (tmp_path / "umxhq-vocals").mkdir()
    (tmp_path / optional["target"]).write_bytes(b"vocal")
    assert model_identity(tmp_path, include_vocals=True)[optional["target"]] == optional["sha256"]


def test_cancellation_after_separation_checks_before_cache_publication(vocal_run):
    run, calls, folder, stages, _identities = vocal_run

    def stop():
        if "separate" in stages:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        run(vocal_mode="isolated", check=stop)
    assert calls["ctc"] == 0
    assert not list(folder.glob("*-vocals.npy"))


def test_pinned_optional_vocal_download_provider_is_restricted():
    item = next(
        f
        for m in catalog()["models"]
        if m["id"] == "umxhq-vocals"
        for f in m["files"]
        if f["filename"].endswith(".pth")
    )
    assert (
        DownloadFile(**item).url == "https://zenodo.org/records/3370489/files/vocals-b62c91ce.pth"
    )
    assert next(m for m in catalog()["models"] if m["id"] == "umxhq-vocals")["optional"]
    with pytest.raises(ValueError):
        DownloadFile(**dict(item, filename="untrusted.pth"))


def test_vocal_workflow_keeps_previous_analysis_contract_usable():
    from wee_todd_mlx.workflows.schema import contracts
    from wee_todd_mlx.workflows.service import builtin
    from wee_todd_mlx.workflows.validation import validate_document

    doc = builtin("music-video-planning")
    analysis = next(s for s in doc["steps"] if s["id"] == "analysis")
    assert analysis["operation"] == "music.analyze@3"
    assert doc["inputs"]["analysis_vocal_mode"]["default"] == "mixed"
    assert "analysis_vocal_mode" not in contracts()[2]["music.analyze@2"]["inputs"]
    assert validate_document(doc)["valid"]


def test_vocal_settings_invalidate_stem_and_ctc_but_keep_mixture_beats(vocal_run):
    import sys

    run, calls, _folder, stages, _identities = vocal_run
    run(vocal_mode="isolated")
    sys.modules["wee_todd_mlx.audio_analysis.separation"].SETTINGS_ID = "changed-window"
    run(vocal_mode="isolated")
    assert stages.count("separate") == 2
    assert calls["ctc"] == 2
    assert calls["beats"] == 1


@pytest.mark.parametrize("field,value", [("lyricAssistedText", 7), ("recognizedText", None)])
def test_bad_cached_transcription_is_rebuilt_without_reloading_models(analysis_run, field, value):
    run, calls, folder = analysis_run
    first = run(lyrics="THEY CAN REMEMBER", lyrics_status="supplied")
    broken = json.loads(json.dumps(first))
    broken["alignment"][field] = value
    (folder / (first["cacheKey"] + ".json")).write_text(json.dumps(broken))
    repaired = run(lyrics="THEY CAN REMEMBER", lyrics_status="supplied")
    assert repaired["alignment"] == first["alignment"]
    assert calls["ctc"] == 1


@pytest.mark.parametrize("field,value", [("verification", []), ("observedText", {})])
def test_bad_cached_lyric_comparison_is_rebuilt(analysis_run, field, value):
    run, calls, folder = analysis_run
    first = run(lyrics="HELLO", lyrics_status="supplied")
    broken = json.loads(json.dumps(first))
    broken["alignment"]["words"][0][field] = value
    (folder / (first["cacheKey"] + ".json")).write_text(json.dumps(broken))
    assert run(lyrics="HELLO", lyrics_status="supplied")["alignment"] == first["alignment"]
    assert calls["ctc"] == 1


def test_heading_only_supplied_lyrics_reuse_transcription_result(analysis_run, monkeypatch):
    from wee_todd_mlx.audio_analysis import ctc_model

    run, calls, _ = analysis_run
    original = ctc_model.ctc_emissions

    def spoken(*args, **kwargs):
        evidence = original(*args, **kwargs)
        evidence["log_probs"][20, :] = np.log(0.001 / 31)
        evidence["log_probs"][20, evidence["vocabulary"]["A"]] = np.log(0.999)
        return evidence

    monkeypatch.setattr(ctc_model, "ctc_emissions", spoken)
    first = run(lyrics="[Verse]", lyrics_status="supplied")
    assert first["alignment"]["recognizedText"] == "A"
    assert run(lyrics="[Verse]", lyrics_status="supplied") == first
    assert calls["decode"] == 1
    assert calls["ctc"] == 1
