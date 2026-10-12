"""Independent native MLX inference for the merged SheetSage2 checkpoint.

No model runtime is imported until transcription. Model weights remain external
and retain their own license; importing this package executes no publisher code.
"""

from .checkpoint import inspect_checkpoint


def transcribe(audio, model_path, melody_only=True, progress=None, cancelled=None):
    """Analyze one finite mono 24 kHz waveform, between 1025 samples and 300 s.

    ``progress`` receives dictionaries with stage/message/fraction; ``cancelled``
    is an optional zero-argument boolean callback. Results are model predictions
    requiring human review, not a claim of an exact score or rights clearance.
    """
    import numpy as np

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise ValueError("SheetSage2 expects mono audio sampled at 24,000 Hz")
    if len(samples) < 1025:
        raise ValueError("SheetSage2 needs at least 1025 audio samples at 24,000 Hz")
    if len(samples) > 7_200_000:
        raise ValueError(
            "Choose a reference excerpt of at most 300 seconds; audio is never truncated"
        )
    if samples.dtype.kind != "f" or not np.isfinite(samples).all():
        raise ValueError("SheetSage2 audio must be finite floating-point samples")
    if not isinstance(melody_only, bool):
        raise ValueError("melody_only must be true or false")
    if cancelled is not None and cancelled():
        raise InterruptedError("SheetSage2 transcription cancelled")
    info = inspect_checkpoint(model_path)
    from .inference import run

    return run(samples.astype(np.float32, copy=False), info, melody_only, progress, cancelled)


__all__ = ["inspect_checkpoint", "transcribe"]
