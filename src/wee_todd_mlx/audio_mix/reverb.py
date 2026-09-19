"""Deterministic stereo ambience with bounded-memory overlap-add convolution.

The independently synthesized response combines irregular early reflections and
three diffuse noise bands with different decay rates. Four decorrelated paths
retain stereo difference information; cross paths spread a panned source gently.
The response has unit total energy per input channel, independent of decay.
Validation deliberately has no NumPy/SciPy import or renderer side effect.
"""

from __future__ import annotations

import math
from numbers import Real

PRESETS = {
    "room": (0.7, 0.012, 0.45),
    "chamber": (1.2, 0.020, 0.45),
    "hall": (2.6, 0.028, 0.35),
    "plate": (1.6, 0.008, 0.65),
}
_BLOCK = 65536


def validate_reverb(value):
    """Validate every supplied control, including on bypassed effects."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Reverb must be an object")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("Reverb enabled must be a boolean")
    preset = value.get("preset", "room")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("Reverb preset must be room, chamber, hall, or plate")
    decay, delay, tone = PRESETS[preset]
    result = {"enabled": enabled, "preset": preset}
    for name, default, low, high in (
        ("mix", 0.18, 0, 1),
        ("decay", decay, 0.2, 6),
        ("tone", tone, 0, 1),
        ("preDelay", delay, 0, 0.1),
    ):
        number = value.get(name, default)
        if (
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not math.isfinite(number)
            or not low <= number <= high
        ):
            raise ValueError(f"Reverb {name} must be a finite number from {low} to {high}")
        result[name] = float(number)
    return result if enabled and result["mix"] else None


def _check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Audio mix cancelled")


def _check_rate(sample_rate):
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or not 8000 <= sample_rate <= 192000
    ):
        raise ValueError("Reverb sample rate must be an integer from 8000 to 192000")


def impulse_response(settings, sample_rate=48000, *, cancelled=lambda: False):
    """Return float32 (samples, output, input) wet response; bypass returns None.

    Decay is the low/mid-band approximate RT60 in seconds. Tone adjusts high-band
    energy and RT60. A short final fade finishes the already sub -70 dB envelope.
    No live source, movie duration, or processing block enters the fixed seed.
    """
    config = validate_reverb(settings)
    _check_rate(sample_rate)
    _check_cancelled(cancelled)
    if config is None:
        return None
    import numpy as np
    from scipy.signal import butter, sosfilt

    preset_index = tuple(PRESETS).index(config["preset"])
    decay, tone = config["decay"], config["tone"]
    count = math.ceil(decay * 1.3 * sample_rate)
    delay = round(config["preDelay"] * sample_rate)
    time = np.arange(count, dtype=np.float64) / sample_rate
    # Preset differences affect density buildup and room reflection scale.
    buildup = (0.011, 0.018, 0.030, 0.004)[preset_index]
    room_scale = (0.026, 0.043, 0.067, 0.014)[preset_index]
    attack = 1 - np.exp(-time / buildup)
    low_filter = butter(2, 450, fs=sample_rate, output="sos")
    mid_filter = butter(
        2, min(1100 + 2000 * tone, sample_rate * 0.35), fs=sample_rate, output="sos"
    )
    high_pass = butter(2, 85, btype="highpass", fs=sample_rate, output="sos")
    response = np.zeros((delay + count, 2, 2), dtype=np.float32)
    for out in range(2):
        for inp in range(2):
            _check_cancelled(cancelled)
            rng = np.random.default_rng(7301 + preset_index * 101 + out * 17 + inp * 43)
            noise = rng.standard_normal(count)
            low = sosfilt(low_filter, noise)
            middle_low = sosfilt(mid_filter, noise)
            middle, high = middle_low - low, noise - middle_low
            envelope = np.exp(-math.log(1000) * time / decay)
            high_envelope = np.exp(-math.log(1000) * time / (decay * (0.18 + 0.68 * tone)))
            tail = attack * (
                (low + middle * 0.8) * envelope + high * (0.08 + 0.92 * tone) * high_envelope
            )
            # Irregular, signed reflections avoid a periodic comb train. Their
            # relative energy stays stable as the number of tail samples varies.
            reflection_gain = math.sqrt(sample_rate * 0.0015)
            for reflection in range(18):
                arrival = 0.003 + room_scale * (reflection + rng.uniform(0.1, 0.9)) / 5
                index = round(arrival * sample_rate)
                if index < count:
                    tail[index] += (
                        rng.choice((-1, 1)) * reflection_gain * math.exp(-3 * arrival / room_scale)
                    )
            tail = sosfilt(high_pass, tail)
            fade = min(count, max(2, round(sample_rate * 0.02)))
            tail[-fade:] *= np.linspace(1, 0, fade)
            energy = float(np.dot(tail, tail))
            # Unit diagonal, modest crossfeed, each input's total energy unity.
            gain = (1 if out == inp else 0.32) / math.sqrt(1 + 0.32**2)
            response[delay:, out, inp] = tail * (gain / math.sqrt(energy))
    _check_cancelled(cancelled)
    return response


def add_reverb(source, destination, settings, *, sample_rate=48000, cancelled=lambda: False):
    """ADD processed source to destination, retaining tails to the movie boundary.

    Source and destination are separate float32 (N, 2) arrays or disk memmaps.
    Dry has no processing delay. Workspace depends on the bounded response length
    and fixed block size, never on N; successive block convolutions add their
    overlapping tails directly into the destination movie bus.
    """
    config = validate_reverb(settings)
    _check_rate(sample_rate)
    _check_cancelled(cancelled)
    import numpy as np

    if (
        source.ndim != 2
        or source.shape[1] != 2
        or destination.shape != source.shape
        or source.dtype != np.float32
        or destination.dtype != np.float32
    ):
        raise ValueError("Reverb requires matching float32 stereo buffers")
    if np.shares_memory(source, destination):
        raise ValueError("Reverb source and destination must be separate buffers")
    if config is None:
        for first in range(0, len(source), _BLOCK):
            _check_cancelled(cancelled)
            destination[first : first + _BLOCK] += source[first : first + _BLOCK]
        return
    from scipy.fft import irfft, next_fast_len, rfft

    response = impulse_response(config, sample_rate, cancelled=cancelled)
    fft_size = next_fast_len(_BLOCK + len(response) - 1)
    kernels = rfft(response, n=fft_size, axis=0)
    del response
    mix = np.float32(config["mix"])
    dry = np.float32(1 - config["mix"])
    for first in range(0, len(source), _BLOCK):
        _check_cancelled(cancelled)
        block = source[first : first + _BLOCK]
        destination[first : first + len(block)] += dry * block
        # This optimization is exact and keeps padded movie silence inexpensive.
        if not np.any(block):
            continue
        spectrum = rfft(block, n=fft_size, axis=0)
        count = min(fft_size, len(source) - first)
        for out in range(2):
            _check_cancelled(cancelled)
            wet = irfft(
                spectrum[:, 0] * kernels[:, out, 0] + spectrum[:, 1] * kernels[:, out, 1],
                n=fft_size,
            )
            destination[first : first + count, out] += mix * wet[:count]
