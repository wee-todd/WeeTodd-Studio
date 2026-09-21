"""Flow-matching Euler with the checkpoint's dynamic exponential time shift."""

import math

import numpy as np


def make_schedule(steps: int, target_tokens: int, scheduler_config: dict):
    if type(steps) is not int or steps < 1 or target_tokens < 1:
        raise ValueError("Steps and target token count must be positive")
    c = scheduler_config
    base, maximum = c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 8192)
    if maximum <= base:
        raise ValueError("Invalid scheduler sequence-length range")
    mu = c.get("base_shift", 0.5) + (target_tokens - base) * (
        c.get("max_shift", 0.9) - c.get("base_shift", 0.5)
    ) / (maximum - base)
    sigma = np.linspace(1, 1 / steps, steps, dtype=np.float64)
    sigma = math.exp(mu) / (math.exp(mu) + 1 / sigma - 1)
    terminal = c.get("shift_terminal", 0.02)
    if terminal is not None and steps > 1:
        sigma = 1 - (1 - sigma) * (1 - terminal) / (1 - sigma[-1])
    return np.concatenate([sigma, [0]]).astype(np.float32)


def euler_step(latents, prediction, sigma, next_sigma):
    return latents + (float(next_sigma) - float(sigma)) * prediction
