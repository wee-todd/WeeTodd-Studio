"""Qwen3-TTS Base/CustomVoice talker and residual-code predictor in native MLX."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from wee_todd_mlx.speech_ops import Weights, attention, gated_mlp, rope, sample


def text_rope(x, offset, base):
    # For text/audio-only prompts all three MRoPE axes use the same positions.
    # Interleaving [24,20,20] therefore reduces exactly to ordinary split-half RoPE.
    return rope(x, offset, base)


def transformer(
    weights,
    x,
    prefix,
    config,
    caches=None,
    *,
    norm_kind="rms",
    scales=False,
    gelu=False,
    final_norm=True,
    window=None,
):
    n = config["num_hidden_layers"]
    caches = [None] * n if caches is None else caches
    eps = config.get("rms_norm_eps", config.get("norm_eps", 1e-6))
    norm = weights.norm if norm_kind == "rms" else weights.layer_norm
    d = config.get("head_dim", 64)
    for i, cache in enumerate(caches):
        p = f"{prefix}.layers.{i}"
        z = norm(x, p + ".input_layernorm", eps)
        q, k, v = [
            weights.linear(z, p + ".self_attn." + name)
            .reshape(1, x.shape[1], -1, d)
            .transpose(0, 2, 1, 3)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        if p + ".self_attn.q_norm.weight" in weights.w:
            q, k = (
                weights.norm(q, p + ".self_attn.q_norm", eps),
                weights.norm(k, p + ".self_attn.k_norm", eps),
            )
        offset = cache[0].shape[-2] if cache else 0
        q, k = [text_rope(a, offset, config.get("rope_theta", 1000000)) for a in (q, k)]
        mask = None
        if window and x.shape[1] > 1:
            r = mx.arange(offset, offset + x.shape[1])[:, None]
            c = mx.arange(offset + x.shape[1])[None]
            mask = (c <= r) & (c > r - window)
        y = attention(q, k, v, cache, mask).transpose(0, 2, 1, 3).reshape(1, x.shape[1], -1)
        y = weights.linear(y, p + ".self_attn.o_proj")
        x = x + y * (weights.w[p + ".self_attn_layer_scale.scale"] if scales else 1)
        z = norm(x, p + ".post_attention_layernorm", eps)
        if gelu:
            y = weights.linear(nn.gelu(weights.linear(z, p + ".mlp.fc1")), p + ".mlp.fc2")
        else:
            y = gated_mlp(weights, z, p + ".mlp", ("gate_proj", "up_proj", "down_proj"))
        x = x + y * (weights.w[p + ".mlp_layer_scale.scale"] if scales else 1)
    return norm(x, prefix + ".norm", eps) if final_norm else x


class QwenTalker(Weights):
    def __init__(self, tensors, config):
        super().__init__(tensors)
        self.config = config
        self.talker = config["talker_config"]

    def text(self, ids):
        z = self.embedding(mx.array([ids]), "talker.model.text_embedding")
        return self.linear(
            nn.silu(self.linear(z, "talker.text_projection.linear_fc1")),
            "talker.text_projection.linear_fc2",
        )

    def codec(self, ids):
        return self.embedding(mx.array([ids]), "talker.model.codec_embedding")

    def combined_codes(self, codes):
        z = self.codec(codes[0].tolist())
        for i in range(15):
            z = z + self.embedding(
                mx.array([codes[i + 1].tolist()]),
                f"talker.code_predictor.model.codec_embedding.{i}",
            )
        return z

    def custom_voice_prompt(self, tokenizer, text, speaker, instruct, language):
        """Named codec timbre plus an optional preceding user instruction turn."""
        cfg = self.talker
        speaker = speaker.lower()
        if speaker not in cfg.get("spk_id", {}):
            raise ValueError("Unsupported Qwen CustomVoice speaker")
        language = language.lower()
        dialect = cfg.get("spk_is_dialect", {}).get(speaker)
        if language in {"auto", "chinese"} and dialect:
            language = dialect
        prompt, trailing, pad = self.prompt(
            tokenizer, text, "", self.codec([cfg["spk_id"][speaker]]), None, language
        )
        if instruct:
            instruction = tokenizer.encode(
                f"<|im_start|>user\n{instruct}<|im_end|>\n", add_special_tokens=False
            )
            prompt = mx.concatenate([self.text(instruction), prompt], axis=1)
        return prompt, trailing, pad

    def prompt(self, tokenizer, text, transcript, speaker, reference_codes, language):
        cfg = self.talker
        language = language.lower()
        if language != "auto" and language not in cfg["codec_language_id"]:
            raise ValueError("Unsupported Qwen speech language")
        tts = self.text(
            [self.config[k] for k in ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id")]
        )
        bos, eos, pad = [tts[:, i : i + 1] for i in range(3)]
        ids = tokenizer.encode(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        role = self.text(ids[:3])
        prefix = (
            [cfg["codec_nothink_id"], cfg["codec_think_bos_id"], cfg["codec_think_eos_id"]]
            if language == "auto"
            else [
                cfg["codec_think_id"],
                cfg["codec_think_bos_id"],
                cfg["codec_language_id"][language],
                cfg["codec_think_eos_id"],
            ]
        )
        prefix = mx.concatenate(
            [
                self.codec(prefix),
                speaker.reshape(1, 1, -1).astype(pad.dtype),
                self.codec([cfg["codec_pad_id"], cfg["codec_bos_id"]]),
            ],
            axis=1,
        )
        combined = (
            mx.concatenate(
                [mx.broadcast_to(pad, (1, prefix.shape[1] - 2, pad.shape[-1])), bos], axis=1
            )
            + prefix[:, :-1]
        )
        if reference_codes is not None:
            ref_ids = tokenizer.encode(
                f"<|im_start|>assistant\n{transcript}<|im_end|>\n", add_special_tokens=False
            )[3:-2]
            words = mx.concatenate([self.text(ref_ids + ids[3:-5]), eos], axis=1) + self.codec(
                [cfg["codec_pad_id"]]
            )
            reference = (
                mx.concatenate(
                    [self.codec([cfg["codec_bos_id"]]), self.combined_codes(reference_codes)],
                    axis=1,
                )
                + pad
            )
            return mx.concatenate([role, combined, words, reference], axis=1), pad, pad
        words = self.text(ids)
        return (
            mx.concatenate([role, combined, words[:, 3:4] + prefix[:, -1:]], axis=1),
            mx.concatenate([words[:, 4:-5], eos], axis=1),
            pad,
        )

    def generate(
        self,
        prompt,
        trailing,
        pad,
        *,
        max_tokens,
        temperature,
        top_p,
        top_k,
        cancelled=lambda: False,
        progress=None,
    ):
        cfg = self.talker
        if prompt.shape[1] + max_tokens > cfg["max_position_embeddings"]:
            raise ValueError("Qwen reference and text exceed the context window")
        caches = [[] for _ in range(cfg["num_hidden_layers"])]
        frames, ended = [], False
        for step in range(max_tokens):
            if cancelled():
                raise InterruptedError("Speech generation cancelled")
            hidden = transformer(self, prompt, "talker.model", cfg, caches)[:, -1:]
            logits = self.linear(hidden, "talker.codec_head").reshape(-1).astype(mx.float32)
            ids = mx.arange(logits.size)
            logits = mx.where((ids < 2048) | (ids == cfg["codec_eos_token_id"]), logits, -mx.inf)
            if frames:
                repeated = mx.array(list({frame[0] for frame in frames}))
                values = logits[repeated]
                logits[repeated] = mx.where(values < 0, values * 1.05, values / 1.05)
            token = sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            if token == cfg["codec_eos_token_id"]:
                ended = True
                break
            codes = [token]
            small_cfg = cfg["code_predictor_config"]
            fast_cache = [[] for _ in range(small_cfg["num_hidden_layers"])]
            x = mx.concatenate([hidden, self.codec([token])], axis=1)
            for j in range(15):
                if j:
                    x = self.embedding(
                        mx.array([[codes[-1]]]),
                        f"talker.code_predictor.model.codec_embedding.{j - 1}",
                    )
                if "talker.code_predictor.small_to_mtp_projection.weight" in self.w:
                    x = self.linear(x, "talker.code_predictor.small_to_mtp_projection")
                z = transformer(self, x, "talker.code_predictor.model", small_cfg, fast_cache)[
                    :, -1:
                ]
                code = sample(
                    self.linear(z, f"talker.code_predictor.lm_head.{j}"),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                codes.append(code)
            frames.append(codes)
            prompt = self.combined_codes(np.array(codes)[:, None]) + (
                trailing[:, step : step + 1] if step < trailing.shape[1] else pad
            )
            mx.eval(prompt)
            if progress and step % 10 == 0:
                progress(step + 1, max_tokens)
        if not frames:
            raise ValueError("Qwen produced no speech frames")
        return np.array(frames, dtype=np.int32).T, not ended
