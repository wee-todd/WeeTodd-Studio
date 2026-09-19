"""Fish S2 Pro dual autoregressive transformer and reference prompt contract."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from wee_todd_mlx.speech_ops import Weights, attention, gated_mlp, rope, sample


def build_prompt(tokenizer, text, reference_codes=None, transcript=""):
    segments = []

    def append(value):
        ids = tokenizer.encode(value).ids
        block = np.zeros((11, len(ids)), dtype=np.int32)
        block[0] = ids
        segments.append(block)

    append("<|im_start|>system\n")
    if reference_codes is None:
        append("convert the provided text to speech")
    else:
        append("convert the provided text to speech reference to the following:\n\nText:\n")
        append(transcript if "<|speaker:" in transcript else "<|speaker:0|>" + transcript)
        append("\n\nSpeech:\n")
        codes = np.asarray(reference_codes, dtype=np.int32)
        segments.append(np.concatenate([codes[:1] + 151678, codes], axis=0))
    append("<|im_end|>\n<|im_start|>user\n")
    append(text)
    append("<|im_end|>\n<|im_start|>assistant\n<|voice|>")
    return mx.array(np.concatenate(segments, axis=1))


class FishTransformer(Weights):
    def __init__(self, tensors, config):
        tensors = dict(tensors)
        if "text_model.model.embeddings.weight" in tensors:
            mapping = {
                "audio_decoder.codebook_embeddings": "codebook_embeddings",
                "audio_decoder.embeddings": "fast_embeddings",
                "audio_decoder.layers": "fast_layers",
                "audio_decoder.norm": "fast_norm",
                "audio_decoder.output": "fast_output",
                "audio_decoder.project_in": "fast_project_in",
            }
            renamed = {}
            for k, v in tensors.items():
                k = k.removeprefix("text_model.model.")
                for source, target in mapping.items():
                    if k.startswith(source + "."):
                        k = target + k[len(source) :]
                        break
                renamed[k] = v
            tensors = renamed
        super().__init__(tensors)
        self.config = config

    def blocks(self, x, config, prefix, caches):
        h, kv, d = config["n_head"], config["n_local_heads"], config["head_dim"]
        for i, cache in enumerate(caches):
            p = f"{prefix}.{i}"
            z = self.norm(x, p + ".attention_norm", config["norm_eps"])
            qkv = self.linear(z, p + ".attention.wqkv")
            q, k, v = mx.split(qkv, [h * d, (h + kv) * d], axis=-1)
            q = q.reshape(1, -1, h, d).transpose(0, 2, 1, 3)
            k, v = [a.reshape(1, -1, kv, d).transpose(0, 2, 1, 3) for a in (k, v)]
            if config["attention_qk_norm"]:
                q, k = (
                    self.norm(q, p + ".attention.q_norm", config["norm_eps"]),
                    self.norm(k, p + ".attention.k_norm", config["norm_eps"]),
                )
            offset = cache[0].shape[-2] if cache else 0
            q, k = [
                rope(a, offset, config["rope_base"], adjacent=True, rounded=True) for a in (q, k)
            ]
            y = attention(q, k, v, cache).transpose(0, 2, 1, 3).reshape(1, x.shape[1], h * d)
            x = x + self.linear(y, p + ".attention.wo")
            x = x + gated_mlp(
                self, self.norm(x, p + ".ffn_norm", config["norm_eps"]), p + ".feed_forward"
            )
        return x

    def slow(self, tokens, caches):
        semantic = tokens[0]
        x = self.embedding(semantic, "embeddings")
        codes = sum(
            self.embedding(tokens[i + 1] + i * 4096, "codebook_embeddings") for i in range(10)
        )
        mask = ((semantic >= 151678) & (semantic <= 155773))[:, None]
        x = mx.where(mask, (x + codes) / 11**0.5, x)[None]
        x = self.blocks(x, self.config["text_config"], "layers", caches)
        x = self.norm(x[:, -1:], "norm", self.config["text_config"]["norm_eps"])
        return self.linear(x, "embeddings").reshape(-1), x

    def fast(self, x, caches):
        x = self.blocks(x, self.config["audio_decoder_config"], "fast_layers", caches)
        return self.linear(
            self.norm(x[:, -1:], "fast_norm", self.config["audio_decoder_config"]["norm_eps"]),
            "fast_output",
        ).reshape(-1)

    def generate(
        self,
        prompt,
        *,
        max_tokens,
        temperature,
        top_p,
        top_k,
        cancelled=lambda: False,
        progress=None,
    ):
        if prompt.shape[1] + max_tokens > self.config["text_config"]["max_seq_len"]:
            raise ValueError("Fish reference and text exceed the context window")
        caches = [[] for _ in range(self.config["text_config"]["n_layer"])]
        logits, hidden = self.slow(prompt, caches)
        frames, history = [], []
        ended = False
        ids = mx.arange(logits.size)
        allowed = ((ids >= 151678) & (ids <= 155773)) | (ids == 151645)
        for step in range(max_tokens):
            if cancelled():
                raise InterruptedError("Speech generation cancelled")
            restricted = mx.where(allowed, logits, -mx.inf)
            token = sample(restricted, temperature=temperature, top_p=top_p, top_k=top_k)
            if token in history[-10:] and token != 151645:
                token = sample(restricted, temperature=1.0, top_p=0.9, top_k=top_k)
            if token == 151645:
                ended = True
                break
            history.append(token)
            code = token - 151678
            codes = [code]
            fast_caches = [[] for _ in range(self.config["audio_decoder_config"]["n_layer"])]
            mx.eval(self.fast(hidden, fast_caches))
            for _ in range(9):
                logits_fast = self.fast(
                    self.embedding(mx.array([[code]]), "fast_embeddings"), fast_caches
                )
                code = sample(logits_fast, temperature=temperature, top_p=top_p, top_k=top_k)
                codes.append(code)
            frames.append(codes)
            if progress and step % 10 == 0:
                progress(step + 1, max_tokens)
            logits, hidden = self.slow(mx.array([[token] + codes]).T, caches)
            mx.eval(logits, hidden)
        if not frames:
            raise ValueError("Fish produced no speech frames")
        return np.array(frames, dtype=np.int32).T, not ended
