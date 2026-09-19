"""Frozen Qwen BPE vocabulary with YuE2's two score delimiters."""

import base64
import unicodedata
from pathlib import Path


class Tokenizer:
    def __init__(self, filename):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "YuE2 text encoding requires the tiktoken runtime dependency"
            ) from exc
        ranks = {}
        for line in Path(filename).read_bytes().splitlines():
            if line.strip():
                token, rank = line.split()
                ranks[base64.b64decode(token, validate=True)] = int(rank)
        if len(ranks) != 151643 or set(ranks.values()) != set(range(151643)):
            raise ValueError("qwen.tiktoken must contain the complete 151643-token Qwen vocabulary")
        names = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<R>",
            "<S>",
            "<X>",
            "<mask>",
            "<sep>",
        ]
        names.extend(f"<extra_{i}>" for i in range(200))
        names[204], names[205] = "<abc>", "</abc>"
        pattern = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        self.encoding = tiktoken.Encoding(
            "weetodd-yue2",
            pat_str=pattern,
            mergeable_ranks=ranks,
            special_tokens={name: 151643 + i for i, name in enumerate(names)},
        )

    def encode(self, text):
        return self.encoding.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, tokens):
        return self.encoding.decode(tokens, errors="replace")
