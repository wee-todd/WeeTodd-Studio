"""Checkpoint-native score/semantic boundary tokens and prompt grammar."""

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
CONTEXT, VOCAB_SIZE = 24576, 184704
INSTRUCTIONS = {
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec "
        "tokens from the given conditions."
    ),
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate "
        "music with codec tokens from the given conditions."
    ),
    "off": "Generate music with codec tokens from the given conditions.",
}


def _score(ids):
    ids = list(ids)
    if any(type(i) is not int or not 0 <= i < EOD for i in ids):
        raise ValueError("Score token IDs must be ordinary text tokens")
    return ids


def prefix_tokens(request, tokenizer, abc_ids=None):
    text = (
        f"{INSTRUCTIONS[request['cot']]}\n[Tags]\n{request['style']}\n"
        f"[Lyrics]\n{request['lyrics']}\n"
    )
    tokens = [EOD, *tokenizer.encode(text), ABC_START]
    if request["cot"] == "off":
        return tokens + [ABC_END, MUSIC_START]
    return tokens if abc_ids is None else tokens + _score(abc_ids) + [ABC_END, MUSIC_START]


def negative_tokens(request, tokenizer, abc_ids):
    tokens = [EOD, *tokenizer.encode(INSTRUCTIONS[request["cot"]])]
    if request["cot"] == "off":
        return tokens + [MUSIC_START]
    return tokens + [ABC_START] + _score(abc_ids) + [ABC_END, MUSIC_START]


def chunk_ranges(frames, prefix_length, context=CONTEXT):
    capacity = (context - prefix_length - 3) // 2
    if frames < 1 or capacity < 1:
        raise ValueError("No acoustic frames fit within the remaining context")
    return [(start, min(start + capacity, frames)) for start in range(0, frames, capacity)]
