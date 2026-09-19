"""WeeTodd's process-local Qwen3-TTS Base engine, with deferred weighted imports."""


def inspect_checkpoint(model_path, *, precision="auto"):
    from wee_todd_mlx.speech_checkpoint import inspect

    return inspect(model_path, engine="qwen3TTS", precision=precision)


def generate(request, output, *, progress=None, cancelled=None):
    from .pipeline import generate as execute

    return execute(request, output, progress=progress, cancelled=cancelled)


def unload():
    from .pipeline import unload as release

    release()
