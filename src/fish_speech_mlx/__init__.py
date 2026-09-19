"""WeeTodd's process-local Fish S2 Pro engine. Importing never loads MLX or weights."""


def inspect_checkpoint(model_path, *, precision="auto"):
    from wee_todd_mlx.speech_checkpoint import inspect

    return inspect(model_path, engine="fishS2Pro", precision=precision)


def generate(request, output, *, progress=None, cancelled=None):
    from .pipeline import generate as execute

    return execute(request, output, progress=progress, cancelled=cancelled)


def unload():
    from .pipeline import unload as release

    release()
