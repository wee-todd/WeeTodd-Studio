"""Small bridge surface for explicit local assistant setup."""

from wee_todd_mlx.assistant_models import catalog, health_check, inspect_model, install


def dispatch(command, request, *, progress=lambda *_: None, cancelled=lambda: False):
    if command == "assistant-model-catalog":
        return catalog()
    if command == "assistant-model-inspect":
        progress("Verifying installed checkpoint…", 0.0)
        return inspect_model(request["path"], cancelled=cancelled)
    if command == "assistant-model-download":
        return install(request["destination"], progress=progress, cancelled=cancelled)
    if command == "assistant-model-health":
        return health_check(
            request["path"],
            request.get("runtime", {}).get("drawThingsHelperPath", ""),
            progress=progress,
            cancelled=cancelled,
        )
    raise ValueError("Unknown assistant model operation")
