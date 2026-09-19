"""HighResFix stays intact from saved Studio selection through estimate adoption."""

import copy

import pytest
from test_drawthings_studio import project

from wee_todd_remote.adapter import DrawThingsAdapter
from wee_todd_remote.studio import compose_drawthings_request

HIRES = {"hiresFix": True, "hiresFixWidth": 1024, "hiresFixHeight": 576, "hiresFixStrength": 0.5}


def test_studio_ltx_hires_recipe_survives_composition_and_effective_configuration():
    value = project()
    clip = value["clips"][0]
    clip.update(generationWidth=2048, generationHeight=1152)
    recipe = {**HIRES, "sampler": 17, "steps": 8, "guidanceScale": 1, "shift": 6.5}
    clip["drawThings"]["configuration"] = recipe
    before = copy.deepcopy(value)
    result = compose_drawthings_request(value, "clip-uuid", [], request_id="hires")
    assert all(result["configuration"][key] == setting for key, setting in recipe.items())
    adopted = DrawThingsAdapter._adopt_estimate_configuration(result, result["configuration"])
    assert adopted is not None and adopted["configuration"] == result["configuration"]
    assert value == before
    changed = {**result["configuration"], "hiresFixWidth": 512}
    assert DrawThingsAdapter._adopt_estimate_configuration(result, changed) is None


def test_h3_cannot_silently_accept_ltx_hires_controls():
    value = project()
    value["clips"][0]["drawThings"].update(modelFamily="minimaxH3", configuration=HIRES)
    with pytest.raises(ValueError, match="HighResFix.*LTX"):
        compose_drawthings_request(value, "clip-uuid", [])


@pytest.mark.parametrize("key", ["stage2Steps", "stage2Guidance", "stage2Shift"])
def test_ltx_stage2_controls_remain_unsupported(key):
    value = project()
    value["clips"][0]["drawThings"]["configuration"] = {key: 1}
    with pytest.raises(ValueError, match="Unsupported Draw Things"):
        compose_drawthings_request(value, "clip-uuid", [])
