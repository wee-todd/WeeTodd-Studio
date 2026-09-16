"""Native scene timing and global image contracts; no model weights required."""

import importlib
import math
import subprocess
import sys

import pytest


def scene_plan(durations, **kwargs):
    module = importlib.import_module("ltx25_mlx.chain_plan")
    return module.plan_ltx25_scene(durations, **kwargs)


def test_six_five_second_segments_keep_exact_delivered_ranges():
    plan = scene_plan([5] * 6)
    assert plan.window_frame_counts == (121, 145, 145, 145, 145, 145)
    assert plan.window_start_frames == (0, 96, 216, 336, 456, 576)
    assert plan.segment_frame_counts == (120,) * 6
    assert plan.segment_start_frames == (0, 120, 240, 360, 480, 600)
    assert plan.total_frames == 721
    assert plan.expected_audio_tokens == 751
    assert plan.as_dict()["delivered_duration_seconds"] == 30


def test_unequal_segments_use_cumulative_grid_rounding_and_audio_clock():
    plan = scene_plan([1.4, 2.3, 3.8], frame_rate=25)
    assert plan.segment_start_frames == (0, 32, 96)
    assert plan.segment_frame_counts == (32, 64, 88)
    assert plan.window_frame_counts == (33, 89, 113)
    cumulative = plan.window_audio_token_counts[0]
    for i, trim in enumerate(plan.join_audio_tokens, 1):
        cumulative += plan.window_audio_token_counts[i] - trim
        assert cumulative == round((sum(plan.segment_frame_counts[: i + 1]) + 1) / 25 * 25)
    assert cumulative == plan.expected_audio_tokens
    assert sum(plan.segment_frame_counts) == round(sum([1.4, 2.3, 3.8]) * 25 / 8) * 8


@pytest.mark.parametrize(
    "durations,kwargs",
    [
        ([5], {}),
        ([1] * 7, {}),
        ([16, 16], {}),
        ([0, 5], {}),
        ([math.nan, 5], {}),
        ([math.inf, 5], {}),
        ([5, 5], {"overlap_frames": 24}),
        ([5, 5], {"overlap_frames": 1}),
        ([5, 5], {"frame_rate": math.nan}),
        ([0.1, 5], {}),
    ],
)
def test_scene_invalid_geometry_rejected(durations, kwargs):
    with pytest.raises(ValueError):
        scene_plan(durations, **kwargs)


def test_planner_import_does_not_import_mlx_or_comfy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ltx25_mlx.chain_plan; "
            "assert 'mlx.core' not in sys.modules; assert 'comfy' not in sys.modules",
        ],
        check=True,
    )


def test_global_images_cover_overlap_and_last_visible_frame(tmp_path):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    module = importlib.import_module("ltx25_mlx.chain_plan")
    plan = scene_plan([5] * 6)
    images = [ImageConditioningInput(str(tmp_path / str(i)), i, 1.0) for i in (0, 100, 719)]
    windows = module.route_scene_images(images, plan)
    assert [x.frame_idx for x in windows[0]] == [0, 100]
    assert [x.frame_idx for x in windows[1]] == [4]
    assert [x.frame_idx for x in windows[-1]] == [143]
    assert windows[-1][-1].frame_idx + plan.window_start_frames[-1] == 719
    # The endpoint uses exact pixel coordinates; arbitrary keyframe positions
    # are supported by VideoConditionByKeyframeIndex, not quantized to 8n.


def test_conflicting_or_invisible_scene_images_rejected():
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    module = importlib.import_module("ltx25_mlx.chain_plan")
    plan = scene_plan([5, 5])
    for images in (
        [ImageConditioningInput("one", 240, 1)],
        [ImageConditioningInput("one", 10, 1), ImageConditioningInput("two", 10, 1)],
    ):
        with pytest.raises(ValueError):
            module.route_scene_images(images, plan)


def test_scene_image_conditioning_limits_are_checked_without_models():
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    module = importlib.import_module("ltx25_mlx.chain_plan")
    plan = scene_plan([5] * 6)
    images = [ImageConditioningInput(str(index), index, 1) for index in range(33)]
    with pytest.raises(ValueError, match="32 image"):
        module.route_scene_images(images, plan)


@pytest.mark.parametrize('durations', [[5, 5, 5], [2, 1/3, 1/3, 2]])
def test_automatic_boundary_guides_apply_each_image_once_and_preserve_strength(durations):
    from collections import namedtuple

    from ltx25_mlx.chain_plan import (
        boundary_image_guidance_report,
        plan_ltx25_scene,
        route_scene_images,
    )

    Image = namedtuple('Image', 'frame_idx strength path')
    plan = plan_ltx25_scene(durations, frame_rate=24)
    frames = sorted({0, plan.total_frames - 2, *plan.segment_start_frames[1:],
                     *(f - 1 for f in plan.segment_start_frames[1:])})
    images = [Image(f, 0.8, str(f)) for f in frames]
    original = list(images)
    balanced = route_scene_images(images, plan, boundary_image_policy='balanced')
    strict = route_scene_images(images, plan, boundary_image_policy='strict')
    assert images == original
    assert balanced[0] == strict[0]
    assert sorted((i.frame_idx + start, i.strength, i.path)
                  for start, group in zip(plan.window_start_frames, balanced, strict=True)
                  for i in group) == [(i.frame_idx, i.strength, i.path) for i in images]
    assert balanced[0][0].strength == balanced[-1][-1].strength == 0.8
    report = boundary_image_guidance_report(images, plan, 'balanced')
    inherited = report['inherited_anchors']
    assert len(inherited) == sum(map(len, strict)) - len(images)
    for item in inherited:
        owner, window, frame = item['source_window'] - 1, item['window'] - 1, item['frame_index']
        assert owner < window
        assert any(i.frame_idx + plan.window_start_frames[owner] == frame for i in balanced[owner])
        assert not any(i.frame_idx + plan.window_start_frames[window] == frame
                       for i in balanced[window])
        assert item['strength'] == 0.8
    assert boundary_image_guidance_report(images, plan, 'strict')['inherited_anchors'] == []


@pytest.mark.parametrize('policy', ['unknown', None, {}, True])
def test_invalid_boundary_image_policy_is_rejected_before_routing(policy):
    from ltx25_mlx.chain_plan import plan_ltx25_scene, route_scene_images
    with pytest.raises(ValueError, match='boundary image'):
        route_scene_images([], plan_ltx25_scene([5, 5], frame_rate=24),
                           boundary_image_policy=policy)
