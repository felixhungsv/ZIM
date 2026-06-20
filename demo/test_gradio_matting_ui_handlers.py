"""Smoke tests for gradio_matting_ui handler return shapes and types."""
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.gradio_matting_ui import (
    DEFAULT_TUNING,
    NUM_IMAGE_OUTPUTS,
    build_outputs,
    build_ui,
    new_session,
    on_clear_prompts,
    on_prepare_exports,
    on_image_upload,
    on_prompter_change,
    on_remove_image,
    on_run_multimask,
    on_run_point,
    on_select_mask,
    _store_mask_stack,
    pack_example_outputs,
    pack_image_outputs,
    wrap,
)

NUM_PANEL_OUTPUTS = 1 + 1 + NUM_IMAGE_OUTPUTS + 1  # session + mask key + images + editor

NUM_EXAMPLE_OUTPUTS = 1 + NUM_IMAGE_OUTPUTS + 2  # session + images + prompter + scribble
NUM_REMOVE_OUTPUTS = NUM_EXAMPLE_OUTPUTS


def _assert_image_outputs(outputs, label):
    assert len(outputs) == NUM_IMAGE_OUTPUTS, f"{label}: expected {NUM_IMAGE_OUTPUTS}, got {len(outputs)}"
    for i, val in enumerate(outputs):
        if i in (9, 10):
            continue
        if i == 11:
            assert isinstance(val, list), f"{label}[{i}] expected list, got {type(val)}"
            continue
        assert isinstance(val, np.ndarray), f"{label}[{i}] expected ndarray, got {type(val)}"


def test_on_image_upload():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = on_image_upload({"image": img, "points": []}, new_session())
    assert len(result) == 1 + NUM_IMAGE_OUTPUTS
    assert wrap(result[0]).image is not None
    _assert_image_outputs(result[1:], "on_image_upload")


def test_pack_example_outputs():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = pack_example_outputs(img, new_session())
    assert len(result) == NUM_EXAMPLE_OUTPUTS
    assert wrap(result[0]).image is not None
    prompter_val, scribble_val = result[-2:]
    assert isinstance(prompter_val, dict) and "image" in prompter_val
    assert isinstance(scribble_val, dict)
    _assert_image_outputs(result[1:-2], "pack_example_outputs")


def test_on_prompter_change_point_click_returns_session_only():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    wrap(session).image = img
    prompter_val = {"image": img, "points": [[10, 10, 1, 0, 0, 4]]}
    session_out = on_prompter_change(prompter_val, session)
    assert isinstance(session_out, dict)


def test_on_clear_prompts():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = np.ones((64, 64), dtype=np.float32)
    s.d["last_prompter"] = {"image": img, "points": [[10, 10, 1, 0, 0, 4]]}
    result = on_clear_prompts(session, *DEFAULT_TUNING)
    assert len(result) == NUM_EXAMPLE_OUTPUTS
    assert wrap(result[0]).raw_alpha is None
    prompter_val, scribble_val = result[-2:]
    assert prompter_val == {"image": img, "points": []}
    assert isinstance(scribble_val, dict)


def test_on_remove_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    wrap(session).image = img
    result = on_remove_image(session, *DEFAULT_TUNING)
    assert len(result) == NUM_REMOVE_OUTPUTS
    assert wrap(result[0]).image is None
    assert result[-2] is None


def test_build_outputs_no_file_writes_by_default():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = np.ones((64, 64), dtype=np.float32)
    outs = build_outputs(s.to_dict(), *DEFAULT_TUNING, run_compare=False)
    assert len(outs) == NUM_IMAGE_OUTPUTS


def test_build_outputs_crop_to_object():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[35:65, 35:65] = 1.0
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = alpha
    tuning = list(DEFAULT_TUNING)
    tuning[14] = True
    outs = build_outputs(s.to_dict(), *tuning, run_compare=False)
    before_after = outs[7]
    left_w = before_after.shape[1] // 2
    assert before_after.shape[0] == outs[2].shape[0]
    assert left_w == outs[2].shape[1]


def test_on_prepare_exports_writes_files():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = np.ones((64, 64), dtype=np.float32)
    paths = on_prepare_exports(session, *DEFAULT_TUNING)
    assert len(paths) == 4
    assert all(os.path.isfile(p) for p in paths)


def test_on_run_point_with_mock():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    wrap(session).image = img
    prompter = {"image": img, "points": [[32, 32, 1, 0, 0, 4]]}

    def fake_predict(session_data, multimask=False):
        s = wrap(session_data)
        s.raw_alpha = np.ones((64, 64), dtype=np.float32) * 0.8
        s.all_masks = np.stack([s.raw_alpha])
        s.iou_scores = np.array([0.9])

    import demo.gradio_matting_ui as ui

    with patch.object(ui, "run_zim_predict", fake_predict):
        result = on_run_point(prompter, session, None, *DEFAULT_TUNING)
    assert len(result) == NUM_PANEL_OUTPUTS
    _assert_image_outputs(result[2:-1], "on_run_point")


def test_on_select_mask_tuple_index():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = np.ones((64, 64), dtype=np.float32) * 0.8
    masks = np.stack([np.ones((64, 64)) * v for v in (0.5, 0.7, 0.9, 0.6)])
    s.all_masks = masks
    s.iou_scores = np.array([0.9, 0.8, 0.95, 0.7])
    key = _store_mask_stack(s, masks)

    class EvtTuple:
        index = (1, 0)
        selected = True

    result = on_select_mask(EvtTuple(), session, key, *DEFAULT_TUNING)
    assert len(result) == NUM_PANEL_OUTPUTS
    assert wrap(result[0]).mask_index == 2

    class EvtInt:
        index = 3
        selected = True

    result = on_select_mask(EvtInt(), session, key, *DEFAULT_TUNING)
    assert wrap(result[0]).mask_index == 3


def test_on_select_mask_uses_cache_when_session_masks_missing():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.raw_alpha = np.ones((64, 64), dtype=np.float32) * 0.8
    masks = np.stack([np.ones((64, 64)) * v for v in (0.5, 0.7, 0.9, 0.6)])
    key = _store_mask_stack(s, masks)
    s.all_masks = None

    class EvtInt:
        index = 1
        selected = True

    result = on_select_mask(EvtInt(), session, key, *DEFAULT_TUNING)
    assert wrap(result[0]).mask_index == 1
    assert abs(wrap(result[0]).raw_alpha.mean() - 0.7) < 0.01


def test_on_run_multimask_with_mock():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    session = new_session()
    s = wrap(session)
    s.image = img
    s.prompts = {"point": [[32, 32, 1]]}

    def fake_predict(session_data, multimask=False):
        assert multimask is True
        s2 = wrap(session_data)
        alphas = [np.ones((64, 64), dtype=np.float32) * v for v in (0.5, 0.6, 0.7, 0.8)]
        s2.all_masks = np.stack(alphas)
        s2.iou_scores = np.array([0.7, 0.8, 0.9, 0.75])
        s2.mask_index = 2
        s2.raw_alpha = alphas[2]
        ui._store_mask_stack(s2, s2.all_masks)

    import demo.gradio_matting_ui as ui

    with patch.object(ui, "run_zim_predict", fake_predict):
        result = on_run_multimask(session, None, *DEFAULT_TUNING)
    assert len(result) == NUM_PANEL_OUTPUTS
    assert result[1] is not None
    assert wrap(result[0]).all_masks.shape[0] == 4


def test_build_ui_wiring():
    demo = build_ui()
    assert demo is not None


if __name__ == "__main__":
    test_on_image_upload()
    test_pack_example_outputs()
    test_on_prompter_change_point_click_returns_session_only()
    test_on_clear_prompts()
    test_on_remove_image()
    test_build_outputs_no_file_writes_by_default()
    test_build_outputs_crop_to_object()
    test_on_prepare_exports_writes_files()
    test_on_run_point_with_mock()
    test_on_select_mask_tuple_index()
    test_on_select_mask_uses_cache_when_session_masks_missing()
    test_on_run_multimask_with_mock()
    test_build_ui_wiring()
    print("All handler smoke tests passed.")
