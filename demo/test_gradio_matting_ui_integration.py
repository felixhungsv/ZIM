"""Integration tests: launch Gradio app and verify endpoints respond."""
import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_PORT = 17860
TEST_HOST = "127.0.0.1"


def _make_image(h=96, w=96):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[24:72, 24:72] = (200, 100, 80)
    return img


def _mock_zim(monkeypatch):
    import demo.gradio_matting_ui as ui

    def fake_predict(session_data, multimask=False):
        s = ui.wrap(session_data)
        h, w = s.image.shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[20:76, 20:76] = 0.9
        s.raw_alpha = alpha
        if multimask:
            masks = np.stack([alpha, alpha * 0.7, alpha * 0.5, alpha * 0.3])
            s.all_masks = masks
            s.iou_scores = np.array([0.91, 0.82, 0.75, 0.68])
            ui._store_mask_stack(s, masks)
        else:
            s.all_masks = np.stack([alpha])
            s.iou_scores = np.array([0.91])
            ui._clear_mask_cache(s)

    predictor = MagicMock()
    predictor.set_image = MagicMock()
    monkeypatch.setattr(ui.MODELS, "load_zim", lambda key: predictor)
    monkeypatch.setattr(ui, "run_zim_predict", fake_predict)
    monkeypatch.setattr(ui, "run_sam_predict", lambda s: np.zeros((96, 96), np.float32))


def _wait_for_server(url, timeout=90):
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Server not ready: {url}")


def test_full_handler_workflow(monkeypatch):
    """End-to-end handler chain without browser (fast, deterministic)."""
    import demo.gradio_matting_ui as ui

    _mock_zim(monkeypatch)
    img = _make_image()
    session = ui.new_session()
    tuning = ui.DEFAULT_TUNING
    prompter = {"image": img, "points": [[48, 48, 1, 0, 0, 4]]}

    # 1 upload
    session, *_ = ui.on_image_upload(prompter, session, None, *tuning)
    assert ui.wrap(session).image is not None

    # 2 example
    session, *outs, pv, scrib = ui.pack_example_outputs(img, session)
    assert isinstance(pv, dict)

    # 3 point click (state only)
    session, _ = ui.on_prompter_change({"image": img, "points": [[40, 40, 1, 0, 0, 4]]}, session, None)
    assert isinstance(session, dict)

    # 4 run
    session, mask_key, *outs = ui.on_run_point(prompter, session, None, *tuning)
    assert ui.wrap(session).raw_alpha is not None
    assert mask_key is None

    # 4b scribble
    layer = np.zeros((96, 96, 4), dtype=np.uint8)
    layer[30:70, 30:70, 3] = 255
    scribble = {"background": img, "layers": [layer], "composite": img}
    session, mask_key, *outs = ui.on_run_scribble(scribble, session, mask_key, *tuning)
    assert ui.wrap(session).raw_alpha is not None

    # 4c multimask + select
    ui.wrap(session).prompts = {"point": [[48, 48, 1]]}
    session, mask_key, *outs = ui.on_run_multimask(session, None, *tuning)
    assert mask_key is not None
    gallery = outs[11]
    assert len(gallery) == 4

    class Evt:
        index = 2
        selected = True

    session, mask_key, *outs = ui.on_select_mask(Evt(), session, mask_key, *tuning)
    assert ui.wrap(session).mask_index == 2

    # 5 tuning
    outs = ui.on_tuning_change(session, *tuning)
    assert len(outs) == ui.NUM_IMAGE_OUTPUTS + 1

    # 6 clear points
    session = ui.new_session()
    ui.wrap(session).image = img
    ui.wrap(session).d["last_prompter"] = prompter
    session, _ = ui.on_prompter_change({"image": img, "points": []}, session, None)
    assert ui.wrap(session).raw_alpha is None

    # 7 remove image
    ui.wrap(session).image = img
    fresh, *outs, p, s = ui.on_remove_image(session, *tuning)
    assert ui.wrap(fresh).image is None

    # 8 undo/redo
    session, mask_key, *outs = ui.on_run_point(prompter, ui.new_session(), None, *tuning)
    ui.wrap(session).image = img
    session, mask_key, *outs = ui.on_undo(session, mask_key, *tuning)
    session, mask_key, *outs = ui.on_redo(session, mask_key, *tuning)

    # 9 export
    paths = ui.on_prepare_exports(session, *tuning)
    assert len(paths) == 4
    assert all(os.path.isfile(p) for p in paths)

    # 10 compare (SAM missing is OK)
    ui.on_compare(session, *tuning)

    # 11 model switch
    session = ui.on_model_change("vit_b (fast)", session)


def test_server_launch_and_api(monkeypatch):
    """Launch app locally and verify Gradio client can connect."""
    from gradio_client import Client

    import demo.gradio_matting_ui as ui

    _mock_zim(monkeypatch)
    demo = ui.build_ui()
    _, local_url, _ = demo.launch(
        server_name=TEST_HOST,
        server_port=TEST_PORT,
        prevent_thread_lock=True,
        quiet=True,
        show_error=True,
    )
    try:
        _wait_for_server(local_url)
        client = Client(local_url)
        api = client.view_api(return_format="dict")
        assert api is not None
        assert len(client.endpoints) > 0
    finally:
        demo.close()


if __name__ == "__main__":
    class MP:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    mp = MP()
    test_full_handler_workflow(mp)
    print("test_full_handler_workflow passed")
    test_server_launch_and_api(mp)
    print("test_server_launch_and_api passed")
    print("All integration tests passed.")
