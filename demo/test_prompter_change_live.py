"""Live test: prompter point click must not hang or rebuild the full panel."""
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_PORT = 17861
TEST_HOST = "127.0.0.1"
TIMEOUT_S = 8


def _mock_zim(monkeypatch):
    import demo.gradio_matting_ui as ui

    predictor = MagicMock()
    predictor.set_image = MagicMock()

    def fake_predict(session_data, multimask=False):
        s = ui.wrap(session_data)
        h, w = s.image.shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[20:76, 20:76] = 0.9
        s.raw_alpha = alpha
        s.all_masks = np.stack([alpha])
        s.iou_scores = np.array([0.91])

    monkeypatch.setattr(ui.MODELS, "load_zim", lambda key: predictor)
    monkeypatch.setattr(ui, "run_zim_predict", fake_predict)


def _wait_for_server(url, timeout=60):
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def test_prompter_point_click_completes_quickly(monkeypatch):
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
        from gradio_client import handle_file

        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[40:88, 40:88] = 200
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            Image.fromarray(img).save(img_path)
        try:
            prompter_val = {
                "image": handle_file(img_path),
                "points": [[64, 64, 1, 0, 0, 4]],
            }
            t0 = time.time()
            client.predict(prompter_val, api_name="/on_prompter_change")
            change_elapsed = time.time() - t0
            assert change_elapsed < TIMEOUT_S, f"prompter change took {change_elapsed:.1f}s"
            print(f"prompter change completed in {change_elapsed:.2f}s")
        finally:
            os.unlink(img_path)
    finally:
        demo.close()


if __name__ == "__main__":
    class MP:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    test_prompter_point_click_completes_quickly(MP())
    print("Live prompter change test passed.")
