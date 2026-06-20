"""
ZIM Matting Studio — interactive zero-shot image matting UI.

Run from the ZIM project root:
    python demo/gradio_matting_ui.py
"""
import copy
import base64
import json
import os
import sys
import tempfile
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from PIL import Image

sys.path.append(os.getcwd())

from zim_anything import ZimPredictor, zim_model_registry

MODEL_REGISTRY = {
    "vit_l (quality)": "results/zim_vit_l_2092",
    "vit_b (fast)": "results/zim_vit_b_2043",
}

SAM_CHECKPOINT = "results/sam_vit_b_01ec64.pth"
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

EXPORT_DIR = os.path.join(tempfile.gettempdir(), "zim_matting_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


class ModelManager:
    """Lazy-load ZIM and SAM predictors (only one ZIM model at a time)."""

    def __init__(self) -> None:
        self.zim_key: Optional[str] = None
        self.zim_predictor: Optional[ZimPredictor] = None
        self.sam_predictor: Any = None
        self._sam_loaded = False

    def load_zim(self, model_key: str) -> ZimPredictor:
        if self.zim_key == model_key and self.zim_predictor is not None:
            return self.zim_predictor

        ckpt = MODEL_REGISTRY[model_key]
        for name in ("encoder.onnx", "decoder.onnx"):
            path = os.path.join(ckpt, name)
            if not os.path.isfile(path):
                raise gr.Error(
                    f"Missing {path}. Download weights from HuggingFace "
                    f"(see README Model Zoo) and place them under {ckpt}/"
                )

        model = zim_model_registry["vit_l"](checkpoint=ckpt)
        if torch.cuda.is_available():
            model.cuda()

        self.zim_predictor = ZimPredictor(model)
        self.zim_key = model_key
        return self.zim_predictor

    def load_sam(self):
        if self._sam_loaded and self.sam_predictor is not None:
            return self.sam_predictor

        # if not os.path.isfile(SAM_CHECKPOINT):
        #     raise gr.Error(
        #         f"SAM weights not found at {SAM_CHECKPOINT}. "
        #         f"Download from {SAM_DOWNLOAD_URL}"
        #     )

        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        if torch.cuda.is_available():
            sam.cuda()
        self.sam_predictor = SamPredictor(sam)
        self._sam_loaded = True
        return self.sam_predictor


MODELS = ModelManager()

# Server-side cache: gr.State often drops large (C,H,W) mask stacks between events.
_MASK_STACK_CACHE: Dict[str, np.ndarray] = {}


def new_session() -> Dict[str, Any]:
    """Create a fresh session dict (must be JSON-serializable-friendly for gr.State)."""
    return {
        "image": None,
        "raw_alpha": None,
        "iou_scores": None,
        "all_masks": None,
        "prompts": {},
        "model_key": "vit_l (quality)",
        "mask_index": 0,
        "history": [],
        "history_index": -1,
        "last_prompter": None,
    }


def ensure_session(session) -> Dict[str, Any]:
    """Normalize gr.State value to a session dict."""
    if not isinstance(session, dict):
        return new_session()
    return session


class MattingSession:
    """Helper wrapper around session dict (do not store this class in gr.State)."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.d = ensure_session(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.d

    @property
    def image(self):
        return self.d["image"]

    @image.setter
    def image(self, value):
        self.d["image"] = value

    @property
    def raw_alpha(self):
        return self.d["raw_alpha"]

    @raw_alpha.setter
    def raw_alpha(self, value):
        self.d["raw_alpha"] = value

    @property
    def iou_scores(self):
        return self.d["iou_scores"]

    @iou_scores.setter
    def iou_scores(self, value):
        self.d["iou_scores"] = value

    @property
    def all_masks(self):
        return self.d["all_masks"]

    @all_masks.setter
    def all_masks(self, value):
        self.d["all_masks"] = value

    @property
    def prompts(self):
        return self.d["prompts"]

    @prompts.setter
    def prompts(self, value):
        self.d["prompts"] = value

    @property
    def model_key(self):
        return self.d["model_key"]

    @model_key.setter
    def model_key(self, value):
        self.d["model_key"] = value

    @property
    def mask_index(self):
        return self.d["mask_index"]

    @mask_index.setter
    def mask_index(self, value):
        self.d["mask_index"] = value

    @property
    def history(self):
        return self.d["history"]

    @history.setter
    def history(self, value):
        self.d["history"] = value

    @property
    def history_index(self):
        return self.d["history_index"]

    @history_index.setter
    def history_index(self, value):
        self.d["history_index"] = value

    def snapshot(self) -> None:
        snap = {
            "prompts": copy.deepcopy(self.prompts),
            "mask_index": self.mask_index,
            "raw_alpha": self.raw_alpha.copy() if self.raw_alpha is not None else None,
        }
        self.history = self.history[: self.history_index + 1]
        self.history.append(snap)
        self.history_index = len(self.history) - 1

    def undo(self) -> None:
        if self.history_index <= 0:
            return
        self.history_index -= 1
        self._restore(self.history[self.history_index])

    def redo(self) -> None:
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self._restore(self.history[self.history_index])

    def _restore(self, snap: Dict) -> None:
        self.prompts = copy.deepcopy(snap["prompts"])
        self.mask_index = snap["mask_index"]
        if snap["raw_alpha"] is not None:
            self.raw_alpha = snap["raw_alpha"].copy()


def wrap(session) -> MattingSession:
    if isinstance(session, MattingSession):
        return session
    return MattingSession(session)


def _pack(session, *outputs):
    """Return session dict plus output tuple for Gradio."""
    if isinstance(session, MattingSession):
        return (session.to_dict(),) + outputs
    return (wrap(session).to_dict(),) + outputs


def logits_to_alpha(mask_logits: np.ndarray) -> np.ndarray:
    if mask_logits.max() > 1.0 or mask_logits.min() < 0.0:
        alpha = 1.0 / (1.0 + np.exp(-mask_logits))
    else:
        alpha = mask_logits.astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def apply_tuning(alpha, opacity=100.0, contrast=100.0, feather=0, choke=0, invert=False):
    a = alpha.copy()
    if contrast != 100.0:
        factor = contrast / 100.0
        a = np.clip((a - 0.5) * factor + 0.5, 0.0, 1.0)
    a = np.clip(a * (opacity / 100.0), 0.0, 1.0)
    if choke != 0:
        k = abs(choke) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = (a * 255).astype(np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel) if choke > 0 else cv2.erode(mask_u8, kernel)
        a = mask_u8.astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        a = cv2.GaussianBlur(a, (k, k), 0)
    if invert:
        a = 1.0 - a
    return np.clip(a, 0.0, 1.0)


def alpha_to_uint8(alpha):
    return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)


def composite_rgba(image, alpha):
    return np.dstack([image, alpha_to_uint8(alpha)])


def normalize_image_for_model(image: np.ndarray, min_side: int = 8) -> np.ndarray:
    """Convert to HxWx3 RGB uint8 for ZIM/SAM `set_image`."""
    if image is None:
        raise gr.Error("Image is missing.")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise gr.Error(f"Expected an HxWx3 image, got shape {getattr(image, 'shape', None)}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    if h < min_side or w < min_side:
        raise gr.Error(f"Image is too small ({w}x{h}). Please upload a larger image.")
    return image


def as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure Gradio RGB image outputs are uint8 in [0, 255]."""
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def normalize_bg_image(bg_image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure background upload is HxWx3 RGB uint8."""
    if bg_image is None:
        return None
    if bg_image.ndim == 2:
        bg_image = np.stack([bg_image] * 3, axis=-1)
    if bg_image.shape[-1] == 4:
        bg_image = bg_image[..., :3]
    return bg_image.astype(np.uint8)


def _resolve_background(bg_mode, bg_color, bg_image):
    """Return (bg_rgb tuple, bg_image array) honoring background source mode."""
    bg_image = normalize_bg_image(bg_image)
    if bg_mode == "Upload image" and bg_image is not None:
        return (255, 255, 255), bg_image
    hex_c = str(bg_color).lstrip("#")
    if len(hex_c) != 6:
        hex_c = "FFFFFF"
    bg_rgb = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4))
    return bg_rgb, None


def _array_to_b64_png(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return ""
    buf = BytesIO()
    if arr.ndim == 3 and arr.shape[2] == 4:
        if np.issubdtype(arr.dtype, np.floating):
            rgb = as_uint8_rgb(arr[..., :3])
            a_chan = arr[..., 3]
            alpha = (
                (np.clip(a_chan, 0, 1) * 255).astype(np.uint8)
                if a_chan.max() <= 1.0 else np.clip(a_chan, 0, 255).astype(np.uint8)
            )
            rgba = np.dstack([rgb, alpha])
        else:
            rgba = arr.astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    else:
        Image.fromarray(as_uint8_rgb(arr)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def rotate_image_alpha(image: np.ndarray, alpha: np.ndarray, angle_deg: float):
    """Rotate RGB image and alpha mask together around the image center."""
    if abs(angle_deg) < 1e-6:
        return image, alpha
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    rotated = cv2.warpAffine(
        image, M, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    rotated_a = cv2.warpAffine(
        alpha, M, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return rotated, rotated_a


def paste_layer(canvas: np.ndarray, fg: np.ndarray, alpha: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Alpha-blend fg onto canvas at top-left (ox, oy). Always returns uint8 RGB."""
    canvas = canvas.astype(np.float32)
    fh, fw = fg.shape[:2]
    ch, cw = canvas.shape[:2]
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(cw, ox + fw)
    y1 = min(ch, oy + fh)
    sx0 = x0 - ox
    sy0 = y0 - oy
    w_paste, h_paste = x1 - x0, y1 - y0
    if w_paste <= 0 or h_paste <= 0:
        return np.clip(canvas, 0, 255).astype(np.uint8)
    a = alpha[sy0:sy0 + h_paste, sx0:sx0 + w_paste]
    a3 = a[:, :, None]
    region = canvas[y0:y1, x0:x1]
    fg_region = fg[sy0:sy0 + h_paste, sx0:sx0 + w_paste].astype(np.float32)
    canvas[y0:y1, x0:x1] = fg_region * a3 + region * (1.0 - a3)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def composite_on_bg(image, alpha, bg_color=(255, 255, 255), bg_image=None,
                    offset_x=0, offset_y=0, scale=1.0, rotation_deg=0.0,
                    flip_h=False, flip_v=False,
                    keep_shadow=False, shadow_thresh=0.15):
    h, w = image.shape[:2]
    a = alpha.copy()
    if keep_shadow:
        shadow_mask = (a > 0) & (a < shadow_thresh)
        a[shadow_mask] = a[shadow_mask] * 0.5
    bg_image = normalize_bg_image(bg_image)
    if bg_image is not None:
        bh, bw = bg_image.shape[:2]
        scale_cover = max(w / bw, h / bh)
        nw, nh = int(bw * scale_cover), int(bh * scale_cover)
        bg_scaled = cv2.resize(bg_image, (nw, nh))
        x0 = max(0, (nw - w) // 2)
        y0 = max(0, (nh - h) // 2)
        bg = bg_scaled[y0:y0 + h, x0:x0 + w]
        if bg.shape[0] != h or bg.shape[1] != w:
            bg = cv2.resize(bg_image, (w, h))
    else:
        bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

    fg = image.astype(np.float32)
    layer_a = a.astype(np.float32)
    if flip_h:
        fg = cv2.flip(fg, 1)
        layer_a = cv2.flip(layer_a, 1)
    if flip_v:
        fg = cv2.flip(fg, 0)
        layer_a = cv2.flip(layer_a, 0)
    has_transform = (
        abs(scale - 1.0) > 1e-6 or abs(rotation_deg) > 1e-6
        or offset_x != 0 or offset_y != 0 or flip_h or flip_v
    )
    if not has_transform:
        a3 = layer_a[:, :, None]
        return np.clip(fg * a3 + bg.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)

    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if abs(scale - 1.0) > 1e-6:
        fg = cv2.resize(fg, (nw, nh))
        layer_a = cv2.resize(layer_a, (nw, nh))
    fg_u8 = np.clip(fg, 0, 255).astype(np.uint8)
    fg_u8, layer_a = rotate_image_alpha(fg_u8, layer_a, rotation_deg)
    nh, nw = fg_u8.shape[:2]
    canvas = bg.astype(np.float32).copy()
    ox = int(offset_x + (w - nw) / 2)
    oy = int(offset_y + (h - nh) / 2)
    return paste_layer(canvas, fg_u8.astype(np.float32), layer_a, ox, oy)


def crop_to_content(image, alpha, pad=16):
    ys, xs = np.where(alpha > 0.05)
    if len(ys) == 0:
        return image, alpha
    y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
    return image[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def render_edge_inspector(image, alpha, zoom=3):
    grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    if grad.max() > 0:
        grad /= grad.max()
    ys, xs = np.where(grad > 0.1)
    if len(ys) == 0:
        return image
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = image.shape[:2]
    band = max(32, min(h, w) // 8)
    y0, y1 = max(0, cy - band), min(h, cy + band)
    x0, x1 = max(0, cx - band), min(w, cx + band)
    patch = image[y0:y1, x0:x1].copy()
    heat = cv2.cvtColor(cv2.applyColorMap((grad[y0:y1, x0:x1] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(patch, 0.6, heat, 0.4, 0)
    return cv2.resize(blended, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def render_alpha_histogram(alpha):
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.hist(alpha.ravel(), bins=50, range=(0, 1), color="#6c00c0", edgecolor="white")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Pixel count")
    ax.set_title("Alpha distribution")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf.copy()


def overlay_mask_vis(image, alpha, color=(108, 0, 192)):
    a = alpha[:, :, None]
    tint = np.array(color, dtype=np.float32)
    blended = image.astype(np.float32) * 0.5 + tint * 0.5
    return np.clip(blended * a + image.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


def inpaint_background(image, alpha):
    mask = (alpha < 0.5).astype(np.uint8) * 255
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)


def parse_prompter_input(img_dict, prompts):
    if img_dict is None or img_dict.get("image") is None:
        raise gr.Error("Upload an image first.")
    image = img_dict["image"]
    point_prompts, box_prompts = [], []
    for prompt in img_dict.get("points", []):
        prompt = [int(p) for p in prompt]
        if prompt[2] == 2 and prompt[5] == 3:
            box_prompts = [[prompt[0], prompt[1], prompt[3], prompt[4]]]
        elif prompt[2] == 1 and prompt[5] == 4:
            point_prompts.append((1, (prompt[0], prompt[1])))
        elif prompt[2] == 0 and prompt[5] == 4:
            point_prompts.append((0, (prompt[0], prompt[1])))
    prompts = copy.deepcopy(prompts)
    prompts.pop("scribble", None)
    if point_prompts:
        prompts["point"] = point_prompts
    else:
        prompts.pop("point", None)
    if box_prompts:
        prompts["bbox"] = box_prompts
    else:
        prompts.pop("bbox", None)
    return image, prompts


def parse_scribble_prompt(image, scribble):
    if not scribble or not scribble.get("layers"):
        raise gr.Error("Please draw at least one scribble stroke.")
    scribble_mask = scribble["layers"][0][..., -1] > 0
    coords = np.argwhere(scribble_mask)
    if len(coords) == 0:
        raise gr.Error("Please draw at least one scribble stroke.")
    n = min(len(coords), 24)
    idx = np.linspace(0, len(coords) - 1, n, dtype=int)
    return image, {"scribble": coords[idx]}


def prompts_to_arrays(prompts):
    point_coords = point_labels = boxes = None
    if "point" in prompts:
        point_coords, point_labels = [], []
        for label, pts in prompts["point"]:
            point_coords.append(pts)
            point_labels.append(label)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
    if "bbox" in prompts:
        boxes = np.array(prompts["bbox"])
    if "scribble" in prompts:
        point_coords, point_labels = [], []
        for pts in prompts["scribble"]:
            point_coords.append(np.flip(pts))
            point_labels.append(1)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
    return point_coords, point_labels, boxes


def is_ambiguous_prompt(prompts):
    """Legacy helper: single positive point with no negative/box (multi-mask use case)."""
    if "bbox" in prompts or "scribble" in prompts:
        return False
    points = prompts.get("point", [])
    return sum(1 for p in points if p[0] == 1) == 1 and sum(1 for p in points if p[0] == 0) == 0


def normalize_mask_stack(masks: Any) -> Optional[np.ndarray]:
    """Ensure mask stack is (C, H, W) float32."""
    if masks is None:
        return None
    if isinstance(masks, list):
        if len(masks) == 0:
            return None
        parts = [np.asarray(m, dtype=np.float32) for m in masks]
        if parts[0].ndim == 2:
            return np.stack(parts, axis=0)
        m = np.asarray(masks, dtype=np.float32)
    else:
        m = np.asarray(masks, dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim != 3:
        return None
    return m


def _mask_stack_path(key: str) -> str:
    return os.path.join(EXPORT_DIR, f"masks_{key}.npz")


def _session_cache_key(session: "MattingSession") -> str:
    key = session.d.get("_sid")
    if not key:
        key = uuid.uuid4().hex
        session.d["_sid"] = key
    return key


def _store_mask_stack(session: "MattingSession", masks: Optional[np.ndarray]) -> Optional[str]:
    """Persist (C,H,W) masks; return cache key stored in gr.State."""
    key = _session_cache_key(session)
    stack = normalize_mask_stack(masks)
    if stack is None or stack.shape[0] == 0:
        _clear_mask_cache(session)
        return None
    _MASK_STACK_CACHE[key] = stack.copy()
    np.savez_compressed(_mask_stack_path(key), masks=stack)
    session.all_masks = stack
    return key


def _load_mask_stack(session: "MattingSession", stack_key: Optional[str] = None) -> Optional[np.ndarray]:
    key = stack_key or session.d.get("_sid")
    if key:
        if key in _MASK_STACK_CACHE:
            return _MASK_STACK_CACHE[key]
        path = _mask_stack_path(key)
        if os.path.isfile(path):
            loaded = np.asarray(np.load(path)["masks"], dtype=np.float32)
            _MASK_STACK_CACHE[key] = loaded
            return loaded
    return normalize_mask_stack(session.all_masks)


def _clear_mask_cache(session: "MattingSession") -> None:
    key = session.d.get("_sid")
    if key:
        _MASK_STACK_CACHE.pop(key, None)
        path = _mask_stack_path(key)
        if os.path.isfile(path):
            os.remove(path)


def build_multimask_gallery(session: MattingSession) -> List[Tuple[np.ndarray, str]]:
    """Overlay previews for each mask candidate (Multi-mask tab)."""
    masks = _load_mask_stack(session)
    if masks is None or masks.shape[0] <= 1 or session.image is None:
        return []
    gallery = []
    scores = session.iou_scores
    for i in range(masks.shape[0]):
        vis = overlay_mask_vis(session.image, logits_to_alpha(masks[i]))
        score = f"{scores[i]:.3f}" if scores is not None and i < len(scores) else "?"
        tag = " (active)" if i == session.mask_index else ""
        if scores is not None and len(scores) > 1 and i == int(np.argmax(scores)):
            tag += " (recommended)"
        gallery.append((vis, f"Mask {i} · IoU {score}{tag}"))
    return gallery


def run_zim_predict(session_data, multimask=False):
    session = wrap(session_data)
    if session.image is None:
        raise gr.Error("Please upload an image first.")
    if not session.prompts:
        raise gr.Error("Please add a point, box, or scribble prompt.")
    session.image = normalize_image_for_model(session.image)
    predictor = MODELS.load_zim(session.model_key)
    predictor.set_image(session.image)
    pc, pl, box = prompts_to_arrays(session.prompts)
    masks, iou, _ = predictor.predict(
        point_coords=pc, point_labels=pl, box=box,
        multimask_output=multimask, return_logits=True,
    )
    masks = normalize_mask_stack(masks)
    iou = np.asarray(iou, dtype=np.float32).reshape(-1)
    session.mask_index = int(np.argmax(iou)) if len(iou) > 1 else 0
    session.raw_alpha = logits_to_alpha(masks[session.mask_index])
    session.iou_scores = iou
    if multimask and masks.shape[0] > 1:
        _store_mask_stack(session, masks)
    else:
        session.all_masks = masks
        _clear_mask_cache(session)


def run_sam_predict(session_data):
    session = wrap(session_data)
    sam = MODELS.load_sam()
    sam.set_image(session.image)
    pc, pl, box = prompts_to_arrays(session.prompts)
    masks, _, _ = sam.predict(point_coords=pc, point_labels=pl, box=box, multimask_output=False)
    return np.squeeze(masks).astype(np.float32)


DEFAULT_TUNING = (
    100, 100, 0, 0, False, "Solid color", "#FFFFFF", None,
    0, 0, 1.0, 0.0, False, False, False, False, False,
)
NUM_TUNING_ARGS = len(DEFAULT_TUNING)


def _layers_for_composite(session_data, tuning: Tuple[Any, ...]):
    """Return image, alpha, bg_rgb, bg_image, offset_x, offset_y, scale, rotation, flip_h, flip_v, keep_shadow."""
    session = wrap(session_data)
    if session.raw_alpha is None or session.image is None:
        return None
    tuning = normalize_tuning_args(tuning)
    opacity, contrast, feather, choke, invert = tuning[:5]
    bg_mode, bg_color, bg_upload = tuning[5:8]
    offset_x, offset_y, scale, rotation, flip_h, flip_v, crop = tuning[8:15]
    keep_shadow, do_inpaint = tuning[15:17]
    alpha = apply_tuning(session.raw_alpha, opacity, contrast, feather, choke, invert)
    image = session.image
    if crop:
        image, alpha = crop_to_content(image, alpha)
    bg_rgb, bg_img = _resolve_background(bg_mode, bg_color, bg_upload)
    return image, alpha, bg_rgb, bg_img, offset_x, offset_y, scale, rotation, flip_h, flip_v, keep_shadow


def _composite_iframe_document(config_json: str) -> str:
    """Self-contained HTML page for the interactive composite canvas (runs inside iframe)."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:#222}}
body{{display:flex;flex-direction:column;align-items:center;justify-content:center}}
canvas{{display:block;max-width:100%;cursor:grab;background:#333}}
canvas:active{{cursor:grabbing}}
.hint{{color:#aaa;font:12px/1.4 sans-serif;padding:6px;text-align:center}}
</style></head><body>
<canvas id="c"></canvas>
<div class="hint">Drag · Scroll=scale · Shift+scroll=rotate</div>
<script>
(function(){{
  const cfg = {config_json};
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const maxDim = 640;
  const viewScale = Math.min(1, maxDim / Math.max(cfg.w, cfg.h));
  canvas.width = Math.round(cfg.w * viewScale);
  canvas.height = Math.round(cfg.h * viewScale);
  const state = {{
    offset_x: cfg.offset_x, offset_y: cfg.offset_y,
    scale: cfg.scale, rotation: cfg.rotation,
    flip_h: !!cfg.flip_h, flip_v: !!cfg.flip_v,
  }};
  const bgImg = new Image();
  const fgImg = new Image();
  let loaded = 0;
  const maybeInit = () => {{ loaded += 1; if (loaded >= 2) draw(); }};
  bgImg.onload = maybeInit;
  fgImg.onload = maybeInit;
  bgImg.onerror = maybeInit;
  fgImg.onerror = maybeInit;
  bgImg.src = "data:image/png;base64," + cfg.bg;
  fgImg.src = "data:image/png;base64," + cfg.fg;

  function draw() {{
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
    ctx.save();
    const cx = canvas.width / 2 + state.offset_x * viewScale;
    const cy = canvas.height / 2 + state.offset_y * viewScale;
    ctx.translate(cx, cy);
    ctx.rotate(state.rotation * Math.PI / 180);
    const sx = state.scale * (state.flip_h ? -1 : 1);
    const sy = state.scale * (state.flip_v ? -1 : 1);
    ctx.scale(sx, sy);
    const dw = fgImg.width * viewScale;
    const dh = fgImg.height * viewScale;
    ctx.drawImage(fgImg, -dw / 2, -dh / 2, dw, dh);
    ctx.restore();
  }}

  function pushTransform() {{
    window.parent.postMessage({{
      type: "zim_composite_transform",
      payload: {{
        offset_x: Math.round(state.offset_x),
        offset_y: Math.round(state.offset_y),
        scale: Math.round(state.scale * 1000) / 1000,
        rotation: Math.round(state.rotation * 10) / 10,
        flip_h: !!state.flip_h,
        flip_v: !!state.flip_v,
      }},
    }}, "*");
  }}

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("mousedown", (e) => {{
    dragging = true; lastX = e.clientX; lastY = e.clientY;
  }});
  window.addEventListener("mouseup", () => {{
    if (!dragging) return;
    dragging = false;
    pushTransform();
  }});
  window.addEventListener("mousemove", (e) => {{
    if (!dragging) return;
    state.offset_x += (e.clientX - lastX) / viewScale;
    state.offset_y += (e.clientY - lastY) / viewScale;
    lastX = e.clientX; lastY = e.clientY;
    draw();
  }});
  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    if (e.shiftKey) {{
      state.rotation += e.deltaY > 0 ? 3 : -3;
    }} else {{
      const f = e.deltaY > 0 ? 0.96 : 1.04;
      state.scale = Math.min(3, Math.max(0.25, state.scale * f));
    }}
    draw();
    clearTimeout(canvas._wheelTimer);
    canvas._wheelTimer = setTimeout(pushTransform, 120);
  }}, {{ passive: false }});
}})();
</script></body></html>"""


def composite_editor_html(session_data, tuning: Tuple[Any, ...]) -> str:
    """Interactive iframe canvas for drag / scroll-scale / shift-scroll-rotate."""
    layers = _layers_for_composite(session_data, tuning)
    if layers is None:
        return (
            '<div class="composite-editor-empty">Run inference first, then drag the cutout here '
            'to move, scroll to scale, Shift+scroll to rotate.</div>'
        )
    image, alpha, bg_rgb, bg_img, ox, oy, scale, rotation, flip_h, flip_v, _keep = layers
    h, w = image.shape[:2]
    fg_rgba = np.dstack([as_uint8_rgb(image), alpha_to_uint8(alpha)])
    if bg_img is not None:
        bh, bw = bg_img.shape[:2]
        scale_cover = max(w / bw, h / bh)
        nw, nh = int(bw * scale_cover), int(bh * scale_cover)
        bg_scaled = cv2.resize(bg_img, (nw, nh))
        x0 = max(0, (nw - w) // 2)
        y0 = max(0, (nh - h) // 2)
        bg_preview = bg_scaled[y0:y0 + h, x0:x0 + w]
        if bg_preview.shape[0] != h or bg_preview.shape[1] != w:
            bg_preview = cv2.resize(bg_img, (w, h))
    else:
        bg_preview = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
    config = {
        "w": int(w),
        "h": int(h),
        "offset_x": float(ox),
        "offset_y": float(oy),
        "scale": float(scale),
        "rotation": float(rotation),
        "flip_h": bool(flip_h),
        "flip_v": bool(flip_v),
        "fg": _array_to_b64_png(fg_rgba),
        "bg": _array_to_b64_png(bg_preview),
    }
    if not config["fg"] or not config["bg"]:
        return '<div class="composite-editor-empty">Could not build composite preview.</div>'
    doc = _composite_iframe_document(json.dumps(config))
    b64 = base64.b64encode(doc.encode("utf-8")).decode("ascii")
    view_h = int(min(520, max(220, h * min(1.0, 640.0 / max(h, w)) + 40)))
    return (
        f'<iframe class="composite-interactive-iframe" '
        f'src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:{view_h}px;border:0;border-radius:6px;background:#222"></iframe>'
    )


def normalize_tuning_args(tuning_args):
    """Pad or trim tuning values so build_outputs always receives 17 args."""
    args = list(tuning_args)
    if len(args) < NUM_TUNING_ARGS:
        args.extend(DEFAULT_TUNING[len(args):NUM_TUNING_ARGS])
    return tuple(args[:NUM_TUNING_ARGS])


def scribble_from_image(image: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    return {"background": image, "layers": [], "composite": image}


def remember_prompter(session: MattingSession, img_dict: Optional[Dict[str, Any]]) -> None:
    if img_dict is not None and img_dict.get("image") is not None:
        session.d["last_prompter"] = copy.deepcopy(img_dict)


def clear_mask_results(session: MattingSession) -> None:
    session.prompts = {}
    session.raw_alpha = None
    session.all_masks = None
    session.iou_scores = None
    session.mask_index = 0
    session.history = []
    session.history_index = -1
    _clear_mask_cache(session)


def _empty_outputs():
    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    empty_l = np.zeros((256, 256), dtype=np.uint8)
    return (empty_l, empty, empty, empty, empty, empty, empty, empty, empty, "", "", [])


def build_outputs(session_data, opacity, contrast, feather, choke, invert, bg_mode, bg_color, bg_image,
                  offset_x, offset_y, scale, rotation, flip_h, flip_v, crop, keep_shadow, do_inpaint,
                  run_compare=False):
    session = wrap(session_data)
    if session.raw_alpha is None or session.image is None:
        return _empty_outputs()
    alpha = apply_tuning(session.raw_alpha, opacity, contrast, feather, choke, invert)
    full_image = session.image
    tuned_alpha = alpha
    image = full_image
    if crop:
        image, alpha = crop_to_content(full_image, tuned_alpha)
    matte_u8 = alpha_to_uint8(alpha)
    rgba = composite_rgba(image, alpha)
    cutout_rgb = overlay_mask_vis(image, alpha)
    bg_rgb, bg_img = _resolve_background(bg_mode, bg_color, bg_image)
    composite = composite_on_bg(
        image, alpha, bg_rgb, bg_img, offset_x, offset_y, scale, rotation, flip_h, flip_v, keep_shadow,
    )
    composite = as_uint8_rgb(composite)
    cutout_rgb = as_uint8_rgb(cutout_rgb)
    hist = render_alpha_histogram(alpha)
    before_after = np.hstack([as_uint8_rgb(image), composite])
    compare_caption = f"SAM weights not found at `{SAM_CHECKPOINT}`. Download from {SAM_DOWNLOAD_URL}"
    if run_compare:
        try:
            sam_vis = overlay_mask_vis(full_image, run_sam_predict(session.to_dict()), color=(0, 180, 255))
            zim_vis = overlay_mask_vis(full_image, tuned_alpha)
            if crop:
                sam_vis, _ = crop_to_content(sam_vis, tuned_alpha)
                zim_vis, _ = crop_to_content(zim_vis, tuned_alpha)
            compare_caption = "Left: SAM (hard binary edges). Right: ZIM (soft alpha matte)."
        except gr.Error as exc:
            sam_vis = np.zeros_like(image)
            zim_vis = overlay_mask_vis(image, alpha)
            compare_caption = str(exc)
    else:
        sam_vis = np.zeros_like(image)
        zim_vis = overlay_mask_vis(image, alpha)
    compare_panel = np.hstack([as_uint8_rgb(sam_vis), as_uint8_rgb(zim_vis)])
    inpaint_vis = as_uint8_rgb(inpaint_background(image, alpha) if do_inpaint else composite)
    edge = render_edge_inspector(image, alpha)
    gallery = build_multimask_gallery(session)
    iou_text = ""
    if session.iou_scores is not None and len(session.iou_scores) > 0:
        scores = ", ".join(f"{s:.3f}" for s in session.iou_scores)
        best = int(np.argmax(session.iou_scores))
        iou_text = (
            f"IoU scores: {scores}  |  active mask: #{session.mask_index}"
            + (f"  |  recommended: #{best}" if len(session.iou_scores) > 1 else "")
            + "\n"
        )
        if len(session.iou_scores) > 1:
            iou_text += (
                "Click a thumbnail in **Multi-mask** to switch candidates. "
                "Higher IoU usually means a better fit."
            )
        else:
            iou_text += (
                "One mask is active. Open **Multi-mask** and click **Generate 4 candidates** "
                "to compare alternatives (uses your current point/box/scribble prompts)."
            )
    return (matte_u8, rgba, composite, cutout_rgb, compare_panel, edge, hist, before_after,
            inpaint_vis, compare_caption, iou_text, gallery)


def save_session_json(session_data, tuning):
    session = wrap(session_data)
    path = os.path.join(EXPORT_DIR, "session.json")
    prompts_serializable = {k: (v.tolist() if k == "scribble" else v) for k, v in session.prompts.items()}
    data = {
        "model_key": session.model_key,
        "prompts": prompts_serializable,
        "mask_index": session.mask_index,
        "iou_scores": session.iou_scores.tolist() if session.iou_scores is not None else None,
        "tuning": tuning,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


NUM_IMAGE_OUTPUTS = 12


def pack_image_outputs(session_data, *tuning_args, run_compare=False, with_editor=False):
    """Session dict + image panel outputs (12 items), optionally + interactive editor HTML."""
    tuning = normalize_tuning_args(tuning_args)
    session = ensure_session(session_data)
    images = build_outputs(session, *tuning, run_compare=run_compare)
    base = _pack(wrap(session_data).to_dict(), *images)
    if with_editor:
        return base + (composite_editor_html(session, tuning),)
    return base


def pack_panel_outputs(session_data, mask_stack_key, *tuning_args, with_editor=True):
    """Session + mask-stack key + right-panel visuals (+ optional composite editor)."""
    tuning = normalize_tuning_args(tuning_args)
    session = ensure_session(session_data)
    images = build_outputs(session, *tuning, run_compare=False)
    out = (wrap(session_data).to_dict(), mask_stack_key, *images)
    if with_editor:
        return out + (composite_editor_html(session, tuning),)
    return out


def pack_example_outputs(example_img, session):
    """Load example into session; refresh panel + prompter + scribble."""
    if example_img is None:
        return pack_image_outputs(session, *DEFAULT_TUNING) + (None, None)
    s = wrap(session)
    s.image = example_img
    clear_mask_results(s)
    remember_prompter(s, {"image": example_img, "points": []})
    prompter_value = {"image": example_img, "points": []}
    return pack_image_outputs(s.to_dict(), *DEFAULT_TUNING) + (
        prompter_value, scribble_from_image(example_img),
    )


def pack_remove_image(session, *tuning_args):
    """Reset workspace after prompter.clear."""
    model_key = wrap(session).model_key
    fresh = new_session()
    fresh["model_key"] = model_key
    return (fresh, *_empty_outputs(), None, None)


def get_examples():
    assets_dir = os.path.join(os.path.dirname(__file__), "examples")
    if not os.path.isdir(assets_dir):
        return []
    return [os.path.join(assets_dir, f) for f in sorted(os.listdir(assets_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]


def on_model_change(model_key, session):
    s = wrap(session)
    s.model_key = model_key
    s.raw_alpha = None
    gr.Info(f"Model set to {model_key}. Re-run inference after uploading an image.")
    return s.to_dict()


def on_image_upload(img_dict=None, session=None):
    s = wrap(session)
    if img_dict is None or (isinstance(img_dict, dict) and img_dict.get("image") is None):
        return pack_image_outputs(s.to_dict(), *DEFAULT_TUNING)
    image = normalize_image_for_model(img_dict["image"] if isinstance(img_dict, dict) else img_dict)
    s.image = image
    clear_mask_results(s)
    remember_prompter(s, {"image": image, "points": []})
    return pack_image_outputs(s.to_dict(), *DEFAULT_TUNING)


def on_example(example_img, session):
    return pack_example_outputs(example_img, session)


def on_prompter_change(img_dict=None, session=None):
    """State-only sync when points/boxes are edited on the canvas."""
    s = wrap(session)
    if img_dict is None or img_dict.get("image") is None:
        return s.to_dict()
    remember_prompter(s, img_dict)
    return s.to_dict()


def pack_clear_prompts(session, *tuning_args):
    """Clear points/scribbles and mask results; keep the image."""
    s = wrap(session)
    if s.image is None:
        return pack_image_outputs(s.to_dict(), *tuning_args) + (None, None)
    clear_mask_results(s)
    prompter_value = {"image": s.image, "points": []}
    remember_prompter(s, prompter_value)
    return pack_image_outputs(s.to_dict(), *tuning_args) + (
        prompter_value, scribble_from_image(s.image),
    )


def on_clear_prompts(session, *tuning_args):
    return pack_clear_prompts(session, *tuning_args)


def on_remove_image(session, *tuning_args):
    return pack_remove_image(session, *tuning_args)


def on_run_point(img_dict=None, session=None, mask_stack_key=None, *tuning_args):
    s = wrap(session)
    image, prompts = parse_prompter_input(img_dict, s.prompts)
    s.image = normalize_image_for_model(image)
    s.prompts = prompts
    remember_prompter(s, img_dict)
    s.snapshot()
    run_zim_predict(s.to_dict(), multimask=False)
    return pack_panel_outputs(s.to_dict(), None, *tuning_args, with_editor=True)


def on_run_multimask(session, mask_stack_key, *tuning_args):
    """Re-run ZIM with multimask_output=True using prompts already in session."""
    s = wrap(session)
    if s.image is None:
        raise gr.Error("Upload an image and click Run first.")
    if not s.prompts:
        raise gr.Error("Add a point, box, or scribble prompt, then click Run before generating candidates.")
    run_zim_predict(s.to_dict(), multimask=True)
    key = s.d.get("_sid")
    gr.Info("Generated 4 mask candidates. Click a thumbnail to switch.")
    return pack_panel_outputs(s.to_dict(), key, *tuning_args, with_editor=True)


def on_run_scribble(scribble, session, mask_stack_key, *tuning_args):
    s = wrap(session)
    if s.image is None and scribble is not None and scribble.get("background") is not None:
        s.image = normalize_image_for_model(scribble["background"])
    if s.image is None:
        raise gr.Error("Upload an image first.")
    if scribble is None:
        raise gr.Error("Please draw at least one scribble stroke.")
    _, prompts = parse_scribble_prompt(s.image, scribble)
    s.prompts = prompts
    s.snapshot()
    run_zim_predict(s.to_dict(), multimask=False)
    return pack_panel_outputs(s.to_dict(), None, *tuning_args, with_editor=True)


def on_tuning_change(session, *tuning_args):
    tuning = normalize_tuning_args(tuning_args)
    s = ensure_session(session)
    return (*build_outputs(s, *tuning, run_compare=False), composite_editor_html(s, tuning))


def on_composite_interact(json_str, session, *tuning_args):
    """Sync drag/scroll transform from the interactive composite canvas to sliders + preview."""
    tuning = normalize_tuning_args(tuning_args)
    s = ensure_session(session)
    noop = (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        *build_outputs(s, *tuning, run_compare=False),
        composite_editor_html(s, tuning),
    )
    if not json_str or not str(json_str).strip():
        return noop
    data = json.loads(json_str)
    tuning = list(tuning)
    tuning[8] = int(round(float(data.get("offset_x", tuning[8]))))
    tuning[9] = int(round(float(data.get("offset_y", tuning[9]))))
    tuning[10] = float(np.clip(float(data.get("scale", tuning[10])), 0.25, 3.0))
    tuning[11] = float(data.get("rotation", tuning[11]))
    tuning[12] = bool(data.get("flip_h", tuning[12]))
    tuning[13] = bool(data.get("flip_v", tuning[13]))
    tuning = tuple(tuning)
    return (
        gr.update(value=tuning[8]),
        gr.update(value=tuning[9]),
        gr.update(value=tuning[10]),
        gr.update(value=tuning[11]),
        gr.update(value=tuning[12]),
        gr.update(value=tuning[13]),
        *build_outputs(s, *tuning, run_compare=False),
        composite_editor_html(s, tuning),
    )


def on_compare(session, *tuning_args):
    tuning = normalize_tuning_args(tuning_args)
    outs = build_outputs(ensure_session(session), *tuning, run_compare=True)
    return outs[4], outs[9]


def _gallery_select_index(evt, *, columns: int = 2) -> Optional[int]:
    """Parse Gradio SelectData.index (int or (row, col) tuple)."""
    if evt is None:
        return 0
    if hasattr(evt, "selected") and evt.selected is False:
        return None
    if hasattr(evt, "index"):
        idx = evt.index
        if isinstance(idx, (tuple, list)):
            if len(idx) >= 2:
                return int(idx[0]) * columns + int(idx[1])
            return int(idx[0])
        return int(idx)
    return int(evt)


def on_select_mask(evt: gr.SelectData, session, mask_stack_key, *tuning_args):
    s = wrap(session)
    stack_key = mask_stack_key or s.d.get("_sid")
    masks = _load_mask_stack(s, stack_key)
    if masks is None or masks.shape[0] <= 1:
        gr.Warning("Generate candidates first, then click a thumbnail.")
        return pack_panel_outputs(s.to_dict(), stack_key, *tuning_args, with_editor=True)
    idx = _gallery_select_index(evt)
    if idx is None:
        return pack_panel_outputs(s.to_dict(), stack_key, *tuning_args, with_editor=True)
    idx = max(0, min(idx, masks.shape[0] - 1))
    s.mask_index = idx
    s.raw_alpha = logits_to_alpha(masks[idx])
    s.snapshot()
    gr.Info(f"Using mask candidate #{idx}.")
    return pack_panel_outputs(s.to_dict(), stack_key, *tuning_args, with_editor=True)


def on_undo(session, mask_stack_key, *tuning_args):
    s = wrap(session)
    s.undo()
    return pack_panel_outputs(s.to_dict(), mask_stack_key, *tuning_args, with_editor=True)


def on_redo(session, mask_stack_key, *tuning_args):
    s = wrap(session)
    s.redo()
    return pack_panel_outputs(s.to_dict(), mask_stack_key, *tuning_args, with_editor=True)


def on_prepare_exports(session, *tuning_args):
    """Write PNG + session JSON to disk; return paths for the download list."""
    s = wrap(ensure_session(session))
    if s.raw_alpha is None or s.image is None:
        raise gr.Error("Run inference first before exporting.")
    tuning = normalize_tuning_args(tuning_args)
    matte_u8, rgba, composite, *_rest = build_outputs(s.to_dict(), *tuning, run_compare=False)
    base = os.path.join(EXPORT_DIR, "zim_export")
    mask_path = base + "_mask.png"
    rgba_path = base + "_cutout.png"
    comp_path = base + "_composite.png"
    Image.fromarray(matte_u8).save(mask_path)
    Image.fromarray(rgba).save(rgba_path)
    Image.fromarray(composite).save(comp_path)
    tuning_dict = {
        "opacity": tuning_args[0], "contrast": tuning_args[1],
        "feather": tuning_args[2], "choke": tuning_args[3], "invert": tuning_args[4],
        "bg_mode": tuning_args[5], "crop": tuning_args[14],
    }
    session_path = save_session_json(session, tuning_dict)
    gr.Info("Export files are ready. Download from the list below.")
    return [mask_path, rgba_path, comp_path, session_path]


CONFIRM_CLEAR_PROMPTS = (
    "() => confirm('Clear all prompts and mask results?\\nThe image will be kept.')"
)

CONFIRM_REMOVE_IMAGE = (
    "() => confirm('Remove the entire image and reset the workspace?\\nThis cannot be undone.')"
)

COMPOSITE_EDITOR_HEAD = """
<script>
window.addEventListener("message", function(ev) {
  if (!ev.data || ev.data.type !== "zim_composite_transform") return;
  var root = document.getElementById("composite_transform");
  if (!root) return;
  var ta = root.querySelector("textarea") || root.querySelector("input");
  if (!ta) return;
  ta.value = JSON.stringify(ev.data.payload);
  ta.dispatchEvent(new Event("input", { bubbles: true }));
});
</script>
"""

STUDIO_CSS = """
<style>
#studio-header {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
  width: 100%;
}
/* Title column grows; controls stay fixed width */
#studio-header > div.studio-header-grow-wrap {
  flex: 1 1 auto !important;
  min-width: 12rem;
}
#studio-header > div:not(.studio-header-grow-wrap) {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}
#studio-header .studio-header-text {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: nowrap;
  white-space: nowrap;
  line-height: 2.375rem;
}
#studio-header .studio-header-text .studio-title {
  font-size: 1.35rem;
  font-weight: 700;
}
#studio-header .studio-header-text .studio-subtitle {
  font-size: 0.95rem;
  font-weight: 400;
  color: var(--body-text-color-subdued, #666);
}
#studio-header .studio-model-wrap {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 0.5rem;
  flex-wrap: nowrap;
}
#studio-header .studio-model-wrap > .block {
  margin: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  border: none !important;
  background: transparent !important;
}
#studio-header .studio-model-label {
  font-size: 0.9rem;
  font-weight: 500;
  white-space: nowrap;
  line-height: 2.375rem;
  color: var(--body-text-color, inherit);
}
#studio-header .studio-model {
  min-width: 11rem;
  max-width: 14rem;
}
#studio-header .studio-model .wrap {
  margin: 0;
}
#studio-header .studio-header-btn {
  min-width: 4.5rem !important;
  max-width: 4.5rem !important;
  height: 2.375rem !important;
  min-height: 2.375rem !important;
  padding-left: 0.5rem !important;
  padding-right: 0.5rem !important;
}
.input-panel-wrap {
  position: relative;
}
.input-toolbar-row {
  position: absolute;
  top: 2.75rem;
  right: 0.75rem;
  z-index: 20;
  display: flex !important;
  gap: 0.25rem;
  margin: 0 !important;
  width: auto !important;
  justify-content: flex-end;
}
.input-toolbar-row .studio-toolbar-btn {
  min-width: 2rem !important;
  width: 2rem !important;
  max-width: 2rem !important;
  height: 2rem !important;
  min-height: 2rem !important;
  padding: 0 !important;
  font-size: 1rem;
  line-height: 1;
  border-radius: var(--radius-sm, 4px);
  box-shadow: var(--block-shadow, 0 1px 2px rgba(0,0,0,.05));
}
.studio-prompter-wrap .icon-buttons {
  display: none !important;
}
.studio-scribble-wrap .icon-buttons {
  display: none !important;
}
.composite-interactive-iframe {
  display: block;
  width: 100%;
  min-height: 220px;
  border-radius: var(--radius-sm, 6px);
}
.composite-interactive-wrap {
  margin-top: 0.5rem;
}
.composite-interactive-wrap canvas {
  display: block;
  max-width: 100%;
  border-radius: var(--radius-sm, 6px);
  border: 1px solid var(--block-border-color, #ddd);
  cursor: grab;
  touch-action: none;
}
.composite-interactive-wrap canvas:active {
  cursor: grabbing;
}
.composite-interactive-hint {
  margin-top: 0.35rem;
  font-size: 0.85rem;
  color: var(--body-text-color-subdued, #666);
}
.composite-editor-empty {
  padding: 0.75rem;
  font-size: 0.9rem;
  color: var(--body-text-color-subdued, #666);
  border: 1px dashed var(--block-border-color, #ccc);
  border-radius: var(--radius-sm, 6px);
}
.export-panel .export-downloads {
  margin-top: 0.5rem;
}
.export-panel .export-downloads .wrap {
  min-height: 0 !important;
}
.multimask-gallery .grid-wrap {
  gap: 0.75rem !important;
}
.multimask-gallery .thumbnail-item,
.multimask-gallery button.thumbnail-item {
  min-height: 11rem !important;
}
.multimask-gallery .thumbnail-item img,
.multimask-gallery button.thumbnail-item img {
  object-fit: contain !important;
  max-height: 10rem !important;
  width: 100% !important;
}
</style>
"""


def _wire_input_toolbar(session, tuning_args, image_outputs,
                        prompter, scribble_editor):
    panel_outputs = [session] + list(image_outputs)
    full_outputs = panel_outputs + [prompter, scribble_editor]
    # undo_btn.click(on_undo, [session] + list(tuning_args), panel_outputs)
    # redo_btn.click(on_redo, [session] + list(tuning_args), panel_outputs)
    # trash_btn.click(
    #     on_clear_prompts,
    #     [session] + list(tuning_args),
    #     full_outputs,
    #     js=CONFIRM_CLEAR_PROMPTS,
    # )
    # cross_btn.click(
    #     on_remove_image,
    #     [session] + list(tuning_args),
    #     full_outputs,
    #     js=CONFIRM_REMOVE_IMAGE,
    # )


def build_ui():
    with gr.Blocks(title="ZIM Matting Studio", head=COMPOSITE_EDITOR_HEAD) as demo:
        session = gr.State(new_session())
        mask_stack_key = gr.State(None)
        gr.HTML(STUDIO_CSS)

        iou_text = gr.Textbox(
            label="Quality scores (IoU)",
            interactive=False,
            lines=3,
            placeholder="Run inference to see IoU scores here.",
            render=False,
        )
        matte_out = gr.Image(label="Alpha matte", image_mode="L", interactive=False, render=False)
        rgba_out = gr.Image(label="RGBA cutout", interactive=False, render=False)
        cutout_vis_out = gr.Image(label="Cutout overlay", interactive=False, render=False)
        composite_out = gr.Image(label="Composite", interactive=False, render=False)
        before_after_out = gr.Image(label="Before | After", interactive=False, render=False)
        inpaint_out = gr.Image(label="Inpaint preview", interactive=False, render=False)
        compare_caption_out = gr.Markdown(f"SAM weights: `{SAM_CHECKPOINT}`", render=False)
        compare_out = gr.Image(label="SAM (left) vs ZIM (right)", interactive=False, render=False)
        edge_out = gr.Image(label="Edge inspector", interactive=False, render=False)
        hist_out = gr.Image(label="Alpha histogram", interactive=False, render=False)
        gallery_out = gr.Gallery(
            label="Candidates (click to select)",
            columns=2,
            height=420,
            object_fit="contain",
            allow_preview=False,
            preview=False,
            elem_classes="multimask-gallery",
            render=False,
        )
        multimask_btn = gr.Button("Generate 4 candidates", variant="secondary", render=False)
        export_btn = gr.Button("Generate export files", variant="primary", render=False)
        export_files = gr.File(
            label="Downloads",
            file_count="multiple",
            interactive=False,
            elem_classes="export-downloads",
            render=False,
        )
        composite_editor = gr.HTML("", render=False)
        composite_transform = gr.Textbox(visible=False, elem_id="composite_transform")

        with gr.Row(elem_id="studio-header", equal_height=True):
            with gr.Column(scale=4, min_width=280, elem_classes="studio-header-grow-wrap"):
                gr.HTML(
                    '<div class="studio-header-text">'
                    '<span class="studio-title">ZIM Matting Studio</span>'
                    '<span class="studio-subtitle">Zero-shot interactive image matting</span>'
                    '</div>'
                )
            with gr.Column(scale=0, min_width=260, elem_classes="studio-model-wrap"):
                gr.HTML('<span class="studio-model-label">Select model</span>')
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value="vit_l (quality)",
                    show_label=False,
                    container=False,
                    elem_classes="studio-model",
                    min_width=180,
                )
            # undo_btn = gr.Button("Undo", elem_classes="studio-header-btn", scale=0, min_width=72)
            # redo_btn = gr.Button("Redo", elem_classes="studio-header-btn", scale=0, min_width=72)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Matte tuning (no re-inference)", open=True):
                    # gr.Markdown(
                    #     "Post-process the matte without re-running the model. "
                    #     "**Opacity** = global transparency; **Edge contrast** = harden/soften edges; "
                    #     "**Gamma** = non-linear edge curve; **Feather** blurs edges; "
                    #     "**Choke/Expand** shrinks or grows the mask."
                    # )
                    opacity = gr.Slider(
                        0, 100, value=100, step=1,
                        label="Opacity %",
                        info="100 = original matte. Lower = more transparent everywhere.",
                    )
                    contrast = gr.Slider(
                        0, 200, value=100, step=1,
                        label="Edge contrast %",
                        info="100 = unchanged. >100 harder edges; <100 softer edges (linear stretch from midpoint).",
                    )
                    feather = gr.Slider(0, 20, value=0, step=1, label="Feather")
                    choke = gr.Slider(-10, 10, value=0, step=1, label="Choke (-) / Expand (+)")
                    invert = gr.Checkbox(label="Invert selection", value=False)
                    crop = gr.Checkbox(label="Crop to object", value=False)
                    keep_shadow = gr.Checkbox(label="Keep soft shadow", value=False)
                    do_inpaint = gr.Checkbox(label="Inpaint removed background (preview)", value=False)
                with gr.Accordion("Composite settings", open=True):
                    gr.Markdown(
                        "Run inference first, then pick a background. Adjust **Scale**, **Offset**, "
                        "and **Rotation** with sliders, or use the interactive editor in the Composite tab "
                        "(drag to move, scroll to scale, Shift+scroll to rotate)."
                    )
                    bg_mode = gr.Radio(["Solid color", "Upload image"], value="Solid color", label="Background source")
                    bg_color = gr.ColorPicker(value="#FFFFFF", label="Background color")
                    bg_upload = gr.Image(label="Background image", type="numpy")
                    scale = gr.Slider(0.25, 3.0, value=1.0, step=0.01, label="Foreground scale")
                    offset_x = gr.Slider(-400, 400, value=0, step=1, label="Offset X")
                    offset_y = gr.Slider(-400, 400, value=0, step=1, label="Offset Y")
                    rotation = gr.Slider(-180, 180, value=0, step=1, label="Rotation (°)")
                    with gr.Row():
                        flip_h = gr.Checkbox(label="Flip horizontal", value=False)
                        flip_v = gr.Checkbox(label="Flip vertical", value=False)
            with gr.Column(scale=1):
                example_img = gr.Image(visible=False)
                with gr.Tab("Point / Box"):
                    with gr.Column(elem_classes=["input-panel-wrap", "studio-prompter-wrap"]):
                        prompter = ImagePrompter(label="Query image", sources="upload")
                        gr.Markdown("**Left** = positive · **Middle** = negative · **Drag** = box")
                        run_btn = gr.Button("Run", variant="primary")
                with gr.Tab("Scribble"):
                    with gr.Column(elem_classes=["input-panel-wrap", "studio-scribble-wrap"]):
                        scribble_editor = gr.ImageEditor(
                            label="Scribble",
                            brush=gr.Brush(colors=["#00FF00"], default_size=15),
                            sources="upload",
                            transforms=None,
                            layers=False,
                        )
                        run_scribble_btn = gr.Button("Run Scribble", variant="primary")
                gr.Examples(
                    examples=[[p] for p in get_examples()],
                    inputs=[example_img, session],
                    outputs=[
                        session, matte_out, rgba_out, composite_out, cutout_vis_out, compare_out,
                        edge_out, hist_out, before_after_out, inpaint_out, compare_caption_out,
                        iou_text, gallery_out, prompter, scribble_editor,
                    ],
                    fn=pack_example_outputs,
                    cache_examples=False,
                    run_on_click=True,
                    label="Examples",
                )
            with gr.Column(scale=1):
                iou_text.render()
                with gr.Tab("Matte"):
                    matte_out.render()
                with gr.Tab("Cutout"):
                    rgba_out.render()
                    cutout_vis_out.render()
                with gr.Tab("Composite"):
                    gr.Markdown(
                        "Foreground blended onto your chosen background. "
                        "Use the interactive editor below or sliders in Composite settings."
                    )
                    composite_out.render()
                    composite_editor.render()
                    before_after_out.render()
                    inpaint_out.render()
                with gr.Tab("ZIM vs SAM"):
                    compare_caption_out.render()
                    compare_btn = gr.Button("Run comparison")
                    compare_out.render()
                with gr.Tab("Edge"):
                    gr.Markdown(
                        "**Edge inspector:** magnified view around the strongest alpha boundary "
                        "(useful for hair/fur). **Alpha histogram:** distribution of transparency "
                        "values. A good ZIM matte often has pixels between 0 and 1 at edges "
                        "(soft transition). A sharp two-peak chart (only 0 and 1) means a hard, "
                        "binary mask — try **Edge contrast** or **Feather** under Matte tuning."
                    )
                    edge_out.render()
                    hist_out.render()
                with gr.Tab("Multi-mask"):
                    gr.Markdown(
                        "1. Upload an image and add prompts, then click **Run** (single matte).  "
                        "2. Click **Generate 4 candidates** here to compare ZIM alternatives.  "
                        "3. **Click a thumbnail** to switch the active mask — Matte / Cutout / Composite update immediately."
                    )
                    multimask_btn.render()
                    gallery_out.render()
                with gr.Tab("Export"):
                    with gr.Column(elem_classes="export-panel"):
                        gr.Markdown(
                            "Run inference first, then click **Generate export files**. "
                            "Matte PNG, RGBA cutout, composite PNG, and session JSON will appear "
                            "in the download list."
                        )
                        export_btn.render()
                        export_files.render()
        tuning_args = (
            opacity, contrast, feather, choke, invert, bg_mode, bg_color, bg_upload,
            offset_x, offset_y, scale, rotation, flip_h, flip_v, crop, keep_shadow, do_inpaint,
        )
        image_outputs = (
            matte_out, rgba_out, composite_out, cutout_vis_out, compare_out, edge_out, hist_out,
            before_after_out, inpaint_out, compare_caption_out, iou_text, gallery_out,
        )
        preview_outputs = list(image_outputs) + [composite_editor]
        transform_outputs = (
            [offset_x, offset_y, scale, rotation, flip_h, flip_v]
            + list(image_outputs) + [composite_editor]
        )
        panel_outputs = [session, mask_stack_key] + preview_outputs

        model_dropdown.change(on_model_change, [model_dropdown, session], [session])
        prompter.upload(on_image_upload, [prompter, session], [session] + list(image_outputs))
        prompter.change(on_prompter_change, [prompter, session], [session])
        run_btn.click(
            on_run_point,
            [prompter, session, mask_stack_key] + list(tuning_args),
            panel_outputs,
        )
        run_scribble_btn.click(
            on_run_scribble,
            [scribble_editor, session, mask_stack_key] + list(tuning_args),
            panel_outputs,
        )
        for ctrl in tuning_args:
            ctrl.change(on_tuning_change, [session] + list(tuning_args), preview_outputs)
        composite_transform.change(
            on_composite_interact,
            [composite_transform, session] + list(tuning_args),
            transform_outputs,
        )
        compare_btn.click(on_compare, [session] + list(tuning_args), [compare_out, compare_caption_out])
        gallery_out.select(
            on_select_mask,
            [session, mask_stack_key] + list(tuning_args),
            panel_outputs,
        )
        multimask_btn.click(
            on_run_multimask,
            [session, mask_stack_key] + list(tuning_args),
            panel_outputs,
        )
        # undo_btn.click(on_undo, [session] + list(tuning_args), panel_outputs)
        # redo_btn.click(on_redo, [session] + list(tuning_args), panel_outputs)
        _wire_input_toolbar(
            session, tuning_args, image_outputs, prompter, scribble_editor,
        )
        _wire_input_toolbar(
            session, tuning_args, image_outputs, prompter, scribble_editor,
        )
        export_btn.click(
            on_prepare_exports,
            [session] + list(tuning_args),
            [export_files],
        )
    return demo


if __name__ == "__main__":
    try:
        MODELS.load_zim("vit_l (quality)")
        print("Loaded default ZIM model (vit_l).")
    except Exception as exc:
        print(f"Warning: could not pre-load model: {exc}")
    app = build_ui()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860)
