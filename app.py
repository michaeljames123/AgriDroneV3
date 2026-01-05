import asyncio
import base64
import os
import shutil
import threading
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from PIL import ImageOps
from ultralytics import YOLO

app = FastAPI()

_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
if _cors_origins_raw.strip() == "*":
    _allow_origins = ["*"]
else:
    _allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DEVICE = os.getenv("DEVICE")
_IMG_SIZE = int(os.getenv("IMG_SIZE", "1024"))
_MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "120"))
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
_MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "2048"))

_model: Optional[YOLO] = None
_model_path: Optional[str] = None

_model_lock = threading.Lock()
_inference_lock = asyncio.Lock()
_ui_path = Path(__file__).resolve().with_name("index.html")


def _normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names)}
    return {}


def _candidate_model_paths() -> list[str]:
    env_path = os.getenv("MODEL_PATH")
    paths: list[str] = []
    if env_path:
        paths.append(env_path)
    paths.extend([os.path.join("models", "best.pt"), "best.pt"])
    seen = set()
    unique_paths: list[str] = []
    for p in paths:
        if p and p not in seen:
            unique_paths.append(p)
            seen.add(p)
    return unique_paths


def _resolve_model_download_target() -> str:
    env_path = os.getenv("MODEL_PATH")
    target = env_path or os.path.join("models", "best.pt")
    if target and os.path.isdir(target):
        target = os.path.join(target, "best.pt")
    return target


def _download_model(url: str, target_path: str) -> None:
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    tmp_path = target_path + ".download"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    try:
        with urllib.request.urlopen(url, timeout=_MODEL_DOWNLOAD_TIMEOUT) as r:
            status = getattr(r, "status", 200)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"HTTP {status}")

            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(r, f)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1024:
            raise RuntimeError("Downloaded model file is empty")

        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _ensure_model_present() -> None:
    if any(os.path.exists(p) for p in _candidate_model_paths()):
        return

    url = os.getenv("MODEL_URL")
    if not url:
        return

    target_path = _resolve_model_download_target()
    try:
        _download_model(url, target_path)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "Model download failed", "message": str(e), "url": url},
        )


def _load_model_if_needed() -> YOLO:
    global _model
    global _model_path
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        _ensure_model_present()

        for path in _candidate_model_paths():
            if os.path.exists(path):
                _model = YOLO(path)
                _model_path = os.path.abspath(path)
                return _model

    raise HTTPException(
        status_code=503,
        detail={
            "error": "Model not found",
            "message": "Place best.pt in ./models/best.pt or ./best.pt, or set MODEL_PATH.",
            "checked_paths": _candidate_model_paths(),
        },
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "agri-drone-seg-api"}


@app.get("/ui")
def ui() -> Any:
    if _ui_path.exists():
        return FileResponse(str(_ui_path))
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/health")
def health(load_model: bool = False) -> dict[str, Any]:
    model_loaded = _model is not None
    model_error = None

    if load_model and not model_loaded:
        try:
            _load_model_if_needed()
            model_loaded = True
        except HTTPException as e:
            model_error = e.detail
        except Exception as e:
            model_error = str(e)

    model_names = _normalize_model_names(getattr(_model, "names", None)) if model_loaded else None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": _model_path,
        "model_names": model_names,
        "model_error": model_error,
        "checked_paths": _candidate_model_paths(),
        "device": _DEVICE,
        "imgsz": _IMG_SIZE,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.7,
    return_image: bool = False,
) -> dict[str, Any]:
    if not (0.0 <= conf <= 1.0):
        raise HTTPException(status_code=422, detail="conf must be between 0 and 1")
    if not (0.0 <= iou <= 1.0):
        raise HTTPException(status_code=422, detail="iou must be between 0 and 1")

    model = _load_model_if_needed()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    image_bytes = await file.read()

    if _MAX_UPLOAD_BYTES > 0 and len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        pil_image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes)))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    if _MAX_IMAGE_EDGE > 0 and max(pil_image.width, pil_image.height) > _MAX_IMAGE_EDGE:
        pil_image.thumbnail((_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)

    pil_image = pil_image.convert("RGB")

    image_np = np.array(pil_image)

    predict_kwargs: dict[str, Any] = {
        "source": image_np,
        "imgsz": _IMG_SIZE,
        "conf": conf,
        "iou": iou,
        "save": False,
        "retina_masks": True,
        "verbose": False,
    }
    if _DEVICE:
        predict_kwargs["device"] = _DEVICE

    try:
        async with _inference_lock:
            results = await asyncio.to_thread(model.predict, **predict_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "inference_failed", "message": str(e)})

    if not results:
        model_names = _normalize_model_names(getattr(model, "names", None))
        return {
            "image": {"width": pil_image.width, "height": pil_image.height},
            "predictions": [],
            "meta": {
                "model_path": _model_path,
                "model_names": model_names,
                "imgsz": _IMG_SIZE,
                "device": _DEVICE,
                "conf": conf,
                "iou": iou,
            },
        }

    r = results[0]

    names = _normalize_model_names(getattr(r, "names", None))
    if not names:
        names = _normalize_model_names(getattr(model, "names", None))

    boxes = getattr(r, "boxes", None)
    masks = getattr(r, "masks", None)

    predictions: list[dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        conf_list = boxes.conf.detach().cpu().numpy().astype(float).tolist()
        xyxy_list = boxes.xyxy.detach().cpu().numpy().astype(float).tolist()

        mask_polys: list[Optional[np.ndarray]]
        if masks is not None and getattr(masks, "xy", None) is not None:
            mask_polys = [np.asarray(p, dtype=float) if p is not None else None for p in masks.xy]
        elif masks is not None and getattr(masks, "xyn", None) is not None:
            wh = np.array([pil_image.width, pil_image.height], dtype=float)
            mask_polys = [np.asarray(p, dtype=float) * wh if p is not None else None for p in masks.xyn]
        else:
            mask_polys = [None] * len(cls_list)

        for i, (cls_id, score, box_xyxy) in enumerate(zip(cls_list, conf_list, xyxy_list)):
            poly = None
            poly_norm = None

            if i < len(mask_polys) and mask_polys[i] is not None:
                poly_arr = mask_polys[i]
                poly = poly_arr.tolist()
                if pil_image.width > 0 and pil_image.height > 0:
                    poly_norm = (poly_arr / np.array([pil_image.width, pil_image.height])).tolist()

            raw_name = names.get(cls_id, cls_id)

            predictions.append(
                {
                    "class_id": cls_id,
                    "class_name": str(raw_name),
                    "confidence": score,
                    "box_xyxy": box_xyxy,
                    "polygon": poly,
                    "polygon_normalized": poly_norm,
                }
            )

    response: dict[str, Any] = {
        "image": {"width": pil_image.width, "height": pil_image.height},
        "predictions": predictions,
        "meta": {
            "model_path": _model_path,
            "model_names": names,
            "imgsz": _IMG_SIZE,
            "device": _DEVICE,
            "conf": conf,
            "iou": iou,
        },
    }

    if return_image:
        annotated = r.plot()
        annotated_rgb = annotated[..., ::-1]
        buf = BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=90)
        response["annotated_image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        response["annotated_image_mime"] = "image/jpeg"

    return response
