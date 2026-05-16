from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import tf_keras as keras

from web_app.model_manifest import IMAGE_SIZE, MAX_UPLOAD_MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

_PREPROCESSORS = {
    "MobileNetV2": keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": keras.applications.efficientnet.preprocess_input,
    "ResNet50": keras.applications.resnet.preprocess_input,
}


def validate_upload(filename: str, data: bytes) -> str | None:
    if not filename:
        return "Please provide an image file."

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return "Unsupported file type. Please upload jpg, jpeg, or png."

    if len(data) > MAX_UPLOAD_BYTES:
        return f"File is larger than {MAX_UPLOAD_MB} MB. Please choose a smaller image."

    return None


def decode_and_prepare_image(data: bytes, backbone: str) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    preprocess = _PREPROCESSORS.get(backbone)
    if preprocess is None:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    return preprocess(arr)


def load_model(model_path: Path, backbone: str) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if _PREPROCESSORS.get(backbone) is None:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    return keras.models.load_model(model_path, compile=False)


def predict_topk(
    model: keras.Model,
    image_tensor: np.ndarray,
    class_ids: list[str],
    class_id_to_name: dict[str, str],
    k: int = 3,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    probs = model.predict(image_tensor, verbose=0)[0]

    if len(probs) != len(class_ids):
        raise ValueError(
            "Model output dimension does not match configured class mapping. "
            f"Expected {len(class_ids)}, got {len(probs)}."
        )

    top_indices = np.argsort(probs)[::-1][:k]
    top_rows = []
    for idx in top_indices:
        class_id = class_ids[idx]
        top_rows.append(
            {
                "class_id": class_id,
                "breed": class_id_to_name[class_id],
                "confidence": float(probs[idx]),
            }
        )

    top1 = top_rows[0]
    return top1, top_rows
