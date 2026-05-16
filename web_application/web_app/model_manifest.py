from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 5

CLASS_ID_TO_NAME = {
    "n02086240": "Shih-Tzu",
    "n02087394": "Rhodesian Ridgeback",
    "n02088364": "Beagle",
    "n02089973": "English Foxhound",
    "n02093754": "Australian Terrier",
    "n02096294": "Border Terrier",
    "n02099601": "Golden Retriever",
    "n02105641": "Old English Sheepdog",
    "n02111889": "Samoyed",
    "n02115641": "Dingo",
}

CLASS_IDS = list(CLASS_ID_TO_NAME.keys())

MODEL_MANIFEST = {
    "MobileNetV2": {
        "run_name": "MobileNetV2",
        "artifact": PROJECT_ROOT / "results" / "MobileNetV2" / "model.keras",
        "backbone": "MobileNetV2",
    },
    "EfficientNetB0": {
        "run_name": "EfficientNetB0_FineTune",
        "artifact": PROJECT_ROOT / "results" / "EfficientNetB0_FineTune" / "model.keras",
        "backbone": "EfficientNetB0",
    },
    "ResNet50": {
        "run_name": "ResNet50",
        "artifact": PROJECT_ROOT / "results" / "ResNet50" / "model.keras",
        "backbone": "ResNet50",
    },
}
