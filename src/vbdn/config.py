"""Shared defaults for the malware classification CLI."""

from pathlib import Path

SEED = 50

RESULTS_DIR = Path("results")
PLOT_DIR = RESULTS_DIR / "plots"
MODEL_DIR = RESULTS_DIR / "models"
CSV_DIR = RESULTS_DIR / "csv"
REPORT_DIR = RESULTS_DIR / "reports"
PREDICTION_DIR = RESULTS_DIR / "predictions"

DEFAULT_DATASETS = ("Malimg", "Malevis", "Blended")
ALL_DATASETS = (*DEFAULT_DATASETS, "BIG2015")
DEFAULT_PRETRAINED_MODELS = (
    "VGG16",
    "AlexNet",
    "DenseNet-121",
    "MobileNetV2",
    "ResNeXt-50",
    "ShuffleNetV2",
)
DEFAULT_BIG2015_PATH = Path.home() / ".cache/kaggle/BIG2015/main"
PAPER_IMAGE_SIZE = 512
PAPER_EPOCHS = 200
PAPER_TRAIN_BATCH_SIZE = 64
PAPER_TEST_BATCH_SIZE = 32
PAPER_PRETRAINED_BATCH_SIZE = 8
PAPER_LEARNING_RATE = 0.01
PAPER_MOMENTUM = 0.5
SAFE_IMAGE_SIZE = 224
SAFE_EPOCHS = 1
SAFE_TRAIN_BATCH_SIZE = 32
SAFE_TEST_BATCH_SIZE = 16
SAFE_PRETRAINED_BATCH_SIZE = 4
SAFE_PRETRAINED_LEARNING_RATE = 0.0001
RUN_ALL_CONVNET_EPOCHS = 80
RUN_ALL_PRETRAINED_EPOCHS = 18

KNOWN_DATASETS = {
    "Malimg": {
        "kaggle": "ikrambenabd/malimg-original",
        "subpath": "malimg_paper_dataset_imgs",
        # Keep the paper's 8,408 training images. The available archive has
        # 9,339 images (four fewer than Table 1), so its test remainder is 931.
        "paper_train_size": 8408,
    },
    "Malevis": {
        "kaggle": "sohamkumar1703/malevis-dataset",
        "subpath": "malevis_train_val_300x300/train",
        "test_subpath": "malevis_train_val_300x300/val",
    },
    "Blended": {
        "kaggle": "gauravpendharkar/blended-malware-image-dataset",
        "subpath": "malware_dataset/train",
        "test_subpath": "malware_dataset/val",
    },
}


def ensure_output_dirs() -> None:
    """Create result folders only when a command is executed."""
    for path in (PLOT_DIR, MODEL_DIR, CSV_DIR, REPORT_DIR, PREDICTION_DIR):
        path.mkdir(parents=True, exist_ok=True)
