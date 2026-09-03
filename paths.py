"""Portable paths for the public UGNS repository.

Data, result, and model roots can live outside the checkout.  Set the matching
environment variable before invoking a script to override a default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _root_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _root_from_env("UGNS_DATA_ROOT", REPO_ROOT / "data")
RESULTS_DIR = _root_from_env("UGNS_RESULTS_ROOT", REPO_ROOT / "results")
MODELS_DIR = _root_from_env("UGNS_MODELS_ROOT", REPO_ROOT / "checkpoints")

CONFIG_DIR = REPO_ROOT / "configs"
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
TRAINING_DIR = REPO_ROOT / "training"
INFERENCE_DIR = REPO_ROOT / "inference"
EVALUATION_DIR = REPO_ROOT / "evaluation"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"

CHECKPOINT_DIR = MODELS_DIR
MYDATA_DIR = DATA_DIR
DRUS_TRAIN_DIR = DATA_DIR / "ultrasound_drus_train_256"
ALL_RESULT_DIR = RESULTS_DIR / "ALL_RESULT"
GD_OUTPUT_NEW = RESULTS_DIR / "guided_diffusion_output_256_new"
GD_OUTPUT_DAS = RESULTS_DIR / "guided_diffusion_output_256_das"
BASE_DIFFUSION_CKPT = MODELS_DIR / "256x256_diffusion_uncond.pt"


def register_paths() -> None:
    """Register first-party modules and the vendored diffusion package."""
    for path in (
        REPO_ROOT,
        PREPROCESSING_DIR,
        TRAINING_DIR,
        INFERENCE_DIR,
        EVALUATION_DIR,
        THIRD_PARTY_DIR,
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


register_paths()
