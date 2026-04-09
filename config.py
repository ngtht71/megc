"""
Configuration file for MEGC project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Dataset paths configuration
DATASET_CONFIG = {
    "casme2": {
        "crop_dir": os.environ.get("CASME2_CROP_DIR", "/path/to/casme2/Cropped/Cropped"),
        "metadata_file": os.environ.get("CASME2_METADATA", "/path/to/CASME2-coding-20140508.xlsx"),
    },
    "samm": {
        "crop_dir": os.environ.get("SAMM_CROP_DIR", "/path/to/samm/SAMM_Cropped"),
        "metadata_file": os.environ.get("SAMM_METADATA", "/path/to/SAMM_Micro_FACS_Codes_v2.xlsx"),
    },
    "smic": {
        "crop_dir": os.environ.get("SMIC_CROP_DIR", "/path/to/smic/SMIC_all_cropped/HS"),
    },
    "qa_file": os.environ.get("QA_FILE", "/path/to/me_vqa_samm_casme2_smic_v2.jsonl"),
    "landmarks_predictor": os.environ.get("LANDMARKS_PREDICTOR", "./shape_predictor_68_face_landmarks.dat"),
}

# Model configuration
MODEL_CONFIG = {
    "vision_encoder": {
        "model_name": "vit_base_patch16_224",
        "out_dim": 512,
        "pretrained": True,
    },
    "text_encoder": {
        "model_name": "openai/clip-vit-base-patch32",
        "freeze": True,
    },
    "clip_alignment": {
        "init_temp": 0.07,
        "epochs": 20,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
    },
    "moe_model": {
        "embed_dim": 512,
        "num_experts": 4,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
    },
}

# Training configuration
TRAINING_CONFIG = {
    "device": "cuda",
    "seed": 42,
    "num_workers": 2,
    "pin_memory": True,
    "train_ratio": 0.9,
    "val_ratio": 0.1,
    "checkpoint_dir": "checkpoints",
}

# Data split
DATA_SPLIT = {
    "train_ratio": 0.9,
    "val_ratio": 0.1,
}

# Task mapping
TASK_MAP = {
    "au_singular": 0,
    "au_plural": 0,
    "au_yes_no": 1,
    "coarse": 2,
    "fine": 3,
    "joint": 4,
    "location": 5,
}

# Emotion classes
ALL_EMOTION = ["happiness", "surprise", "fear", "disgust", "anger", "sadness", "repression"]
ALL_COARSE = ["positive", "negative", "surprise"]

# AU mapping
AU_MAPPING = {
    1: {"name": "inner brow raiser", "landmarks": [20, 21, 22, 23]},
    2: {"name": "outer brow raiser", "landmarks": [17, 18, 19, 24, 25, 26]},
    4: {"name": "brow lowerer", "landmarks": [19, 20, 21, 22, 23, 24, 27]},
    5: {"name": "upper lid raiser", "landmarks": [37, 38, 43, 44]},
    6: {"name": "cheek raiser", "landmarks": [39, 40, 41, 45, 46, 47]},
    7: {"name": "lid tightener", "landmarks": list(range(36, 48))},
    9: {"name": "nose wrinkler", "landmarks": list(range(27, 36))},
    10: {"name": "upper lip raiser", "landmarks": [49, 50, 51, 52, 53]},
    12: {"name": "lip corner puller", "landmarks": [48, 54]},
    14: {"name": "dimpler", "landmarks": [48, 54]},
    15: {"name": "lip corner depressor", "landmarks": [48, 54, 56, 57, 58]},
    17: {"name": "chin raiser", "landmarks": [6, 7, 8, 9, 10, 55, 56, 57, 58, 59]},
    20: {"name": "lip stretcher", "landmarks": [48, 54]},
    23: {"name": "lip tightener", "landmarks": list(range(48, 68))},
    24: {"name": "lip pressor", "landmarks": list(range(48, 68))},
    25: {"name": "lip part", "landmarks": list(range(60, 68))},
    26: {"name": "jaw drop", "landmarks": list(range(48, 68))},
}
