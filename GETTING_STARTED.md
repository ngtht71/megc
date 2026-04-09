# Quick Start Guide

This guide will help you get started with training the MEGC VQA model.

## Step 1: Environment Setup

```bash
# Install Python 3.8+
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download facial landmarks predictor
wget https://github.com/italojs/facial-landmarks-recognition/raw/refs/heads/master/shape_predictor_68_face_landmarks.dat
```

## Step 2: Prepare Your Data

### Option A: Using Pre-processed Data

If you already have processed data with apex, flow, ROI images, and QA annotations:

1. Organize your data as shown in the [Data Format](#data-format) section
2. Update `config.py` with your dataset paths
3. Use `data_loader.py` to load your data

### Option B: Process Raw Videos

If you have raw video sequences:

1. Extract frames from video sequences
2. Detect apex frames using optical flow
3. Extract facial landmarks and AU ROIs
4. Prepare QA annotations in JSONL format

See the [Data Preparation](#data-preparation) section below.

## Step 3: Configure Paths

Edit `config.py` and set your dataset paths:

```python
DATASET_CONFIG = {
    "casme2": {
        "crop_dir": "/your/path/to/casme2/Cropped",
        "metadata_file": "/your/path/to/CASME2-coding.xlsx",
    },
    "qa_file": "/your/path/to/me_vqa_data.jsonl",
    "landmarks_predictor": "/your/path/shape_predictor_68_face_landmarks.dat",
}
```

## Step 4: Load Data in main.py

Edit `main.py` and uncomment the data loading section:

```python
def main():
    # ... (other code) ...

    # Uncomment this section:
    from data_loader import load_all_datasets

    data_infos = load_all_datasets(
        qa_file=DATASET_CONFIG["qa_file"],
        casme2_metadata=DATASET_CONFIG["casme2"]["metadata_file"],
        casme2_crop_dir=DATASET_CONFIG["casme2"]["crop_dir"],
        landmarks_predictor=DATASET_CONFIG["landmarks_predictor"]
    )

    # Continue with training...
```

## Step 5: Run Training

```bash
python main.py
```

The training will:
1. Pre-train CLIP alignment model (Stage 1) → saves to `checkpoints/best_visual_encoder.pth`
2. Train Hierarchical MoE (Stage 2) → saves to `checkpoints/best_moe_megc.pth`

## Data Format

Your data should be organized with each sample containing:

```
sample_dir/
├── apex.jpg           # Apex frame
├── flow.jpg           # Optical flow visualization
├── roi_01.jpg         # Action Unit ROI 1
├── roi_02.jpg         # Action Unit ROI 2
└── ...                # More ROIs (typically 14-17 total)
```

Each sample in `data_infos` should have this structure:

```python
{
    "dataset": "casme2",              # Dataset name
    "subject": "casme2_sub01",        # Subject ID
    "filename": "EP02_01f",           # Video/sequence name
    "apex": "path/to/apex.jpg",
    "flow": "path/to/flow.jpg",
    "roi_paths": [                    # List of ROI paths
        "path/to/roi_01.jpg",
        "path/to/roi_02.jpg",
        # ... more ROIs
    ],
    "au_list": [                      # Action units
        "cheek raiser",
        "lip corner puller"
    ],
    "emotion": "happiness",           # Fine-grained emotion
    "coarse": "positive",             # Coarse emotion (positive/negative/surprise)
    "qa_list": [                      # VQA pairs
        {
            "question": "What is the coarse expression class?",
            "answer": "positive"
        },
        {
            "question": "What are the action units?",
            "answer": "cheek raiser, lip corner puller"
        }
    ]
}
```

## Data Preparation

### For CASME2 Dataset

```python
from data_loader import load_casme2

casme2_data = load_casme2(
    df_path="/path/to/CASME2-coding-20140508.xlsx",
    crop_dir="/path/to/Cropped/Cropped",
    qa_dict_full=qa_dict_full,
    qa_dict_base=qa_dict_base,
    au_extractor=au_extractor
)
```

### QA Data Format (JSONL)

```jsonl
{"dataset": "casme2", "image_id": "sub01/EP02_01f", "question": "What is the coarse expression class?", "answer": "positive"}
{"dataset": "casme2", "image_id": "sub01/EP02_01f", "question": "What are the action units?", "answer": "cheek raiser, lip corner puller"}
```

## Customization

### Modify Training Parameters

Edit `config.py`:

```python
MODEL_CONFIG = {
    "clip_alignment": {
        "epochs": 20,        # Increase for longer training
        "batch_size": 4,     # Reduce if OOM, increase for faster training
        "learning_rate": 5e-5,
    },
    "moe_model": {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 3e-5,
    },
}
```

### Use Different Vision Models

```python
MODEL_CONFIG = {
    "vision_encoder": {
        "model_name": "vit_large_patch16_224",  # Use larger ViT
        "out_dim": 768,  # Adjust output dimension
    }
}
```

## Troubleshooting

### 1. CUDA Out of Memory

Reduce batch size:
```python
"batch_size": 2  # or 1
```

### 2. Module Import Errors

Make sure you're in the correct directory:
```bash
cd /path/to/megc
python main.py
```

### 3. Missing Data Files

Check that all paths in `config.py` are correct:
```bash
ls /your/path/to/data_files
```

### 4. QA Data Not Loading

Verify JSONL format and dataset names match expected values (casme2, samm, smic)

## Next Steps

1. **Run Inference**: Use the trained models for prediction
2. **Evaluate**: Calculate metrics (UF1, UAR) on test set
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Deploy**: Integrate model into your application

See `README.md` for more details on each step.
