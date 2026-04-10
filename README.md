# MEGC: Micro-Expression Grand Challenge 2026

This repository contains a hierarchical Mixture of Experts (MoE) architecture for Visual Question Answering (VQA) on micro-expressions from the MEGC datasets (CASME2, SAMM, SMIC).

## Overview

The project implements a two-stage training pipeline:

1. **Stage 1: CLIP Alignment** - Pre-train vision encoders using CLIP-based contrastive learning
2. **Stage 2: Hierarchical MoE** - Train a hierarchical mixture of experts model for VQA tasks

### Key Features

- **Vision Encoder**: Vision Transformer (ViT) for robust feature extraction
- **Text Encoder**: CLIP model for semantic understanding
- **Hierarchical MoE**: 4 specialized experts for different aspects of micro-expressions
  - **ExpertAULocal**: Action Unit detection with ROI attention
  - **ExpertEmotionHolistic**: Holistic emotion recognition
  - **ExpertSpatial**: Spatial reasoning with positional embeddings
  - **ExpertRelation**: Relational reasoning across different features
- **Multiple Loss Functions**: InfoNCE for CLIP, cross-entropy for task routing
- **Optical Flow**: Motion analysis using Farneback algorithm
- **Facial Landmarks**: 68-point detection for AU extraction

## Project Structure

```
megc/
├── config.py                 # Configuration for paths and model parameters
├── main.py                   # Main training script
├── data_loader.py            # Data loading utilities
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── GETTING_STARTED.md        # Quick start guide
├── checkpoints/              # Saved model checkpoints (created after training)
│   ├── best_visual_encoder.pth
│   ├── best_moe_megc.pth
│   ├── clip_training_curves.png
│   └── moe_training_curves.png
├── dataset/                  # Raw datasets (download separately)
│   ├── casme2/
│   │   ├── Cropped/          # Cropped face images
│   │   │   ├── sub01/
│   │   │   ├── sub02/
│   │   │   └── ...
│   │   └── CASME2-coding-20140508.xlsx
│   ├── samm/
│   │   ├── SAMM_Cropped/     # Cropped face images
│   │   │   ├── 001/
│   │   │   ├── 002/
│   │   │   └── ...
│   │   └── SAMM_Micro_FACS_Codes_v2.xlsx
│   ├── smic/
│   │   ├── SMIC_all_cropped/
│   │   │   └── HS/           # Cropped face images
│   │   │       ├── s1/
│   │   │       ├── s2/
│   │   │       └── ...
│   │   └── (metadata from QA file)
│   └── unseen/               # Unseen test videos (optional)
│       ├── ME_VQA_MEGC_2025_Test/
│       └── me_vqa_test_to_answer.jsonl
├── me_vqa_samm_casme2_smic_v2.jsonl  # QA annotations (download)
├── shape_predictor_68_face_landmarks.dat  # Facial landmarks model (download)
├── data/
│   ├── __init__.py
│   ├── preprocessing.py      # Optical flow, AU extraction, landmarks
│   └── dataset.py            # Dataset classes for training
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py     # ViT-based visual encoder
│   ├── text_encoder.py       # CLIP-based text encoder
│   ├── clip_alignment.py     # CLIP alignment model
│   └── moe_model.py          # Hierarchical MoE architecture
└── utils/
    ├── __init__.py
    ├── prompt_builder.py     # Text prompt generation
    └── inference.py          # Inference utilities
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### Setup

1. Clone the repository
```bash
cd megc
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download required models and data
```bash
# Download facial landmarks predictor
wget https://github.com/italojs/facial-landmarks-recognition/raw/refs/heads/master/shape_predictor_68_face_landmarks.dat

# Download QA dataset
wget https://megc2026.github.io/files/me_vqa_samm_casme2_smic_v2.jsonl
```

## Dataset Setup

This section explains how to download and organize the datasets for training.

### Step 1: Download and Setup Datasets

First, create the dataset directory structure:

```bash
# Create dataset directory
mkdir -p dataset/{casme2,samm,smic,unseen}

# Download QA annotations (place in root directory)
wget https://megc2026.github.io/files/me_vqa_samm_casme2_smic_v2.jsonl
```

Then download the three micro-expression datasets from their official sources:

#### CASME II Dataset
To download the dataset, please visit: [CASME II Official Website](https://melabipcas.github.io/melab/en/db/casme2.html). Download and fill in the license agreement form, submit throuth the website.

#### SAMM Dataset
To download the dataset, please visit: [SAMM Official Website](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php). Download and fill in the license agreement form, email to M.Yap@mmu.ac.uk with email subject: SAMM videos.

#### SMIC Dataset
To download the dataset, please visit: [SMIC Official Website](https://www.oulu.fi/cmvs/node/41319). Download and fill in the license agreement form (please indicate which version/subset you need), email to Xiaobai.Li@oulu.fi.

#### Unseen Test Data (Optional)
To obtain the test sets, please required to download and fill in the license agreement forms for the SAMM Challenge dataset and CAS(ME)3 in [MEGC 2026 Challenge website](https://megc2026.github.io/challenge.html), and upload them to the organisers through a query link.

### Step 2: Organize Folder Structure

After downloading, organize your datasets as follows:

```bash
# Create dataset directory
mkdir -p dataset

# CASME II
mkdir -p dataset/casme2

# SAMM
mkdir -p dataset/samm

# SMIC
mkdir -p dataset/smic

# Unseen test data (optional)
mkdir -p dataset/unseen
```

### Step 3: Verify Dataset Organization

Your dataset folder should look like:

```
dataset/
├── casme2/
│   ├── Cropped/
│   │   ├── sub01/
│   │   │   ├── EP02_01f/
│   │   │   │   ├── reg_img46.jpg
│   │   │   │   ├── reg_img47.jpg
│   │   │   │   └── ...
│   │   │   ├── EP02_02f/
│   │   │   └── ...
│   │   └── CASME2-coding-20140508.xlsx
├── samm/
│   ├── SAMM_Cropped/
│   │   ├── 006/
│   │   │   ├── 006_1_2/
│   │   │   │   ├── crop_5562.jpg
│   │   │   │   ├── crop_5563.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── SAMM_Micro_FACS_Codes_v2.xlsx
├── smic/
│   ├── SMIC_all_cropped/
│   │   ├── s1/
│   │   │   ├── micro/
│   │   │   │   ├── positive/
│   │   │   │   ├── negative/
│   │   │   │   └── surprise/
│   │   │   └── ...
│   │   └── ...
├── unseen/ (optional)
│   ├── ME_VQA_MEGC_2025_Test/
│   │   ├── CAS-1/
│   │   ├── CAS-2/
│   │   └── ...
│   ├── me_vqa_samm_v2_test_to_answer.jsonl
│   └── me_vqa_casme3_v2_test_to_answer.jsonl
└── me_vqa_samm_casme2_smic_v2.jsonl
```

## Configuration

Edit `config.py` to set your dataset paths and training parameters:

```python
DATASET_CONFIG = {
    "casme2": {
        "crop_dir": "dataset/casme2/Cropped",
        "metadata_file": "dataset/casme2/CASME2-coding-20140508.xlsx",
    },
    "samm": {
        "crop_dir": "dataset/samm/SAMM_Cropped",
        "metadata_file": "dataset/samm/SAMM_Micro_FACS_Codes_v2.xlsx",
    },
    "smic": {
        "crop_dir": "dataset/smic/SMIC_all_cropped",
    },
    "qa_file": "me_vqa_samm_casme2_smic_v2.jsonl",
    "landmarks_predictor": "./shape_predictor_68_face_landmarks.dat",
}
```

Or use absolute paths if datasets are stored elsewhere:

```python
DATASET_CONFIG = {
    "casme2": {
        "crop_dir": "/absolute/path/to/CASME2/Cropped",
        "metadata_file": "/absolute/path/to/CASME2-coding-20140508.xlsx",
    },
    # ... rest of config
}
```

### Using Environment Variables

You can also set paths via environment variables (takes precedence over config.py):

```bash
export CASME2_CROP_DIR="/path/to/casme2"
export CASME2_METADATA="/path/to/metadata.xlsx"
export SAMM_CROP_DIR="/path/to/samm"
export SAMM_METADATA="/path/to/metadata.xlsx"
export SMIC_CROP_DIR="/path/to/smic"
export QA_FILE="/path/to/qa_data.jsonl"
export LANDMARKS_PREDICTOR="./shape_predictor_68_face_landmarks.dat"
```

## Data Preparation

### Expected Data Format

Each sample should contain:
- **apex**: Reference apex frame image
- **onset**: Onset frame image
- **flow**: Pre-computed optical flow image
- **roi_paths**: List of Action Unit ROI images
- **au_list**: List of action unit names
- **emotion**: Fine-grained emotion label
- **coarse**: Coarse emotion label (positive/negative/surprise)
- **qa_list**: List of QA pairs with questions and answers

### Data Structure Example

```python
data_info = {
    "dataset": "casme2",
    "subject": "casme2_sub01",
    "filename": "EP02_01f",
    "apex": "path/to/apex.jpg",
    "flow": "path/to/flow.jpg",
    "roi_paths": ["path/to/roi_1.jpg", ..., "path/to/roi_14.jpg"],
    "au_list": ["cheek raiser", "lip corner puller"],
    "emotion": "happiness",
    "coarse": "positive",
    "qa_list": [
        {"question": "What is the coarse expression class?", "answer": "positive"},
        {"question": "What are the action units?", "answer": "cheek raiser, lip corner puller"}
    ]
}
```

## Pre-trained Models

We provide pre-trained checkpoints for both stages of the model trained on CASME2, SAMM, and SMIC datasets.

### Download Pre-trained Checkpoints

You can download the pre-trained models from Google Drive:

**[Download Pre-trained Models](https://drive.google.com/drive/folders/1pCcKMU3QvcM9APbDjV-zfgesc2sqyy2U?usp=sharing)**
 
### Setup Pre-trained Models

1. **Download the checkpoint files** from the Google Drive link above

2. **Create checkpoints directory and copy files**:
   ```bash
   mkdir -p checkpoints
   # Download and place the .pth files into checkpoints/
   ~/checkpoints/best_visual_encoder.pth
   ~/checkpoints/best_moe_megc.pth
   ```

3. **Verify file structure**:
   ```bash
   ls -lh checkpoints/
   # Expected output:
   # best_visual_encoder.pth
   # best_moe_megc.pth
   # clip_training_curves.png
   # moe_training_curves.png
   ```

## Evaluation & Metrics

After training your model, evaluation metrics are automatically computed on the validation set!

### Metrics Explanation

- **UF1 (Coarse)**: Unweighted F1-Score for Coarse Emotion Classification (positive/negative/surprise)
- **UAR (Coarse)**: Unweighted Recall for Coarse Emotion (macro-averaged)
- **UF1 (Fine-grained)**: Unweighted F1-Score for fine-grained emotion
- **UAR (Fine-grained)**: Unweighted Recall for fine-grained emotion
- **BLEU**: Bilingual Evaluation Understudy - measures text generation quality
- **ROUGE**: Recall-Oriented Understudy for Gisting - measures text matching quality

### VQA Tasks Supported

The model supports the following VQA tasks:

- **au_singular**: Single action unit detection ("What is the action unit?")
- **au_plural**: Multiple action units ("What are the action units?")
- **fine**: Fine-grained emotion ("What is the fine-grained emotion?")
- **coarse**: Coarse emotion ("What is the coarse emotion?")
- **joint**: Combined analysis ("Please analyze the micro-expression in detail")
- **location**: Spatial localization ("Is this on the left or right?")
- **au_yes_no**: AU presence ("Is inner brow raiser shown on the face?")

## Model Architecture Details

### Stage 1: CLIP Alignment

Pre-trains visual encoder using:
- Apex frame + optical flow fusion
- ROI-based attention for action units
- 4 contrastive losses (AU, fine emotion, coarse emotion, joint)

### Stage 2: Hierarchical MoE

4 specialized experts with dynamic routing:

1. **Action Unit Expert**: ROI self-attention + cross-attention with question
2. **Emotion Holistic Expert**: Transformer-based holistic feature fusion
3. **Spatial Expert**: Positional embedding-aware ROI analysis
4. **Relation Expert**: Graph-based reasoning across experts

Task Router learns to dynamically weight expert contributions based on question type.

## Output

After training, the following files are generated:

```
checkpoints/
├── best_visual_encoder.pth          # Pre-trained visual encoder
├── best_moe_megc.pth                # Final trained MoE model
├── clip_training_curves.png         # CLIP training loss plot
└── moe_training_curves.png          # MoE training loss plot
```

## Results & Metrics

The pre-trained model is evaluated on CASME2, SAMM, and SMIC datasets:

### Performance Metrics

The model is evaluated using:
- **Unweighted F1 Score (UF1)**: Macro-averaged F1 across classes
- **Unweighted Accuracy Rate (UAR)**: Macro-averaged recall across classes
- **BLEU**: Bilingual Evaluation Understudy for text generation quality
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation for text matching quality

### Detailed Evaluation Results

See **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)** for comprehensive metrics.

### Troubleshooting

#### CUDA Out of Memory

Reduce batch size in `config.py`:
```python
"batch_size": 4  # Try 2 or 1
```

#### Model Checkpoint Not Found

If you get an error saying checkpoint file is not found:

1. Make sure you downloaded the files from [Google Drive](https://drive.google.com/drive/folders/1pCcKMU3QvcM9APbDjV-zfgesc2sqyy2U?usp=sharing)
2. Place them in the `checkpoints/` directory
3. Verify file paths match:
   ```bash
   ls -lh checkpoints/
   ```

#### Missing Dependencies

Reinstall with verbose output:
```bash
pip install -r requirements.txt -v
```

#### Data Loading Errors

Make sure dataset paths are correctly configured in `config.py`:
```bash
# Verify paths exist
ls -d dataset/casme2/Cropped/sub* | head -3
ls -d dataset/samm/SAMM_Cropped/* | head -3
ls -d dataset/smic/SMIC_all_cropped/s* | head -3
```

## Support

For issues or questions, please check:
1. Dataset paths in `config.py`
2. All required files are downloaded
3. CUDA/GPU is properly configured
4. Python and dependency versions match requirements
