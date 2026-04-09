# MEGC: Micro-Expression and Emotion Recognition with VQA

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
├── requirements.txt          # Python dependencies
├── README.md                 # This file
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

# Download QA dataset (example)
wget https://megc2026.github.io/files/me_vqa_samm_casme2_smic_v2.jsonl
```

## Configuration

Edit `config.py` to set your dataset paths and training parameters:

```python
DATASET_CONFIG = {
    "casme2": {
        "crop_dir": "/path/to/casme2/Cropped/Cropped",
        "metadata_file": "/path/to/CASME2-coding-20140508.xlsx",
    },
    "samm": {
        "crop_dir": "/path/to/samm/SAMM_Cropped",
        "metadata_file": "/path/to/SAMM_Micro_FACS_Codes_v2.xlsx",
    },
    "smic": {
        "crop_dir": "/path/to/smic/SMIC_all_cropped/HS",
    },
    "qa_file": "/path/to/me_vqa_samm_casme2_smic_v2.jsonl",
    "landmarks_predictor": "./shape_predictor_68_face_landmarks.dat",
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
- **roi_paths**: List of Action Unit ROI images (typically 14-17 images)
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

## Training

### Run Training Pipeline

```bash
python main.py
```

The training pipeline will:

1. **Stage 1**: Train CLIP alignment model (20 epochs by default)
   - Pre-trains visual encoder using contrastive learning
   - Saves best checkpoint to `checkpoints/best_visual_encoder.pth`

2. **Stage 2**: Train Hierarchical MoE model (10 epochs by default)
   - Uses pre-trained visual encoder
   - Trains hierarchical mixture of experts
   - Saves best model to `checkpoints/best_moe_megc.pth`

### Customize Training

Edit training parameters in `config.py`:

```python
MODEL_CONFIG = {
    "clip_alignment": {
        "epochs": 20,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
    },
    "moe_model": {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
    },
}
```

## Inference

### Load and Use Trained Model

```python
import torch
from models.moe_model import HierarchicalMoE

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("checkpoints/best_moe_megc.pth", map_location=device, weights_only=False)
model.eval()

# Use model for predictions
with torch.no_grad():
    logits = model.predict(apex, flow, roi, question, candidate_texts)
    predictions = torch.argmax(logits, dim=1)
```

### VQA Tasks Supported

The model supports the following VQA tasks:

- **au_singular**: Single action unit detection ("What is the action unit?")
- **au_plural**: Multiple action units ("What are the action units?")
- **fine**: Fine-grained emotion ("What is the fine-grained emotion?")
- **coarse**: Coarse emotion ("What is the coarse emotion?")
- **joint**: Combined analysis ("Please analyze the micro-expression in detail")
- **location**: Spatial localization ("Is this on the left or right?")
- **au_yes_no**: AU presence ("Is AU12 shown on the face?")

## Model Architecture Details

### Stage 1: CLIP Alignment

Pre-trains visual encoder using:
- Apex frame + optical flow fusion
- ROI-based attention for action units
- 4 contrastive losses (AU, fine emotion, coarse emotion, joint)

### Stage 2: Hierarchical MoE

4 specialized experts with dynamic routing:

1. **Action Unit Expert**: ROI self-attention + cross-attention with question
2. **Emotion Expert**: Transformer-based holistic feature fusion
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

The model is evaluated on:
- **Unweighted F1 Score (UF1)**: Macro-averaged F1 across classes
- **Unweighted Accuracy Rate (UAR)**: Macro-averaged recall across classes

## Citation

If you use this work, please cite:

```bibtex
@article{megc2026,
  title={Micro-Expression and Emotion Recognition with Hierarchical Mixture of Experts},
  year={2026}
}
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.py`:
```python
"batch_size": 4  # Try 2 or 1
```

### Model Not Converging

- Increase learning rate gradually
- Check dataset balance
- Ensure all data paths are correct

### Missing Dependencies

Reinstall with verbose output:
```bash
pip install -r requirements.txt -v
```

## References

- [CLIP: Learning Transferable Models for Computational Linguistics](https://arxiv.org/abs/2103.00020)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Mixture of Experts](https://arxiv.org/abs/2002.30057)

## License

This project is provided for research and educational purposes.

## Support

For issues or questions, please check:
1. Dataset paths in `config.py`
2. All required files are downloaded
3. CUDA/GPU is properly configured
4. Python and dependency versions match requirements

## Contributing

Contributions are welcome! Please follow the existing code structure and style.
