# Project Structure Summary

## Successfully Created Files

### Core Configuration
- **config.py** - Centralized configuration for all paths, model parameters, and training settings

### Data Processing
- **data/preprocessing.py** - Utilities for optical flow, facial landmark extraction, and AU ROI processing
- **data/dataset.py** - PyTorch Dataset classes (MEDataset for CLIP, MEVQA_Dataset for VQA)
- **data_loader.py** - Data loading utilities for CASME2, SAMM, SMIC datasets

### Models
- **models/vision_encoder.py** - Vision Transformer (ViT) based visual feature encoder
- **models/text_encoder.py** - CLIP-based text feature encoder
- **models/clip_alignment.py** - CLIP alignment pre-training model
- **models/moe_model.py** - Hierarchical Mixture of Experts (MoE) for VQA

### Utilities
- **utils/prompt_builder.py** - Text prompt generation for micro-expression descriptions
- **utils/inference.py** - Inference utilities and VQA question parsing

### Training & Documentation
- **main.py** - Complete training pipeline (CLIP → MoE stages)
- **evaluate.py** - Evaluation script for computing metrics on validation/test sets
- **requirements.txt** - Python dependencies
- **README.md** - Comprehensive project documentation (English)
- **GETTING_STARTED.md** - Quick start guide for new users
- **EVALUATION_RESULTS.md** - Validation and test metrics tracking for all models and datasets

### Package Initialization
- **data/__init__.py**
- **models/__init__.py**
- **utils/__init__.py**

## Project Configuration

All configuration consolidated in **config.py**:
- Dataset paths (set via environment variables or config file)
- Model hyperparameters (vision encoder, text encoder, clip alignment, moe)
- Training parameters (device, batch size, learning rate, epochs)
- Emotion and AU mappings
- Task routing configuration