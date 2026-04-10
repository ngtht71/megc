# Validation and Test Set - Evaluation Results

This document tracks evaluation metrics for the MEGC VQA models across different datasets and tasks.

## Evaluation Metrics Definition

- **UF1 (Coarse)**: Unweighted F1-score for coarse emotion classification (positive/negative/surprise)
- **UAR (Coarse)**: Unweighted accuracy rate for coarse emotion classification
- **UF1 (Fine-grained)**: Unweighted F1-Score for fine-grained emotion classification
- **UAR (Fine-grained)**: Unweighted accuracy rate for fine-grained emotion classification
- **BLEU**: Bilingual Evaluation Understudy (text generation quality)
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation (text matching quality)


## Overall Results

CLIP-AL denotes the baseline model utilizing only the CLIP-Alignment module for direct prediction, whereas CQ-MoE represents the complete proposed architecture

### Validation Set

| Method | UF1 (Coarse) | UAR (Coarse) | UF1 (Fine) | UAR (Fine) | BLEU | ROUGE |
|--------|--------------|--------------|-----------|-----------|------|-------|
| CLIP_AL | 0.2046 | 0.2771 | 0.3580 | 0.3944 | 0.1056 | 0.3436 |
| **CQ-MoE** | 0.2122 | 0.2853 | 0.3661 | 0.4370 | 0.1165 | 0.5617 |


### Unseen Test Set

| Method | UF1 (Coarse) | UAR (Coarse) | UF1 (Fine) | UAR (Fine) | BLEU | ROUGE |
|--------|--------------|--------------|-----------|-----------|------|-------|
| CLIP_AL | 0.250 | 0.267 | 0.148 | 0.192 | 0.129 | 0.348 |
| **CQ-MoE** | 0.374 | 0.400 | 0.108 | 0.192 | 0.139 | 0.403 |


