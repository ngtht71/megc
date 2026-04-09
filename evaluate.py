"""
Evaluation script for MEGC VQA model
Runs inference on validation/test set and computes metrics
"""
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

from config import DATASET_CONFIG, TASK_MAP, ALL_EMOTION, ALL_COARSE, AU_MAPPING
from data.dataset import MEVQA_Dataset, vqa_collate_fn
from data_loader import load_all_datasets
from models.moe_model import HierarchicalMoE
from utils.inference import parse_vqa_question, generate_answer

# Metrics libraries
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: NLTK or rouge_score not installed. Install with:")
    print("  pip install nltk rouge-score")
    METRICS_AVAILABLE = False


def compute_classification_metrics(predictions, targets, average='weighted'):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average=average, zero_division=0),
        'recall': recall_score(targets, predictions, average=average, zero_division=0),
        'f1': f1_score(targets, predictions, average=average, zero_division=0),
        'uf1': f1_score(targets, predictions, average='macro', zero_division=0),  # Unweighted F1
        'uar': recall_score(targets, predictions, average='macro', zero_division=0),  # Unweighted Recall
    }
    return metrics


def compute_bleu_score(predictions, references):
    """Compute BLEU score."""
    if not METRICS_AVAILABLE or len(predictions) == 0:
        return 0.0

    # Convert to list of tokens
    ref_tokens = [[ref.split()] for ref in references]
    pred_tokens = [pred.split() for pred in predictions]

    try:
        bleu_score = corpus_bleu(ref_tokens, pred_tokens)
        return bleu_score
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return 0.0


def compute_rouge_score(predictions, references):
    """Compute ROUGE score."""
    if not METRICS_AVAILABLE or len(predictions) == 0:
        return 0.0

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score['rouge1'].fmeasure)
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        return 0.0


def extract_emotion_from_answer(answer, emotion_type='coarse'):
    """Extract emotion label from answer text."""
    answer = answer.lower().strip()

    if emotion_type == 'coarse':
        for emotion in ALL_COARSE:
            if emotion.lower() in answer:
                return emotion
    else:  # fine-grained
        for emotion in ALL_EMOTION:
            if emotion.lower() in answer:
                return emotion

    return None


def run_evaluation(model_path, data_infos, split='validation', batch_size=8, device='cuda'):
    """Run evaluation on validation/test set."""

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"Evaluating on {split} set")
    print(f"{'='*60}\n")

    # Split data
    from collections import defaultdict
    subject_dict = defaultdict(list)
    for item in data_infos:
        subject_dict[item["subject"]].append(item)

    subjects = list(subject_dict.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)

    n = len(subjects)
    split_idx = int(n * 0.9)  # 90/10 split

    if split == 'validation':
        split_subjects = subjects[split_idx:]
    else:  # test
        split_subjects = subjects[:split_idx]

    split_data = []
    for s in split_subjects:
        split_data.extend(subject_dict[s])

    print(f"Using {len(split_data)} samples from {len(split_subjects)} subjects for {split}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # Create dataset
    dataset = MEVQA_Dataset(split_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=0  # Disable multiprocessing for evaluation
    )

    # Storage for metrics
    results = {
        'coarse_emotion': {
            'predictions': [],
            'targets': [],
            'answers': [],
            'references': []
        },
        'fine_emotion': {
            'predictions': [],
            'targets': [],
            'answers': [],
            'references': []
        },
        'au_detection': {
            'predictions': [],
            'targets': [],
            'answers': [],
            'references': []
        }
    }

    # Get AU names
    au_names = [info["name"] for info in AU_MAPPING.values()]

    candidate_dict = {
        "au": au_names,
        "fine": ALL_EMOTION,
        "coarse": ALL_COARSE,
        "location": ["left", "right"],
        "au_yes_no": ["yes", "no"]
    }

    # Run inference
    print(f"\nRunning inference on {len(loader)} batches...\n")

    with torch.no_grad():
        for batch_idx, (apex, flow, roi, questions, answers, tasks) in enumerate(tqdm(loader)):
            apex = apex.to(device)
            flow = flow.to(device)
            roi = roi.to(device)

            for i in range(len(questions)):
                question = questions[i]
                answer = answers[i]
                task = tasks[i]

                try:
                    # Generate prediction
                    pred_answer = generate_answer(
                        model,
                        apex[i:i+1],
                        flow[i:i+1],
                        roi[i:i+1],
                        question,
                        candidate_dict,
                        device
                    )

                    # Classify task
                    task_type, _ = parse_vqa_question(question)

                    # Extract emotion from answers
                    if task_type == 'coarse':
                        pred_emotion = extract_emotion_from_answer(pred_answer, 'coarse')
                        target_emotion = extract_emotion_from_answer(answer, 'coarse')

                        if pred_emotion and target_emotion:
                            results['coarse_emotion']['predictions'].append(pred_emotion)
                            results['coarse_emotion']['targets'].append(target_emotion)
                            results['coarse_emotion']['answers'].append(pred_answer)
                            results['coarse_emotion']['references'].append(answer)

                    elif task_type == 'fine':
                        pred_emotion = extract_emotion_from_answer(pred_answer, 'fine')
                        target_emotion = extract_emotion_from_answer(answer, 'fine')

                        if pred_emotion and target_emotion:
                            results['fine_emotion']['predictions'].append(pred_emotion)
                            results['fine_emotion']['targets'].append(target_emotion)
                            results['fine_emotion']['answers'].append(pred_answer)
                            results['fine_emotion']['references'].append(answer)

                    elif task_type in ['au_singular', 'au_plural']:
                        results['au_detection']['answers'].append(pred_answer)
                        results['au_detection']['references'].append(answer)

                except Exception as e:
                    print(f"Error processing sample {batch_idx}-{i}: {e}")
                    continue

    # Compute metrics
    print("\n" + "="*60)
    print("Computing metrics")
    print("="*60 + "\n")

    metrics_summary = {}

    # Coarse Emotion Metrics
    if results['coarse_emotion']['predictions']:
        coarse_metrics = compute_classification_metrics(
            results['coarse_emotion']['predictions'],
            results['coarse_emotion']['targets'],
            average='macro'
        )
        metrics_summary['coarse_emotion'] = coarse_metrics
        print(f"Coarse emotion classification:")
        print(f"  UF1:  {coarse_metrics['uf1']:.4f}")
        print(f"  UAR:  {coarse_metrics['uar']:.4f}")
        print(f"  F1:   {coarse_metrics['f1']:.4f}")
        print(f"  Acc:  {coarse_metrics['accuracy']:.4f}\n")

    # Fine-grained Emotion Metrics
    if results['fine_emotion']['predictions']:
        fine_metrics = compute_classification_metrics(
            results['fine_emotion']['predictions'],
            results['fine_emotion']['targets'],
            average='macro'
        )
        metrics_summary['fine_emotion'] = fine_metrics
        print(f"Fine-grained emotion classification:")
        print(f"  UF1:  {fine_metrics['uf1']:.4f}")
        print(f"  UAR:  {fine_metrics['uar']:.4f}")
        print(f"  F1:   {fine_metrics['f1']:.4f}")
        print(f"  Acc:  {fine_metrics['accuracy']:.4f}\n")

    # Text Generation Metrics
    if results['au_detection']['answers']:
        bleu = compute_bleu_score(results['au_detection']['answers'], results['au_detection']['references'])
        rouge = compute_rouge_score(results['au_detection']['answers'], results['au_detection']['references'])

        metrics_summary['text_generation'] = {
            'bleu': bleu,
            'rouge': rouge
        }
        print(f"Text Generation Quality:")
        print(f"  BLEU:  {bleu:.4f}")
        print(f"  ROUGE: {rouge:.4f}\n")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate MEGC VQA model')
    parser.add_argument('--model', type=str, default='checkpoints/best_moe_megc.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation', 
                        help='Which split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        default='cuda', help='Device to use')
    parser.add_argument('--no-load-data', action='store_true',
                        help='Skip loading data (for testing)')

    args = parser.parse_args()

    # Load data
    if not args.no_load_data:
        print("Loading datasets...")
        try:
            data_infos = load_all_datasets(
                qa_file=DATASET_CONFIG["qa_file"],
                casme2_metadata=DATASET_CONFIG["casme2"]["metadata_file"],
                casme2_crop_dir=DATASET_CONFIG["casme2"]["crop_dir"],
                samm_metadata=DATASET_CONFIG["samm"]["metadata_file"],
                samm_crop_dir=DATASET_CONFIG["samm"]["crop_dir"],
                smic_crop_dir=DATASET_CONFIG["smic"]["crop_dir"],
                landmarks_predictor=DATASET_CONFIG["landmarks_predictor"]
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Make sure datasets are properly configured in config.py")
            return
    else:
        data_infos = []

    # Run evaluation
    if data_infos or args.no_load_data:
        run_evaluation(
            args.model,
            data_infos,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        print("No data loaded. Please configure datasets in config.py")


if __name__ == "__main__":
    main()
