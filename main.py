"""
Main training script for MEGC VQA model
"""
import os
import torch
import gc
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, TASK_MAP,
    ALL_EMOTION, ALL_COARSE, AU_MAPPING
)
from data.preprocessing import (
    get_optical_flow_image, AUExtractor, process_all_roi, get_name_au_label
)
from data.dataset import MEDataset, MEVQA_Dataset, collate_fn, vqa_collate_fn
from models.clip_alignment import CLIPAlignment
from models.moe_model import HierarchicalMoE
from utils.prompt_builder import PromptBuilder
from utils.inference import compute_multi_task_loss


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_clip_model(
    model: CLIPAlignment,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str = "cuda"
):
    """Train CLIP alignment model."""
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=MODEL_CONFIG["clip_alignment"]["learning_rate"],
        weight_decay=MODEL_CONFIG["clip_alignment"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    save_dir = TRAINING_CONFIG["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for apex, flow, roi, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
            apex, flow, roi = apex.to(device), flow.to(device), roi.to(device)

            optimizer.zero_grad()
            loss = model(apex, flow, roi, texts)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for apex, flow, roi, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"):
                apex, flow, roi = apex.to(device), flow.to(device), roi.to(device)
                loss = model(apex, flow, roi, texts)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.visual_encoder, os.path.join(save_dir, "best_visual_encoder.pth"))
            print(f"*** Saved best model: (Val Loss: {best_val_loss:.4f}) ***\n")
        else:
            print()

    # Plot training curves
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    f.suptitle('CLIP Training Results')
    ax1.plot(range(len(train_loss_history)), train_loss_history, color='Blue', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(len(val_loss_history)), val_loss_history, color='Red', label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clip_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("CLIP training completed")


def train_moe_model(
    model: HierarchicalMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str = "cuda"
):
    """Train Hierarchical MoE model."""
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=MODEL_CONFIG["moe_model"]["learning_rate"],
        weight_decay=MODEL_CONFIG["moe_model"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    save_dir = TRAINING_CONFIG["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for apex, flow, roi, q_list, a_list, task_strs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
            apex, flow, roi = apex.to(device), flow.to(device), roi.to(device)
            true_task_ids = torch.tensor([TASK_MAP.get(t, 0) for t in task_strs]).to(device)

            optimizer.zero_grad()

            logits_vqa, task_logits, expert_weights = model(apex, flow, roi, q_list, target_answers=a_list)

            loss = compute_multi_task_loss(logits_vqa, task_logits, true_task_ids, device)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for apex, flow, roi, q_list, a_list, task_strs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"):
                apex, flow, roi = apex.to(device), flow.to(device), roi.to(device)
                true_task_ids = torch.tensor([TASK_MAP.get(t, 0) for t in task_strs]).to(device)

                logits_vqa, task_logits, _ = model(apex, flow, roi, q_list, target_answers=a_list)
                loss = compute_multi_task_loss(logits_vqa, task_logits, true_task_ids, device)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.7f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model, os.path.join(save_dir, "best_moe_megc.pth"))
            print(f"*** Saved best MoE model: (Loss: {best_loss:.4f}) ***\n")
        else:
            print()

    # Plot training curves
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    f.suptitle('MoE Training Results')
    ax1.plot(range(len(train_loss_history)), train_loss_history, color='Blue', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(len(val_loss_history)), val_loss_history, color='Red', label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'moe_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("MoE training completed")


def main():
    """Main training pipeline."""
    # Setup
    set_seed(TRAINING_CONFIG["seed"])
    device = torch.device(TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dataset paths loaded from config")

    # Initialize prompt builder
    prompt_builder = PromptBuilder(ALL_EMOTION, ALL_COARSE)

    # Load data using data_loader
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60 + "\n")

    # Uncomment the following to load data using data_loader
    # from data_loader import load_all_datasets
    # data_infos = load_all_datasets(
    #     qa_file=DATASET_CONFIG["qa_file"],
    # )

    # For now, create placeholder (user needs to implement data loading)
    data_infos = []  # User should load actual data here

    if len(data_infos) == 0:
        print("\n" + "!"*60)
        print("ERROR: No training data loaded!")
        print("!"*60)
        print("\nTo load data, uncomment the data_loader section in main():")
        print("  from data_loader import load_all_datasets")
        print("  data_infos = load_all_datasets(...)")
        print("\nSee data_loader.py and README.md for more details.")
        print("="*60 + "\n")
        return

    # Split data
    from collections import defaultdict

    subject_dict = defaultdict(list)
    for item in data_infos:
        subject_dict[item["subject"]].append(item)

    subjects = list(subject_dict.keys())
    random.shuffle(subjects)

    n = len(subjects)
    train_ratio = 0.9
    n_train = int(n * train_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:]

    def collect(sub_list):
        data = []
        for s in sub_list:
            data.extend(subject_dict[s])
        return data

    train_data = collect(train_subjects)
    val_data = collect(val_subjects)

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # CLIP Training
    print("\n" + "="*60)
    print("STAGE 1: CLIP Alignment Training")
    print("="*60)

    train_dataset = MEDataset(train_data, prompt_builder)
    val_dataset = MEDataset(val_data, prompt_builder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG["clip_alignment"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG["clip_alignment"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )

    clip_model = CLIPAlignment(device=device, init_temp=MODEL_CONFIG["clip_alignment"]["init_temp"])
    train_clip_model(
        clip_model,
        train_loader,
        val_loader,
        MODEL_CONFIG["clip_alignment"]["epochs"],
        device
    )

    # Clear memory
    del clip_model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # MoE Training
    print("\n" + "="*60)
    print("STAGE 2: Hierarchical MoE Training")
    print("="*60)

    train_dataset = MEVQA_Dataset(train_data)
    val_dataset = MEVQA_Dataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG["moe_model"]["batch_size"],
        shuffle=True,
        collate_fn=vqa_collate_fn,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG["moe_model"]["batch_size"],
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )

    visual_encoder_path = os.path.join(TRAINING_CONFIG["checkpoint_dir"], "best_visual_encoder.pth")
    if not os.path.exists(visual_encoder_path):
        raise FileNotFoundError(f"Visual encoder checkpoint not found: {visual_encoder_path}")

    moe_model = HierarchicalMoE(
        device=device,
        pretrain_visual_path=visual_encoder_path,
        embed_dim=MODEL_CONFIG["moe_model"]["embed_dim"]
    )

    train_moe_model(
        moe_model,
        train_loader,
        val_loader,
        MODEL_CONFIG["moe_model"]["epochs"],
        device
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
