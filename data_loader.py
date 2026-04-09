"""
Data preparation guide and utilities for loading MEGC datasets

This script provides utilities to load and preprocess CASME2, SAMM, and SMIC datasets
"""
import os
import cv2
import json
import pandas as pd
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path

from data.preprocessing import (
    get_optical_flow_image,
    AUExtractor,
    process_all_roi,
    get_name_au_label
)
from config import DATASET_CONFIG, AU_MAPPING


def norm_text(s: str) -> str:
    """Normalize text for matching."""
    s = str(s).strip().lower().replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", s)
    return s


def get_basename_noext(s: str) -> str:
    """Get basename without extension."""
    s = norm_text(s)
    return os.path.basename(s)


def load_qa_data(jsonl_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load QA data from JSONL file and build lookup dictionaries."""
    qa_dict_full = defaultdict(list)
    qa_dict_base = defaultdict(list)
    qa_dict_smic_pair = defaultdict(list)
    all_smic_keys = []

    print(f"Loading QA data from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            dataset = str(item.get("dataset", "")).strip().lower()
            if dataset not in ["casme2", "samm", "smic"]:
                continue

            image_id = item.get("image_id", "")
            image_id_norm = norm_text(image_id)
            base = get_basename_noext(image_id_norm)

            qa_item = {
                "question": item["question"],
                "answer": item["answer"]
            }

            qa_dict_full[f"{dataset}::{image_id_norm}"].append(qa_item)
            qa_dict_base[f"{dataset}::{base}"].append(qa_item)

            if dataset == "smic":
                all_smic_keys.append(image_id_norm)

    print(f"Loaded {len(qa_dict_full)} exact keys, {len(qa_dict_base)} base keys")
    return qa_dict_full, qa_dict_base, all_smic_keys


def get_qa_list(
    dataset: str,
    qa_dict_full: Dict,
    qa_dict_base: Dict,
    *candidate_ids
) -> List[Dict]:
    """Get QA list for a given dataset and candidates."""
    dataset = str(dataset).lower()
    hits = []

    # Try full-key match
    for cid in candidate_ids:
        cid_norm = norm_text(cid)
        hits.extend(qa_dict_full.get(f"{dataset}::{cid_norm}", []))

    if hits:
        return hits

    # Try basename match
    for cid in candidate_ids:
        base = get_basename_noext(cid)
        hits.extend(qa_dict_base.get(f"{dataset}::{base}", []))

    return hits


def load_casme2(
    df_path: str,
    crop_dir: str,
    qa_dict_full: Dict,
    qa_dict_base: Dict,
    au_extractor: AUExtractor,
    output_dir: str = "processed_data/casme2"
) -> List[Dict]:
    """Load CASME2 dataset."""
    print(f"\nLoading CASME2 from {df_path}...")

    df = pd.read_excel(df_path)
    df = df[pd.to_numeric(df["ApexFrame"], errors="coerce").notna()]

    data_infos = []
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        try:
            subject = int(row["Subject"])
            filename = str(row["Filename"]).strip()
            onset_frame = int(row["OnsetFrame"])
            apex_frame = int(row["ApexFrame"])
            au_raw = str(row.get("Action Units", ""))
            emotion = str(row.get("Estimated Emotion", "unknown")).lower()
            coarse = str(row.get("coarse emotion", "unknown")).lower()

            sub_folder = f"sub{subject:02d}"
            onset_path = os.path.join(crop_dir, sub_folder, filename, f"reg_img{onset_frame}.jpg")
            apex_path = os.path.join(crop_dir, sub_folder, filename, f"reg_img{apex_frame}.jpg")

            if not os.path.exists(onset_path) or not os.path.exists(apex_path):
                continue

            # Get QA data
            qa_candidates = [
                f"{sub_folder}_{filename}",
                filename,
                os.path.join(sub_folder, filename)
            ]
            qa_list = get_qa_list("casme2", qa_dict_full, qa_dict_base, *qa_candidates)

            if not qa_list:
                continue

            # Process images
            onset = cv2.imread(onset_path)
            apex = cv2.imread(apex_path)

            if onset is None or apex is None:
                continue

            flow_img = get_optical_flow_image(onset, apex)

            landmarks = au_extractor.extract_landmarks(apex)
            if landmarks is None:
                continue

            au_names, roi_imgs = process_all_roi(apex, landmarks, AU_MAPPING)
            if not roi_imgs:
                continue

            # Save processed images
            subject_id = f"casme2_{sub_folder}"
            sample_id = os.path.join(subject_id, filename)
            sample_out_dir = os.path.join(output_dir, sample_id)
            os.makedirs(sample_out_dir, exist_ok=True)

            out_apex_path = os.path.join(sample_out_dir, "apex.jpg")
            out_flow_path = os.path.join(sample_out_dir, "flow.jpg")

            cv2.imwrite(out_apex_path, apex)
            cv2.imwrite(out_flow_path, flow_img)

            out_roi_paths = []
            for i, (name, roi_img) in enumerate(roi_imgs.items()):
                roi_path = os.path.join(sample_out_dir, f"roi_{i+1:02d}.jpg")
                cv2.imwrite(roi_path, roi_img)
                out_roi_paths.append(roi_path)

            data_infos.append({
                "dataset": "casme2",
                "subject": subject_id,
                "filename": filename,
                "onset": onset_path,
                "apex": out_apex_path,
                "flow": out_flow_path,
                "roi_paths": out_roi_paths,
                "au_list": au_names,
                "emotion": emotion,
                "coarse": coarse,
                "qa_list": qa_list
            })

        except Exception as e:
            print(f"Error processing CASME2 sample {idx}: {e}")
            continue

    print(f"Loaded {len(data_infos)} CASME2 samples")
    return data_infos


def load_all_datasets(
    qa_file: str,
    casme2_metadata: str = None,
    casme2_crop_dir: str = None,
    samm_metadata: str = None,
    samm_crop_dir: str = None,
    smic_crop_dir: str = None,
    landmarks_predictor: str = None,
) -> List[Dict]:
    """Load all datasets (CASME2, SAMM, SMIC)."""

    # Use config paths if not provided
    if casme2_metadata is None:
        casme2_metadata = DATASET_CONFIG["casme2"]["metadata_file"]
    if casme2_crop_dir is None:
        casme2_crop_dir = DATASET_CONFIG["casme2"]["crop_dir"]
    if landmarks_predictor is None:
        landmarks_predictor = DATASET_CONFIG["landmarks_predictor"]

    # Initialize AU extractor
    if not os.path.exists(landmarks_predictor):
        raise FileNotFoundError(f"Landmarks predictor not found: {landmarks_predictor}")

    au_extractor = AUExtractor(landmarks_predictor)

    # Load QA data
    if not os.path.exists(qa_file):
        raise FileNotFoundError(f"QA file not found: {qa_file}")

    qa_dict_full, qa_dict_base, _ = load_qa_data(qa_file)

    # Load datasets
    all_data = []

    if os.path.exists(casme2_metadata) and os.path.exists(casme2_crop_dir):
        casme2_data = load_casme2(
            casme2_metadata,
            casme2_crop_dir,
            qa_dict_full,
            qa_dict_base,
            au_extractor
        )
        all_data.extend(casme2_data)

    print(f"\nTotal samples loaded: {len(all_data)}")
    return all_data


# Example usage
if __name__ == "__main__":
    print("MEGC Data Preparation Guide")
    print("="*60)
    print("\nUsage: Comment out the example code and use the functions")
    print("to load your datasets into data_infos.\n")

    print("Example:")
    print("""
    from data_loader import load_all_datasets

    # Load all datasets
    data_infos = load_all_datasets(
        qa_file=DATASET_CONFIG["qa_file"],
        casme2_metadata=DATASET_CONFIG["casme2"]["metadata_file"],
        casme2_crop_dir=DATASET_CONFIG["casme2"]["crop_dir"],
        landmarks_predictor=DATASET_CONFIG["landmarks_predictor"]
    )

    # Now use data_infos in main.py
    """)
