"""
Dataset classes for MEGC training
"""
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
import re


class MEDataset(Dataset):
    """Micro-Expression Dataset for CLIP alignment training."""

    def __init__(self, data_infos: List[Dict], prompt_builder):
        self.data_infos = data_infos
        self.prompt_builder = prompt_builder

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.data_infos)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Tuple:
        info = self.data_infos[idx]

        apex_img = self._load_image(info['apex'])
        flow_img = self._load_image(info['flow'])
        roi_imgs = [self._load_image(p) for p in info['roi_paths']]

        apex_tensor = self.transform(apex_img)
        flow_tensor = self.transform(flow_img)
        roi_tensor = torch.stack([self.transform(roi) for roi in roi_imgs])

        text_prompts = self.prompt_builder.build_sample_prompts(
            info['au_list'],
            info['emotion'],
            info['coarse']
        )

        return apex_tensor, flow_tensor, roi_tensor, text_prompts


class MEVQA_Dataset(Dataset):
    """VQA Dataset for Hierarchical MoE training."""

    def __init__(self, data_infos: List[Dict]):
        self.flat_data = []
        for info in data_infos:
            for qa in info.get('qa_list', []):
                self.flat_data.append({
                    "video_info": info,
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.flat_data)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.flat_data[idx]
        info = item["video_info"]

        apex_img = self._load_image(info['apex'])
        flow_img = self._load_image(info['flow'])
        roi_imgs = [self._load_image(p) for p in info['roi_paths']]

        apex_tensor = self.transform(apex_img)
        flow_tensor = self.transform(flow_img)
        roi_tensor = torch.stack([self.transform(roi) for roi in roi_imgs])

        question = item["question"]
        answer = item["answer"]

        task_str, _ = self._parse_vqa_question(question)

        return apex_tensor, flow_tensor, roi_tensor, question, answer, task_str

    @staticmethod
    def _parse_vqa_question(question: str) -> Tuple[str, any]:
        """Parse VQA question to determine task type."""
        q = question.lower().strip()

        if "left or right" in q:
            return "location", None

        yes_no_match = re.search(r"is the action unit (.*?) shown on the face", q)
        if yes_no_match:
            target_au = yes_no_match.group(1).strip()
            return "au_yes_no", target_au

        if "analyse" in q or "detailed" in q or "analysis" in q:
            return "joint", None

        if "coarse" in q:
            return "coarse", None

        if "fine-grained" in q or "fine" in q:
            return "fine", None

        if "what are the action units" in q or "action units present" in q:
            return "au_plural", None

        if "action unit" in q:
            return "au_singular", None

        return "unknown", None


def collate_fn(batch: List[Tuple]) -> Tuple:
    """Collate function for MEDataset."""
    apex_list, flow_list, roi_list, text_list = zip(*batch)

    apex = torch.stack(apex_list)
    flow = torch.stack(flow_list)

    # Pad ROIs to same size
    max_rois = max(r.size(0) for r in roi_list)
    padded_rois = []
    for r in roi_list:
        pad_size = max_rois - r.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, 3, 224, 224)
            r = torch.cat([r, pad], dim=0)
        padded_rois.append(r)

    roi = torch.stack(padded_rois)

    text_dict = {
        "au_prompts": [t["au_prompt"] for t in text_list],
        "fine_prompts": [t["fine_prompt"] for t in text_list],
        "coarse_prompts": [t["coarse_prompt"] for t in text_list],
        "joint_prompts": [t["joint_prompt"] for t in text_list]
    }

    return apex, flow, roi, text_dict


def vqa_collate_fn(batch: List[Tuple]) -> Tuple:
    """Collate function for MEVQA_Dataset."""
    apex_list, flow_list, roi_list, q_list, a_list, task_list = zip(*batch)

    apex = torch.stack(apex_list)
    flow = torch.stack(flow_list)

    # Pad ROIs
    max_rois = max(r.size(0) for r in roi_list)
    padded_rois = []
    for r in roi_list:
        pad_size = max_rois - r.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, 3, 224, 224)
            r = torch.cat([r, pad], dim=0)
        padded_rois.append(r)
    roi = torch.stack(padded_rois)

    questions = list(q_list)
    answers = list(a_list)
    tasks = list(task_list)

    return apex, flow, roi, questions, answers, tasks
