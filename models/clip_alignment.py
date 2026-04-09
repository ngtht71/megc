"""
CLIP Alignment model for pre-training visual encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vision_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from typing import Dict


class CLIPAlignment(nn.Module):
    """CLIP-based alignment model for pre-training."""

    def __init__(self, device: str, init_temp: float = 0.07):
        super().__init__()
        self.device = device
        self.visual_encoder = VisualEncoder()
        self.text_encoder = TextEncoder(device=device)

        # Temperature for InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

    def encode_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """Encode visual features."""
        visual_features = self.visual_encoder(visual)
        return F.normalize(visual_features, p=2, dim=-1)

    def encode_text(self, text_list: list) -> torch.Tensor:
        """Encode text features."""
        text_features = self.text_encoder(text_list)
        return F.normalize(text_features, p=2, dim=-1)

    def compute_infonce(self, v_embeds: torch.Tensor, t_embeds: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss."""
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * v_embeds @ t_embeds.t()
        labels = torch.arange(v_embeds.size(0), device=self.device)
        loss_v = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_v + loss_t) / 2

    def forward(
        self,
        apex_imgs: torch.Tensor,
        flow_imgs: torch.Tensor,
        roi_imgs: torch.Tensor,
        text_prompts_dict: Dict[str, list]
    ) -> torch.Tensor:
        """Forward pass for training."""
        B = apex_imgs.size(0)

        # Extract visual features
        v_apex = self.visual_encoder(apex_imgs)
        v_flow = self.visual_encoder(flow_imgs)

        # Process ROIs
        num_roi = roi_imgs.shape[1]
        v_rois_flat = self.visual_encoder(roi_imgs.view(-1, 3, 224, 224))
        v_roi = v_rois_flat.view(B, num_roi, -1).mean(dim=1)
        v_roi = F.normalize(v_roi, p=2, dim=-1)

        # Fusion
        v_global = F.normalize(v_apex + v_flow, p=2, dim=-1)
        v_joint = F.normalize(v_apex + v_flow + v_roi, p=2, dim=-1)

        # Extract text features
        t_au = self.encode_text(text_prompts_dict['au_prompts'])
        t_fine = self.encode_text(text_prompts_dict['fine_prompts'])
        t_coarse = self.encode_text(text_prompts_dict['coarse_prompts'])
        t_joint = self.encode_text(text_prompts_dict['joint_prompts'])

        # Compute losses
        loss_au = self.compute_infonce(v_roi, t_au)
        loss_fine = self.compute_infonce(v_global, t_fine)
        loss_coarse = self.compute_infonce(v_global, t_coarse)
        loss_joint = self.compute_infonce(v_joint, t_joint)

        total_loss = (loss_au + loss_fine + loss_coarse + loss_joint) / 4.0

        return total_loss

    def predict_logits(
        self,
        apex_imgs: torch.Tensor,
        flow_imgs: torch.Tensor,
        roi_imgs: torch.Tensor,
        candidate_dict: Dict[str, list]
    ) -> tuple:
        """Predict logits for inference."""
        self.eval()
        B = apex_imgs.size(0)

        with torch.no_grad():
            v_apex = self.visual_encoder(apex_imgs)
            v_flow = self.visual_encoder(flow_imgs)

            num_roi = roi_imgs.shape[1]
            v_rois_flat = self.visual_encoder(roi_imgs.view(-1, 3, 224, 224))
            v_roi = v_rois_flat.view(B, num_roi, -1).mean(dim=1)
            v_roi = F.normalize(v_roi, p=2, dim=-1)

            v_global = F.normalize(v_apex + v_flow, p=2, dim=-1)

            t_au = self.encode_text(candidate_dict['au'])
            t_fine = self.encode_text(candidate_dict['fine'])
            t_coarse = self.encode_text(candidate_dict['coarse'])

            logit_scale = self.logit_scale.exp()

            logits_au = logit_scale * v_roi @ t_au.t()
            logits_fine = logit_scale * v_global @ t_fine.t()
            logits_coarse = logit_scale * v_global @ t_coarse.t()

        return logits_au, logits_fine, logits_coarse
