"""
Hierarchical Mixture of Experts (MoE) model for VQA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_encoder import TextEncoder
from typing import Tuple, Dict, Optional


class ExpertAULocal(nn.Module):
    """Expert for action unit detection with self and cross attention."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.roi_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, v_roi: torch.Tensor, q_text: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        roi_context, _ = self.roi_self_attn(v_roi, v_roi, v_roi)
        v_roi = self.norm1(v_roi + roi_context)

        q = q_text.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=q, key=v_roi, value=v_roi)
        out = self.norm2(q + attn_out).squeeze(1)

        return self.ffn(out)


class ExpertEmotionHolistic(nn.Module):
    """Expert for holistic emotion recognition with Transformer fusion."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
            activation='gelu',
            dropout=0.1
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, v_global: torch.Tensor, q_text: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B = v_global.size(0)

        v = v_global.unsqueeze(1)
        q = q_text.unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, v, q], dim=1)

        fused_tokens = self.fusion_transformer(tokens)
        out = fused_tokens[:, 0, :]
        return out


class ExpertSpatial(nn.Module):
    """Expert for spatial attention with ROI positional embeddings."""

    def __init__(self, embed_dim: int = 512, num_rois: int = 17, num_heads: int = 8):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_rois, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, v_roi: torch.Tensor, q_text: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        v_roi_spatial = v_roi + self.pos_embed

        q = q_text.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=q, key=v_roi_spatial, value=v_roi_spatial)

        out = self.layer_norm(q + attn_out).squeeze(1)
        return self.ffn(out)


class ExpertRelation(nn.Module):
    """Expert for relational reasoning across AU and emotion."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, out_au: torch.Tensor, out_emo: torch.Tensor, q_text: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        nodes = torch.stack([out_au, out_emo, q_text], dim=1)
        reasoned_nodes = self.transformer_encoder(nodes)
        out = reasoned_nodes.mean(dim=1)
        return out


class HierarchicalMoE(nn.Module):
    """Hierarchical Mixture of Experts for VQA."""

    def __init__(self, device: str, pretrain_visual_path: str, embed_dim: int = 512, init_temp: float = 0.07):
        super().__init__()
        self.device = device

        # Load pretrained visual encoder
        from models.vision_encoder import VisualEncoder
        self.visual_encoder = torch.load(pretrain_visual_path, map_location=device, weights_only=False)
        self.text_encoder = TextEncoder(device=device)

        print(f"Loading pre-trained Visual Encoder from {pretrain_visual_path} and freeze...")
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

        self.router = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.task_to_expert_proj = nn.Linear(6, 4)

        self.e_au_local = ExpertAULocal(embed_dim)
        self.e_emo_holistic = ExpertEmotionHolistic(embed_dim)
        self.e_spatial = ExpertSpatial(embed_dim)
        self.e_relation = ExpertRelation(embed_dim)

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        apex_imgs: torch.Tensor,
        flow_imgs: torch.Tensor,
        roi_imgs: torch.Tensor,
        questions_list: list,
        target_answers: Optional[list] = None
    ) -> Tuple:
        """Forward pass."""
        B = apex_imgs.size(0)

        # Extract features (no gradients for frozen encoder)
        with torch.no_grad():
            v_apex = self.visual_encoder(apex_imgs)
            v_flow = self.visual_encoder(flow_imgs)

            num_roi = roi_imgs.shape[1]
            v_rois_flat = self.visual_encoder(roi_imgs.view(-1, 3, 224, 224))
            v_roi = v_rois_flat.view(B, num_roi, -1)

            v_global = F.normalize(v_apex + v_flow, p=2, dim=-1)

            q_text = self.text_encoder(questions_list)

        # Router
        task_logits = self.router(q_text)
        task_probs = F.softmax(task_logits, dim=-1)

        expert_logits = self.task_to_expert_proj(task_probs)
        expert_weights = F.softmax(expert_logits, dim=-1)

        w_au, w_emo, w_spa, w_rel = torch.split(expert_weights, 1, dim=-1)

        # Expert processing
        out_au = self.e_au_local(v_roi, q_text)
        out_emo = self.e_emo_holistic(v_global, q_text)
        out_spa = self.e_spatial(v_roi, q_text)

        out_rel = self.e_relation(out_au, out_emo, q_text)

        # Expert fusion
        v_final = (w_au * out_au) + (w_emo * out_emo) + (w_spa * out_spa) + (w_rel * out_rel)
        v_final = self.final_norm(v_final)

        v_final = F.normalize(v_final, p=2, dim=-1)

        if target_answers is not None:
            t_answers = self.text_encoder(target_answers)
            t_answers = F.normalize(t_answers, p=2, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_vqa = logit_scale * v_final @ t_answers.t()
            return logits_vqa, task_logits, expert_weights

        return v_final, task_logits, expert_weights

    def predict(
        self,
        apex_imgs: torch.Tensor,
        flow_imgs: torch.Tensor,
        roi_imgs: torch.Tensor,
        question: str,
        candidate_texts: list
    ) -> torch.Tensor:
        """Predict for inference."""
        self.eval()
        with torch.no_grad():
            v_fused, _, _ = self.forward(apex_imgs, flow_imgs, roi_imgs, [question] * apex_imgs.size(0))

            t_candidates = self.text_encoder(candidate_texts)
            t_candidates = F.normalize(t_candidates, p=2, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * v_fused @ t_candidates.t()

        return logits
