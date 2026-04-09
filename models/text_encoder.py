"""
Text Encoder using CLIP
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union


class TextEncoder(nn.Module):
    """CLIP-based text feature encoder."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model ({model_name}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Extract text features."""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)
        return outputs.pooler_output
