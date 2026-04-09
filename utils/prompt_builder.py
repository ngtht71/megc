"""
Prompt builder for generating text descriptions
"""
import random
from typing import List, Dict


class PromptBuilder:
    """Build text prompts for micro-expressions."""

    def __init__(self, all_emotions: List[str], all_coarse: List[str]):
        self.fine_emotions = all_emotions
        self.coarse_emotions = all_coarse

        self.au_templates = [
            "The {} is activated.",
            "The micro-expression shows the activation of the {}.",
            "A subtle facial movement activates the {}.",
            "The {} action unit is present on the face.",
            "A micro-expression involves the {}."
        ]

        self.fine_emo_templates = [
            "The micro-expression suggests {}.",
            "This is a micro-expression of {}.",
            "The expression indicates {}.",
            "The face briefly shows {}.",
            "A subtle expression of {} appears.",
            "This facial movement indicates {}."
        ]

        self.coarse_emo_templates = [
            "This micro-expression belongs to the {} emotion class.",
            "The emotion expressed is {}.",
            "The face shows a {} emotional state.",
            "This micro-expression is {}."
        ]

        self.joint_templates = [
            "The action units present is {}, which indicates a {} micro-expression, belonging to the {} emotion category.",
            "The facial action unit {} is a key indicator of {} micro-expression, which is a {} state.",
            "The presence of {} suggests a {} expression that belongs to the {} category."
        ]

    def format_au_list(self, au_list: List[str]) -> str:
        """Format action unit list as readable string."""
        if not au_list:
            return "no prominent action units"
        elif len(au_list) == 1:
            return au_list[0]
        elif len(au_list) == 2:
            return f"{au_list[0]} and {au_list[1]}"
        else:
            return ", ".join(au_list[:-1]) + f", and {au_list[-1]}"

    def get_au_prompt(self, au_names: List[str]) -> str:
        """Generate AU prompt."""
        if isinstance(au_names, str):
            au_names = [au_names]

        au_string = self.format_au_list(au_names)
        template = random.choice(self.au_templates)
        return template.format(au_string)

    def get_fine_emotion_prompt(self, fine_emotion: str) -> str:
        """Generate fine-grained emotion prompt."""
        template = random.choice(self.fine_emo_templates)
        return template.format(fine_emotion)

    def get_coarse_emotion_prompt(self, coarse_emotion: str) -> str:
        """Generate coarse emotion prompt."""
        template = random.choice(self.coarse_emo_templates)
        return template.format(coarse_emotion)

    def get_joint_prompt(self, au_names: List[str], fine_emotion: str, coarse_emotion: str) -> str:
        """Generate joint prompt combining AU and emotions."""
        if isinstance(au_names, str):
            au_names = [au_names]

        au_string = self.format_au_list(au_names)

        template = random.choice(self.joint_templates)
        return template.format(au_string, fine_emotion, coarse_emotion)

    def build_sample_prompts(
        self,
        au_list: List[str],
        fine_emotion: str,
        coarse_emotion: str
    ) -> Dict[str, str]:
        """Build all prompts for a sample."""
        return {
            "au_prompt": self.get_au_prompt(au_list),
            "fine_prompt": self.get_fine_emotion_prompt(fine_emotion),
            "coarse_prompt": self.get_coarse_emotion_prompt(coarse_emotion),
            "joint_prompt": self.get_joint_prompt(au_list, fine_emotion, coarse_emotion)
        }
