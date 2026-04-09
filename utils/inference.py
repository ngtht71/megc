"""
Inference utilities for VQA
"""
import torch
import re
from typing import Tuple, Dict


def parse_vqa_question(question: str) -> Tuple[str, any]:
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


def generate_answer(
    model,
    apex: torch.Tensor,
    flow: torch.Tensor,
    roi: torch.Tensor,
    question: str,
    candidate_dict: Dict,
    device: str
) -> str:
    """Generate answer for VQA question."""
    model.eval()
    task, target_au = parse_vqa_question(question)

    if task == "unknown":
        return "Sorry, I cannot understand the question format."

    # Ensure batch dimension
    if apex.dim() == 3:
        apex = apex.unsqueeze(0)
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    if roi.dim() == 4:
        roi = roi.unsqueeze(0)

    apex, flow, roi = apex.to(device), flow.to(device), roi.to(device)

    with torch.no_grad():
        if task == "au_plural":
            candidate_texts = candidate_dict.get("au", [])
            logits = model.predict(apex, flow, roi, question, candidate_texts)

            probs = torch.softmax(logits, dim=1)[0]
            max_prob = probs.max().item()
            threshold = max_prob * 0.6

            selected_indices = torch.where(probs >= threshold)[0].tolist()
            predicted_aus = [candidate_texts[i] for i in selected_indices]
            answer = ", ".join(predicted_aus)
            return answer

        elif task == "au_singular":
            candidate_texts = candidate_dict.get("au", [])
            logits = model.predict(apex, flow, roi, question, candidate_texts)

            pred_idx = torch.argmax(logits, dim=1).item()
            return candidate_texts[pred_idx]

        elif task == "joint":
            logits_au = model.predict(apex, flow, roi, question, candidate_dict["au"])
            pred_au = candidate_dict["au"][torch.argmax(logits_au, dim=1).item()]

            logits_fine = model.predict(apex, flow, roi, question, candidate_dict["fine"])
            pred_fine = candidate_dict["fine"][torch.argmax(logits_fine, dim=1).item()]

            logits_coarse = model.predict(apex, flow, roi, question, candidate_dict["coarse"])
            pred_coarse = candidate_dict["coarse"][torch.argmax(logits_coarse, dim=1).item()]

            answer = f"The observed facial movement corresponds to action unit {pred_au}, which is associated with {pred_fine} and an overall {pred_coarse} emotion."
            return answer

        else:
            candidate_texts = candidate_dict.get(task, [])
            if not candidate_texts:
                return "unknown"

            logits = model.predict(apex, flow, roi, question, candidate_texts)
            pred_idx = torch.argmax(logits, dim=1).item()
            return candidate_texts[pred_idx]


def compute_multi_task_loss(
    logits_vqa: torch.Tensor,
    task_logits: torch.Tensor,
    true_task_ids: torch.Tensor,
    device: str
) -> torch.Tensor:
    """Compute combined VQA and router loss."""
    # VQA InfoNCE Loss
    labels_vqa = torch.arange(logits_vqa.size(0)).to(device)
    loss_v_to_t = torch.nn.functional.cross_entropy(logits_vqa, labels_vqa)
    loss_t_to_v = torch.nn.functional.cross_entropy(logits_vqa.t(), labels_vqa)
    loss_vqa = (loss_v_to_t + loss_t_to_v) / 2.0

    # Router Guidance Loss
    loss_router = torch.nn.functional.cross_entropy(task_logits, true_task_ids)

    return loss_vqa + 0.5 * loss_router
