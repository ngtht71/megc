"""
Data preprocessing utilities: optical flow, facial landmarks, AU extraction
"""
import os
import cv2
import numpy as np
import dlib
import re
from typing import Tuple, Dict, List, Optional


def compute_flow(onset_img: np.ndarray, apex_img: np.ndarray) -> np.ndarray:
    """Compute optical flow using Farneback algorithm."""
    onset_img = cv2.resize(onset_img, (224, 224))
    apex_img = cv2.resize(apex_img, (224, 224))

    gray1 = cv2.cvtColor(onset_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(apex_img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to RGB image."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((224, 224, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def get_optical_flow_image(onset_img: np.ndarray, apex_img: np.ndarray) -> np.ndarray:
    """Get optical flow as RGB image."""
    flow = compute_flow(onset_img, apex_img)
    flow_rgb = flow_to_rgb(flow)
    return flow_rgb


class AUExtractor:
    """Extract facial landmarks and Action Unit ROIs."""

    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 68-point facial landmarks."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        shape = self.predictor(gray, faces[0])
        coords = np.array([[p.x, p.y] for p in shape.parts()])
        return coords


def extract_au_roi_by_indices(
    image: np.ndarray,
    landmarks: np.ndarray,
    point_indices: List[int],
    padding_ratio: float = 0.15,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Extract AU ROI based on landmark indices."""
    points = landmarks[point_indices]

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    face_width = landmarks[16][0] - landmarks[0][0]
    pad = int(face_width * padding_ratio)

    box_height = y_max - y_min
    box_width = x_max - x_min
    if box_height < pad:
        y_min -= pad // 2
        y_max += pad // 2
    if box_width < pad:
        x_min -= pad // 2
        x_max += pad // 2

    start_x = max(0, int(x_min - pad))
    start_y = max(0, int(y_min - pad))
    end_x = min(image.shape[1], int(x_max + pad))
    end_y = min(image.shape[0], int(y_max + pad))

    roi = image[start_y:end_y, start_x:end_x]

    if roi.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    img = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    return img


def get_name_au_label(raw_au_string: str, au_mapping: Dict) -> List[str]:
    """Extract AU names from raw AU string."""
    extracted_numbers = re.findall(r'\d+', str(raw_au_string))
    au_numbers = []
    for num in extracted_numbers:
        n = int(num)
        if n not in au_numbers:
            au_numbers.append(n)

    au_names = []
    for num in au_numbers:
        if num in au_mapping:
            name = au_mapping[num]["name"]
            au_names.append(name)

    return au_names


def process_roi_with_raw_au(
    raw_au_string: str,
    image: np.ndarray,
    landmarks: np.ndarray,
    au_mapping: Dict
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Process ROI for specific AUs from raw AU string."""
    if landmarks is None:
        print("Landmarks detection failed")
        return [], {}

    extracted_numbers = re.findall(r'\d+', str(raw_au_string))
    au_numbers = []
    for num in extracted_numbers:
        n = int(num)
        if n not in au_numbers:
            au_numbers.append(n)

    au_names = []
    au_rois = {}

    for num in au_numbers:
        if num in au_mapping:
            name = au_mapping[num]["name"]
            au_names.append(name)

            point_indices = au_mapping[num]["landmarks"]
            roi_img = extract_au_roi_by_indices(image, landmarks, point_indices)
            au_rois[name] = roi_img

    return au_names, au_rois


def process_all_roi(
    image: np.ndarray,
    landmarks: np.ndarray,
    au_mapping: Dict
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Process ROI for all AUs."""
    au_names = []
    au_rois = {}
    au_numbers = sorted(au_mapping.keys())

    for num in au_numbers:
        if num in au_mapping:
            name = au_mapping[num]["name"]
            au_names.append(name)

            point_indices = au_mapping[num]["landmarks"]
            roi_img = extract_au_roi_by_indices(image, landmarks, point_indices)
            au_rois[name] = roi_img

    return au_names, au_rois
