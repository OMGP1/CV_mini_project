"""
Traffic Light Color Detector
Core HSV thresholding logic for detecting red, yellow, and green regions.

Precision improvements:
- Tighter HSV thresholds (especially yellow saturation)
- Circularity filter to reject non-round contours
- Aspect ratio filter (lights are roughly square bounding boxes)
- Adaptive min area based on image size
- Stronger morphological cleanup
- Keep only top detections per color (reject noise)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── HSV Color Thresholds ──────────────────────────────────────────────
# Balanced for real-world robustness + precision
# Shape filters (circularity, aspect ratio) prevent false positives
# even with wider HSV ranges

# Red wraps around hue 0/180, so we need two ranges
HSV_RANGES = {
    "Red": [
        (np.array([0, 120, 100]), np.array([10, 255, 255])),
        (np.array([165, 120, 100]), np.array([180, 255, 255])),
    ],
    "Yellow": [
        # High saturation + value to only catch actual lit lights
        (np.array([18, 150, 160]), np.array([32, 255, 255])),
    ],
    "Green": [
        (np.array([40, 80, 100]), np.array([85, 255, 255])),
    ],
}

# Colors for drawing (BGR format)
DRAW_COLORS = {
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
}

# Colors for mask tinting (BGR format)
TINT_COLORS = {
    "Red": (0, 0, 200),
    "Yellow": (0, 200, 200),
    "Green": (0, 200, 0),
}

# ── Filter Constants ──────────────────────────────────────────────────
MIN_CIRCULARITY = 0.50        # reject arrows, triangles, irregular shapes
MAX_ASPECT_RATIO = 1.6        # traffic lights are nearly circular (square bbox)
MIN_AREA_FRACTION = 0.0005    # reject tiny noise regions
MIN_SOLIDITY = 0.70           # reject hollow/irregular shapes (arrows, signs)
MAX_DETECTIONS_PER_COLOR = 3  # max lights per color in frame


@dataclass
class Detection:
    """Represents a single color detection result."""
    color: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    confidence: float       # percentage of image area
    circularity: float      # how circular the contour is (0-1)


@dataclass
class DetectionResult:
    """Complete detection result for an image."""
    detections: List[Detection] = field(default_factory=list)
    masks: dict = field(default_factory=dict)        # color -> raw binary mask
    filtered_masks: dict = field(default_factory=dict)  # color -> mask after filtering
    dominant_color: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None
    original_image: Optional[np.ndarray] = None


def create_mask(hsv_image: np.ndarray, color: str) -> np.ndarray:
    """
    Create a binary mask for a specific color using HSV thresholding.

    Args:
        hsv_image: Image in HSV color space
        color: One of 'Red', 'Yellow', 'Green'

    Returns:
        Binary mask where detected regions are white (255)
    """
    ranges = HSV_RANGES[color]
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        mask |= cv2.inRange(hsv_image, lower, upper)

    # Stronger morphological cleanup
    # Step 1: Open (erode then dilate) to remove small noise spots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Step 2: Close (dilate then erode) to fill small holes in valid regions
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Step 3: Final Gaussian blur + threshold to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def compute_circularity(contour: np.ndarray) -> float:
    """
    Compute circularity of a contour.
    circularity = 4π × area / perimeter²
    Perfect circle = 1.0

    Args:
        contour: OpenCV contour

    Returns:
        Circularity value (0.0 to 1.0)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter * perimeter)


def find_detections(mask: np.ndarray, color: str, image_area: int) -> Tuple[List[Detection], np.ndarray]:
    """
    Find contours in a mask, apply shape filters, and return valid Detection objects.

    Args:
        mask: Binary mask
        color: Color name
        image_area: Total image area in pixels (for adaptive thresholding)

    Returns:
        Tuple of (list of Detection objects, filtered mask)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Adaptive minimum area based on image size
    min_area = max(int(image_area * MIN_AREA_FRACTION), 400)

    detections = []
    # Create filtered mask — only includes contours that pass all filters
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter 1: Minimum area
        if area < min_area:
            continue

        # Filter 2: Circularity — traffic lights are round, arrows are not
        circularity = compute_circularity(cnt)
        if circularity < MIN_CIRCULARITY:
            continue

        # Filter 3: Aspect ratio — bounding box should be roughly square
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w / h, h / w) if min(w, h) > 0 else 999
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # Filter 4: Solidity — reject hollow/irregular shapes (arrows, signs)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        confidence = (area / image_area) * 100

        detections.append(Detection(
            color=color,
            contour=cnt,
            bbox=(x, y, w, h),
            area=area,
            confidence=confidence,
            circularity=circularity,
        ))

        # Draw accepted contour onto filtered mask
        cv2.drawContours(filtered_mask, [cnt], -1, 255, cv2.FILLED)

    # Sort by area (largest first) and keep only top N
    detections.sort(key=lambda d: d.area, reverse=True)
    detections = detections[:MAX_DETECTIONS_PER_COLOR]

    return detections, filtered_mask


def annotate_image(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: Original BGR image
        detections: List of Detection objects

    Returns:
        Annotated image copy
    """
    annotated = image.copy()

    for det in detections:
        x, y, w, h = det.bbox
        color_bgr = DRAW_COLORS[det.color]

        # Draw bounding box with rounded appearance
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2, cv2.LINE_AA)

        # Draw label background
        label = f"{det.color} {det.confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y - 8, th + 4)
        # Semi-transparent label background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, label_y - th - 4), (x + tw + 8, label_y + 4), color_bgr, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.putText(annotated, label, (x + 4, label_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated


def create_tinted_mask(mask: np.ndarray, color: str) -> np.ndarray:
    """
    Create a color-tinted version of a binary mask for visualization.

    Args:
        mask: Binary mask
        color: Color name for tinting

    Returns:
        BGR image with color-tinted mask
    """
    tinted = np.zeros((*mask.shape, 3), dtype=np.uint8)
    tint = TINT_COLORS[color]
    tinted[mask > 0] = tint
    return tinted


def detect(image: np.ndarray) -> DetectionResult:
    """
    Main detection pipeline.

    Args:
        image: Input image in BGR format (as read by OpenCV)

    Returns:
        DetectionResult with detections, masks, and annotated image
    """
    result = DetectionResult()
    result.original_image = image.copy()

    # ── Preprocessing for real-world robustness ──
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # to handle overexposed/underexposed real-world photos
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Slight Gaussian blur to reduce sensor noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    image_area = image.shape[0] * image.shape[1]

    all_detections = []

    for color in HSV_RANGES:
        # Create raw mask
        mask = create_mask(hsv, color)
        result.masks[color] = mask

        # Find detections with shape filtering
        detections, filtered_mask = find_detections(mask, color, image_area)
        result.filtered_masks[color] = filtered_mask
        all_detections.extend(detections)

    # Sort all detections by area (largest first)
    all_detections.sort(key=lambda d: d.area, reverse=True)
    result.detections = all_detections

    # Determine dominant color — the color with the largest single detection
    if all_detections:
        result.dominant_color = all_detections[0].color

    # Annotate image
    result.annotated_image = annotate_image(image, all_detections)

    return result
