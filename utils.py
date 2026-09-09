"""
Utility functions for the Traffic Light Detection System.
Handles image loading, stats computation, and display helpers.
"""

import cv2
import numpy as np
from typing import List, Dict
from detector import Detection


def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Load an image from raw bytes (e.g., from Streamlit file uploader).

    Args:
        file_bytes: Raw image bytes

    Returns:
        Image in BGR format (OpenCV convention)
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (display)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_stats(detections: List[Detection]) -> Dict:
    """
    Compute detection statistics.

    Args:
        detections: List of Detection objects

    Returns:
        Dictionary with stats per color and overall
    """
    stats = {
        "Red": {"count": 0, "total_area": 0.0, "max_confidence": 0.0, "avg_circularity": 0.0},
        "Yellow": {"count": 0, "total_area": 0.0, "max_confidence": 0.0, "avg_circularity": 0.0},
        "Green": {"count": 0, "total_area": 0.0, "max_confidence": 0.0, "avg_circularity": 0.0},
    }

    for det in detections:
        color = det.color
        stats[color]["count"] += 1
        stats[color]["total_area"] += det.area
        stats[color]["max_confidence"] = max(stats[color]["max_confidence"], det.confidence)
        stats[color]["avg_circularity"] += det.circularity

    # Compute averages
    for color in stats:
        if stats[color]["count"] > 0:
            stats[color]["avg_circularity"] /= stats[color]["count"]

    return stats


def resize_for_display(image: np.ndarray, max_width: int = 600) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum display width

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_color_emoji(color: str) -> str:
    """Return emoji for a traffic light color."""
    return {"Red": "🔴", "Yellow": "🟡", "Green": "🟢"}.get(color, "⚪")


def get_color_hex(color: str) -> str:
    """Return hex color code for UI elements."""
    return {"Red": "#FF4B4B", "Yellow": "#FACA15", "Green": "#21C55D"}.get(color, "#888888")
