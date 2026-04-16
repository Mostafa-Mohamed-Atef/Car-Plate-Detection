"""
Car Plate Detection Utilities
Uses OpenCV-based methods for license plate detection and analysis.
No TensorFlow dependency — works on all Python versions.
"""

import numpy as np
import cv2
import os


class PlateDetector:
    """
    Detects license plates in vehicle images using OpenCV.
    Uses YOLO-format labels when available, falls back to
    contour-based detection for uploaded images.
    """

    def __init__(self, labels_dir=None):
        self.labels_dir = labels_dir

    def _parse_yolo_label(self, label_path, img_h, img_w):
        """Parse a YOLO-format label file and return bounding boxes."""
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_c, y_c, box_w, box_h = map(float, parts)
                    x1 = int((x_c - box_w / 2) * img_w)
                    y1 = int((y_c - box_h / 2) * img_h)
                    w = int(box_w * img_w)
                    h = int(box_h * img_h)
                    boxes.append({
                        "class_id": int(class_id),
                        "x": max(x1, 0),
                        "y": max(y1, 0),
                        "w": w,
                        "h": h,
                        "confidence": 1.0,
                        "source": "ground_truth",
                    })
        return boxes

    def detect_plates_opencv(self, image):
        """
        Detect potential license plate regions using OpenCV techniques:
        morphological operations, edge detection, and contour filtering.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # --- Method 1: Adaptive threshold + contour filtering ---
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 9
        )

        # Morphological close to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = bw / bh if bh > 0 else 0

            # Filter: plate-like aspect ratio & reasonable size
            if 1.5 < aspect < 6.0 and area > 800 and bw > 60 and bh > 15:
                # Confidence heuristic based on how "plate-like" the region is
                ideal_aspect = 3.5
                conf = max(0.3, 1.0 - abs(aspect - ideal_aspect) / ideal_aspect)
                candidates.append({
                    "class_id": 0,
                    "x": x, "y": y, "w": bw, "h": bh,
                    "confidence": round(conf, 2),
                    "source": "opencv_detection",
                })

        # --- Method 2: Edge-based detection (Canny + contours) ---
        edges = cv2.Canny(blur, 50, 200)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)
        contours2, _ = cv2.findContours(
            edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours2:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = bw / bh if bh > 0 else 0
            if 1.5 < aspect < 6.0 and area > 1000 and bw > 70 and bh > 18:
                # Avoid duplicates (overlapping with existing candidates)
                is_dup = False
                for c in candidates:
                    iou = _compute_iou(
                        (c["x"], c["y"], c["w"], c["h"]),
                        (x, y, bw, bh)
                    )
                    if iou > 0.3:
                        is_dup = True
                        break
                if not is_dup:
                    ideal_aspect = 3.5
                    conf = max(0.25, 1.0 - abs(aspect - ideal_aspect) / ideal_aspect)
                    candidates.append({
                        "class_id": 0,
                        "x": x, "y": y, "w": bw, "h": bh,
                        "confidence": round(conf, 2),
                        "source": "edge_detection",
                    })

        # Sort by confidence
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates

    def detect(self, image, image_name=None):
        """
        Detect plates. Uses ground-truth labels if available,
        otherwise falls back to OpenCV detection.
        """
        # Try ground-truth labels first
        if self.labels_dir and image_name:
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)
            h, w = image.shape[:2]
            gt_boxes = self._parse_yolo_label(label_path, h, w)
            if gt_boxes:
                return gt_boxes

        # Fall back to OpenCV detection
        return self.detect_plates_opencv(image)

    def annotate_image(self, image, detections):
        """Draw detection boxes on the image."""
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            conf = det["confidence"]
            source = det.get("source", "")

            # Color coding
            if source == "ground_truth":
                color = (0, 220, 80)   # green
                label = f"Plate (GT)"
            else:
                color = (80, 160, 255) # blue
                label = f"Plate {conf:.0%}"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(
                annotated, label, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        return annotated

    def extract_plate_crops(self, image, detections):
        """Crop and return individual plate regions."""
        crops = []
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = image[y : y + h, x : x + w]
            if crop.size > 0:
                crops.append(crop)
        return crops


def compute_image_stats(image):
    """Compute basic image statistics for display."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    return {
        "width": w,
        "height": h,
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
        "mean_brightness": round(float(np.mean(gray)), 1),
        "contrast": round(float(np.std(gray)), 1),
        "edge_density": round(float(np.sum(edges > 0) / (h * w)) * 100, 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
    }


def _compute_iou(box1, box2):
    """Compute Intersection over Union between two (x,y,w,h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0