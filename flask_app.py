"""
Traffic Light Detection System — Flask Backend
REST API for image detection, video processing, and webcam frames.
"""

import cv2
import numpy as np
import base64
import time
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

from detector import detect, create_tinted_mask
from commands import CommandEngine
from utils import load_image, bgr_to_rgb, compute_stats

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload
app.config["UPLOAD_FOLDER"] = "/tmp"

# Global command engine for webcam mode (per-session in production)
command_engine = CommandEngine(stability_frames=3)

SAMPLE_DIR = Path(__file__).parent / "sample_images"
VIDEO_DIR = Path(__file__).parent / "sample_videos"

def encode_image_base64(image_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode BGR image to base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", image_bgr, encode_params)
    return base64.b64encode(buffer).decode("utf-8")

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to BGR image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples")
def get_samples():
    samples = []
    if SAMPLE_DIR.exists():
        for f in sorted(SAMPLE_DIR.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                samples.append({"name": f.stem.replace("_", " ").title(), "filename": f.name})
    return jsonify(samples)

@app.route("/api/sample/<filename>")
def get_sample_image(filename):
    return send_from_directory(str(SAMPLE_DIR), filename)

@app.route("/api/detect", methods=["POST"])
def detect_image():
    image = None
    if "image" in request.files:
        file = request.files["image"]
        image = load_image(file.read())
    elif request.is_json:
        data = request.get_json()
        if "image" in data:
            image = decode_base64_image(data["image"])

    if image is None: return jsonify({"error": "No image provided"}), 400

    start_time = time.time()
    result = detect(image)
    elapsed_ms = (time.time() - start_time) * 1000

    stats = compute_stats(result.detections)
    masks = {}
    for color in ["Red", "Yellow", "Green"]:
        raw = result.masks.get(color)
        filtered = result.filtered_masks.get(color)
        if raw is not None:
            tinted = create_tinted_mask(filtered if filtered is not None else raw, color)
            masks[color] = {
                "raw": encode_image_base64(cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)),
                "filtered": encode_image_base64(tinted),
                "raw_pixels": int(np.sum(raw > 0)),
                "filtered_pixels": int(np.sum(filtered > 0)) if filtered is not None else 0,
            }

    return jsonify({
        "dominant_color": result.dominant_color,
        "original_image": encode_image_base64(result.original_image),
        "annotated_image": encode_image_base64(result.annotated_image),
        "processing_time_ms": round(elapsed_ms, 1),
        "detections": [
            {"color": d.color, "bbox": d.bbox, "area": d.area, "confidence": round(d.confidence, 2), "circularity": round(d.circularity, 2)}
            for d in result.detections
        ],
        "stats": {
            color: {"count": s["count"], "total_area": s["total_area"], "max_confidence": round(s["max_confidence"], 2)}
            for color, s in stats.items()
        },
        "masks": masks,
    })

@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    data = request.get_json()
    if not data or "image" not in data: return jsonify({"error": "No image provided"}), 400
    image = decode_base64_image(data["image"])
    if image is None: return jsonify({"error": "Invalid image"}), 400

    start_time = time.time()
    result = detect(image)
    elapsed_ms = (time.time() - start_time) * 1000

    confidence = result.detections[0].confidence if result.detections else 0.0
    cmd = command_engine.update(result.dominant_color, confidence)

    return jsonify({
        "dominant_color": result.dominant_color,
        "annotated_image": encode_image_base64(result.annotated_image, quality=70),
        "processing_time_ms": round(elapsed_ms, 1),
        "command": {"text": cmd.command, "color": cmd.color, "emoji": cmd.emoji, "confidence": round(cmd.confidence, 2), "duration": round(command_engine.get_state_duration(), 1)},
        "detection_count": len(result.detections),
    })

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400
    file = request.files["video"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_video.mp4")
    file.save(path)
    return jsonify({"stream_url": f"/api/stream_video?path={path}"})

@app.route("/api/stream_video")
def stream_video():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "Video not found", 404

    def generate():
        cap = cv2.VideoCapture(path)
        command_engine.reset()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detect(frame)
            confidence = result.detections[0].confidence if result.detections else 0.0
            cmd = command_engine.update(result.dominant_color, confidence)
            
            # Draw HUD
            cv2.putText(result.annotated_image, f"CMD: {cmd.command}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd.text_color_bgr, 2, cv2.LINE_AA)
            cv2.putText(result.annotated_image, f"Conf: {cmd.confidence:.1f}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033) # ~30 FPS
            
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/reset_commands", methods=["POST"])
def reset_commands():
    command_engine.reset()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("\n🚦 Traffic Light Detection System")
    print("   http://localhost:5001\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
