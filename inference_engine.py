"""
inference_engine.py  — FIXED v2

Key fixes in this version:
  1. VideoWriter codec changed from mp4v to avc1/H264 with ffmpeg fallback
     so the output video is actually browser/player-playable.
  2. After VideoWriter finishes, re-encode with ffmpeg (if available) to ensure
     a proper moov atom is written (mp4v often writes it at the end, making
     the file unplayable if the process is interrupted).
  3. CSV is now written INCREMENTALLY every 100 frames so data is never lost
     even if processing crashes mid-video.
  4. Added verbose frame logging so you can see progress in terminal.
  5. Empty detection handling made bullet-proof.
"""

import os
import uuid
import time
import subprocess
from datetime import datetime

import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PIPELINE_MODEL_PATH = os.path.join(MODELS_DIR, "pipeline_best.pt")
UNDERWATER_MODEL_PATH = os.path.join(MODELS_DIR, "underwater_best.pt")

MASTER_SUMMARY_CSV = os.path.join(SESSIONS_DIR, "sessions_summary.csv")

os.makedirs(SESSIONS_DIR, exist_ok=True)

DETECTION_COLUMNS = [
    "frame_number", "timestamp", "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
    "width", "height", "area", "severity", "zone",
]


# ============================================================
# SEVERITY / ZONE HELPERS
# ============================================================

def compute_severity(class_name: str, confidence: float) -> str:
    cls = class_name.lower()
    if cls == "crack":
        return "CRITICAL" if confidence > 0.85 else "HIGH"
    elif cls == "deformation":
        return "HIGH"
    elif cls in ("rust", "corrosion"):
        return "MODERATE"
    elif cls == "debris":
        return "HIGH"
    elif cls == "coral":
        return "LOW"
    else:
        return "LOW"


def assign_zone(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "Zone_A"
    elif center_x < (2 * frame_width / 3):
        return "Zone_B"
    else:
        return "Zone_C"


# ============================================================
# MODEL REGISTRY (singleton)
# ============================================================

class ModelRegistry:
    _pipeline_model = None
    _underwater_model = None

    @classmethod
    def get_pipeline_model(cls):
        if cls._pipeline_model is None:
            if not os.path.exists(PIPELINE_MODEL_PATH):
                raise FileNotFoundError(
                    f"Pipeline model not found at {PIPELINE_MODEL_PATH}. "
                    "Place pipeline_best.pt in the models/ folder."
                )
            cls._pipeline_model = YOLO(PIPELINE_MODEL_PATH)
        return cls._pipeline_model

    @classmethod
    def get_underwater_model(cls):
        if cls._underwater_model is None:
            if not os.path.exists(UNDERWATER_MODEL_PATH):
                raise FileNotFoundError(
                    f"Underwater model not found at {UNDERWATER_MODEL_PATH}. "
                    "Place underwater_best.pt in the models/ folder."
                )
            cls._underwater_model = YOLO(UNDERWATER_MODEL_PATH)
        return cls._underwater_model


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def create_session(source_type: str, module: str) -> dict:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    session_id = f"{module}_{source_type}_{timestamp_str}_{short_uuid}"
    session_folder = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)
    return {
        "session_id": session_id,
        "session_folder": session_folder,
        "module": module,
        "source_type": source_type,
        "created_at": datetime.now().isoformat(),
    }


# ============================================================
# CODEC HELPER — find best available codec
# ============================================================

def _get_video_writer(output_path, fps, width, height):
    """
    Try codecs in order of preference.
    avc1/H264 = browser playable.
    mp4v = fallback, works everywhere but not browser-streamable.
    """
    # Try H264 first (browser playable)
    for codec in ["avc1", "H264", "X264"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[engine] Using codec: {codec}")
            return writer, codec
        writer.release()

    # Fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[engine] Using codec: mp4v (fallback)")
    return writer, "mp4v"


def _reencode_with_ffmpeg(input_path, output_path):
    """
    Re-encode with ffmpeg to fix moov atom placement and ensure
    the file is browser/player playable. Skips silently if ffmpeg
    is not installed.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-c:v", "libx264", "-preset", "fast",
             "-movflags", "+faststart",   # puts moov atom at start = streamable
             "-pix_fmt", "yuv420p",
             output_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            os.replace(output_path, input_path)  # replace original with re-encoded
            print(f"[engine] ffmpeg re-encode success: {input_path}")
            return True
        else:
            print(f"[engine] ffmpeg failed: {result.stderr[-300:]}")
            return False
    except FileNotFoundError:
        print("[engine] ffmpeg not found — skipping re-encode (video may not be browser-streamable)")
        return False
    except Exception as e:
        print(f"[engine] ffmpeg error: {e}")
        return False


# ============================================================
# FRAME-LEVEL DETECTION
# ============================================================

def run_detection_on_frame(model, frame, frame_number, fps, frame_width):
    """
    Runs YOLO + ByteTrack on a single frame.
    Returns (list_of_detection_dicts, annotated_frame_bgr).
    """
    results = model.track(
        source=frame,
        show=False,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            try:
                track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else -1
            except Exception:
                track_id = -1

            timestamp = frame_number / fps if fps > 0 else frame_number

            detections.append({
                "frame_number": frame_number,
                "timestamp": round(timestamp, 2),
                "track_id": track_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "width": round(box_width, 2),
                "height": round(box_height, 2),
                "area": round(area, 2),
                "severity": compute_severity(class_name, confidence),
                "zone": assign_zone(center_x, frame_width),
            })

    # result.plot() returns RGB → convert to BGR for OpenCV
    annotated_frame = result.plot()
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    return detections, annotated_frame_bgr


# ============================================================
# VIDEO FILE PROCESSING
# ============================================================

def process_video_file(video_path: str, module: str, frame_skip: int = 1):
    """
    Processes an uploaded video end-to-end.
    Returns session info dict.
    """
    model = (
        ModelRegistry.get_pipeline_model()
        if module == "pipeline"
        else ModelRegistry.get_underwater_model()
    )

    session = create_session(source_type="upload", module=module)
    session_folder = session["session_folder"]

    cap = cv2.VideoCapture(video_path)
    print(f"[engine] Opening video: {video_path}  opened={cap.isOpened()}")
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    print(f"[engine] Video info: {width}x{height} @ {fps}fps  total_frames={total_frames}")

    output_video_path = os.path.join(session_folder, "annotated_output.mp4")
    out, codec_used = _get_video_writer(output_video_path, fps, width, height)

    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter — cannot write output video")

    # CSV written incrementally — data is never lost if crash mid-video
    detections_csv_path = os.path.join(session_folder, "detections.csv")
    csv_written = False

    all_logs = []
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Progress log every 50 frames
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[engine] Frame {frame_count}/{total_frames}  elapsed={elapsed:.1f}s")

        if frame_skip > 1 and frame_count % frame_skip != 0:
            out.write(frame)
            continue

        try:
            detections, annotated_frame = run_detection_on_frame(
                model, frame, frame_count, fps, width
            )
            all_logs.extend(detections)
            out.write(annotated_frame)
        except Exception as e:
            print(f"[engine] ERROR on frame {frame_count}: {e}")
            out.write(frame)  # write original frame so video isn't broken

        # Flush CSV every 100 frames so data is safe if server crashes
        if frame_count % 100 == 0 and all_logs:
            df_partial = pd.DataFrame(all_logs)
            df_partial = df_partial[DETECTION_COLUMNS]
            df_partial.to_csv(detections_csv_path, index=False)
            csv_written = True
            print(f"[engine] CSV checkpoint: {len(all_logs)} detections saved")

    cap.release()
    out.release()
    processing_time = round(time.time() - start_time, 2)
    print(f"[engine] Processing done: {frame_count} frames in {processing_time}s")

    # Re-encode with ffmpeg for browser compatibility
    ffmpeg_tmp = output_video_path.replace(".mp4", "_tmp.mp4")
    _reencode_with_ffmpeg(output_video_path, ffmpeg_tmp)

    # Final CSV write
    if all_logs:
        df = pd.DataFrame(all_logs)
        df = df[DETECTION_COLUMNS]
    else:
        df = pd.DataFrame(columns=DETECTION_COLUMNS)

    df.to_csv(detections_csv_path, index=False)
    print(f"[engine] CSV saved: {detections_csv_path}  rows={len(df)}")

    _append_session_summary(session, df, frame_count, fps, processing_time, video_path)

    return {
        "session_id": session["session_id"],
        "detections_csv": detections_csv_path,
        "annotated_video": output_video_path,
        "frames_processed": frame_count,
        "detections_logged": len(df),
        "processing_time_sec": processing_time,
    }


# ============================================================
# WEBCAM SESSION
# ============================================================

class WebcamSession:
    def __init__(self, module: str):
        self.module = module
        self.session = create_session(source_type="webcam", module=module)
        self.model = (
            ModelRegistry.get_pipeline_model()
            if module == "pipeline"
            else ModelRegistry.get_underwater_model()
        )
        self.all_logs = []
        self.frame_count = 0
        self.fps_estimate = 15.0
        self.start_time = time.time()

    def process_frame(self, frame, frame_width):
        self.frame_count += 1
        detections, annotated_frame = run_detection_on_frame(
            self.model, frame, self.frame_count, self.fps_estimate, frame_width
        )
        self.all_logs.extend(detections)
        return detections, annotated_frame

    def finalize(self):
        processing_time = round(time.time() - self.start_time, 2)

        if self.all_logs:
            df = pd.DataFrame(self.all_logs)
            df = df[DETECTION_COLUMNS]
        else:
            df = pd.DataFrame(columns=DETECTION_COLUMNS)

        detections_csv_path = os.path.join(self.session["session_folder"], "detections.csv")
        df.to_csv(detections_csv_path, index=False)

        _append_session_summary(
            self.session, df, self.frame_count, self.fps_estimate,
            processing_time, source_label="live_webcam"
        )

        return {
            "session_id": self.session["session_id"],
            "detections_csv": detections_csv_path,
            "frames_processed": self.frame_count,
            "detections_logged": len(df),
            "processing_time_sec": processing_time,
        }


# ============================================================
# MASTER SUMMARY
# ============================================================

def _append_session_summary(session, df, frame_count, fps, processing_time,
                            video_path=None, source_label=None):
    if not df.empty:
        unique_defects_by_class = df.groupby("class_name")["track_id"].nunique().to_dict()
        sev_per_track = (
            df.sort_values("severity")
              .groupby("track_id")["severity"]
              .last()
        )
        severity_counts = sev_per_track.value_counts().to_dict()
        total_unique_defects = int(df["track_id"].nunique())
    else:
        unique_defects_by_class = {}
        severity_counts = {}
        total_unique_defects = 0

    summary_row = {
        "session_id":              session["session_id"],
        "module":                  session["module"],
        "source_type":             session["source_type"],
        "video_source":            video_path or source_label or "unknown",
        "created_at":              session["created_at"],
        "frames_processed":        frame_count,
        "fps":                     fps,
        "processing_time_sec":     processing_time,
        "total_detections_logged": len(df),
        "total_unique_defects":    total_unique_defects,
        "crack_count":             unique_defects_by_class.get("crack", 0),
        "corrosion_count":         unique_defects_by_class.get("corrosion", 0),
        "rust_count":              unique_defects_by_class.get("rust", 0),
        "deformation_count":       unique_defects_by_class.get("deformation", 0),
        "coral_count":             unique_defects_by_class.get("coral", 0),
        "debris_count":            unique_defects_by_class.get("debris", 0),
        "critical_severity_count": severity_counts.get("CRITICAL", 0),
        "high_severity_count":     severity_counts.get("HIGH", 0),
        "moderate_severity_count": severity_counts.get("MODERATE", 0),
        "low_severity_count":      severity_counts.get("LOW", 0),
    }

    summary_df = pd.DataFrame([summary_row])
    if os.path.exists(MASTER_SUMMARY_CSV):
        summary_df.to_csv(MASTER_SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(MASTER_SUMMARY_CSV, mode="w", header=True, index=False)