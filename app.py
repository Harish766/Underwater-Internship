"""
app.py (FastAPI version) - FIXED

Fixes applied:
  1. TEMPLATES_DIR now correctly points to ./templates folder
  2. analytics page route serves analytics.html (which is the __init__.py content renamed)
  3. Import guard for pandas moved inside functions (avoids cold-import errors)
  4. WebSocket: annotated_frame RGB→BGR already correct via cv2.imencode
  5. Added CORS middleware for local dev
  6. /api/sessions/{session_id} properly returns 404 JSON
"""

import os
import json
import base64

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import inference_engine as engine
from analytics.pipeline_analytics import run_full_analytics

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Infrastructure Monitoring System",
    description="Pipeline defect detection + underwater environment monitoring backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")   # FIX: was missing correct path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# ============================================================
# PAGE ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page():
    # FIX: analytics.html is the renamed __init__.py (the analytics dashboard HTML)
    path = os.path.join(TEMPLATES_DIR, "analytics.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ============================================================
# REST API - VIDEO UPLOAD
# ============================================================

@app.post("/api/process_video")
async def process_video(file: UploadFile = File(...), module: str = Form(...)):
    """
    Accepts a video file + module name ("pipeline" or "underwater").
    Runs video processing in a thread pool so the event loop isn't blocked.
    """
    if module not in ("pipeline", "underwater"):
        return JSONResponse(status_code=400, content={"error": "module must be 'pipeline' or 'underwater'"})

    if not file.filename or not allowed_video_file(file.filename):
        return JSONResponse(status_code=400, content={"error": "Invalid or missing video file"})

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": "File too large (max 500MB)"})

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        result = await run_in_threadpool(engine.process_video_file, save_path, module)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": f"Model not found: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"status": "success", **result}


# ============================================================
# REST API - SESSION LIST
# ============================================================

@app.get("/api/sessions")
async def list_sessions():
    """Returns all past sessions from sessions_summary.csv."""
    import pandas as pd  # FIX: local import avoids startup failure if pandas missing
    summary_path = engine.MASTER_SUMMARY_CSV
    if not os.path.exists(summary_path):
        return {"sessions": []}
    df = pd.read_csv(summary_path)
    return {"sessions": df.to_dict("records")}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Returns the raw detections CSV rows for a specific session."""
    import pandas as pd
    csv_path = os.path.join(engine.SESSIONS_DIR, session_id, "detections.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    df = pd.read_csv(csv_path)
    return {"session_id": session_id, "rows": len(df), "data": df.to_dict("records")}


# ============================================================
# REST API - ANALYTICS
# ============================================================

@app.get("/api/sessions/{session_id}/analytics")
async def get_analytics(session_id: str, fps: float = 30.0):
    """
    Runs the full analytics engine on a session's detections CSV.
    Returns all metrics + Plotly chart JSON in a single response.
    """
    detections_csv = os.path.join(engine.SESSIONS_DIR, session_id, "detections.csv")
    summary_csv = engine.MASTER_SUMMARY_CSV

    if not os.path.exists(detections_csv):
        return JSONResponse(status_code=404, content={"error": f"No detections found for session {session_id}"})

    result = await run_in_threadpool(
        run_full_analytics,
        detections_csv,
        summary_csv,
        fps,
    )
    return result


# ============================================================
# WEBSOCKET - LIVE WEBCAM
# ============================================================

@app.websocket("/ws/webcam")
async def webcam_websocket(websocket: WebSocket):
    """
    Protocol (plain JSON text messages over the WebSocket):

    Client -> Server:
      {"type": "start", "module": "pipeline"}
      {"type": "frame", "image": "data:image/jpeg;base64,..."}
      {"type": "stop"}

    Server -> Client:
      {"type": "started", "session_id": "..."}
      {"type": "result", "annotated_image": "...", "detections": [...], "frame_number": N}
      {"type": "stopped", ...summary...}
      {"type": "error", "error": "..."}
    """
    await websocket.accept()
    webcam_session = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            msg_type = message.get("type")

            if msg_type == "start":
                module = message.get("module", "pipeline")
                try:
                    webcam_session = engine.WebcamSession(module=module)
                    await websocket.send_json({
                        "type": "started",
                        "session_id": webcam_session.session["session_id"],
                        "module": module,
                    })
                except FileNotFoundError as e:
                    await websocket.send_json({"type": "error", "error": str(e)})

            elif msg_type == "frame":
                if webcam_session is None:
                    await websocket.send_json({"type": "error", "error": "No active session. Send 'start' first."})
                    continue

                try:
                    # FIX: strip data URI prefix safely
                    image_data_str = message["image"]
                    if "," in image_data_str:
                        image_data_str = image_data_str.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_data_str)
                    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send_json({"type": "error", "error": "Could not decode frame"})
                        continue

                    frame_width = frame.shape[1]

                    detections, annotated_frame = await run_in_threadpool(
                        webcam_session.process_frame, frame, frame_width
                    )

                    # FIX: ensure annotated_frame is uint8 BGR before encoding
                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8)

                    _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                    await websocket.send_json({
                        "type": "result",
                        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                        "detections": detections,
                        "frame_number": webcam_session.frame_count,
                    })

                except Exception as e:
                    await websocket.send_json({"type": "error", "error": str(e)})

            elif msg_type == "stop":
                if webcam_session is not None:
                    result = await run_in_threadpool(webcam_session.finalize)
                    await websocket.send_json({"type": "stopped", **result})
                    webcam_session = None
                else:
                    await websocket.send_json({"type": "error", "error": "No active session to stop"})

    except WebSocketDisconnect:
        if webcam_session is not None:
            await run_in_threadpool(webcam_session.finalize)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)