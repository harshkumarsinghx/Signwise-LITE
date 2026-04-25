import os
import cv2
import base64
import pickle
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SignWise Lite Backend")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MediaPipe Tasks – Hand Landmarker (new API, 0.10.31+) ──────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    # Download the model at startup if it is missing
    import urllib.request
    _url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"Downloading hand_landmarker.task model from {_url} …")
    urllib.request.urlretrieve(_url, MODEL_PATH)
    print("Model downloaded successfully.")

_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_hand_landmarker = mp_vision.HandLandmarker.create_from_options(_options)


def extract_landmarks(hand_landmarks_list) -> list[float]:
    """Flatten one hand's landmarks into a 63-element list (21 × x,y,z)."""
    landmarks: list[float] = []
    for lm in hand_landmarks_list:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


# ── Model ──────────────────────────────────────────────────────────────────
model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"Warning: {model_file} not found. Predictions will not work until the model is trained.")


# ── WebSocket endpoint ─────────────────────────────────────────────────────
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if model is None:
        await websocket.send_json({"error": "Model not loaded. Train the model first."})
        await websocket.close()
        return

    try:
        while True:
            # Receive base64-encoded JPEG frame
            data = await websocket.receive_text()

            # Strip data-URL header if present
            if "," in data:
                data = data.split(",")[1]

            try:
                img_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Convert to RGB for MediaPipe (no flip — frontend mirrors via canvas)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                detection_result = _hand_landmarker.detect(mp_image)

                prediction_label = "No Hand Detected"
                confidence = 0.0

                landmarks_for_client = None
                if detection_result.hand_landmarks:
                    # Process only the first detected hand
                    hand_lm = detection_result.hand_landmarks[0]
                    extracted = extract_landmarks(hand_lm)
                    features = np.array([extracted])

                    prediction = model.predict(features)
                    prediction_label = str(prediction[0])

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)
                        confidence = float(np.max(proba)) * 100

                    # Send normalized (x, y) landmark coords for frontend drawing
                    landmarks_for_client = [[lm.x, lm.y] for lm in hand_lm]

                await websocket.send_json(
                    {
                        "prediction": prediction_label,
                        "confidence": f"{confidence:.1f}%" if confidence > 0 else "",
                        "landmarks": landmarks_for_client,
                    }
                )

            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")


# ── Serve frontend static files ────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "Backend is running. Frontend not found."}
