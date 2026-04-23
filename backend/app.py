import os
import cv2
import base64
import pickle
import numpy as np
import mediapipe as mp
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

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load Model
model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pkl')
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"Warning: {model_file} not found. Predictions will not work until the model is trained.")

def extract_landmarks(hand_landmarks):
    landmarks = []
    # Using raw MediaPipe coordinates to match the Kaggle dataset
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if model is None:
        await websocket.send_json({"error": "Model not loaded. Train the model first."})
        await websocket.close()
        return

    try:
        while True:
            # Receive base64 encoded image
            data = await websocket.receive_text()
            
            # Remove header if present (e.g., "data:image/jpeg;base64,...")
            if "," in data:
                data = data.split(",")[1]
                
            try:
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # Process frame
                # Flip and convert color for MediaPipe
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                prediction_label = "No Hand Detected"
                confidence = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        extracted_landmarks = extract_landmarks(hand_landmarks)
                        features = np.array([extracted_landmarks])
                        prediction = model.predict(features)
                        
                        # Get confidence if model supports predict_proba
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(features)
                            confidence = float(np.max(proba)) * 100
                        
                        prediction_label = str(prediction[0])
                        break # Only process first hand for now

                await websocket.send_json({
                    "prediction": prediction_label,
                    "confidence": f"{confidence:.1f}%" if confidence > 0 else ""
                })

            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(frontend_dir, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "Backend is running. Frontend not found."}
