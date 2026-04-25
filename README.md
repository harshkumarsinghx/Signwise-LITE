# SignWise Lite (Full-Stack)

SignWise Lite is a lightweight, real-time hand gesture recognition system featuring a stunning **glassmorphism UI**. It leverages MediaPipe for hand landmark extraction and Scikit-Learn (K-Nearest Neighbors) for robust, lightweight classification.

## Project Structure

- `backend/app.py`: A FastAPI application that serves the frontend and handles real-time WebSocket communication for MediaPipe predictions.
- `frontend/`: Contains the modern HTML/CSS/JS frontend application.
- `dataset.csv`: The Kaggle ASL landmark dataset (63 features per row).
- `model.pkl`: The trained Scikit-Learn KNN model.
- `train_model.py`: Script to train a new model from `dataset.csv`.
- `data_collection.py` & `predict.py`: Standalone Python scripts for offline testing and data collection.

## Quick Start (Web App)

1. Make sure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the full-stack web application:
   ```bash
   python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

4. Allow the browser to access your webcam. The UI will instantly connect to the backend and start displaying real-time predictions!

## Training the Model

If you ever want to retrain the model on new data, simply run:
```bash
python train_model.py
```
This will overwrite `model.pkl`. The backend will use the newly trained model on its next startup.

#Screenshots(LOCAL)
<img width="1918" height="867" alt="Screenshot 2026-04-25 193824" src="https://github.com/user-attachments/assets/ac627426-362d-4872-90f6-38b3b4a745cf" />
<img width="525" height="708" alt="Screenshot 2026-04-25 195057" src="https://github.com/user-attachments/assets/28291bbe-9583-4456-a1ab-d2fda9240356" />
<img width="461" height="138" alt="Screenshot 2026-04-25 195046" src="https://github.com/user-attachments/assets/8338ca7a-4b52-40ef-931c-fa34d52b1c29" />

