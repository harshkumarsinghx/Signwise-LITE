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
