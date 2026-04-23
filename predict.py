import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def main():
    model_file = 'model.pkl'
    
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found. Please run train_model.py first.")
        return

    print("Loading model...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    print("Starting real-time prediction... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        prediction_label = "No Hand Detected"
        color = (0, 0, 255) # Red for no hand

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                extracted_landmarks = extract_landmarks(hand_landmarks)
                
                # Predict
                features = np.array([extracted_landmarks])
                prediction = model.predict(features)
                prediction_label = f"Gesture: {prediction[0]}"
                color = (0, 255, 0) # Green for detected gesture

        # Display prediction
        cv2.putText(frame, prediction_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        cv2.imshow('SignWise Lite Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
