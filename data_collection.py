import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    landmarks = []
    # Use raw MediaPipe coordinates to match the Kaggle ASL dataset
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def main():
    cap = cv2.VideoCapture(0)
    csv_file = 'dataset.csv'
    
    print("Starting data collection...")
    print("Press 'a', 'b', 'c', etc. to save the gesture with that label.")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        extracted_landmarks = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                extracted_landmarks = extract_landmarks(hand_landmarks)
                
        cv2.putText(frame, "Press key to save. 'q' to quit.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('a') <= key <= ord('z') or ord('0') <= key <= ord('9'):
            if extracted_landmarks is not None:
                label = chr(key).upper()
                # Save to CSV
                file_exists = os.path.isfile(csv_file)
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # Write header to match Kaggle dataset
                        header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
                        writer.writerow(header)
                    row = extracted_landmarks + [label]
                    writer.writerow(row)
                print(f"Saved gesture '{label}'")
            else:
                print("No hand detected. Cannot save.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
