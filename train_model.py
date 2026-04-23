import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def main():
    csv_file = 'dataset.csv'
    model_file = 'model.pkl'

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run data_collection.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("Dataset is empty. Please collect some data first.")
        return

    # Check if there's enough data for multiple classes
    if len(df['label'].unique()) < 2:
        print(f"Error: Dataset only has one class '{df['label'].unique()[0]}'. You need at least 2 classes to train a classifier.")
        return

    X = df.drop('label', axis=1).values
    y = df['label'].values

    print(f"Dataset contains {len(df)} samples.")
    print(f"Classes: {np.unique(y)}")

    # Split dataset
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print("Warning: Not enough data in some classes to stratify. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training K-Nearest Neighbors classifier...")
    # Adjust n_neighbors if there are very few samples
    n_neighbors = min(3, len(X_train))
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Saving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print("Training complete!")

if __name__ == '__main__':
    main()
