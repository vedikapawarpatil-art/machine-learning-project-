"""
Machine Learning Model Training Pipeline
Author: Vedika Pawar-Patil
Description: End-to-end classification pipeline utilizing Scikit-Learn with
             data preprocessing, model training, evaluation metrics, and model persistence.
"""

import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def run_pipeline():
    print("=" * 60)
    print("      Machine Learning Classification Pipeline")
    print("=" * 60)

    # 1. Load Dataset
    print("\n[1/5] Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    print(f" -> Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f" -> Classes: {target_names[0]} (0) vs {target_names[1]} (1)")

    # 2. Train-Test Split
    print("\n[2/5] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f" -> Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 3. Feature Scaling
    print("\n[3/5] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    print("\n[4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print(" -> Model training complete.")

    # 5. Evaluation
    print("\n[5/5] Evaluating model performance on unseen test data...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f" -> Overall Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "random_forest_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[OK] Model successfully saved to: {model_path}")
    print(f"[OK] Scaler successfully saved to: {scaler_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()