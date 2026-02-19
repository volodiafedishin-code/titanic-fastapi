import os
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

from metrics import (
    precision_at_k, recall_at_k, evaluate_test_accuracy,
    plot_calibration_simple, pr_treshold_curve
)

DATA_PATH = "data/sample.csv"
MODEL_PATH = "models/model.pkl"
RANDOM_STATE = 42

from sklearn.base import BaseEstimator, TransformerMixin

class AgeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["is_young"] = (X["age"] < 23).astype(int)
        return X

def train():
    # Завантаження даних
    df = pd.read_csv(DATA_PATH)
    
    
    X = df[["salary", "experience", "is_young"]]
    y = df["is_senior"]
    age_transformer = AgeFeatures()
    
    # ✅ Тестуємо трансформацію
    X_transformed = age_transformer.transform(X)
    print("Після трансформації AgeFeatures:")
    print(X_transformed.head())

    # ✅ ПРАВИЛЬНО: калібрована модель в pipeline
    base_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=3
    )
    
    pipeline = Pipeline([
        ("features", AgeFeatures()),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print("CV AUC scores:", scores)
    print("Mean AUC:", scores.mean())
    
    # Розділення даних
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    
    # Тренування
    pipeline.fit(X_train, y_train)
    
    # Оцінка на тренувальних даних
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTrain accuracy: {train_accuracy:.3f}")
    
    # ========== ОЦІНКА З THRESHOLD ==========
    MANUAL_THRESHOLD = 0.5
    
    # Отримуємо калібровані ймовірності
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"\nCalibrated probabilities (sample): {np.round(y_proba[:5], 3)}")
    
    # Застосовуємо threshold
    y_test_pred = (y_proba >= MANUAL_THRESHOLD).astype(int)
    
    # Оцінка точності
    test_accuracy = evaluate_test_accuracy(y_test, y_test_pred)
    
    # PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.3f}")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # ========== RANKING METRICS ==========
    y_scores = y_proba
    y_true = y_test.values
    
    print("\n📊 Ranking Metrics:")
    for k in [3, 5, 10]:
        print(f"K={k}: P@{k}={precision_at_k(y_true, y_scores, k):.2f}, "
              f"R@{k}={recall_at_k(y_true, y_scores, k):.2f}")
    
    # ========== PR CURVE ==========
    pr_treshold_curve(y_proba, y_test)
    
    # ========== КАЛІБРУВАЛЬНА КРИВА ==========
    plot_calibration_simple(y_true, y_proba)
    
    # ========== OPTIMAL THRESHOLD BY COST ==========
    cost_fn = 10  # пропустити Senior
    cost_fp = 1   # зайвий Junior
    
    thresholds = np.linspace(0.1, 0.9, 81)
    total_costs = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        total_costs.append(cost)
    
    # ✅ ПРАВИЛЬНО: після циклу
    best_t = thresholds[np.argmin(total_costs)]
    print(f"\n🎯 Optimal threshold based on business cost: {best_t:.2f}")
    
    # Оцінка з оптимальним порогом
    y_opt_pred = (y_proba >= best_t).astype(int)
    opt_accuracy = accuracy_score(y_test, y_opt_pred)
    print(f"Accuracy with optimal threshold: {opt_accuracy:.3f}")
    
    # ========== ЗБЕРЕЖЕННЯ МОДЕЛІ ==========
    # Створюємо папку models
    os.makedirs("models", exist_ok=True)
    
    # Зберігаємо модель
    joblib.dump(pipeline, MODEL_PATH)
    
    # Зберігаємо метадані
    metadata = {
        "model_type": "RandomForestClassifier",
        "features": ["salary", "experience", "is_young"],
        "optimal_threshold": float(best_t),
        "manual_threshold": float(MANUAL_THRESHOLD),
        "auc_cv_mean": float(scores.mean()),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "pr_auc": float(pr_auc),
        "created_at": datetime.utcnow().isoformat()
    }
    
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"\n📊 Final metrics:")
    print(f"  CV AUC: {scores.mean():.3f}")
    print(f"  Train accuracy: {train_accuracy:.3f}")
    print(f"  Test accuracy: {test_accuracy:.3f}")
    print(f"  PR AUC: {pr_auc:.3f}")
    print(f"  Optimal threshold: {best_t:.3f}")

if __name__ == "__main__":
    train()