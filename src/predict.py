import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"


def predict():
    pipeline = joblib.load(MODEL_PATH)

    # Новий кандидат
    new_person = pd.DataFrame(
        [[90000,8,1]],
        columns=["salary", "experience","is_young"]
    )

    prediction = pipeline.predict(new_person)[0]
    probability = pipeline.predict_proba(new_person)[0][1]
    THRESHOLD = 0.7  # ми самі вирішуємо, наскільки бути строгими

    prediction = 1 if probability >= THRESHOLD else 0

    print(f"Probability of Senior: {probability:.2f}")
    print(f"Final decision with threshold {THRESHOLD}: {prediction}")
    print(new_person)
    print(f"Prediction: {prediction} (1=Senior, 0=Junior)")
    print(f"Probability of Senior: {probability:.2f}")

    from metrics import precision_at_k, recall_at_k

# predicted probabilities
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    for k in [3, 5, 10]:
        print(f"Precision@{k}: {precision_at_k(y_true, y_scores, k):.2f}")
        print(f"Recall@{k}:    {recall_at_k(y_true, y_scores, k):.2f}")


if __name__ == "__main__":
    predict()
