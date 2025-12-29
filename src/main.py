import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text  # Для перегляду одного дерева
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "model.pkl"

# 1. Load data
df = pd.read_csv("data/sample.csv")

X = df[["age", "salary", "experience"]]
y = df["is_senior"]

# 2. If model exists → load it
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = joblib.load(MODEL_PATH)

# 3. Else → train and save
else:
    print("Training new model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print("Model saved.")

# 4. Prediction for new person
new_person = [[28, 1200, 4]]
prediction = model.predict(new_person)

print("Prediction (1=Senior, 0=Junior):", prediction[0])

# 5. Візуалізація одного дерева з Random Forest
print("\n=== Rules from first tree in Random Forest ===")

# Отримуємо перше дерево з ансамблю
first_tree = model.estimators_[24]  # model.estimators_ - список всіх дерев

# Тепер експортуємо правила для цього одного дерева
tree_rules = export_text(
    first_tree,  # передаємо одне дерево, а не весь RandomForest
    feature_names=["age", "salary", "experience"]
)
print(tree_rules)
import pandas as pd

importances = model.feature_importances_

features = ["age", "salary", "experience"]

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature importance:")
print(importance_df)
