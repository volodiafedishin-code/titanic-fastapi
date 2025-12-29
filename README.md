# ML Engineer Portfolio Project — Employee Level Classification

## 📌 Project Overview
This project demonstrates a simple Machine Learning pipeline that classifies employees as Junior or Senior based on:
- Age
- Salary
- Years of experience

The goal is to show ML Engineer thinking: data handling, model training, persistence, and explainability.

---

## 🧠 Problem Statement
HR teams often need a quick and consistent way to estimate candidate seniority.
This model provides a recommendation, not an automatic decision.

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- RandomForestClassifier
- Joblib

---

## 🔬 ML Pipeline
1. Load structured data from CSV
2. Train a RandomForest model
3. Persist the trained model
4. Load the model for predictions
5. Explain decisions using feature importance

---

## 📊 Model Features Importance
Experience is the most important feature, followed by salary and age.

---

## ⚠️ Limitations
- Small dataset
- Not suitable for real hiring decisions
- Requires regular retraining

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/main.py

