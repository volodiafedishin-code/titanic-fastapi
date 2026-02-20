from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os


app = FastAPI()



Base_Dir=os.path.dirname(__file__)
print(f"📍 Папка скрипта (BASE_DIR): {Base_Dir}")
MODEL_PATH=os.path.join(Base_Dir,"..","models","titanic_model.pkl")
print(f"📂 Шукаю модель за цим шляхом: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Модель успішно завантажена!")
except Exception as e:
    print(f"❌ Помилка при завантаженні моделі: {e}")
    model = None # Позначаємо, що моделі немає
# 1. Створюємо "Трафарет" (згадай нашу розмову про Class)
class Passenger_tytanic(BaseModel):
    pclass:int
    age:float
    fare:float

@app.post("/get-result")
def process_data(data: Passenger_tytanic):
    if model is None: 
        return {"error": "Model not loaded"}
    else:
        features = [[data.pclass, data.age, data.fare]]
        result=model.predict(features)
        result=int(result[0])
        if result==1:
            res="passenger survived"
        else:
            res="passenger died"
        return {"Status":res,
                "Prediction":result,
                "Passenger class":data.pclass
                }

@app.get("/INFO")
def model_info():
    return {
        "model_name": "Titanic Survival Predictor",
        "version": "1.0.0",
        "created_at": "2023-10-27" # Можеш вказати дату створення своєї моделі
    }