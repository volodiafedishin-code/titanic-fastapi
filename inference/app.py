from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # Якщо будуть картинки
import uvicorn
import joblib
import numpy as np

app = FastAPI()

# Вказуємо шлях до папки з твоїм HTML
import os

# Отримуємо шлях до папки, де лежить цей файл (inference)
import os
from fastapi.templating import Jinja2Templates

# Отримуємо шлях до папки, де лежить app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Чітко вказуємо шлях до папки з шаблонами
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Завантажуємо твою навчену модель ШІ
# Переконайся, що файл model.pkl лежить у тій же папці
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Виходимо на один рівень вгору (у папку /src) і заходимо в /models
# '../models/titanic_model.pkl' — тут важливо вказати розширення .pkl
model_path = os.path.join(BASE_DIR, "..", "models", "titanic_model.pkl")

# 3. Тепер завантажуємо, використовуючи точну «карту» до файлу
model = joblib.load(model_path)

# 1. ГОЛОВНА СТОРІНКА (показуємо твій крутий дизайн)
@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. ЛОГІКА ПЕРЕДБАЧЕННЯ (обробка форми)
@app.post("/predict")
async def predict(pclass: int = Form(...), age: float = Form(...), fare: float = Form(...), sex: int = Form(...)):
    features = np.array([[pclass, age, fare, sex]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        text = "Przeżyje! 🟢"  
    else:
        text = "Nie przeżyje... 🔴"
    return {"prediction_text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)