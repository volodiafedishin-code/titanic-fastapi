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
async def predict(
    request: Request,
    pclass: int = Form(...),
    age: float = Form(...),
    fare: float = Form(...)
):
    # Готуємо дані для моделі (перетворюємо в масив numpy)
    features = np.array([[pclass, age, fare]])
    
    # Робимо прогноз: 0 - загинув, 1 - вижив
    prediction = model.predict(features)[0]
    
    # Визначаємо текст результату
    result_text = "Pasażer mógł przeżyć!" if prediction == 1 else "Pasażer prawdopodobnie by zginął."
    
    # Повертаємо результат на нову або ту ж саму сторінку
    return templates.TemplateResponse("result.html", {
        "request": request, 
        "prediction_text": result_text
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)