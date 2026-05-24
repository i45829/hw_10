import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# 1. Загрузка обученной модели и скейлера (1 балл за использование модели из репозитория)
# Файлы берутся из списка файлов вашего репозитория [3]
with open("model.pkl", 'rb') as file:
    model = joblib.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = joblib.load(file)

# Определение структуры входных данных на основе признаков из traning.py [1]
class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float

class Result(BaseModel):
    result: float

# Реализация liveness-пробы (health-check) — 1 балл
@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

# Реализация предсказания через GET-запрос по адресу /predict_get — 2 балла
@app.get("/predict_get", response_model=Result)
def predict_get(total_square: float, rooms: float, floor: float):
    # Формируем словарь с именами признаков, на которых обучалась модель [1]
    input_data = {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor
    }
    input_df = pd.DataFrame([input_data])
    
    # Масштабирование данных перед предсказанием [2]
    scaled_data = scaler.transform(input_df)
    
    # Получение предсказания моделью
    prediction = model.predict(scaled_data)
    return Result(result=float(prediction))

# Реализация предсказания через POST-запрос по адресу /predict_post — 2 балла
@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])
    
    # Масштабирование и предсказание
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    return Result(result=float(prediction))

# Запуск через uvicorn (в стиле предоставленного образца [4])
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)