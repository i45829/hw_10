import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("model.pkl", 'rb') as file:
    model = joblib.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = joblib.load(file)

class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float

class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get", response_model=Result)
def predict_get(total_square: float, rooms: float, floor: float):
    input_data = {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor
    }
    input_df = pd.DataFrame([input_data])
    
    scaled_data = scaler.transform(input_df)
    
    prediction = model.predict(scaled_data)
    return Result(result=float(prediction))

@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])
    
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    return Result(result=float(prediction))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
