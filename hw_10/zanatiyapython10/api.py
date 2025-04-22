from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from model_utils import load_model

app = FastAPI()
model = load_model()

class PropertyInput(BaseModel):
    total_square: float
    rooms: int
    floor: int
    lat: float
    lon: float
    city: str
    district: str

@app.get("/health")
def health_check():
    return {"status": "alive"}

@app.get("/predict_get")
def predict_get(
    total_square: float = Query(...),
    rooms: int = Query(...),
    floor: int = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    city: str = Query(...),
    district: str = Query(...)
):
    input_df = pd.DataFrame([{
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city,
        "district": district
    }])
    prediction = model.predict(input_df)
    return {"predicted_price": float(prediction[0])}

@app.post("/predict_post")
def predict_post(data: PropertyInput):
    input_df = pd.DataFrame([data.model_dump()])  # это для pydantic 2.x
    prediction = model.predict(input_df)
    return {"predicted_price": float(prediction[0])}