from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("triage_model.pkl")


class Input(BaseModel):
    message: str
    age: int = 30
    gender: str = "male"


@app.post("/predict")
def predict(data: Input):
    text = f"{data.message} Age:{data.age} Gender:{data.gender}"
    result = model.predict([text])[0]

    return {
        "specialty": result,
        "status": "success"
    }


@app.get("/")
def home():
    return {"message": "Medical Triage API is running!"}