from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/model.joblib")
CLASSES = ["setosa", "versicolor", "virginica"]

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    y = int(model.predict([[f.sepal_length, f.sepal_width,
                            f.petal_length, f.petal_width]])[0])
    return {"prediction": y, "label": CLASSES[y]}