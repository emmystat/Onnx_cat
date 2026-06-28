from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from contextlib import asynccontextmanager
from typing import List

model_session=None
input_name=None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_session, input_name
    try:
        model_session = ort.InferenceSession('model.onnx')
        input_name = model_session.get_input()[0].name
    except Exception as e:
        print(f"Error:{e}")
    yield
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

class InferenceRequest(BaseModel):
    features: List[float]

@app.post('/predict')
async def predict(request: InferenceRequest):
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail='Model requires exactly four input features')
    
    input_data = np.array([request.features], dtype=np.float32)
    try:
        raw_predictions = model_session.run(None, {input_name: input_data})[0]
        prediction = int(np.argmax(raw_predictions[0]))
        probabilities = raw_predictions[0].tolist()
        return {
            'prediction':prediction,
            'probabilities': probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400)

@app.get('/')
def index():
    return {'status':'Ok'}