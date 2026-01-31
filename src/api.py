from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

# Load the model from the local folder
MODEL_PATH = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get the predicted class index (0 or 1)
    idx = torch.argmax(logits, dim=-1).item()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][idx].item()
    
    # Standard Mapping: 0 is Negative, 1 is Positive
    label_map = {0: "negative", 1: "positive"}
    
    return {
        "sentiment": label_map[idx],
        "confidence": round(confidence, 4)
    }
