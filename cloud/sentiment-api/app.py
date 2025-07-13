# app.py
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/predict/")
def predict_sentiment(text: str):
    result = classifier(text)
    return {"label": result[0]['label'], "score": result[0]['score']}



