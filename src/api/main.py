from fastapi import FastAPI
from src.api.schemas import SentimentRequest
from src.models.inference import sentiment_model

app = FastAPI(
    title="Customer Insights API",
    description="Sentiment analysis using fine-tuned DistilBERT",
    version="1.0.0"
)

@app.post("/predict/sentiment")
def predict_sentiment(request: SentimentRequest):
    preds, probs = sentiment_model.predict(request.texts)

    results = []
    for text, pred, prob in zip(request.texts, preds, probs):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        results.append({
            "text": text,
            "label": label,
            "probabilities": prob
        })

    return {"results": results}


@app.get("/")
def health_check():
    return {"status": "ok"}
