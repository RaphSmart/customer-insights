from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert"

class SentimentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

    def predict(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        preds = probs.argmax(dim=1)

        return preds.tolist(), probs.tolist()


# Singleton instance (loaded once)
sentiment_model = SentimentModel()
