from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert"

# Fallback model from HuggingFace if local model doesn't exist
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentModel:
    def __init__(self):

        # Create models folder if missing
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Check if model exists locally
        if not MODEL_PATH.exists() or not any(MODEL_PATH.iterdir()):
            print("Local model not found. Downloading from HuggingFace...")

            # Download model
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME)

            # Save locally for future use
            MODEL_PATH.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(MODEL_PATH)
            self.model.save_pretrained(MODEL_PATH)

            print(f"Model saved to {MODEL_PATH}")

        else:
            print(f"Loading model from local path: {MODEL_PATH}")

            # Load local model
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
