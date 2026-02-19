from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/distilbert"

def predict(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    preds = probs.argmax(dim=1)
    return preds.tolist(), probs.tolist()


if __name__ == "__main__":
    samples = [
        "I absolutely love this product, it works perfectly!",
        "This was a terrible experience, very disappointed.",
        "Customer support was okay, nothing special."
    ]

    preds, probs = predict(samples)

    for text, pred, prob in zip(samples, preds, probs):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print("\nText: ", text)
        print("Prediction:", label)
        print("Probabilities:", prob)