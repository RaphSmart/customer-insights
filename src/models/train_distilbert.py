# DistilBERT Fine-Tuning
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# ------------------
# Paths
# ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/amazon_reviews_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models/distilbert"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# ------------------
# Metrics
# ------------------
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return f1_metric.compute(
        predictions=preds,
        references=labels,
        average="weighted"
    )

# ------------------
# Main
# ------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    test_ds = test_ds.remove_columns(
        [c for c in test_ds.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    print("Training DistilBERT...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(MODEL_DIR)

    print("Done.")

if __name__ == "__main__":
    main()
