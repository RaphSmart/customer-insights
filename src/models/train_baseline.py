# Baseline Training Script 
# Logistic Regression text classifier

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


# Resolve project root safely
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data/processed/amazon_reviews_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models/baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluation...")
    preds = model.predict(X_test_vec)
    print(classification_report(y_test, preds))

    print("Saving model...")
    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        MODEL_DIR / "sentiment_baseline.pkl"
    )

    print("Done.")


if __name__ == "__main__":
    main()