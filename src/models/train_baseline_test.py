# Model Training
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



PROCESSED_DATA_PATH = Path("data/processed/amazon_reviews_clean.csv")
MODEL_DIR = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df["clean_text"]
    y = df["label"]
    return X, y


def train():
    print("Loading processed data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    