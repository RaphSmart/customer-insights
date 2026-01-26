# customer-insights
End-to-end machine learning project demonstrating:
- data pipelines
- model training
- model serving (API)
- basic MLOps practices

## Tech stack
- Python
- PyTorch + Hugging Face
- FastAPI
- Docker (later)
- MLflow (later)

## Project structure
See `/src` for implementation and `/docs` for documentation.

## Status
🚧 In progress — actively developing

## Model Performance

| Model                     | F1 Score | Notes |
|---------------------------|----------|-------|
| TF-IDF + Logistic Reg     | 0.80     | Strong classical baseline |
| DistilBERT (fine-tuned)   | 0.896    | 2 epochs, Hugging Face Transformers |


# Project demo
A live demo built with Streamlit is included in this project.
