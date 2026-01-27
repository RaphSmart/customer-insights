# Project Title
Customer Insights – End-to-End NLP Sentiment Analysis System

## What this project does

This is an End-to-end NLP system that trains, serves, and deploys a sentiment analysis model using DistilBERT. It includes data preprocessing, model training, evaluation, FastAPI inference API, and Dockerized deployment and Streamlit UI.

## Tech stack
- Python 3.10
- PyTorch + Hugging Face
- Scikit-learn
- FastAPI
- Docker 
- MLflow 


## Project structure
See `/src` for implementation and `/docs` for documentation.

## Status
🚧 In progress — actively developing

## Model Performance

| Model                     | F1 Score | Notes |
|---------------------------|----------|-------|
| TF-IDF + Logistic Reg     | 0.80     | Strong classical baseline |
| DistilBERT (fine-tuned)   | 0.896    | 2 epochs, Hugging Face Transformers |


## Project demo

![Customer Sentiment Analyzer](assets/demo_screenshot.png)

This application demonstrates an end-to-end AI system:
- Fine-tuned DistilBERT sentiment model
- FastAPI inference service
- Streamlit UI
- Docker + Docker Compose deployment