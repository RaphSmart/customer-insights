import os
import streamlit as st
import requests


# API URL:
# - docker-compose: http://api:8000/predict/sentiment
# - local fallback: http://localhost:8000/predict/sentiment
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict/sentiment"
)

st.set_page_config(
    page_title="Customer Sentiment Analyzer",
    layout="centered"
)

st.title("Customer Sentiment Analyzer")
st.write(
    "Enter customer feedback and analyze sentiment using a fine-tuned DistilBERT model."
)

text_input = st.text_area(
    "Customer text",
    height=150,
    placeholder="Type or paste customer feedback here..."
)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"texts": [text_input]},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()["results"][0]

                st.subheader(f"Prediction: **{result['label']}**")

                st.write("Probabilities:")
                st.json(result["probabilities"])

            else:
                st.error(
                    f"API error ({response.status_code}). "
                    "Check FastAPI logs."
                )

        except requests.exceptions.RequestException as e:
            st.error(
                "Could not connect to the API. "
                "Make sure FastAPI is running."
            )
            st.exception(e)
