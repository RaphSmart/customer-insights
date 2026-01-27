import os
import streamlit as st
import requests

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

# ---- Example inputs ----
examples = [
    "I absolutely love this product, it works perfectly!",
    "Terrible experience, customer support was useless.",
    "The service was okay, nothing special."
]

if "example_text" not in st.session_state:
    st.session_state["example_text"] = ""

if st.button("Use example text"):
    st.session_state["example_text"] = examples[0]

text_input = st.text_area(
    "Customer text",
    height=150,
    placeholder="Type or paste customer feedback here...",
    key="example_text"
)

# ---- Analyze button ----
if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                response = requests.post(
                    API_URL,
                    json={"texts": [text_input]},
                    timeout=10
                )

            if response.status_code == 200:
                result = response.json()["results"][0]

                label = result["label"]
                probs = result["probabilities"]

                st.subheader(f"Prediction: **{label}**")

                col1, col2 = st.columns(2)
                col1.metric("Negative", f"{probs[0]*100:.2f}%")
                col2.metric("Positive", f"{probs[1]*100:.2f}%")

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
