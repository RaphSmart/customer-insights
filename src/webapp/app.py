import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict/sentiment"

st.set_page_config(page_title="Customer Sentiment Analyzer", layout="centered")


st.title("🧠 Customer Sentiment Analyzer")
st.write("Enter customer feedback and analyze sentiment using a fine-tuned DistilBERT model.")


text_input = st.text_area(
    "Customer text",
    height=150,
    placeholder="Type or paste customer feedback here..."
)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            API_URL,
            json={"texts": [text_input]}
        )

        if response.status_code == 200:
            result = response.json()["results"][0]
            st.subheader(f"Presiction: **{result['label']}**")
            st.write("Probabilities:")
            st.json(result["probabilities"])
        else:
            st.error("API error. Make sure FastAPI is running.")