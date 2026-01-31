import streamlit as st
import requests
import os

# Get API URL from environment variable (Docker) or default to localhost (Local)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖")
st.title("Twitter Sentiment Analysis (BERT)")
st.markdown("Enter text below to analyze its emotional tone.")

user_input = st.text_area("Tweet/Review Text:", placeholder="Type something here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(f"{API_URL}/predict", json={"text": user_input}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data["sentiment"]
                    confidence = data["confidence"]
                    
                    # Visual feedback
                    color = "green" if sentiment == "positive" else "red"
                    st.subheader(f"Result: :{color}[{sentiment.upper()}]")
                    st.progress(confidence)
                    st.write(f"Confidence: {round(confidence * 100, 2)}%")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to API at {API_URL}. Ensure the API container is running.")
