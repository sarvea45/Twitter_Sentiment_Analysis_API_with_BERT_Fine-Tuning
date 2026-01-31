import streamlit as st
import requests
import pandas as pd
import time

# --- Page Config ---
st.set_page_config(
    page_title="Sentimates AI | Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
API_URL = "http://api:8000/predict"

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border: 1px solid #ced4da;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg", width=60)
    st.title("Sentimates AI")
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "89.7%", "+1.2%")
    col2.metric("Latency", "45ms", "-12ms")
    
    st.markdown("### 🛠️ Backend")
    st.code("DistilBERT-Uncased\nFastAPI Server\nDocker Container", language="text")
    
    st.markdown("---")
    st.info("This system analyzes text sentiment using a fine-tuned Transformer model deployed on a local Docker cluster.")

# --- Main Content ---
st.title("📢 Sentiment Analysis Engine")
st.markdown("Enter a tweet, review, or statement below to analyze its emotional tone in real-time.")

# Layout: Input on left, Results on right (Desktop view)
col_input, col_result = st.columns([2, 1])

with col_input:
    st.subheader("✍️ Input Text")
    user_text = st.text_area("Type your text here...", height=150, placeholder="e.g., The cinematography was breathtaking, but the plot was a bit slow.")
    
    if st.button("🔍 Analyze Sentiment"):
        if user_text.strip():
            with st.spinner("Processing with BERT..."):
                try:
                    # Artificial delay for visual effect in video (0.5s)
                    time.sleep(0.5) 
                    response = requests.post(API_URL, json={"text": user_text}, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        sentiment = data['sentiment']
                        confidence = data['confidence']
                        
                        # Save to history
                        st.session_state.history.insert(0, {
                            "Text": user_text,
                            "Sentiment": sentiment,
                            "Confidence": confidence,
                            "Time": pd.Timestamp.now().strftime("%H:%M:%S")
                        })
                        
                        # --- Display Results in Right Column ---
                        with col_result:
                            st.subheader("🎯 Analysis Result")
                            
                            # Color Logic
                            color = "green" if sentiment == "positive" else "red"
                            emoji = "😊" if sentiment == "positive" else "😠"
                            
                            # Result Card
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 5px solid {color};">
                                <h2 style="color: {color}; margin:0;">{emoji} {sentiment.upper()}</h2>
                                <p style="font-size: 1.2em; color: gray;">Confidence Score</p>
                                <h1 style="font-size: 3em; margin:0;">{confidence*100:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence Bar
                            st.progress(confidence)
                            
                            # JSON details expander
                            with st.expander("View Raw JSON Response"):
                                st.json(data)
                                
                    else:
                        st.error("Error: Could not connect to API.")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please enter some text first.")

# --- History Section (Dynamic Table) ---
st.markdown("---")
st.subheader("🕒 Recent Analysis History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    
    # Styled Dataframe
    def highlight_sentiment(val):
        color = '#d4edda' if val == 'positive' else '#f8d7da'
        return f'background-color: {color}'

    st.dataframe(
        df.style.applymap(highlight_sentiment, subset=['Sentiment']),
        use_container_width=True
    )
else:
    st.caption("No analysis performed yet. Try the example above!")