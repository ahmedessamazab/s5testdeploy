import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(page_title="Ahmed Azab Portfolio", page_icon="😊", layout="centered")

st.title("😊 Ahmed Azab - Sentiment Analysis")

st.markdown(
    """
    Welcome to my AI Sentiment Analysis application!

    Analyze text in multiple languages using a
    pretrained Hugging Face Transformer model.
    """
)

# ---------------------------------
# DEVICE CONFIGURATION
# ---------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.caption(f"Running on: {DEVICE.upper()}")

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"


# ---------------------------------
# LOAD SENTIMENT MODEL
# ---------------------------------


@st.cache_resource
def load_sentiment_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
    )

    return sentiment_pipeline


# ---------------------------------
# MAIN APPLICATION
# ---------------------------------

st.header("🌍 Multilingual Sentiment Analysis")

st.write(
    """
    Enter any text and let AI identify its sentiment.

    The model supports five sentiment categories:
    """
)

# Sentiment categories

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("😡")
    st.caption("Very Negative")

with col2:
    st.markdown("😞")
    st.caption("Negative")

with col3:
    st.markdown("😐")
    st.caption("Neutral")

with col4:
    st.markdown("😊")
    st.caption("Positive")

with col5:
    st.markdown("😍")
    st.caption("Very Positive")


# ---------------------------------
# TEXT INPUT
# ---------------------------------

user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really love this product. It is amazing!",
    height=150,
)


# ---------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------

if st.button("🔍 Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment_model = load_sentiment_model()

                result = sentiment_model(user_text, truncation=True)[0]

                label = result["label"]
                confidence = result["score"]

                # Sentiment emojis

                sentiment_emojis = {
                    "Very Negative": "😡",
                    "Negative": "😞",
                    "Neutral": "😐",
                    "Positive": "😊",
                    "Very Positive": "😍",
                }

                emoji = sentiment_emojis.get(label, "🤖")

                # Results

                st.success("Analysis completed successfully!")

                st.divider()

                st.subheader("📊 Analysis Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Sentiment", f"{emoji} {label}")

                with col2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.progress(float(confidence))

                st.subheader("📝 Original Text")

                st.info(user_text)

            except Exception as e:
                st.error("Sentiment analysis failed.")
                st.exception(e)


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>Developed by <b>Ahmed Azab</b></p>
        <p>Powered by Hugging Face, PyTorch and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
