import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Ahmed Azab Portfolio")

st.title("emotion classification")


# Load emotion classification model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device=-1,
    )


pipe = load_model()

st.header("Emotion Detection")

sentence = st.text_area("Enter a sentence:", "I am not having a great day")

if st.button("Analyze Emotion"):
    if sentence.strip():
        results = pipe(sentence)[0]

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.subheader("Predicted Emotions")

        for result in results:
            if result["score"] > 0.1:
                st.write(f"**{result['label']}**: {result['score']:.2%}")
                st.progress(float(result["score"]))
