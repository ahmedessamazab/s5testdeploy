import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Ahmed Azab Portfolio")

st.title("Ahmed Azab Portfolio")

st.image("./images/cat.jpg", alt="Sunrise over a mountain ridge", width=300)

st.write(
    "Welcome to my portfolio! Here you can find information "
    "about my projects, skills, and experience."
)

name = st.text_input("Enter your name:")
age = st.slider("Select your age:", 0, 100, 25)

if st.button("Submit"):
    st.write(f"Hello {name}, you are {age} years old!")

df = pd.DataFrame(
    {
        "Project": ["Project A", "Project B", "Project C"],
        "Description": [
            "Description of Project A",
            "Description of Project B",
            "Description of Project C",
        ],
        "Link": [
            "https://example.com/project-a",
            "https://example.com/project-b",
            "https://example.com/project-c",
        ],
    }
)

st.dataframe(df)


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
