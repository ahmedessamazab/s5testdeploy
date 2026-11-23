import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Page configuration
st.set_page_config(page_title="Ahmed Azab Portfolio", page_icon="🎨", layout="centered")

st.title("AI Image Generator")
st.header("OpenJourney Text-to-Image")

# Detect available device
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


@st.cache_resource
def load_model():
    # switch to "mps" for apple devices
    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", dtype=torch.bfloat16, device_map="cuda"
    )
    pipe = pipe.to(DEVICE)
    return pipe


prompt = st.text_area(
    "Enter your prompt:",
    "cat in a jungle, cold color palette, muted colors, detailed, 8k",
)


if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        pipe = load_model()
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
