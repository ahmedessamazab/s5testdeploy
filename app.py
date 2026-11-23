import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Ahmed Azab Portfolio", page_icon="🎨", layout="centered")

st.title("🎨 AI Image Generator")
st.header("OpenJourney Text-to-Image")

DEVICE = "cpu"
DTYPE = torch.float32


@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    return pipe


prompt = st.text_area(
    "Enter your prompt", "cat in a jungle, cold color palette, muted colors, detailed"
)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            pipe = load_model()
            image = pipe(prompt, num_inference_steps=20).images[0]

            st.image(image, caption="Generated Image", use_container_width=True)

        except Exception as e:
            st.exception(e)
