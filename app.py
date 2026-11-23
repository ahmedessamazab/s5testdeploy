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


# Load model
@st.cache_resource
def load_model():

    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", torch_dtype=DTYPE, use_safetensors=True
    )

    pipe = pipe.to(DEVICE)

    return pipe


# Load pipeline
try:
    with st.spinner("Loading OpenJourney model..."):
        pipe = load_model()

    st.success(f"Model loaded successfully on {DEVICE.upper()}")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Prompt
prompt = st.text_area(
    "Enter your prompt:",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)

# Generate
if st.button("Generate Image"):
    if prompt.strip():
        try:
            with st.spinner("Generating image... Please wait"):
                with torch.inference_mode():
                    image = pipe(prompt).images[0]

                st.success("Image generated successfully!")

                st.image(image, caption=prompt, use_container_width=True)

                # Download image
                import io

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")

                st.download_button(
                    "Download Image",
                    data=buffer.getvalue(),
                    file_name="openjourney.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Image generation failed: {e}")

    else:
        st.warning("Please enter a prompt.")
