import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Page configuration
st.set_page_config(page_title="Ahmed Azab Portfolio", page_icon="🎨", layout="centered")

st.title("AI Image Generator")
st.header("Generate Images with OpenJourney")

st.write("Enter a text prompt below and let AI generate an image for you.")


# Load the OpenJourney model
@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", torch_dtype=dtype
    )

    pipe = pipe.to(device)

    return pipe


# Load model
with st.spinner("Loading AI model... Please wait"):
    pipe = load_model()


# Text prompt
prompt = st.text_area(
    "Enter your image prompt:",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)

# Generate image
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating your image..."):
            try:
                image = pipe(prompt).images[0]

                st.success("Image generated successfully!")

                st.subheader("Generated Image")

                st.image(image, caption=prompt, use_container_width=True)

                # Download generated image
                import io

                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")

                st.download_button(
                    label="Download Image",
                    data=img_buffer.getvalue(),
                    file_name="openjourney_generated.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(f"Error generating image: {e}")

    else:
        st.warning("Please enter a prompt first.")
