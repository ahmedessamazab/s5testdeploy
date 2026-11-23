import io

import streamlit as st
import torch
from diffusers import DDIMScheduler, DiffusionPipeline

# Page configuration
st.set_page_config(page_title="Ahmed Azab Portfolio", page_icon="🎨", layout="centered")

st.title("AI Image Generator")
st.header("Generate Images with OpenJourney")

st.write("Enter a prompt and generate an AI image.")

# Device configuration
DEVICE = "cpu"


# Load model
@st.cache_resource
def load_model():

    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney", torch_dtype=torch.float32, use_safetensors=True
    )

    # Use DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(DEVICE)

    # Enable memory optimization
    pipe.enable_attention_slicing()

    return pipe


# Load model
try:
    with st.spinner("Loading OpenJourney model..."):
        pipe = load_model()

    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Prompt input
prompt = st.text_area(
    "Enter your image prompt:",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)

# Generation settings
steps = st.slider("Generation Steps", min_value=10, max_value=30, value=15)

guidance = st.slider(
    "Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5
)


# Generate button
if st.button("Generate Image"):
    if prompt.strip():
        try:
            with st.spinner(
                "Generating your image... This may take several minutes on CPU."
            ):
                generator = torch.Generator(device=DEVICE).manual_seed(42)

                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        height=512,
                        width=512,
                    )

                image = result.images[0]

                st.success("Image generated successfully!")

                st.subheader("Generated Image")

                st.image(image, caption=prompt, use_container_width=True)

                # Download image
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")

                st.download_button(
                    "Download Image",
                    data=img_buffer.getvalue(),
                    file_name="openjourney_generated.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Error generating image: {e}")

    else:
        st.warning("Please enter a prompt.")
