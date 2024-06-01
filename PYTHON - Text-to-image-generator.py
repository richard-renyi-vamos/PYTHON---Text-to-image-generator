import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

st.title("Text to Image Generator 🌈🖼️")

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    return image

prompt = st.text_input("Enter a text prompt:")
if st.button("Generate Image"):
    if prompt:
        with st.spinner('Generating image... ⏳'):
            image = generate_image(prompt)
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a text prompt!")
