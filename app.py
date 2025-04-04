import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Generator  # Ensure this file contains your Generator class

# Set title
st.title("🎨 Anime Face Generator (GAN)")
st.write("Generate AI-powered anime faces using a trained GAN model.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
@st.cache_resource
def load_generator():
    G = Generator().to(device)
    G.load_state_dict(torch.load("generator.pth", map_location=device))
    G.eval()
    return G

G = load_generator()

# User inputs
num_images = st.slider("Number of faces to generate", min_value=1, max_value=64, value=16, step=1)
latent_dim = st.number_input("Latent dimension (Z vector)", value=100)

# Button to generate
if st.button("Generate Faces"):
    z = torch.randn(num_images, latent_dim).to(device)

    with torch.no_grad():
        fake_imgs = G(z).detach().cpu()

    # Denormalize from [-1, 1] to [0, 1]
    fake_imgs = fake_imgs * 0.5 + 0.5

    # Display images
    grid = make_grid(fake_imgs, nrow=int(np.sqrt(num_images)))
    npimg = grid.permute(1, 2, 0).numpy()

    st.image(npimg, caption="Generated Anime Faces", use_column_width=True)

