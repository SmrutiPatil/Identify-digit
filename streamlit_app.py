# imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image

from model import CNN

# load data
from torchvision import datasets
from torchvision.transforms import ToTensor

# config
st.set_page_config(page_title='Identify Digits',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
    initial_sidebar_state='expanded')

alt.themes.enable('fivethirtyeight')


data = datasets.MNIST(
    root="data", 
    transform=ToTensor()
)


def data_plot(data, target):
    indices = np.where(data.targets == target)[0]

    if "index" not in st.session_state:
        st.session_state.index = 0

    # Calculate the range of images to display
    start_idx = st.session_state.index * 100
    end_idx = min(start_idx + 100, len(indices))

    num_images = end_idx - start_idx
    rows = 10
    cols = 10
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i in range(num_images):
        image = data.data[
            indices[start_idx + i]
        ].numpy()  # Convert tensor to numpy array
        label = data.targets[
            indices[start_idx + i]
        ].item()  # Get the label as a Python scalar

        ax = axes[i // cols, i % cols]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label}")

    # Hide any unused subplots
    for j in range(num_images, rows * cols):
        ax = axes[j // cols, j % cols]
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # Button to show the next set of images
    if st.button("Next"):
        st.session_state.index += 1


def identify_digit():
    st.title('Identify Digits')
    number = st.slider('Select a number', 0, 9, 0)
    data_plot(data, number)

def draw():
    st.title('Handwritten Digits')
    canvas_result = st_canvas(
        stroke_color="yellow",
        stroke_width=3,
        height=100,
        width=100,
        drawing_mode="freedraw",
        key="draw_and_identify",   
    )
    
    
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    
        model = torch.load("cnn.pth", map_location=torch.device('cpu'))
        model.eval()
        
        # Preprocess the image
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        image = np.array(image)
        image = 255 - image  # Invert image colors (if needed)
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        
        if st.button("Identify"):
            st.write("Identifying...")
            with torch.no_grad():
                output = model(image)
            number = output[0].argmax().item()
            st.write("The model identified the digit as:", number)
    else:
        st.write("Draw a digit in the canvas above and click 'Identify' to see the result.")

def about():
    st.title('About')
    st.write('This is a simple web app that allows you to identify digits from the MNIST dataset.')
    st.write('The MNIST dataset contains 70,000 images of handwritten digits (0-9) and is commonly used for training image processing models.')
    st.write('This app allows you to select a digit and view a sample of images from the dataset.')
    st.write('You can also draw your own digit and use the model to identify it.')
    st.write('This app was created using Streamlit, a Python library for building web apps.')


def main():
    PAGES = {
        "About": about,
        "Identify Digits": identify_digit,
        "Draw": draw
    }
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", options=list(PAGES.keys()))
    PAGES[page]()

    st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
