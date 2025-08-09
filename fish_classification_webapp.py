# fish_app.py
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.models import load_model
import plotly.express as px

import numpy as np
from PIL import Image
import gdown
import os

# --------- LOAD MODEL ----------

file_id = "1BVjbVtjGkORGiqCZkYJA-hts5hj3ulUB"
output_path = "model.keras"

# Download model from Google Drive
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the model
model = load_model(output_path)

# Class names 
class_names = ['fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']

# --------- STREAMLIT UI ----------
st.title("🐟 Fish Image Classifier")
st.write("Upload a fish image and to classify it into one of the categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # same as during training


    # Let user select the true label from a dropdown
    true_label = st.selectbox("Select the true label (for comparison):", class_names)

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # Output both predicted and true labels
    st.markdown(f"### Predicted Label: **{class_names[pred_class]}**")
    st.markdown(f"### True Label: **{true_label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    # Create DataFrame for probabilities
    probs = preds[0]
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probs
    }).sort_values(by="Probability", ascending=False)

    # Plot with Plotly Express
    fig = px.bar(
        df,
        x='Class',
        y='Probability',
        text=df['Probability'].apply(lambda x: f"{x:.2%}"),
        title="Class Probabilities",
        labels={'Probability': 'Probability', 'Class': 'Fish Class'},
        color='Probability',
        color_continuous_scale='Blues'
    )

    fig.update_traces(textposition='outside')
    fig.update_yaxes(range=[0, 1])  
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)