import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(r"D:\Troy Msc\3rd Semester (Spring 25)\Specialized Study in CS\Project\plant_leaf_diseases_model.keras")

    return model

model = load_trained_model()

# Define class labels (ensure they match the classes your model was trained on)
class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
                'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
                'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
                'Tomato_Tomato_YellowLeaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

# Function to preprocess the image
def preprocess_image(img, target_size=(256, 256)):
    img = img.resize(target_size)  # Resize the image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Streamlit UI
st.title("🌿 Plant Disease Detection Web App")
st.write("Upload an image of a plant leaf to predict the disease.")

# Upload file
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Get prediction
        predictions = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display result
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")