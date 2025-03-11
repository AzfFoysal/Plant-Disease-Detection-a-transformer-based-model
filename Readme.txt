Plant Disease Detection - README

Project Overview

This project focuses on detecting plant diseases using a Vision Transformer (ViT)-based deep learning model. The model is trained on the PlantVillage dataset and deployed as a web application using Streamlit. Users can upload an image of a plant leaf, and the model will predict the disease class with a confidence score.

Features

Uses Vision Transformer (ViT) for plant disease classification

Supports multiple plant disease categories

Web-based application for real-time disease prediction

User-friendly interface built with Streamlit

High accuracy in plant disease detection

Dependencies

To run this project, you need to install the following dependencies:

pip install streamlit tensorflow numpy opencv-python pillow

Alternatively, you can install all dependencies using the requirements.txt file:

pip install -r requirements.txt

How to Run the Project

1. Clone the Repository

git clone https://github.com/AzfFoysal/Plant-Disease-Detection-a-transformer-based-model.git

2. Load the Pre-Trained Model

Ensure you have the trained ViT model saved as plant_leaf_diseases_model.keras in the project directory. If not, you need to train the model before running the application.

3. Run the Web Application

streamlit run app.py

This will start the web application, and you can access it in your browser at http://localhost:8501/.

Usage

Open the web application.

Upload an image of a plant leaf (JPG, PNG, or JPEG format).

Click the Predict button.

The model will display the predicted disease class along with the confidence score.

Sample Prediction

Model Output:

Prediction: Potato___Early_blight
Confidence: 100.00%

Future Improvements

Extend support for more plant species.

Optimize the model for real-time inference on edge devices.

Improve UI/UX for better usability.



