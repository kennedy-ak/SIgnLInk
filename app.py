import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pyttsx3

# Load the model
model = tf.keras.models.load_model('sign_language_model.h5')

# Map the class indices to their corresponding labels (0-9 and A-Z)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the target size expected by the model
    image = np.array(image)
    image = image / 255.0  # Rescale pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction and return the corresponding class name
def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
    predicted_label = class_names[predicted_class]  # Map the index to the label
    return predicted_label

# Streamlit interface
st.title("Sign Language Classifier")
st.write("Upload an image of a sign language gesture to classify")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button('Classify'):
        predicted_label = make_prediction(image)
        st.write(f"Predicted Sign Language Character: {predicted_label}")

        # Optional: Text-to-Speech output
        engine = pyttsx3.init()
        engine.say(f" The Predicted Character is {predicted_label}")
        engine.runAndWait()
