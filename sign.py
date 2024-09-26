import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pyttsx3
import os

# Load the model
model = tf.keras.models.load_model('sign_language_model.h5')

# Map the class indices to their corresponding labels (0-9 and A-Z)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Folder where images for each class are stored (example: 'images/0.jpg' for class '0')
image_folder = 'images'

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
        engine.say(f"The predicted character is {predicted_label}")
        engine.runAndWait()

# Text input to type a letter or number
st.write("Or, type a letter or number to see its corresponding image:")
input_character = st.text_input("Enter a letter or number:")

# Check if the input is valid
if input_character:
    input_character = input_character.upper()  # Ensure uppercase for consistency
    if input_character in class_names:
        # Construct the path to the corresponding image
        image_path = os.path.join(image_folder, f"{input_character}.jpeg")
        
        # Check if the image exists
        if os.path.exists(image_path):
            # Load and display the corresponding image
            char_image = Image.open(image_path)
            st.image(char_image, caption=f'Image for {input_character}', use_column_width=True)
        else:
            st.write(f"Image for {input_character} not found.")
    else:
        st.write("Please enter a valid character (0-9 or A-Z).")



# Clear button to reset the interface
if st.button('Clear'):
    st.session_state.uploaded_image = None
    st.session_state.input_character = ""
    st.experimental_rerun()
