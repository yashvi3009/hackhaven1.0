import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load the VGG16 model
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))

# Create a Sequential model
model = tf.keras.models.Sequential([
    vgg16_model,  # Add the VGG16 base model
    tf.keras.layers.Flatten(),    # Flatten layer to convert 3D feature maps to 1D
    tf.keras.layers.Dense(2, activation="softmax")  # Output layer with softmax activation for binary classification
])

# Set the VGG16 layers to be non-trainable
for layer in vgg16_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Function to preprocess the image
def preprocess_image(uploaded_file):
    # Open the uploaded file as an image
    img = Image.open(uploaded_file)
    # Resize the image to match the model input shape
    img = img.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the input image
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title('Deepfake Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        prediction = predict(uploaded_file)
        predicted_class = np.argmax(prediction[0])  # Get class index

        if predicted_class == 0:
            prediction_text = "Real"
        else:
            prediction_text = "Fake"

        st.write("Prediction:")
        st.write(prediction_text)
    except Exception as e:
        st.error(f"Error making prediction: {e}")