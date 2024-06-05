import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import librosa
from keras.models import load_model
import cv2

# Custom CSS to style the sidebar and main content
st.markdown(
    """
    <style>
    .main {
        background-color: #87cefa;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to top, #ffffff, #e6e6e6);
        color: black;
    }
    .sidebar .sidebar-content h2 {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the VGG16 model for image classification
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

# Load the audio classification model
model_audio = load_model('C:\\Users\\Yashvi\\Hackathon\\audio_model.h5')

# Load the image classification model
model_image = load_model('C:\\Users\\Yashvi\\Hackathon\\image_model.h5')

# Function to preprocess the image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess audio files
def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)
    return mfccs_features_scaled.reshape(1, -1)

# Function to make image predictions
def predict_image(image_array):
    prediction = model_image.predict(image_array)
    return prediction

# Function to make audio predictions
def predict_audio(audio_file):
    audio_features = preprocess_audio(audio_file)
    prediction = model_audio.predict(audio_features)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# Streamlit app
st.title('Multi-modal Classification App')

# Sidebar for feature selection and additional options
st.sidebar.title("Options")
feature = st.sidebar.radio("Select Feature", ("Image Classification", "Audio Classification"))

if feature == "Image Classification":
    st.subheader("Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        try:
            img_array = preprocess_image(uploaded_file)
            prediction = predict_image(img_array)
            predicted_class = np.argmax(prediction[0])
            if predicted_class == 0:
                prediction_text = "Real"
            else:
                prediction_text = "Fake"
            st.write("Prediction:")
            st.markdown(f"<h3 style='color:#333333;'>{prediction_text}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if feature == "Audio Classification":
    st.subheader("Audio Classification")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        predicted_class_index = predict_audio(uploaded_file)
        st.write("Predicted Class Index:", predicted_class_index)
