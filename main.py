import numpy as np
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import librosa
from keras.models import load_model
import cv2
import os
import datetime
import matplotlib.pyplot as plt

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

# Function to perform Error Level Analysis on an image
def error_level_analysis(image_path):
    try:
        # Load the image
        image = Image.open(image_path)

        # Convert the image to a numpy array
        image_array = np.array(image)
        
        # Get the pixel values of the image
        pixel_values = image_array.flatten()

        # Calculate the error level
        error_level = np.std(pixel_values) / 10

        # Calculate the percentage of pixel error level
        pixel_error_percentage = (error_level / np.mean(pixel_values)) * 100

        # Create a histogram for displaying the error
        fig, ax = plt.subplots()
        ax.hist(pixel_values, bins=range(0, 256), color='r', alpha=0.7, label='Original Image')
        ax.axvline(np.mean(pixel_values), color='k', linestyle='dashed', linewidth=1, label='Mean Pixel Value')
        ax.axvline(np.mean(pixel_values) + error_level, color='b', linestyle='dashed', linewidth=1, label='Error Level +')
        ax.axvline(np.mean(pixel_values) - error_level, color='g', linestyle='dashed', linewidth=1, label='Error Level -')
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Pixel Value Distribution")
        ax.legend()
        st.pyplot(fig)

        # Print the error level and percentage of pixel error level
        st.write("Error level:", error_level)
        st.write("Percentage of pixel error level:", pixel_error_percentage)

    except Exception as e:
        st.write("Error:", str(e))

# Function to get metadata of an image
def get_metadata(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Get basic information
            st.write("Image Format:", img.format)
            st.write("Image Mode:", img.mode)
            st.write("Image Size:", img.size)
            st.write("MIME Type:", Image.MIME[img.format])

            # Get creation date
            creation_date = os.path.getctime(image_path)
            creation_date_readable = datetime.datetime.fromtimestamp(creation_date).strftime('%Y-%m-%d %H:%M:%S')
            st.write("Creation Date (Original):", creation_date_readable)

            # Get modification date
            modification_date = os.path.getmtime(image_path)
            modification_date_readable = datetime.datetime.fromtimestamp(modification_date).strftime('%Y-%m-%d %H:%M:%S')
            st.write("Modification Date (Digitalized):", modification_date_readable)

            # Get compression rate
            compression_rate = os.path.getsize(image_path) / (img.size[0] * img.size[1])
            st.write("Compression Rate:", compression_rate)

            # Calculate bits per pixel
            bits_per_pixel = img.bits
            st.write("Bits per Pixel:", bits_per_pixel)

            # Calculate noise level (example: using standard deviation)
            img_cv2 = cv2.imread(image_path)
            noise_level = cv2.meanStdDev(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY))[1][0][0]
            st.write("Noise Level:", noise_level)

            # Perform Error Level Analysis
            error_level_analysis(image_path)

    except Exception as e:
        st.write("Error:", str(e))

# Streamlit app
st.title('Multi-modal DeepFake Classification Web App')

# Sidebar for feature selection and additional options
st.sidebar.title("Options")
feature = st.sidebar.radio("Select Feature", ("Image Classification", "Audio Classification", "Image Forensic Analysis"))

if feature == "Image Classification":
    st.subheader("Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True, width=300)  # Adjust the width parameter for resizing
        
        with col2:
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

if feature == "Image Forensic Analysis":
    st.subheader("Image Forensic Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create columns for layout
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, width=300)  # Adjust the width parameter for resizing
        
        with col2:
            # Get Metadata of the Image
            get_metadata("temp_image.jpg")