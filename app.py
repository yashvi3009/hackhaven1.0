import streamlit as st
import numpy as np
import librosa
from keras.models import load_model

# Load your trained model
model = load_model('C:\\Users\\Yashvi\\Hackathon\\audio_model.h5')

# Define function for preprocessing audio files
def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)
    return mfccs_features_scaled.reshape(1, -1)

# Function to make predictions
def predict_audio(audio_file):
    # Preprocess the audio file
    audio_features = preprocess_audio(audio_file)
    # Make prediction using the loaded model
    prediction = model.predict(audio_features)
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# Streamlit app
st.title('Audio Classification App')

# Audio file upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Display audio file details
    st.audio(uploaded_file, format='audio/wav')
    # Make prediction
    predicted_class_index = predict_audio(uploaded_file)
    st.write("Predicted Class Index:", predicted_class_index)