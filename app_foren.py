import numpy as np
import streamlit as st
import os
import cv2
import datetime
from PIL import Image
import matplotlib.pyplot as plt

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

# Streamlit UI
st.title("Image Forensic Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Get Metadata of the Image
    get_metadata("temp_image.jpg")
