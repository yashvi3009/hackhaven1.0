#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install resampy')


# In[3]:


get_ipython().system('pip install librosa')


# In[5]:


get_ipython().system('pip install imbalanced-learn')


# In[6]:


import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# In[7]:


audio_files_path = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\KAGGLE\\AUDIO"


# In[8]:


folders = os.listdir(audio_files_path)
print(folders)


# In[9]:


real_audio = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\DEMONSTRATION\\DEMONSTRATION\\linus-original-DEMO.mp3"
fake_audio = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\DEMONSTRATION\\DEMONSTRATION\\linus-to-musk-DEMO.mp3"


# In[11]:


real_ad, real_sr = librosa.load(real_audio)
plt.figure(figsize=(12, 4))
plt.plot(real_ad)
plt.title("Real Audio Data")
plt.show()


# In[12]:


real_spec = np.abs(librosa.stft(real_ad))
real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_spec, sr=real_sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Real Audio Spectogram")
plt.show()


# In[13]:


real_mel_spect = librosa.feature.melspectrogram(y=real_ad, sr=real_sr)
real_mel_spect = librosa.power_to_db(real_mel_spect, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_mel_spect, y_axis="mel", x_axis="time")
plt.title("Real Audio Mel Spectogram")
plt.colorbar(format="%+2.0f dB")
plt.show()


# In[14]:


real_chroma = librosa.feature.chroma_cqt(y=real_ad, sr=real_sr, bins_per_octave=36)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_chroma, sr=real_sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
plt.colorbar()
plt.title("Real Audio Chromagram")
plt.show()


# In[15]:


real_mfccs = librosa.feature.mfcc(y=real_ad, sr=real_sr)

plt.figure(figsize=(14, 5))
librosa.display.specshow(real_mfccs, sr=real_sr, x_axis="time")
plt.colorbar()
plt.title("Real Audio Mel-Frequency Cepstral Coefficients (MFCCs)")
plt.show()


# In[16]:


fake_ad, fake_sr = librosa.load(fake_audio)
plt.figure(figsize=(12, 4))
plt.plot(fake_ad)
plt.title("Fake Audio Data")
plt.show()


# In[17]:


fake_spec = np.abs(librosa.stft(fake_ad))
fake_spec = librosa.amplitude_to_db(fake_spec, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_spec, sr=fake_sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Fake Audio Spectogram")
plt.show()


# In[18]:


fake_mel_spect = librosa.feature.melspectrogram(y=fake_ad, sr=fake_sr)
fake_mel_spect = librosa.power_to_db(fake_mel_spect, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_mel_spect, y_axis="mel", x_axis="time")
plt.title("Fake Audio Mel Spectogram")
plt.colorbar(format="%+2.0f dB")
plt.show()


# In[19]:


fake_chroma = librosa.feature.chroma_cqt(y=fake_ad, sr=fake_sr, bins_per_octave=36)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_chroma, sr=fake_sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
plt.colorbar()
plt.title("Fake Audio Chromagram")
plt.show()


# In[20]:


fake_mfccs = librosa.feature.mfcc(y=fake_ad, sr=fake_sr)

plt.figure(figsize=(14, 5))
librosa.display.specshow(fake_mfccs, sr=fake_sr, x_axis="time")
plt.colorbar()
plt.title("Fake Audio Mel-Frequency Cepstral Coefficients (MFCCs)")
plt.show()


# In[21]:


#preprocess
data = []
labels = []

for folder in folders:
    files = os.listdir(os.path.join(audio_files_path, folder))
    for file in tqdm(files):
        file_path = os.path.join(audio_files_path, folder, file)
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)
        data.append(mfccs_features_scaled)
        labels.append(folder)


# In[22]:


feature_df = pd.DataFrame({"features": data, "class": labels})
feature_df.head()


# In[23]:


feature_df["class"].value_counts()


# In[24]:


def label_encoder(column):
    le = LabelEncoder().fit(column)
    print(column.name, le.classes_)
    return le.transform(column)


# In[25]:


feature_df["class"] = label_encoder(feature_df["class"])


# In[26]:


#feature scaling
X = np.array(feature_df["features"].tolist())
y = np.array(feature_df["class"].tolist())


# In[27]:


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)


# In[28]:


y_resampled = to_categorical(y_resampled)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[30]:


num_labels = len(feature_df["class"].unique())
num_labels


# In[31]:


input_shape = feature_df["features"][0].shape
input_shape


# In[39]:


#model 
model = Sequential()
model.add(Dense(128, input_shape=input_shape))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation(activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation(activation="softmax"))


# In[40]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[41]:


early = EarlyStopping(monitor="val_loss", patience=5)


# In[42]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=2, epochs=100, callbacks=[early])


# In[43]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# In[44]:


plt.figure()
plt.title("Model Accuracy")
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()


# In[45]:


plt.figure()
plt.title("Model Loss")
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()


# In[64]:


from keras.layers import Conv1D, MaxPooling1D, Flatten

# Adjust the input shape for 1D convolutional layers
input_shape = (X_train.shape[1], 1)  # Assuming X_train.shape[1] is the number of time steps in your audio data

# Define the model with 1D convolutional layers
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=2, epochs=100, callbacks=[early])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# In[69]:


plt.figure()
plt.title("Model Accuracy")
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()
plt.figure()
plt.title("Model Loss")
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.legend()
plt.ylim([0, 1])
plt.show()


# In[65]:


#test
def detect_fake(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    result_array = model.predict(mfccs_features_scaled)
    print(result_array)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    print("Result:", result_classes[result])


# In[66]:


test_real = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\DEMONSTRATION\\DEMONSTRATION\\linus-original-DEMO.mp3"
test_fake = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\DEMONSTRATION\\DEMONSTRATION\\linus-to-musk-DEMO.mp3"


# In[67]:


detect_fake(test_real)


# In[68]:


detect_fake(test_fake)


# In[72]:


from keras.models import load_model

# Step 1: Train your model and store it in the variable 'model'

# Save the model to a file using Keras save_model function
model.save("audio_model.h5")

# Step 2: Load the saved model for predictions
# Load the saved model
saved_model = load_model("audio_model.h5")


# In[73]:


from keras.models import save_model, load_model

# Step 1: Train your model and store it in the variable 'model'

# Save the model to a file using the native Keras format
save_model(model, "audio_model.keras")

# Step 2: Load the saved model for predictions
# Load the saved model
saved_model = load_model("audio_model.keras")


# In[74]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Print the file path where the model is saved
print("Model saved at:", os.path.join(current_directory, "audio_model.keras"))


# In[75]:


from keras.models import load_model

# Load the saved model
loaded_model = load_model("C:\\Users\\Yashvi\\audio_model.keras")

# Define a function to preprocess the input data
def preprocess_input_data(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    return mfccs_features_scaled

# Define a function to make predictions using the loaded model
def predict_with_loaded_model(input_data):
    result_array = loaded_model.predict(input_data)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    return result_classes[result]

# Example usage:
test_audio_file = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\KAGGLE\\AUDIO\\REAL\\musk-original.wav"
preprocessed_input = preprocess_input_data(test_audio_file)
prediction = predict_with_loaded_model(preprocessed_input)
print("Prediction:", prediction)


# In[79]:


from keras.models import load_model
import librosa
import numpy as np

# Load the saved model
loaded_model = load_model("audio_model.keras")

# Define a function to preprocess the input data
def preprocess_input_data(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    return mfccs_features_scaled

# Define a function to make predictions using the loaded model
def predict_with_loaded_model(input_data):
    result_array = loaded_model.predict(input_data)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    return result_classes[result]

# Example usage:
test_audio_file = "C:\\Users\\Yashvi\\Downloads\\audiodeepf\\KAGGLE\\AUDIO\\FAKE\\ryan-to-obama.wav"
preprocessed_input = preprocess_input_data(test_audio_file)
prediction = predict_with_loaded_model(preprocessed_input)
print("Prediction:", prediction)


# In[ ]:




