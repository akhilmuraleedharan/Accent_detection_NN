import pickle 
import streamlit as st
import warnings
import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import re
import requests
from moviepy import VideoFileClip
import tensorflow.keras 

label_encoder = LabelEncoder()


warnings.filterwarnings('ignore')

model = pickle.load(open('NN_Accent_prediction.pkl','rb'))
import requests
def download_video(url, save_path):
    headers = {
        "User-Agent": "Mozilla/5.0",  
        "Accept": "*/*",
        "Connection": "keep-alive"
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

st.title("English Accent Detection from video URL")

st.sidebar.text("This is an accent detection AI agent for English Language")
st.sidebar.text("You can paste the url in the given box and use the analyze button to check for the english accent in the video.")
st.sidebar.text("The categorization of the accent is in the following format.")
st.sidebar.text("1. English")
st.sidebar.text('2. Indian')
st.sidebar.text('3. East Asian')
st.sidebar.text('4. South East Asian')
st.sidebar.text('5. African')
st.sidebar.text('6. Middle Eastern, Central Asian or Southern Europe')
st.sidebar.text('7. Western European')
st.sidebar.text('8. Northern European')
st.sidebar.text('9. Eastern European')
st.sidebar.text('10. Oceanian or Other')

url = "https://github.com/akhilmuraleedharan/Accent_detection_NN"
st.sidebar.write("The full code for the agent is avilable in this [link](%s)" % url)
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

accent_url = st.text_input("Paste your link here")
temp_file_1 = "./video.mp4"
temp_file_2 = "./audio.mp3"
if st.button("Analyze"):
    download_video(accent_url,temp_file_1)
    cvt_video = VideoFileClip(temp_file_1)
    ext_audio = cvt_video.audio

    ext_audio.write_audiofile(temp_file_2)
    def extract_mfcc(file_path, n_mfcc=13):
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # Rata-rata untuk setiap koefisien MFCC
        return mfcc_mean
    
    mfcc_features = extract_mfcc(temp_file_2)
    mfcc_features_reshaped = mfcc_features.reshape(1, mfcc_features.shape[0], 1)
    predicted_prob = model.predict(mfcc_features_reshaped)
    predicted_class = np.argmax(predicted_prob, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    st.text(f"Predicted label for the audio file: {predicted_label[0]} with a probability of : {max(predicted_prob[0])}")

