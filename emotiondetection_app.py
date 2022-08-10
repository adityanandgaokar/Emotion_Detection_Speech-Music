
import streamlit as st
import os
import io
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import operator
import librosa
import joblib
# path to save wav files in the local machine
wav_files_path = "C:/Users/AdityaNandgaokar/Aimpower/Main/Emotion_Recognition/uploaded_wav_files_app/"

# trained model path
model_path = 'C:/Users/AdityaNandgaokar/Aimpower/Main/Emotion_Recognition/Models/CNN/model_CNN.h5'

# path to save audio files splitted by spleeter
output_path = 'output/'

# scaled data path
ss = joblib.load('C:/Users/AdityaNandgaokar/Aimpower/Main/Emotion_Recognition/scaled_train_data.bin')


def feature_extractor_audio_file(audio, sample_rate, mfcc, chroma, mel):
    
    if chroma:
        stft = np.abs(librosa.stft(audio))
        result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio,
                        sr= sample_rate, n_mfcc = 40).T, axis = 0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio,
                         sr = sample_rate).T, axis = 0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(audio, 
                      sr=sample_rate).T, axis = 0)
        result = np.hstack((result, mel))
        
    return result 

def spleeter(path_audio_file, output_path):

    file_name = os.path.basename(os.path.normpath(path_audio_file))
    file_name = os.path.splitext(file_name)[0]

    print(path_audio_file)
    cmd = 'spleeter separate -p spleeter:2stems -o output/ ' +  str(path_audio_file)
    print(cmd) 
    os.system(cmd)
    directory = os.getcwd()
    audio_files = os.path.join(directory, output_path, file_name)
    music_file = os.path.join(audio_files, 'accompaniment.wav')
    speech_file = os.path.join(audio_files, 'vocals.wav')
                
    return music_file, speech_file

wav_files = []
st.title('Emotion Detection (wav)')
uploaded_files = st.file_uploader('upload two or more wav files', type=['wav'], accept_multiple_files=True)


run = st.button('run emotion detection')

if run:
    if uploaded_files is not None:
        for i in range(len(uploaded_files)):
            name = uploaded_files[i].name
            name = name.replace(' ', '')
            wav_file_path = os.path.join(wav_files_path, name)
            g = io.BytesIO(uploaded_files[i].read())
            ########## SAVE ADUIO BYTES
            with open(wav_file_path, 'wb') as out:
                audio_bytes = g.read()
                out.write(audio_bytes)   
            wav_files.append(wav_file_path)
            print(wav_files)

        if wav_files is not None:

            loaded_model = load_model(model_path)
            emotions = ['angry', 'calm', 'bore', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

            label_encoder = LabelEncoder()
            encoded_emotions = label_encoder.fit_transform(emotions)

            
            feature_test = []
            test_files = []

            for wav_file in wav_files:
                test_files.append(wav_file)
                print(wav_file)
                music_file, speech_file = spleeter(wav_file, output_path)
                audio, sample_rate = librosa.core.load(speech_file, res_type='kaiser_fast')
                feature = feature_extractor_audio_file(audio, sample_rate, mfcc = True, chroma = True, mel=True)
                feature_test.append(feature)

            test_arr = np.array(feature_test)
            scaled_test_feature = ss.transform(test_arr)
            final_feature = np.expand_dims(scaled_test_feature, -1)
            with st.spinner("detecting emotion"): 
                pred = loaded_model.predict(final_feature)

                for i in range(len(pred)):
                    
                    label = np.argmax(pred[i], axis=None, out=None)
                    emotion = label_encoder.inverse_transform([label])
                    
                    print('Predicted Emotion is {}'. format(emotion[0]))
                    
                    prediction = pred[i] * 100
                    print(prediction)
                    prediction_dict = {}
                    
                    for j in range(len(prediction)):
                        value = ("%.17f" % prediction[j]).rstrip('0').rstrip('.')
                        key_emotion = label_encoder.inverse_transform([j])

                        prediction_dict[key_emotion[0]] = value
                    
                    sorted_pred_dict = dict(sorted(prediction_dict.items(),
                                                key=operator.itemgetter(1),
                                                reverse=True))
                    folder_name = os.path.basename(os.path.normpath(wav_file_path))
                    file_name = os.path.basename(os.path.normpath(test_files[i]))
                    st.write('Predicted Emotion for file {} is {}'. format(file_name, emotion[0]))
                    st.write('Confidence For All Emotions')
                   
                    for k in sorted_pred_dict:
                        st.write('{} = {}'.format(k, sorted_pred_dict[k]))  
                    st.markdown('*************************************************************************************************')
        




    else:
        st.write("could not extract emotion from wav file")