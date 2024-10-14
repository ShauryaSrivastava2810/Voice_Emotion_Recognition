import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Masking
from tensorflow.keras.optimizers import Adam

# Function to extract features from audio using librosa
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)  # Set n_mfcc to 20
    return mfccs.T  # Transpose to have time_steps first

# Function to convert speech to text
def speech_to_text(file_path, model, label_encoder):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)
    # Decode the predicted label using label_encoder if applicable
    predicted_text = label_encoder.inverse_transform([predicted_label])[0]
    # Split the predicted text into individual words
    words = predicted_text.split()
    # Print each word separately
    for word in words:
        print("Predicted Emotions:", word)
    return predicted_text

# Load your trained LSTM model
# Replace 'your_model.h5' with the actual path to your model file
model = tf.keras.models.load_model('custom_speech_recognition_model.h5')

# Load your label encoder
# Replace 'your_label_encoder.pkl' with the actual path to your label encoder file
with open('your_label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Replace 'your_audio_file.wav' with the actual path to your audio file
audio_file_path = 'sound1.wav'
predicted_text = speech_to_text(audio_file_path, model, label_encoder)

print("Predicted Emotions:", predicted_text)
