import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking  # Added Masking import
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your CSV dataset
dataset = pd.read_csv('features.csv')

# Extract features and labels from the dataset
X = dataset.drop(columns=['labels']).values  # Assuming 'labels' is the column to predict
y = dataset['labels'].values

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape the input data to include time_steps dimension
num_features = X_train.shape[1]  # Number of numerical features
time_steps = 1  # Assuming each sample represents a single time step

X_train_reshaped = X_train.reshape(X_train.shape[0], time_steps, num_features)
X_test_reshaped = X_test.reshape(X_test.shape[0], time_steps, num_features)

# Build the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(time_steps, num_features)))  # Added Masking layer
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=128)))
model.add(Dropout(0.2))
model.add(Dense(units=len(np.unique(y_encoded)), activation='softmax'))  # Output units set to number of unique labels

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Save the model
model.save('custom_speech_recognition_model.h5')

# Save the label encoder for later use
with open('your_label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
