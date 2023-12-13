# svm_service/app.py
from flask import Flask, request, jsonify
import time
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Get the absolute path to the directory of this script
base_path = os.path.abspath(os.path.dirname(__file__))

# Load the pre-trained model using joblib
model_path = os.path.join(base_path, "svm-classification-model.h5")
model = load_model(model_path)

genre_dict={0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

# Function to retrieve the latest audio file from a directory
def get_latest_audio_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(
        directory) if f.endswith('.wav')]  # Change extension if different
    if files:
        # Get the latest file based on creation time
        return max(files, key=os.path.getctime)
    else:
        return None

def extract_features(file_path):
    # Load audio file with librosa
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Extract MFCCs and other features...
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T,axis=0)

    return mfccs_processed

def predict_genre(file_path):
    # Extract features
    features = extract_features(file_path)

    # Reshape features to fit the model input format
    features = np.reshape(features, (1, -1)) # Reshape for model

    # Predict genre
    prediction = model.predict(features)
    print("Raw Prediction:", prediction)
    # Convert prediction to genre
    predicted_genre = np.argmax(prediction)
    predicted_genre_name = genre_dict.get(predicted_genre, "Unknown Genre")
    return predicted_genre_name


@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded file from the request
    uploaded_file = request.files['musicFile']
      
    # Save the file to the shared volume
    file_path = '/Nouvarch/shared_volume/' + uploaded_file.filename
    uploaded_file.save(file_path)

    genre = predict_genre(file_path)
    result = "Predicted Genre:" + genre

    # Respond with the file name
    response_data = {"received_message": "File received successfully", "response": result}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
