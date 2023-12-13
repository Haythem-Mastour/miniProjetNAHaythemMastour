from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2

app = Flask(__name__)

# Get the absolute path to the directory of this script
base_path = os.path.abspath(os.path.dirname(__file__))

# Load the pre-trained VGG16 model without the top (fully connected) layers
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Flatten the output of VGG16 and add dense layers for classification
x = vgg16_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(genre_dict), activation='softmax')(x)

# Create a new model with VGG16 as the base and the added dense layers
model = Model(inputs=vgg16_model.input, outputs=predictions)

# Load the weights of your pre-trained SVM model
model_path = os.path.join(base_path, "vgg-classification-model.h5")
model.load_weights(model_path)

genre_dict = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

def extract_features(file_path):
    # Load audio file with librosa
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Convert audio to spectrogram (image)
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spec = librosa.power_to_db(spec, ref=np.max)

    # Resize the spectrogram to match VGG16 input shape (224x224)
    spec = cv2.resize(spec, (224, 224), interpolation=cv2.INTER_CUBIC)
    spec = np.stack((spec, spec, spec), axis=-1)  # Convert to 3-channel image

    # Preprocess the input for VGG16
    spec = preprocess_input(spec)

    return spec

def predict_genre(file_path):
    # Extract features
    features = extract_features(file_path)

    # Reshape features to fit the model input format
    features = np.reshape(features, (1, 224, 224, 3))  # Reshape for VGG16

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
    result = "Predicted Genre: " + genre

    # Respond with the file name
    response_data = {"received_message": "File received successfully", "response": result}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
