# File: deploy.py

import tensorflow as tf
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/my_model/saved_model.pb')

def preprocess_input(input_data):
    # Implement any necessary preprocessing here
    # Assume input_data is a JSON object
    preprocessed_data = input_data  # Simplifying for illustration
    return preprocessed_data

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    output = {
        'prediction': prediction.tolist()
    }
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
