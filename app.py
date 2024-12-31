from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import io

# Define the Recognize_Item class
class Recognize_Item:
    def __init__(self, model_path='models/consolidated_model.h5'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            item_reco_model = load_model(self.model_path)
            return item_reco_model
        except Exception:
            raise Exception('#-----Failed to load model file-----#')

    def process_img(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((112, 112))
        x = img_to_array(img).astype('float32') / 255.0
        return x

    def process_webcam_img(self, img):
        img = img.convert("RGB").resize((112, 112))
        x = img_to_array(img).astype('float32') / 255.0
        return x

    def class_map(self, e):
        # Updated path to the label map JSON file
        label_map = json.load(open('models/label_map.json'))
        gender = label_map['gen_names'][e[0]]
        subCategory = label_map['sub_names'][e[1]]
        articleType = label_map['art_names'][e[2]]
        baseColour = label_map['col_names'][e[3]]
        season = label_map['sea_names'][e[4]]
        usage = label_map['use_names'][e[5]]
        return [gender, subCategory, articleType, baseColour, season, usage]

    def tmp_fn(self, one_hot_labels):
        flatten_labels = []
        for i in range(len(one_hot_labels)):
            flatten_labels.append(np.argmax(one_hot_labels[i], axis=-1)[0])
        return self.class_map(flatten_labels)

    def predict_all(self, image):
        x_image = np.expand_dims(image, axis=0)  # Add batch dimension
        return self.tmp_fn(self.model.predict(x_image))

# Initialize Flask app
app = Flask(__name__)
reco = Recognize_Item()

# Decode base64 image
def decode_base64_image(base64_str):
    decoded = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(decoded))
    return img

# Routes
@app.route('/')
def index():
    return render_template('web.html')  # This will load web.html on initial load

@app.route('/get-started')
def get_started():
    return render_template('index.html')  # Redirecting to index.html when the "Get Started" button is clicked

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Process and predict
    try:
        image = reco.process_img(file_path)
        predictions = reco.predict_all(image)
        os.remove(file_path)  # Clean up uploaded file
        return jsonify({'predictions': predictions})  # Return predictions as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    try:
        # Get base64 image from request
        data = request.get_json()
        base64_image = data.get("image", "")
        if not base64_image:
            return jsonify({"error": "No image data provided"}), 400

        # Decode and process the image
        img = decode_base64_image(base64_image)
        img_array = reco.process_webcam_img(img)
        predictions = reco.predict_all(img_array)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
