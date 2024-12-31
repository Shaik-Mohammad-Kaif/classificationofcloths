from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import io
import json
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load model and label map once at startup
MODEL_PATH = 'models/consolidated_model.h5'
LABEL_MAP_PATH = 'models/label_map.json'

try:
    MODEL = load_model(MODEL_PATH)
    LABEL_MAP = json.load(open(LABEL_MAP_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load model or label map: {e}")

# Prediction helper class
class Recognize_Item:
    @staticmethod
    def process_img(image):
        img = image.convert("RGB").resize((112, 112))
        x = img_to_array(img).astype('float32') / 255.0
        return x

    @staticmethod
    def class_map(indices):
        gender = LABEL_MAP['gen_names'][indices[0]]
        subCategory = LABEL_MAP['sub_names'][indices[1]]
        articleType = LABEL_MAP['art_names'][indices[2]]
        baseColour = LABEL_MAP['col_names'][indices[3]]
        season = LABEL_MAP['sea_names'][indices[4]]
        usage = LABEL_MAP['use_names'][indices[5]]
        return [gender, subCategory, articleType, baseColour, season, usage]

    @staticmethod
    def tmp_fn(predictions):
        indices = [np.argmax(pred, axis=-1)[0] for pred in predictions]
        return Recognize_Item.class_map(indices)

    @staticmethod
    def predict_all(image_array):
        x_image = np.expand_dims(image_array, axis=0)  # Add batch dimension
        predictions = MODEL.predict(x_image)
        return Recognize_Item.tmp_fn(predictions)

# Decode base64 image
def decode_base64_image(base64_str):
    decoded = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(decoded))

# Routes
@app.route('/')
def index():
    return render_template('web.html')

@app.route('/get-started')
def get_started():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save and process uploaded file securely
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', secure_filename(file.filename))
    file.save(file_path)

    try:
        image = Image.open(file_path)
        processed_image = Recognize_Item.process_img(image)
        predictions = Recognize_Item.predict_all(processed_image)
    finally:
        os.remove(file_path)  # Clean up uploaded file

    return jsonify({'predictions': predictions})

@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    data = request.get_json()
    base64_image = data.get("image", "")
    if not base64_image:
        return jsonify({"error": "No image data provided"}), 400

    try:
        img = decode_base64_image(base64_image)
        processed_image = Recognize_Item.process_img(img)
        predictions = Recognize_Item.predict_all(processed_image)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
