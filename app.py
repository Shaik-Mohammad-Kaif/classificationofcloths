from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Disable GPU to avoid cuDNN/cuBLAS issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the Recognize_Item class
class Recognize_Item:
    def __init__(self, model_path=None):
        # Use a relative path for the model
        if model_path is None:
            self.model_path = os.path.join(os.getcwd(), 'data', 'consolidated_model.h5')
        else:
            self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            item_reco_model = load_model(self.model_path)
            return item_reco_model
        except Exception:
            raise Exception('#-----Failed to load model file-----#')

    def process_img(self, image_path):
        img = image.load_img(image_path, color_mode='rgb', target_size=(112, 112, 3))
        x = image.img_to_array(img).astype('float32')
        return x / 255.0

    def class_map(self, e):
        label_map_path = os.path.join(os.getcwd(), 'data', 'label_map.json')
        label_map = json.load(open(label_map_path))
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
        x_image = np.expand_dims(image, axis=0)
        return self.tmp_fn(self.model.predict(x_image))

# Initialize Flask app
app = Flask(__name__)
reco = Recognize_Item()

# Route to upload image and predict
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
        return render_template('index.html', predictions=predictions)
    except Exception as e:
        return render_template('index.html', error=str(e))

# Route for testing server
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
