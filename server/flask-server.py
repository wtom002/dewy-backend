from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)


try:
    model = tf.keras.models.load_model('models/my_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  
    return image

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .jpg, .jpeg, .png, and .bmp are allowed.'}), 400
    
    try:
        image = Image.open(file.stream)
        image.verify()  #verify image
        file.stream.seek(0) 
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        response = {'predictions': predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    if model:
        app.run(debug=True)
    else:
        print("Model could not be loaded. Exiting.")
