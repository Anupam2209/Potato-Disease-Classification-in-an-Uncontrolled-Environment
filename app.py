import sys
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow import keras

if sys.stdout.encoding != 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='UTF-8', buffering=1)

app = Flask(__name__, static_url_path='/static')

# Load your trained model
model = keras.models.load_model("training\model_best_F1score.h5")

# Define class labels 
class_labels = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Phytophthora', 'Pest', 'Virus']

# Define route to serve HTML page
@app.route('/')
def index():
    return render_template('index.html', prediction=None, error=None)

# Define route to handle API requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    
    # Open and preprocess the image
    image = Image.open(file)
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0    # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction using your model
    prediction = model.predict(image)
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index])
    
    # Prepare response
    response = {'predicted_class': predicted_class, 'confidence': confidence}
    
    # Return the prediction as JSON
    return render_template('index.html', prediction=response, error=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
