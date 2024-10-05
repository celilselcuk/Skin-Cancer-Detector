from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.secret_key = 'supersecretkey' 

# Load both models
benign_model = load_model('ben.keras')  # Model good at predicting benign
malignant_model = load_model('mal.keras')  # Model good at predicting malignant

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config.from_mapping(
    UPLOAD_FOLDER=os.path.join(app.static_folder, 'uploads')
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('//')
def go_back():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')


# Preprocess function to prepare the image for both models
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Match model input size
    img = img_to_array(img)
    img = img.astype('float32')  # Ensure float32 type
    img = img / 255.0  # Normalize like in training
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict/<path:filename>')
def predict(filename):
    keras.backend.clear_session()  # Clear session to avoid memory issues
    
    # Preprocess the image
    img_array = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Get predictions from both models
    benign_pred = benign_model.predict(img_array)[0][0]
    malignant_pred = malignant_model.predict(img_array)[0][0]
    print(f'Benign : {benign_pred}')
    print(f'Malignant: {malignant_pred}')
    # Logic for conditional prediction
    if abs(benign_pred - malignant_pred) < 0.2:
       decision = "Inconclusive"
       confidence = None
    else:  
        if (benign_pred < 0.5 and malignant_pred > 0.992) or (benign_pred < 0.35 and malignant_pred < 0.85):
        # Use benign model's prediction directly if its confidence is high and malignant model's is low
            final_pred = benign_pred
            decision = "Benign"
            confidence = (1 - benign_pred) * 100
        else:
        # Average the predictions otherwise
            final_pred = (benign_pred + malignant_pred) / 2
            if final_pred >= 0.5:
                decision = "Malignant"
                confidence = final_pred * 100
            else:
                decision = "Benign"
                confidence = (1 - final_pred) * 100

    if confidence is not None:
        confidence = str(round(confidence)) + "%"
    return render_template('results.html', prediction=confidence, decision=decision, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
