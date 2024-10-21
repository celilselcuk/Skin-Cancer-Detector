from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.secret_key = 'supersecretkey' 

benign_model = load_model('ben.keras')
malignant_model = load_model('mal.keras')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config.from_mapping(UPLOAD_FOLDER=os.path.join(app.static_folder, 'uploads'))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict/<path:filename>')
def predict(filename):
    keras.backend.clear_session()    
    img_array = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    benign_pred = benign_model.predict(img_array)[0][0]
    malignant_pred = malignant_model.predict(img_array)[0][0]
    difference = abs(benign_pred - malignant_pred)
    if difference < 0.15:
        if benign_pred >= 0.70 or malignant_pred >= 0.70:
            confidence = max(benign_pred, malignant_pred) * 100
            decision = "Malignant"
        if benign_pred <= 0.30 or malignant_pred <= 0.30:
            confidence = (1 - min(benign_pred, malignant_pred)) * 100
            decision = "Benign"
        else:
            decision = "Inconclusive"
            confidence = None
    else:  
        if (benign_pred < 0.5 and malignant_pred > 0.992) or (benign_pred < 0.35 and malignant_pred < 0.89):
            final_pred = benign_pred
            decision = "Benign"
            confidence = (1 - benign_pred) * 100
        else:
            final_pred = (benign_pred + malignant_pred) / 2
            if final_pred >= 0.5:
                decision = "Malignant"
                confidence = final_pred * 100
            else:
                decision = "Benign"
                confidence = (1 - final_pred) * 100
    if confidence is not None:
        confidence = str(round(confidence)) + "%"
    
    return render_template('results.html', prediction=confidence, decision=decision, malignant_pred=malignant_pred, benign_pred=benign_pred, filename=filename)

if __name__ == '__main__':
    #public_url = ngrok.connect(name = 'flask').public_url
    #print(f'* Public ngrok url --> {public_url} *')
    app.run(debug=True)
    #app.run()
