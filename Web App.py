from flask import Flask, flash, request, redirect, url_for, render_template
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model

# Loading the trained model
lung_cancer_model = load_model('models/trained_lung_cancer_model.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

IMAGE_SIZE = (350, 350)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/lung cancer')
def brain_tumor():
    return render_template('lung cancer.html')


def load_and_preprocess_image(img, target_size):
    if isinstance(img, np.ndarray):
        img = np.resize(img, (target_size[0], target_size[1], 3))
    else:
        img = image.load_img(img, target_size=target_size)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    img = cv2.imread('static/uploads/' + filename)

    firstname = request.form['firstname']
    lastname = request.form['lastname']
    email = request.form['email']
    phone = request.form['phone']
    gender = request.form['gender']
    age = request.form['age']

    try:
        img = load_and_preprocess_image(file_path, IMAGE_SIZE)
        preds = lung_cancer_model.predict(img)
        predicted_class = np.argmax(preds[0])

        # You can modify this logic based on your model's output
        if predicted_class == 2:
            pred = 0  # Class 2 indicates 'normal' in your case
        else:
            pred = 1  # Other classes indicate 'tumor'

        return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred,
                               gender=gender)

    except Exception as e:
        flash(f'Error occurred: {str(e)}')
        return redirect(request.url)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
