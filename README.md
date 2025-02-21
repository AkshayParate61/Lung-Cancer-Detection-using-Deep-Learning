# Lung Cancer Detection using Deep Learning

This project uses deep learning to detect lung cancer from medical images such as chest X-rays and CT scans. The application leverages a trained model to classify images and provide users with a reliable prediction.

## **Download Data Set From Kaggle**

Link - https://www.kaggle.com/datasets/khalidelmaria/ct-lung-cancer-data-augmentation

## 🧑‍⚕️ **Introduction**

Lung cancer is one of the deadliest types of cancer worldwide, and early detection can save lives. This project aims to create a web application that can detect lung cancer using a deep learning model trained on medical images. The application is built using Flask, Keras, and TensorFlow.

## 📊 **Technologies Used**

- **Flask**: A lightweight web framework for Python to build the web application.
- **Keras**: A high-level neural networks API for deep learning in Python.
- **TensorFlow**: An open-source machine learning library used for training and deploying deep learning models.
- **OpenCV**: A computer vision library used to process images.
- **NumPy**: A library for numerical computations.

## 🚀 **Features**

- **User-friendly Interface**: A simple web form where users can upload medical images (JPG, PNG, JPEG) to detect lung cancer.
- **Model Integration**: Uses a pre-trained deep learning model to classify whether the uploaded image indicates a lung tumor or not.
- **Result Display**: Once the prediction is made, the result (cancer or no cancer) is shown to the user with relevant details.
- **Error Handling**: Gracefully handles errors such as unsupported file formats or failed predictions.

## ⚙️ **Setup Instructions**

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended) and install the required libraries.

```bash
# Create virtual environment (optional)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` file should include the following:
```
Flask
tensorflow
keras
numpy
opencv-python
werkzeug
```

### 3. Download Pre-trained Model
Make sure you have the pre-trained lung cancer model (`trained_lung_cancer_model.h5`) stored in a `models/` folder. You can train this model on your own or find a pre-trained model online. The model file should be placed in the following directory:
```
/models/trained_lung_cancer_model.h5
```

### 4. Run the Application
Once you've set up the environment, you can start the Flask application.

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to access the application.

## 📸 **How it Works**

1. The user uploads a medical image (X-ray or CT scan) through the web form.
2. The image is processed by OpenCV and resized to the appropriate dimensions for the model.
3. The processed image is passed through the pre-trained lung cancer model for classification.
4. Based on the model's output, the application predicts whether the image indicates the presence of lung cancer.
5. The result is displayed to the user along with basic details (e.g., age, gender).

## 🧠 **Model Explanation**

The deep learning model used in this project was trained on medical imaging data, and it classifies images into two categories:

- **Normal (0)**: No signs of lung cancer detected.
- **Tumor (1)**: Tumor detected in the lungs.

The model's predictions are returned as probabilities, and the class with the highest probability is chosen as the result.

## 🖼️ **Web Interface**

The application has two main pages:

1. **Homepage** (`/`): Displays the introduction to the application and provides the option to upload an image.
2. **Lung Cancer Detection** (`/lung cancer`): This page allows the user to submit a medical image and other details (e.g., name, age, gender, etc.). It also displays the result after the image is processed.

### Example of uploading an image:

```html
<form action='/resultbt' method="POST" enctype="multipart/form-data">
    <input type="file" name="file" class="form-control" required>
    <button type="submit">Submit</button>
</form>
```

After submission, the result page will show whether the image indicates a lung tumor or not.

## 🧑‍💻 **Code Explanation**

- **Flask Web Application**: The `Flask` framework is used to build the web application. It includes routes for the homepage, lung cancer detection page, and result page.
- **Image Processing**: The `load_and_preprocess_image` function resizes and normalizes the image before passing it to the model.
- **Model Prediction**: The model (`lung_cancer_model`) is loaded and used to predict the presence of lung cancer.
- **Error Handling**: The application handles missing or invalid files gracefully, providing feedback to the 



