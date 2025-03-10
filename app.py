import os
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import cv2

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MEME_FOLDER'] = 'static/memes'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MEME_FOLDER'], exist_ok=True)

# Load the CNN model
try:
    model = tf.keras.models.load_model('model/meme_cnn_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent app crash

# Emotion labels matching training set (FER-2013)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing function for image input
def preprocess_image(filepath):
    """ Prepares an image for the CNN model. """
    img = cv2.imread(filepath)
    img = cv2.resize(img, (48, 48))  # Match the trained model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def predict_emotion(filepath):
    """ Predict emotion from an uploaded image. """
    if model is None:
        return "Error: Model not loaded"

    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    
    if class_index >= len(emotions):
        return "Unknown"

    return emotions[class_index]

# Meme generation function
def create_meme(filepath, caption):
    """Overlay text on an image to create a meme."""
    img = Image.open(filepath).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_path = "arial.ttf"  # Ensure this font exists
    font = ImageFont.truetype(font_path, size=40)

    text_width, text_height = draw.textsize(caption, font=font)
    top_x = (img.width - text_width) // 2
    top_y = 10  # Padding from the top

    draw.text((top_x, top_y), caption, fill="white", font=font, stroke_width=2, stroke_fill="black")

    meme_filename = "meme_" + os.path.basename(filepath)
    meme_path = os.path.join(app.config['MEME_FOLDER'], meme_filename)
    img.save(meme_path)
    return meme_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    """ Route to upload, process the image, and return meme. """
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return {"error": "No file uploaded"}, 400

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict emotion
        emotion = predict_emotion(filepath)

        # Updated captions
        captions = {
            'happy': "Smile, it's contagious!",
            'sad': "When you see the last slice disappear...",
            'angry': "When your code doesn't compile!",
            'surprise': "Wait... what?!",
            'neutral': "Just another day!",
            'fear': "That moment you realize your project is due tomorrow.",
            'disgust': "Eww, who wrote this spaghetti code?",
        }
        caption = captions.get(emotion, "No caption available.")

        # Generate meme
        meme_filename = create_meme(filepath, caption)
        meme_url = url_for('static', filename='memes/' + meme_filename)

        return {"meme_url": meme_url, "caption": caption}

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
