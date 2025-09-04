import base64
import io
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from game import HangmanGame

# ------------------- TensorFlow tweaks -------------------
# Limit GPU/CPU memory growth to prevent worker timeout
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.get_logger().setLevel('ERROR')  # Reduce TF logging overhead
# ----------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load EMNIST model once
model = load_model("emnist_model.h5")
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Start a Hangman game
game = HangmanGame()

def add_lives_info(state):
    state["lives"] = state["attempts_left"]
    state["max_lives"] = game.max_attempts
    return state

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"].split(",")[1]

    # Process image
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    img_array = np.array(image) / 255.0

    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    coords = np.argwhere(img_array > 0.05)
    if coords.size != 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img_array = img_array[y0:y1, x0:x1]
    else:
        img_array = np.zeros((20, 20))

    img_pil = Image.fromarray((img_array * 255).astype(np.uint8)).resize((20, 20))
    img_array = np.array(img_pil) / 255.0
    padded_img = np.zeros((28, 28))
    padded_img[4:24, 4:24] = img_array
    img_array = padded_img.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array, verbose=0)  # suppress TF messages
    predicted_letter = labels[np.argmax(prediction)]

    game_state = add_lives_info(game.get_game_state())
    return jsonify({"prediction": predicted_letter, "game_state": game_state})

@app.route("/confirm", methods=["POST"])
def confirm_letter():
    data = request.get_json()
    letter = data["letter"]
    game_state = add_lives_info(game.guess_letter(letter.lower()))
    return jsonify({"game_state": game_state})

@app.route("/reset", methods=["POST"])
def reset_game():
    game.reset_game()
    game_state = add_lives_info(game.get_game_state())
    return jsonify({"game_state": game_state})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)