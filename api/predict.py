from flask import Flask, request, jsonify
from preprocessing import preprocess
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# WSGI Flask App
app = Flask(__name__)

# Load model dan tokenizer
model = load_model(os.path.join("..", "model", "model_lstm_stress.h5"))
with open(os.path.join("..", "model", "tokenizer_stress.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text input is required"}), 400

        # Preprocess input
        cleaned_text = preprocess(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len)

        # Predict
        prob = float(model.predict(padded)[0][0])
        prediction = "Negative" if prob < 0.5 else "Positive"
        stress_percent = float(round((1 - prob) * 100, 2))

        return jsonify({
            "prediction": prediction,
            "stress_percent": stress_percent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel handler
def handler(request):
    with app.app_context():
        return app.full_dispatch_request()
