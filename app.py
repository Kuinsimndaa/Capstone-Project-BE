from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os
import sqlite3

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://stress-chat-detector.vercel.app", "methods": ["POST", "OPTIONS"], "allow_headers": "*"}}, supports_credentials=True)

# Load model dan tokenizer dengan path absolut
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model_lstm_stress.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'tokenizer_stress.pkl')

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100  # sesuai saat training

DB_PATH = os.path.join(os.path.dirname(__file__), 'stresslog.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        prediction TEXT NOT NULL,
        stress_percent REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text input is required"}), 400

        cleaned_text = preprocess(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len)

        prob = float(model.predict(padded)[0][0])
        prediction = "Negative" if prob < 0.5 else "Positive"
        stress_percent = float(round((1 - prob) * 100, 2))

        # Simpan ke database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO predictions (text, prediction, stress_percent) VALUES (?, ?, ?)',
                  (text, prediction, stress_percent))
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": prediction,
            "stress_percent": stress_percent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/logs", methods=["GET"])
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, text, prediction, stress_percent, created_at FROM predictions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    logs = [
        {
            "id": row[0],
            "text": row[1],
            "prediction": row[2],
            "stress_percent": row[3],
            "created_at": row[4]
        } for row in rows
    ]
    return jsonify(logs)

@app.route("/logs/<int:log_id>", methods=["DELETE"])
def delete_log(log_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE id = ?', (log_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted", "id": log_id})

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://stress-chat-detector.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Gunakan PORT dari Railway
    app.run(debug=False, host="0.0.0.0", port=port)