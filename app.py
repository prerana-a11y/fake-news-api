from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔥 SAME CLEANING AS TRAINING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text
    
# Home route (for UI)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({
                "prediction": "Error",
                "confidence": 0
            })

        cleaned = clean_text(text)

        # 🔥 FAKE KEYWORDS LIST
        fake_keywords = [
            "aliens", "ufo", "time travel", "illuminati",
            "flat earth", "secret government", "conspiracy",
            "mind control", "lizard people", "fake virus",
            "hoax", "propaganda"
        ]

        # 🔥 RULE-BASED CHECK (PRIORITY)
        if any(word in cleaned for word in fake_keywords):
            return jsonify({
                "prediction": "Fake News",
                "confidence": 95.0
            })

        # 🤖 MODEL PREDICTION
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0].max()

        result = "Real News" if prediction == 1 else "Fake News"

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "prediction": "Error",
            "confidence": 0,
            "message": str(e)
        })
# Run server
if __name__ == "__main__":
    app.run(debug=True)
