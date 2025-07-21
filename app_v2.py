from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model_v2.pkl"
FALLBACK_MODEL_PATH = "model.pkl"  # legacy single-feature


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), "v2"
    elif os.path.exists(FALLBACK_MODEL_PATH):
        with open(FALLBACK_MODEL_PATH, "rb") as f:
            return pickle.load(f), "v1"
    else:
        raise FileNotFoundError("No model file found. Train a model before running the app.")


model, model_version = load_model()
print(f"Loaded model version: {model_version}")


LOCATIONS = ["Vizag", "Hyderabad", "Bengaluru", "Chennai", "Mumbai"]


def format_rupees(x):
    try:
        return f"₹{int(round(x)):,}"
    except Exception:
        return "₹N/A"


@app.route("/", methods=["GET"])
def home():
    # Read and return the HTML content directly
    with open("index.html", "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])

        if model_version == "v2":
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            location = request.form["location"]
            X = [[area, bedrooms, bathrooms, location]]
        else:
            X = np.array([[area]])

        pred_price = model.predict(X)[0]
        return f"<h2>Estimated Price: {format_rupees(pred_price)}</h2><a href='/'>Back</a>"

    except Exception as e:
        return f"<h2>Error: {e}</h2><a href='/'>Back</a>"


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    try:
        area = float(data["area"])

        if model_version == "v2":
            bedrooms = int(data["bedrooms"])
            bathrooms = int(data["bathrooms"])
            location = data["location"]
            X = [[area, bedrooms, bathrooms, location]]
        else:
            X = np.array([[area]])

        pred_price = model.predict(X)[0]
        return jsonify({
            "predicted_price": int(round(pred_price)),
            "predicted_price_formatted": format_rupees(pred_price),
            "model_version": model_version
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
