"""
Area→Price Prediction Flask App (no Jinja templates)

- Serves a static index.html (no template rendering).
- Accepts form submit via frontend JavaScript (POST /api/predict JSON).
- Has an optional /predict (form POST) fallback that returns a simple result page.
- Loads either model_v2.pkl (multi-feature) or model.pkl (area-only legacy).

Place this file in the SAME FOLDER as:
    index.html
    model_v2.pkl (or model.pkl)
    requirements.txt
    Procfile

Then deploy (Render start command: gunicorn app_v2:app).
"""

from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
import pandas as pd  # Added for DataFrame input
import os

# ------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
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
        raise FileNotFoundError(
            "No model file found. Please train and save model_v2.pkl or model.pkl."
        )


model, model_version = load_model()
print(f"[INFO] Loaded model version: {model_version}")

# ------------------------------------------------------------------
# Allowed locations (must match what model was trained on)
# ------------------------------------------------------------------
LOCATIONS = ["Vizag", "Hyderabad", "Bengaluru", "Chennai", "Mumbai"]


def format_rupees(x):
    try:
        return f"₹{int(round(x)):,}"
    except Exception:
        return "₹N/A"


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Serve the static HTML page."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    except FileNotFoundError:
        return (
            "<h2>Error: index.html not found in app directory.</h2>",
            500,
            {"Content-Type": "text/html"},
        )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Fallback route: handles traditional HTML form POST (no JS).
    Returns a simple HTML result page.
    """
    try:
        area = float(request.form.get("area", 0))

        if model_version == "v2":
            bedrooms = int(request.form.get("bedrooms", 0))
            bathrooms = int(request.form.get("bathrooms", 0))
            location = request.form.get("location", LOCATIONS[0])
            X = pd.DataFrame(
                [[area, bedrooms, bathrooms, location]],
                columns=["area", "bedrooms", "bathrooms", "location"]
            )
        else:
            X = np.array([[area]])

        pred_price = model.predict(X)[0]
        formatted = format_rupees(pred_price)
        html = f"""
        <html><body style="font-family:Arial;text-align:center;margin-top:50px;">
            <h2>Estimated Price: {formatted}</h2>
            <p><a href='/'>Back</a></p>
        </body></html>
        """
        return Response(html, mimetype="text/html")

    except Exception as e:
        html = f"""
        <html><body style="font-family:Arial;text-align:center;margin-top:50px;">
            <h2>Error: {e}</h2>
            <p><a href='/'>Back</a></p>
        </body></html>
        """
        return Response(html, mimetype="text/html", status=400)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API
    Expected body (multi-feature model):
    {
      "area": 1200,
      "bedrooms": 3,
      "bathrooms": 2,
      "location": "Hyderabad"
    }

    For legacy area-only model, only send "area".
    """
    data = request.get_json(force=True, silent=False)
    try:
        area = float(data["area"])

        if model_version == "v2":
            bedrooms = int(data.get("bedrooms", 0))
            bathrooms = int(data.get("bathrooms", 0))
            location = data.get("location", LOCATIONS[0])
            X = pd.DataFrame(
                [[area, bedrooms, bathrooms, location]],
                columns=["area", "bedrooms", "bathrooms", "location"]
            )
        else:
            X = np.array([[area]])

        pred_price = model.predict(X)[0]
        return jsonify(
            {
                "predicted_price": int(round(pred_price)),
                "predicted_price_formatted": format_rupees(pred_price),
                "model_version": model_version,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ------------------------------------------------------------------
# Entrypoint for local dev (Render uses gunicorn start command)
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Debug off by default in production; set DEBUG=1 to enable
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
