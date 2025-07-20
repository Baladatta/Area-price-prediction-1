from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Debugging: print paths (helps detect template issues on Render)
print("=== DEBUG: Flask root path:", app.root_path)
print("=== DEBUG: Template folder:", app.template_folder)
try:
    print("=== DEBUG: Root contents:", os.listdir(app.root_path))
    tpl_path = os.path.join(app.root_path, "templates")
    if os.path.exists(tpl_path):
        print("=== DEBUG: templates/ contents:", os.listdir(tpl_path))
    else:
        print("=== DEBUG: templates/ folder not found!")
except Exception as e:
    print("=== DEBUG ERROR listing templates:", e)

# Model paths
MODEL_PATH = "model_v2.pkl"
FALLBACK_MODEL_PATH = "model.pkl"  # legacy single-feature model

# Load the model
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

# Dropdown values
LOCATIONS = ["Vizag", "Hyderabad", "Bengaluru", "Chennai", "Mumbai"]

def format_rupees(x):
    try:
        return f"₹{int(round(x)):,}"
    except Exception:
        return "₹N/A"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", locations=LOCATIONS)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission from index.html.
    Works for both v2 (multi-feature) and v1 (area-only) models.
    """
    try:
        area = float(request.form["area"])

        if model_version == "v2":
            bedrooms = int(request.form.get("bedrooms", 2))
            bathrooms = int(request.form.get("bathrooms", 2))
            location = request.form.get("location", "Vizag")
            X = [[area, bedrooms, bathrooms, location]]
        else:
            X = np.array([[area]])

        pred_price = model.predict(X)[0]
        message = f"Estimated Price: {format_rupees(pred_price)}"
        return render_template("index.html", prediction_text=message, locations=LOCATIONS)
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {e}",
                               locations=LOCATIONS)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API
    Example:
    {
      "area": 1200,
      "bedrooms": 3,
      "bathrooms": 2,
      "location": "Hyderabad"
    }
    """
    data = request.get_json(force=True)
    try:
        area = float(data["area"])

        if model_version == "v2":
            bedrooms = int(data.get("bedrooms", 2))
            bathrooms = int(data.get("bathrooms", 2))
            location = data.get("location", "Vizag")
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
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
