from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Absolute-safe paths (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "wine_cultivar_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["alcohol"]),
                float(request.form["malic_acid"]),
                float(request.form["total_phenols"]),
                float(request.form["flavanoids"]),
                float(request.form["color_intensity"]),
                float(request.form["proline"])
            ]

            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            prediction = f"Cultivar {pred + 1}"

        except Exception as e:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
