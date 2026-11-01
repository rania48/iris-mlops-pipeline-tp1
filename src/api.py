from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
import joblib
import json
import math

app = Flask(__name__)

model = joblib.load("models/iris_model.pkl")

with open("models/model_info.json", encoding="utf-8") as f:
    model_info = json.load(f)

class_names = load_iris().target_names


def features_valides(features):
    """Vérifie qu'on a bien 4 nombres finis."""
    if not isinstance(features, list) or len(features) != 4:
        return False

    for valeur in features:
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            return False
        if not math.isfinite(valeur):
            return False

    return True


@app.get("/")
def home():
    return jsonify({"service": "Iris MLOps API", "status": "running"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.get("/version")
def version():
    return jsonify(model_info)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not features_valides(data.get("features")):
        return jsonify({"error": "'features' doit contenir 4 nombres"}), 400

    prediction = int(model.predict([data["features"]])[0])

    return jsonify({
        "prediction": prediction,
        "class_name": str(class_names[prediction]),
        "model_version": model_info["version"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
