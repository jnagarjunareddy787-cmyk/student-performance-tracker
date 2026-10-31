"""Flask REST API for Student Performance Tracker."""

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "app.py" else Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

FEATURES = ["age", "studytime", "absences", "failures", "goout", "freetime", "health", "Dalc"]


def _create_default_models():
    X = np.array([
        [17, 2, 5, 0, 3, 3, 3, 1],
        [16, 3, 2, 0, 2, 4, 4, 1],
        [18, 1, 8, 1, 4, 2, 2, 2],
        [15, 4, 1, 0, 2, 5, 5, 1],
        [17, 2, 9, 2, 4, 1, 2, 3],
        [16, 1, 12, 1, 5, 1, 1, 2],
        [19, 3, 3, 0, 3, 3, 4, 1],
        [18, 2, 6, 0, 2, 4, 3, 1],
        [15, 1, 15, 2, 5, 1, 2, 4],
        [17, 3, 4, 0, 3, 3, 4, 1],
    ], dtype=float)
    y_grade = np.array([15.2, 16.4, 12.1, 17.8, 11.0, 9.6, 16.6, 15.8, 8.9, 16.2], dtype=float)
    y_risk = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_model = LinearRegression()
    reg_model.fit(X_scaled, y_grade)

    clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_model.fit(X_scaled, y_risk)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_model, MODEL_DIR / "regression_model.pkl")
    joblib.dump(clf_model, MODEL_DIR / "classification_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "features.pkl")

    return reg_model, clf_model, scaler, FEATURES


def load_models():
    required = [
        MODEL_DIR / "regression_model.pkl",
        MODEL_DIR / "classification_model.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "features.pkl",
    ]

    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        print("⚠️ Model files missing or empty. Building default demo models.")
        return _create_default_models()

    try:
        reg = joblib.load(MODEL_DIR / "regression_model.pkl")
        clf = joblib.load(MODEL_DIR / "classification_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        features = joblib.load(MODEL_DIR / "features.pkl")
        return reg, clf, scaler, features
    except Exception as exc:
        print(f"⚠️ Model loading failed: {exc}. Building default demo models.")
        return _create_default_models()


reg_model, clf_model, scaler, features = load_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": reg_model is not None,
        "n_features": len(features) if features else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if reg_model is None:
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.get_json(force=True)
        input_vector = [float(data.get(field, 0)) for field in features]
        X = scaler.transform([input_vector])

        grade = float(reg_model.predict(X)[0])
        risk = int(clf_model.predict(X)[0])
        prob = float(clf_model.predict_proba(X)[0][1])

        return jsonify({
            "status": "success",
            "predicted_grade": round(grade, 2),
            "at_risk": bool(risk),
            "risk_probability": round(prob, 3),
            "recommendation": _recommend(grade, risk, prob, data),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


def _recommend(grade, risk, prob, data):
    tips = []
    if risk:
        tips.append("⚠️ Student is at risk — schedule counseling session.")
    if data.get("absences", 0) > 10:
        tips.append("📅 High absences — improve attendance.")
    if data.get("studytime", 0) < 2:
        tips.append("📚 Increase weekly study time.")
    if data.get("failures", 0) > 0:
        tips.append("🎯 Consider remedial classes for past failures.")
    if not tips:
        tips.append("✅ On track! Keep up the good work.")
    return tips


if __name__ == "__main__":
    print("🚀 Starting Student Performance Tracker API...")
    print(f"   URL: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)