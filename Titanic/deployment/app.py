"""
Flask API for Titanic survival prediction — deploy this on Render.

Render par deploy karne ke liye:
  Start Command: gunicorn app:app
  (requirements.txt me flask, gunicorn, scikit-learn, pandas, numpy already hain)

Endpoints:
  GET  /            -> health check
  POST /predict     -> JSON body se prediction deta hai
"""

import pickle
import os
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "titanic_model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES = bundle["features"]          # ['Pclass', 'Sex', 'Age', 'Fare', 'Family']
AGE_MEAN = bundle["age_mean"]
SEX_MAPPING = bundle["sex_mapping"]    # {'male': 0, 'female': 1}


def prepare_input(payload: dict) -> pd.DataFrame:
    """Raw user-friendly input ko model ke expected features me convert karta hai."""
    pclass = int(payload["Pclass"])
    fare = float(payload["Fare"])

    # Sex: accept either 'male'/'female' or already 0/1
    sex_raw = payload["Sex"]
    if isinstance(sex_raw, str):
        sex = SEX_MAPPING.get(sex_raw.strip().lower())
        if sex is None:
            raise ValueError("Sex must be 'male' or 'female'")
    else:
        sex = int(sex_raw)

    # Age: optional, missing -> training-set mean (same as notebook)
    age = payload.get("Age")
    age = AGE_MEAN if age in (None, "", "null") else float(age)

    # Family: either provided directly, or derived from SibSp + Parch (+1 for self)
    if "Family" in payload and payload["Family"] not in (None, ""):
        family = int(payload["Family"])
    else:
        sibsp = int(payload.get("SibSp", 0))
        parch = int(payload.get("Parch", 0))
        family = sibsp + parch + 1

    row = {"Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare, "Family": family}
    return pd.DataFrame([row], columns=FEATURES)


@app.get("/")
def health():
    return jsonify({"status": "ok", "message": "Titanic survival model is live."})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        X = prepare_input(payload)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return jsonify({
        "survived": bool(pred),
        "prediction": pred,
        "probability_survived": round(float(proba[1]), 4),
        "probability_not_survived": round(float(proba[0]), 4),
    })


if __name__ == "__main__":
    # Local testing only; Render will use gunicorn instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
