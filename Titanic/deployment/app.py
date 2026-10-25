"""
Flask API for Titanic survival prediction — deploy this on Render.

Render par deploy karne ke liye:
  Root Directory: Titanic/deployment
  Build Command : pip install -r requirements.txt
  Start Command : gunicorn app:app

Routes:
  GET  /            -> browser-friendly HTML form (fill values, see prediction)
  POST /            -> same form's submit handler (renders result on the page)
  GET  /health      -> plain JSON health check (for scripts/monitoring)
  POST /predict     -> JSON API: send JSON, get JSON prediction back (curl/Postman/requests)
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

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


PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Titanic Survival Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0e1117; color:#f0f0f0;
           max-width: 480px; margin: 40px auto; padding: 0 16px; }
    h1 { font-size: 1.6em; }
    label { display:block; margin-top: 14px; margin-bottom: 4px; font-size: 0.9em; color:#ccc; }
    input, select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #444;
                    background:#1c1f26; color:#f0f0f0; box-sizing: border-box; }
    button { margin-top: 20px; padding: 10px 20px; border-radius: 6px; border: none;
             background:#e53935; color:white; font-size:1em; cursor:pointer; width: 100%; }
    button:hover { background:#c62828; }
    .result { margin-top: 24px; padding: 14px; border-radius: 8px; }
    .survived { background: #1b5e20; }
    .died { background: #4a0f0f; }
    .caption { color:#999; font-size:0.85em; margin-top: 30px; }
  </style>
</head>
<body>
  <h1>🚢 Titanic Survival Predictor</h1>
  <p class="caption">Random Forest model (~82% CV accuracy)</p>

  <form method="POST">
    <label>Passenger Class (Pclass)</label>
    <select name="Pclass">
      <option value="1" {{ 'selected' if form.Pclass=='1' else '' }}>1</option>
      <option value="2" {{ 'selected' if form.Pclass=='2' else '' }}>2</option>
      <option value="3" {{ 'selected' if form.Pclass=='3' else '' }}>3</option>
    </select>

    <label>Sex</label>
    <select name="Sex">
      <option value="male" {{ 'selected' if form.Sex=='male' else '' }}>male</option>
      <option value="female" {{ 'selected' if form.Sex=='female' else '' }}>female</option>
    </select>

    <label>Age</label>
    <input type="number" step="0.1" name="Age" value="{{ form.Age }}">

    <label>Fare</label>
    <input type="number" step="0.01" name="Fare" value="{{ form.Fare }}">

    <label>Siblings/Spouses aboard (SibSp)</label>
    <input type="number" name="SibSp" value="{{ form.SibSp }}">

    <label>Parents/Children aboard (Parch)</label>
    <input type="number" name="Parch" value="{{ form.Parch }}">

    <button type="submit">Predict</button>
  </form>

  {% if result %}
    <div class="result {{ 'survived' if result.survived else 'died' }}">
      {% if result.survived %}
        ✅ <b>Survived</b> — probability: {{ result.prob_survived }}%
      {% else %}
        ❌ <b>Did not survive</b> — probability: {{ result.prob_died }}%
      {% endif %}
    </div>
  {% elif error %}
    <div class="result died">⚠️ {{ error }}</div>
  {% endif %}
</body>
</html>
"""

DEFAULT_FORM = {"Pclass": "3", "Sex": "male", "Age": "29.7", "Fare": "32.0", "SibSp": "0", "Parch": "0"}


@app.route("/", methods=["GET", "POST"])
def home():
    form_values = DEFAULT_FORM.copy()
    result = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            X = prepare_input(form_values)
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            result = {
                "survived": bool(pred),
                "prob_survived": round(float(proba[1]) * 100, 1),
                "prob_died": round(float(proba[0]) * 100, 1),
            }
        except (KeyError, ValueError, TypeError) as e:
            error = f"Invalid input: {e}"

    return render_template_string(PAGE_TEMPLATE, form=form_values, result=result, error=error)


@app.get("/health")
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

