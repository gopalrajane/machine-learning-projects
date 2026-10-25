"""
Streamlit app for Titanic survival prediction — deploy this on Streamlit Cloud.

Run locally:
    streamlit run streamlit_app.py

Streamlit Cloud par deploy karte time, repo me yeh file + titanic_model.pkl +
requirements.txt same folder me hone chahiye.
"""

import os
import pickle
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model.pkl")


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
AGE_MEAN = bundle["age_mean"]
SEX_MAPPING = bundle["sex_mapping"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.caption("Random Forest model trained on the Titanic dataset (~82% CV accuracy)")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input(
            "Age", min_value=0.0, max_value=100.0, value=round(AGE_MEAN, 1), step=1.0
        )

    with col2:
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    family = sibsp + parch + 1
    row = {
        "Pclass": pclass,
        "Sex": SEX_MAPPING[sex],
        "Age": age,
        "Fare": fare,
        "Family": family,
    }
    X = pd.DataFrame([row], columns=FEATURES)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    if pred == 1:
        st.success(f"✅ Survived — probability: {proba[1]*100:.1f}%")
    else:
        st.error(f"❌ Did not survive — probability: {proba[0]*100:.1f}%")

    st.progress(float(proba[1]))
    st.caption(f"P(survive) = {proba[1]:.3f} | P(not survive) = {proba[0]:.3f}")

st.divider()
st.caption("Model: RandomForestClassifier · Features used: Pclass, Sex, Age, Fare, Family")
