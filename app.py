import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.title("Heart Disease Prediction (Teaching Demo)")
st.caption("Trained on a small synthetic dataset. Not a medical diagnosis.")

# Recreate the encoders exactly as in the notebook (same category lists)
cats = {
    "Sex": ["Male", "Female"],
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "ExerciseAngina": ["Yes", "No"],
    "FastingBloodSugar": ["High", "Normal"],
}
encoders = {k: LabelEncoder().fit(v) for k, v in cats.items()}

with open("svc_trained_model.pkl", "rb") as f:
    model = pickle.load(f)

values = {k: st.selectbox(k, v) for k, v in cats.items()}

if st.button("Predict"):
    row = pd.DataFrame({k: [encoders[k].transform([val])[0]] for k, val in values.items()})
    if model.predict(row)[0] == 1:
        st.error("Heart disease likely (teaching model only)")
    else:
        st.success("Heart disease not likely (teaching model only)")