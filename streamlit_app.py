from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import joblib
import streamlit as st

from database.db import insert_candidate
from src.features.preprocess import clean_resume_text


MODEL_PATH = Path("models/resume_classifier.pkl")
TFIDF_PATH = Path("models/tfidf.pkl")


def require_module(module_name: str, package_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        st.error(
            f"{package_name} is not available in the Python environment running Streamlit.\n\n"
            f"Current interpreter: `{sys.executable}`\n\n"
            f"Start the app with your project environment, for example:\n"
            f"`venv\\Scripts\\streamlit.exe run streamlit_app.py`"
        )
        st.stop()


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf
    except Exception as exc:
        st.error(
            "Model files are missing or invalid. Run `python -m src.models.train_model` first."
        )
        st.stop()


def extract_text(uploaded_file) -> str:
    text = ""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdfplumber = require_module("pdfplumber", "pdfplumber")
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file_name.endswith(".docx"):
        docx = require_module("docx", "python-docx")
        document = docx.Document(uploaded_file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    else:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    return text.strip()


def predict(text: str):
    model, tfidf = load_artifacts()
    cleaned = clean_resume_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0]) if hasattr(model, "predict_proba") else None
    return prediction, confidence


def decision(confidence: float | None) -> str:
    if confidence is None:
        return "Prediction generated"
    if confidence > 0.80:
        return "Highly Suitable"
    if confidence > 0.60:
        return "Moderate Fit"
    return "Not Suitable"


st.set_page_config(page_title="Resume Analyzer", page_icon=":page_facing_up:", layout="centered")

st.title("AI Resume Analyzer")
st.write("Upload a resume and get a role prediction with a suitability score.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.success("Resume uploaded successfully.")
    name = st.text_input("Enter Candidate Name")

    if st.button("Analyze Resume"):
        text = extract_text(uploaded_file)

        if not text:
            st.error("The uploaded file is empty or no text could be extracted.")
            st.stop()

        role, confidence = predict(text)
        result = decision(confidence)

        st.subheader("🔍 Results")
        st.write(f"**Role:** {role}")
        if confidence is None:
            st.write("**Confidence:** Not available")
        else:
            st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Decision:** {result}")

        with st.expander("View Extracted Text"):
            st.write(text[:2000])

        if confidence is not None and confidence > 0.70:
            insert_candidate(name, role, confidence, result)
            st.success("✅ Candidate added to dashboard")
        else:
            st.warning("❌ Not shortlisted")
