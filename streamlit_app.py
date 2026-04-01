from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features.preprocess import clean_resume_text, normalize_columns

DATA_PATH = Path("data") / "resume_data.csv"
MODEL_DIR = Path("models")
CLASSIFIER_PATH = MODEL_DIR / "resume_classifier.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf.pkl"
MATCH_MODEL_PATH = MODEL_DIR / "xgb_resume_model.pkl"
SBERT_DIR_PATH = MODEL_DIR / "sbert_model"
SBERT_PICKLE_PATH = MODEL_DIR / "sbert_model.pkl"

TARGET_ROLE_COLUMN = "job_position_name"
JOB_TEXT_COLUMN = "skills_required"
MATCH_THRESHOLD = 0.75
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="AI",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #060816 0%, #0b1220 100%);
            color: #e5edf7;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-card, .metric-card, .result-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 18px;
            box-shadow: 0 12px 32px rgba(2, 6, 23, 0.35);
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #0f172a 0%, #162033 100%);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #94a3b8;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 0.15rem;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .section-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .result-card {
            padding: 1.2rem 1.3rem;
            margin-top: 1rem;
        }
        .decision-pill {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .decision-selected {
            background: rgba(34, 197, 94, 0.16);
            color: #86efac;
        }
        .decision-rejected {
            background: rgba(248, 113, 113, 0.16);
            color: #fca5a5;
        }
        .caption-text {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return None

    if SBERT_DIR_PATH.exists():
        return SentenceTransformer(str(SBERT_DIR_PATH))
    if SBERT_PICKLE_PATH.exists():
        return joblib.load(SBERT_PICKLE_PATH)

    return None


@st.cache_resource(show_spinner=False)
def load_assets():
    classifier = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    match_model = None
    if MATCH_MODEL_PATH.exists():
        try:
            match_model = joblib.load(MATCH_MODEL_PATH)
        except ModuleNotFoundError:
            match_model = None
    sbert_model = _load_sentence_transformer()
    return classifier, vectorizer, match_model, sbert_model


@st.cache_data(show_spinner=False)
def build_role_lookup() -> dict[str, str]:
    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)
    required_columns = {TARGET_ROLE_COLUMN, JOB_TEXT_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {', '.join(sorted(missing))}")

    filtered = df[[TARGET_ROLE_COLUMN, JOB_TEXT_COLUMN]].copy()
    filtered[TARGET_ROLE_COLUMN] = filtered[TARGET_ROLE_COLUMN].fillna("").astype(str).str.strip()
    filtered[JOB_TEXT_COLUMN] = filtered[JOB_TEXT_COLUMN].fillna("").astype(str).str.strip()
    filtered = filtered[(filtered[TARGET_ROLE_COLUMN] != "") & (filtered[JOB_TEXT_COLUMN] != "")]

    role_lookup: dict[str, str] = {}
    for role, group in filtered.groupby(TARGET_ROLE_COLUMN):
        job_texts = [text for text in group[JOB_TEXT_COLUMN].tolist() if text]
        if not job_texts:
            continue
        role_lookup[role] = Counter(job_texts).most_common(1)[0][0]

    return role_lookup


@st.cache_data(show_spinner=False)
def get_default_job_reference() -> str:
    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)
    if JOB_TEXT_COLUMN not in df.columns:
        return "general business communication teamwork problem solving technical skills"

    values = (
        df[JOB_TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    values = values[values != ""]
    if values.empty:
        return "general business communication teamwork problem solving technical skills"
    return Counter(values.tolist()).most_common(1)[0][0]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
    except ImportError:
        return ""

    document = docx.Document(BytesIO(file_bytes))
    return " ".join(paragraph.text for paragraph in document.paragraphs)


def extract_resume_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if suffix == ".docx":
        return extract_text_from_docx(file_bytes)

    return ""


def score_resume(resume_text: str) -> dict[str, str | float]:
    classifier, vectorizer, match_model, sbert_model = load_assets()
    role_lookup = build_role_lookup()

    cleaned_resume = clean_resume_text(resume_text)
    if not cleaned_resume:
        raise ValueError("No readable resume text found in the uploaded file.")

    predicted_role = classifier.predict(vectorizer.transform([cleaned_resume]))[0]
    job_text = role_lookup.get(predicted_role)
    if not job_text:
        job_text = f"{predicted_role} {get_default_job_reference()}"

    cleaned_job_text = clean_resume_text(job_text)

    if sbert_model is not None and match_model is not None:
        resume_embedding = sbert_model.encode([cleaned_resume], show_progress_bar=False)
        job_embedding = sbert_model.encode([cleaned_job_text], show_progress_bar=False)
        features = np.hstack((resume_embedding, job_embedding))
        raw_score = float(match_model.predict(features)[0])
        score = float(np.clip(raw_score, 0.0, 1.0))
        scoring_mode = "Model score"
    elif sbert_model is not None:
        resume_embedding = sbert_model.encode([cleaned_resume], show_progress_bar=False)
        job_embedding = sbert_model.encode([cleaned_job_text], show_progress_bar=False)
        resume_vector = resume_embedding[0]
        job_vector = job_embedding[0]
        denominator = np.linalg.norm(resume_vector) * np.linalg.norm(job_vector)
        cosine_score = 0.0 if denominator == 0 else float(np.dot(resume_vector, job_vector) / denominator)
        score = float(np.clip((cosine_score + 1.0) / 2.0, 0.0, 1.0))
        scoring_mode = "Embedding similarity"
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf_vectors = vectorizer.transform([cleaned_resume, cleaned_job_text])
        cosine_score = float(cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0])
        score = float(np.clip(cosine_score, 0.0, 1.0))
        scoring_mode = "TF-IDF similarity"
    decision = "Selected" if score >= MATCH_THRESHOLD else "Rejected"

    return {
        "predicted_role": str(predicted_role),
        "job_reference": str(job_text),
        "score": score,
        "decision": decision,
        "scoring_mode": scoring_mode,
    }


def format_score(score: float) -> str:
    return f"{score * 100:.1f}%"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result: dict[str, str | float]) -> None:
    decision = str(result["decision"])
    pill_class = "decision-selected" if decision == "Selected" else "decision-rejected"
    st.markdown(
        f"""
        <div class="result-card">
            <span class="decision-pill {pill_class}">{decision}</span>
            <div class="section-title">Match Score: {format_score(float(result["score"]))}</div>
            <div class="caption-text">Predicted role: {result["predicted_role"]}</div>
            <div class="caption-text">Scoring mode: {result["scoring_mode"]}</div>
            <div class="caption-text">Selection rule: 75% and above is Selected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View matched role context", expanded=False):
        st.write(result["job_reference"])


def render_single_resume_tab() -> None:
    st.markdown('<div class="section-title">Single Resume Check</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload one resume and get a clear Selected or Rejected decision.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="single_resume_uploader",
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing resume..."):
            try:
                resume_text = extract_resume_text(uploaded_file)
                result = score_resume(resume_text)
                render_result_card(result)
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Upload a `.txt`, `.pdf`, or `.docx` resume to see the screening decision.")


def render_bulk_resume_tab() -> None:
    st.markdown('<div class="section-title">Bulk Screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload multiple resumes and get simple Selected or Rejected outputs using the fixed 75% rule.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="bulk_resume_uploader",
    )

    if uploaded_files:
        results: list[dict[str, str]] = []
        progress = st.progress(0.0)

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                resume_text = extract_resume_text(uploaded_file)
                result = score_resume(resume_text)
                results.append(
                    {
                        "Resume": uploaded_file.name,
                        "Predicted Role": str(result["predicted_role"]),
                        "Match Score": format_score(float(result["score"])),
                        "Scoring Mode": str(result["scoring_mode"]),
                        "Decision": str(result["decision"]),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "Resume": uploaded_file.name,
                        "Predicted Role": "N/A",
                        "Match Score": "N/A",
                        "Decision": f"Error: {exc}",
                    }
                )

            progress.progress(index / len(uploaded_files))

        result_df = pd.DataFrame(results)
        selected_count = int((result_df["Decision"] == "Selected").sum())
        rejected_count = int((result_df["Decision"] == "Rejected").sum())

        summary_cols = st.columns(3)
        with summary_cols[0]:
            metric_card("Uploaded Resumes", str(len(uploaded_files)))
        with summary_cols[1]:
            metric_card("Selected", str(selected_count))
        with summary_cols[2]:
            metric_card("Rejected", str(rejected_count))

        st.dataframe(result_df, use_container_width=True, hide_index=True)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results",
            data=csv_bytes,
            file_name="bulk_screening_results.csv",
            mime="text/csv",
            use_container_width=False,
        )
    else:
        st.info("Upload one or more resumes to run bulk screening.")


def validate_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Required file(s) missing: {joined}")


def main() -> None:
    inject_styles()
    validate_paths([DATA_PATH, CLASSIFIER_PATH, TFIDF_PATH])

    st.markdown(
        """
            <div class="hero-card">
            <div class="hero-title">Resume Screening Dashboard</div>
            <div class="hero-subtitle">
                Dark mode screening workflow with fixed decisioning. Any score at 75% or above is marked Selected,
                and anything lower is marked Rejected.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_bulk, tab_single = st.tabs(["Bulk Screening", "Single Resume Check"])

    with tab_bulk:
        render_bulk_resume_tab()

    with tab_single:
        render_single_resume_tab()


if __name__ == "__main__":
    main()
