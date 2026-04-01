from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.features.preprocess import normalize_columns

DATA_PATH = Path("data") / "resume_data.csv"
MODEL_DIR = Path("models")
XGB_MODEL_PATH = MODEL_DIR / "xgb_resume_model.pkl"
SBERT_DIR_PATH = MODEL_DIR / "sbert_model"
SBERT_PICKLE_PATH = MODEL_DIR / "sbert_model.pkl"

RESUME_COLUMNS = ["career_objective", "skills", "responsibilities"]
JOB_COLUMN = "skills_required"
TARGET_COLUMN = "matched_score"


def build_text(df: pd.DataFrame):
    for col in RESUME_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if JOB_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {JOB_COLUMN}")

    resume_text = (
        df["career_objective"].fillna("").astype(str)
        + " "
        + df["skills"].fillna("").astype(str)
        + " "
        + df["responsibilities"].fillna("").astype(str)
    ).str.strip()

    job_text = df[JOB_COLUMN].fillna("").astype(str).str.strip()
    return resume_text, job_text


def train_match_model(
    data_path: Path = DATA_PATH,
    model_path: Path = XGB_MODEL_PATH,
    sbert_dir_path: Path = SBERT_DIR_PATH,
    sbert_pickle_path: Path = SBERT_PICKLE_PATH,
):
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(data_path)
    df = normalize_columns(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    resume_text, job_text = build_text(df)
    y = df[TARGET_COLUMN].astype(float)

    valid_mask = (resume_text != "") & (job_text != "") & y.notna()
    filtered = df.loc[valid_mask].copy()
    resume_text = resume_text.loc[valid_mask]
    job_text = job_text.loc[valid_mask]
    y = y.loc[valid_mask]

    if filtered.empty:
        raise ValueError("No valid rows available after filtering empty text and target values.")

    print(f"Rows used for training: {len(filtered)}")
    print("Loading SBERT model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating resume embeddings...")
    resume_embeddings = sbert.encode(
        resume_text.tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    print("Creating job embeddings...")
    job_embeddings = sbert.encode(
        job_text.tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    X = np.hstack((resume_embeddings, job_embeddings))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Training XGBoost regressor...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Preferred save format for SentenceTransformer at inference time.
    sbert.save(str(sbert_dir_path))

    # Optional pickle save for compatibility with older code paths.
    try:
        joblib.dump(sbert, sbert_pickle_path)
    except Exception:
        pass

    return {
        "rows_used": int(len(filtered)),
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "model_path": str(model_path),
        "sbert_dir": str(sbert_dir_path),
    }


if __name__ == "__main__":
    metrics = train_match_model()
    print("Training complete")
    print(f"Rows used: {metrics['rows_used']}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Saved model: {metrics['model_path']}")
    print(f"Saved SBERT: {metrics['sbert_dir']}")
