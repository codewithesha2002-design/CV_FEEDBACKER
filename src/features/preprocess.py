from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_COLUMN = "job_position_name"
RAW_TARGET_CANDIDATES = (TARGET_COLUMN, "\ufeffjob_position_name")
TEXT_SOURCE_COLUMNS = (
    "career_objective",
    "skills",
    "major_field_of_studies",
    "positions",
    "responsibilities",
)


def clean_resume_text(text: str) -> str:
    """Apply basic text cleanup for resume content."""
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM markers and surrounding whitespace from incoming column names."""
    renamed_columns = {
        column: column.replace("\ufeff", "").strip()
        for column in df.columns
    }
    return df.rename(columns=renamed_columns)


def _validate_available_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
) -> list[str]:
    available = [column for column in required_columns if column in df.columns]
    if available:
        return available

    joined = ", ".join(required_columns)
    raise ValueError(f"Missing expected {label} columns. Looked for: {joined}")


def assemble_resume_text(
    df: pd.DataFrame,
    source_columns: Iterable[str],
) -> pd.Series:
    """Concatenate multiple profile fields into one training text column."""
    source_columns = list(source_columns)
    combined = df[source_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return combined.map(clean_resume_text)


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    text_columns: Iterable[str] = TEXT_SOURCE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create a cleaned dataset ready for training."""
    df = pd.read_csv(input_path)
    df = normalize_columns(df.copy())

    resolved_text_columns = _validate_available_columns(
        df,
        text_columns,
        "text source",
    )
    _validate_available_columns(df, (target_column,), "target")

    df[target_column] = df[target_column].fillna("").astype(str).str.strip()
    df["cleaned_resume"] = assemble_resume_text(df, resolved_text_columns)

    cleaned = df.loc[
        (df[target_column] != "") & (df["cleaned_resume"] != ""),
        [target_column, "cleaned_resume", *resolved_text_columns],
    ].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


if __name__ == "__main__":
    processed = preprocess_dataset(
        "data/resume_data.csv",
        "data/processed/cleaned_resume.csv",
    )
    print(processed[[TARGET_COLUMN, "cleaned_resume"]].head())
