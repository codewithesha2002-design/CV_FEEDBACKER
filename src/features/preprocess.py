from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def clean_resume_text(text: str) -> str:
    """Apply basic text cleanup for resume content."""
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = "Resume",
) -> pd.DataFrame:
    """Create a cleaned dataset ready for training."""
    df = pd.read_csv(input_path)
    df = df.copy()
    df["cleaned_resume"] = df[text_column].fillna("").map(clean_resume_text)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    processed = preprocess_dataset(
        "data/raw/UpdatedResumeDataSet.csv",
        "data/processed/cleaned_resume.csv",
    )
    print(processed[["Category", "cleaned_resume"]].head())
