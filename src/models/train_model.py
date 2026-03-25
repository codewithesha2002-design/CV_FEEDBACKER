from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.features.preprocess import preprocess_dataset


def train_model(
    data_path: str | Path,
    classifier_path: str | Path,
    tfidf_path: str | Path,
    processed_path: str | Path,
) -> float:
    df = preprocess_dataset(data_path, processed_path)

    feature_column = "cleaned_resume"
    target_column = "Category"

    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_column],
        df[target_column],
        test_size=0.2,
        random_state=42,
        stratify=df[target_column],
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train_tfidf, y_train)

    predictions = classifier.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    classifier_path = Path(classifier_path)
    tfidf_path = Path(tfidf_path)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    tfidf_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, tfidf_path)
    return accuracy


if __name__ == "__main__":
    score = train_model(
        Path("data/raw/UpdatedResumeDataSet.csv"),
        Path("models/resume_classifier.pkl"),
        Path("models/tfidf.pkl"),
        Path("data/processed/cleaned_resume.csv"),
    )
    print(f"Artifacts saved. Validation accuracy: {score:.4f}")
