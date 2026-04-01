from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.features.preprocess import TARGET_COLUMN, preprocess_dataset


def _filter_rare_classes(df, target_column: str, minimum_count: int = 2):
    """Drop classes that cannot support a stratified train/test split."""
    label_counts = df[target_column].value_counts()
    valid_labels = label_counts[label_counts >= minimum_count].index
    filtered = df[df[target_column].isin(valid_labels)].copy()
    dropped_rows = len(df) - len(filtered)
    dropped_labels = int((label_counts < minimum_count).sum())
    return filtered, dropped_rows, dropped_labels


def train_model(
    data_path: str | Path,
    classifier_path: str | Path,
    tfidf_path: str | Path,
    processed_path: str | Path,
) -> dict[str, float | int]:
    df = preprocess_dataset(data_path, processed_path)

    feature_column = "cleaned_resume"
    target_column = TARGET_COLUMN
    df, dropped_rows, dropped_labels = _filter_rare_classes(df, target_column)

    if df.empty:
        raise ValueError("No rows remain for training after preprocessing and rare-class filtering.")

    if df[target_column].nunique() < 2:
        raise ValueError("Need at least two target classes to train a classifier.")

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

    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(x_train_tfidf, y_train)

    predictions = classifier.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    classifier_path = Path(classifier_path)
    tfidf_path = Path(tfidf_path)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    tfidf_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, tfidf_path)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "rows_used": int(len(df)),
        "class_count": int(df[target_column].nunique()),
        "dropped_rows": int(dropped_rows),
        "dropped_labels": int(dropped_labels),
    }


if __name__ == "__main__":
    metrics = train_model(
        Path("data/resume_data.csv"),
        Path("models/resume_classifier.pkl"),
        Path("models/tfidf.pkl"),
        Path("data/processed/cleaned_resume.csv"),
    )
    print("Artifacts saved.")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {metrics['macro_f1']:.4f}")
    print(
        "Training summary: "
        f"rows_used={metrics['rows_used']}, "
        f"class_count={metrics['class_count']}, "
        f"dropped_rows={metrics['dropped_rows']}, "
        f"dropped_labels={metrics['dropped_labels']}"
    )
