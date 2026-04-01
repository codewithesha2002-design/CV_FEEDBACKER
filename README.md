# Resume Analyzer

Resume analysis project with two model tracks:
- Role classification (TF-IDF + Logistic Regression)
- Resume-job match scoring (SBERT + XGBoost) for Selected/Rejected screening

## Project Layout

```text
resume-analyzer/
|-- data/
|   `-- resume_data.csv
|   |-- processed/
|   |   `-- cleaned_resume.csv
|-- models/
|   |-- resume_classifier.pkl
|   `-- tfidf.pkl
|   |-- xgb_resume_model.pkl
|   `-- sbert_model/
|-- notebooks/
|   `-- eda.ipynb
|-- src/
|   |-- features/
|   |   `-- preprocess.py
|   |-- models/
|   |   `-- train_model.py
|   |   `-- train_match_model.py
|-- streamlit_app.py
|-- README.md
`-- .gitignore
```

## Dataset Assumptions

- Input data lives at `data/resume_data.csv`.
- The training target is `job_position_name`. The loader normalizes the BOM-prefixed source column name automatically.
- Text features are assembled from `career_objective`, `skills`, `major_field_of_studies`, `positions`, and `responsibilities`.

## Quick Start

1. Run the EDA notebook:
   Open `notebooks/eda.ipynb` in Jupyter or VS Code and run the cells top-to-bottom.
2. Train the model:
   ```bash
   python -m src.models.train_model
   ```
3. Train the match-score model (for screening dashboard):
   ```bash
   python -m src.models.train_match_model
   ```
4. Run Streamlit dashboard:
   ```bash
   python -m streamlit run streamlit_app.py
   ```

## Training Outputs

- `data/processed/cleaned_resume.csv`: processed dataset with normalized target and assembled text
- `models/resume_classifier.pkl`: trained Logistic Regression classifier
- `models/tfidf.pkl`: fitted TF-IDF vectorizer
- `models/xgb_resume_model.pkl`: trained XGBoost regressor for `matched_score`
- `models/sbert_model/`: SentenceTransformer encoder used for resume/job embeddings

The training script prints validation accuracy, macro F1, and a short summary of retained rows and classes.
"# RESUME_ANALYZER" 
