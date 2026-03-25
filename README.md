# Resume Analyzer

Resume classification project with preprocessing, model training, API serving, and a Streamlit UI.

## Project Layout

```text
resume-analyzer/
|-- data/
|   |-- raw/
|   |   `-- UpdatedResumeDataSet.csv
|   |-- processed/
|   |   `-- cleaned_resume.csv
|   `-- uploads/
|-- models/
|   |-- resume_classifier.pkl
|   `-- tfidf.pkl
|-- notebooks/
|   `-- eda.ipynb
|-- src/
|   |-- features/
|   |   `-- preprocess.py
|   |-- models/
|   |   `-- train_model.py
|   |-- services/
|   |   |-- parser.py
|   |   |-- predictor.py
|   |   `-- decision.py
|   `-- api/
|       `-- main.py
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python -m src.models.train_model
   ```
3. Run the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
