# Copilot Instructions for Resume Analyzer

## Goal
Enable AI assistance to be instantly productive in this repository by capturing the key workflows, file structure, and project-specific conventions.

## Do not overwrite user code
- never change existing project files (scripts, models, notebooks, data) unless explicitly asked.
- avoid edits that create ambiguity in training/inference behavior.

## Project root layout
- `streamlit_app.py`: UI entrypoint; run with `python -m streamlit run streamlit_app.py`.
- `src/models/train_model.py`: trains role classification model (TF-IDF + LogisticRegression).
- `src/models/train_match_model.py`: trains score matching model (SBERT + XGBoost).
- `data/resume_data.csv`: raw input data; pipeline outputs to `data/processed/cleaned_resume.csv`.
- `models/`: persisted artifacts `resume_classifier.pkl`, `tfidf.pkl`, `xgb_resume_model.pkl`, and `sbert_model/`.

## Key workflows
1. EDA: open `notebooks/eda.ipynb`.
2. Train classification model: `python -m src.models.train_model`.
3. Train matching model: `python -m src.models.train_match_model`.
4. Run dashboard: `python -m streamlit run streamlit_app.py`.

## Agent behavior guidance
- Prefer existing commands from README and shell scripts to run/validate tasks.
- If asked to add features, keep data/paths consistent with README conventions.
- For model training questions, point to `src/models/*` scripts and explain expected outputs.
- For streamlit deployment questions, cite `streamlit_app.py` and command above.

## Safeguards
- do not add or assume hidden config not found in this repo.
- do not introduce ambiguous migrations of model artifacts (`*.pkl`, `sbert_model/`).
- keep suggestions deterministic and minimal; use raw values from `README` where possible.

## Example user prompts
- "How do I retrain the role classifier?"
- "How can I add a new skill column for preprocessing?"
- "I need to debug the Streamlit resume matching UI; which file to inspect?"
