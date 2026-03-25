import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.db import get_candidates


st.title("📊 HR Dashboard - Shortlisted Candidates")

data = get_candidates()

if data:
    df = pd.DataFrame(data, columns=["ID", "Name", "Role", "Confidence", "Decision"])

    # Sort by best candidates
    df = df.sort_values(by="Confidence", ascending=False)

    st.dataframe(df)

    st.subheader("Top Candidates")
    st.write(df.head(5))
else:
    st.warning("No candidates shortlisted yet")
