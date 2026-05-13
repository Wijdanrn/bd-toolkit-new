# Big Data Toolkit 2 (Rebuilt)

This repository contains a clean rebuild scaffold of the Big Data Toolkit 2 Streamlit app.

Files:
- `app.py` — main Streamlit entry with sidebar navigation and session state initialization.
- `input_data.py`, `eda.py`, `data_cleansing.py`, `data_preprocessing.py`, `data_modeling.py`, `data_visualization.py`, `competition_page.py` — page stubs.
- `requirements.txt` — minimal dependencies.

How to run:

1. Create and activate a Python environment (recommended Python 3.9+).
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app from this folder:

```
streamlit run app.py
```

Notes:
- This is Phase 1 scaffold only. No data upload or processing logic is implemented yet.
- Phase 2 will add upload controls and dataset-specific behaviors.
