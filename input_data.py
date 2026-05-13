import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
from ui_components import render_info_panel, fix_arrow_compatibility


def _clear_derived_keys():
    """Clear all derived session state keys while preserving a minimal set.

    Keeps only `global_seed` and `page`. This avoids keeping duplicate large
    DataFrame copies in session state when a new file is uploaded.
    """
    keep_keys = {"global_seed", "page"}
    for k in list(st.session_state.keys()):
        if k in keep_keys:
            continue
        try:
            del st.session_state[k]
        except Exception:
            # best-effort removal
            st.session_state[k] = None


def render_input_data():
    st.header("📊 Data")
    st.write("Upload a train CSV file with header. Uploading a new file resets derived data.")

    uploaded_file = st.file_uploader("Upload Train CSV (with header)", type=["csv"])

    if uploaded_file is not None:
        # read bytes and compute hash to detect new uploads
        try:
            file_bytes = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        prev_hash = st.session_state.get("upload_hash")
        if prev_hash != file_hash:
            # New file detected -> clear previous derived state
            _clear_derived_keys()
            st.session_state["upload_hash"] = file_hash

        st.subheader("Preview")
        st.dataframe(fix_arrow_compatibility(df.head()))

        target_col = st.selectbox("Select Target Column", options=list(df.columns), key="target_col_select")
        task_type = st.radio("Task Type", ["Classification", "Regression"])

        if st.button("✅ Confirm & Proceed"):
            # Finalize dataset into session_state without forcing a rerun or clearing all derived keys
            st.session_state["df"] = df.copy()
            st.session_state["target_column"] = target_col
            st.session_state["task_type"] = task_type
            st.success("Dataset saved to session. You can proceed to EDA.")

    else:
        # If a dataset is already confirmed in session, show it and allow replacement
        if st.session_state.get("df") is not None:
            df_confirmed = st.session_state.get("df")
            st.subheader("Current dataset (in session)")
            st.dataframe(fix_arrow_compatibility(df_confirmed.head()))
            st.markdown(f"**Target column:** {st.session_state.get('target_column')}  ")
            st.markdown(f"**Task type:** {st.session_state.get('task_type')}  ")
            st.write("To replace this dataset, upload a new CSV above or click Replace Dataset.")
            if st.button("🔁 Replace Dataset"):
                # Clear derived state and allow uploading a new file
                _clear_derived_keys()
                st.success("Ready for new dataset upload.")
                try:
                    st.rerun()
                except AttributeError:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        else:
            st.info("No file uploaded yet. Use the uploader above to upload a CSV file.")

    # Info panel at bottom (outside other expanders)
    render_info_panel("Data")
