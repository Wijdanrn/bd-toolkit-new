import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ui_components import render_info_panel, fix_arrow_compatibility, add_plot_to_session


def render_data_visualization():
    st.header("📋 Evaluation Summary")

    val = st.session_state.get("validation_summary")
    train_eval = st.session_state.get("training_evaluation_summary")

    if st.button("Show random 5 rows", key="viz_random5"):
        df = st.session_state.get("df")
        try:
            st.dataframe(fix_arrow_compatibility(df.sample(5)))
        except Exception:
            if df is not None:
                st.dataframe(fix_arrow_compatibility(df.head()))

    if not val and not train_eval:
        st.info("No validation or training evaluation summaries found in session.")
    else:
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Validation Summary")
            if val:
                st.write(val.get("metrics"))
                imp = val.get("feature_importance")
                if imp is not None:
                    fig, ax = plt.subplots()
                    imp.head(20).plot.bar(ax=ax)
                    ax.set_title("Validation Feature Importance")
                    st.pyplot(fig)
                    try:
                        add_plot_to_session(fig, title="validation_feature_importance_compact", page="Validation", kind="feature_importance")
                    except Exception:
                        pass
            else:
                st.info("No validation summary available.")

        with cols[1]:
            st.subheader("Training Evaluation")
            if train_eval:
                st.write(train_eval)
            else:
                st.info("No training evaluation available.")

    render_info_panel("Data Visualization")
