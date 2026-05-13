import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import io

from ui_components import render_info_panel, fix_arrow_compatibility, annotate_bar_values, add_plot_to_session


def render_eda():
    st.header("📈 EDA")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("No dataset loaded. Please upload a dataset on the Data page first.")
        if st.button("Open Data Page"):
            st.session_state["page"] = "Data"
        return

    df_master = st.session_state.get("df")

    # If a split exists, allow previewing Train/Test subsets instead of the master df
    if st.session_state.get("split_done"):
        preview_choice = st.radio("Preview subset", ["Train (subset)", "Test (subset)"], index=0, key="eda_preview_subset")
        if preview_choice.startswith("Train"):
            px = st.session_state.get("pre_X_train")
            py = st.session_state.get("pre_y_train")
            if px is None:
                df_view = pd.DataFrame()
            else:
                df_view = px.copy()
                if py is not None and st.session_state.get("target_column"):
                    try:
                        df_view[st.session_state.get("target_column")] = py
                    except Exception:
                        pass
        else:
            px = st.session_state.get("pre_X_test")
            py = st.session_state.get("pre_y_test")
            if px is None:
                df_view = pd.DataFrame()
            else:
                df_view = px.copy()
                if py is not None and st.session_state.get("target_column"):
                    try:
                        df_view[st.session_state.get("target_column")] = py
                    except Exception:
                        pass
    else:
        df_view = df_master.copy()

    for c in df_view.select_dtypes(include=["object"]).columns:
        try:
            df_view[c] = df_view[c].astype("category")
        except Exception:
            pass

    # Dataset overview (kept at top)
    st.subheader("Dataset Overview")
    info_df = pd.DataFrame({
        "dtype": df_view.dtypes.astype(str),
        "non_null_count": df_view.count(),
        "unique": df_view.nunique(),
    })
    st.dataframe(fix_arrow_compatibility(info_df))

    # Add missing-values row under describe()
    try:
        desc = df_view.describe(include="all")
        missing_row = df_view.isnull().sum()
        missing_df = pd.DataFrame([missing_row], index=["missing"])
        desc_with_missing = pd.concat([desc, missing_df])
        st.dataframe(fix_arrow_compatibility(desc_with_missing))
    except Exception:
        st.dataframe(fix_arrow_compatibility(df_view.describe(include="all")))

    # Top-level horizontal navbar simulated with tabs (with emojis)
    top_tabs = st.tabs(["📊 Univariat", "🔗 Bivariat", "🧩 Custom"])

    # --- Univariat ---
    with top_tabs[0]:
        sub = st.tabs(["📈 Histogram", "📦 Boxplot", "🥧 Piechart", "📊 Barplot"])

        # Histogram
        with sub[0]:
            st.subheader("Histogram")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for histogram.")
            else:
                col = st.selectbox("Numeric column", options=num_cols, key="uni_hist_col")
                bins = st.slider("Bins", 5, 100, 30, key="uni_hist_bins")
                fig, ax = plt.subplots()
                df_view[col].dropna().hist(bins=bins, ax=ax)
                ax.set_title(f"Histogram: {col}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_hist_{col}")
                if st.button("Add plot to bundle", key=f"eda_add_hist_{col}"):
                    added = add_plot_to_session(fig, title=(save_name or f"hist_{col}"), page="EDA", kind="histogram")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Boxplot (single column)
        with sub[1]:
            st.subheader("Boxplot")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for boxplot.")
            else:
                col = st.selectbox("Numeric column (boxplot)", options=num_cols, key="uni_box_col")
                fig, ax = plt.subplots()
                sns.boxplot(y=df_view[col], ax=ax)
                ax.set_title(f"Boxplot: {col}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_box_{col}")
                if st.button("Add plot to bundle", key=f"eda_add_box_{col}"):
                    added = add_plot_to_session(fig, title=(save_name or f"box_{col}"), page="EDA", kind="boxplot")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Piechart
        with sub[2]:
            st.subheader("Piechart")
            cat_cols = df_view.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
            if not cat_cols:
                st.info("No categorical columns available for pie chart.")
            else:
                col = st.selectbox("Categorical column (pie)", options=cat_cols, key="uni_pie_col")
                vc = df_view[col].value_counts().nlargest(10)
                fig, ax = plt.subplots()
                ax.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%")
                ax.set_title(f"Pie chart: {col}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_pie_{col}")
                if st.button("Add plot to bundle", key=f"eda_add_pie_{col}"):
                    added = add_plot_to_session(fig, title=(save_name or f"pie_{col}"), page="EDA", kind="piechart")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Barplot (univariate counts)
        with sub[3]:
            st.subheader("Barplot")
            cat_cols = df_view.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
            if not cat_cols:
                st.info("No categorical columns available for bar chart.")
            else:
                col = st.selectbox("Categorical column (bar)", options=cat_cols, key="uni_bar_col")
                vc = df_view[col].value_counts().nlargest(20)
                fig, ax = plt.subplots()
                vc.plot(kind="bar", ax=ax)
                annotate_bar_values(ax, integer=True)
                ax.set_title(f"Bar chart: {col}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_bar_{col}")
                if st.button("Add plot to bundle", key=f"eda_add_bar_{col}"):
                    added = add_plot_to_session(fig, title=(save_name or f"bar_{col}"), page="EDA", kind="barplot")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

    # --- Bivariat ---
    with top_tabs[1]:
        sub = st.tabs(["🔵 Scatterplot - 2 column", "📊 Barplot - 2 column", "📦 Boxplot - 2 column", "🌡️ Heatmap"])

        # Scatterplot
        with sub[0]:
            st.subheader("Scatterplot - 2 column")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for a scatterplot.")
            else:
                x = st.selectbox("X", options=num_cols, key="bi_scatter_x")
                y = st.selectbox("Y", options=[c for c in num_cols if c != x], key="bi_scatter_y")
                fig, ax = plt.subplots()
                ax.scatter(df_view[x], df_view[y], alpha=0.6)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"Scatter: {x} vs {y}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_scatter_{x}_{y}")
                if st.button("Add plot to bundle", key=f"eda_add_scatter_{x}_{y}"):
                    added = add_plot_to_session(fig, title=(save_name or f"scatter_{x}_vs_{y}"), page="EDA", kind="scatter")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Barplot - 2 column (categorical x, numeric y)
        with sub[1]:
            st.subheader("Barplot - 2 column")
            cat_cols = df_view.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not cat_cols or not num_cols:
                st.info("Need at least one categorical and one numeric column for this plot.")
            else:
                x = st.selectbox("Categorical X", options=cat_cols, key="bi_bar_x")
                y = st.selectbox("Numeric Y", options=num_cols, key="bi_bar_y")
                agg = df_view.groupby(x, observed=False)[y].mean().nlargest(30)
                fig, ax = plt.subplots()
                agg.plot(kind="bar", ax=ax)
                annotate_bar_values(ax, fmt="{:.2f}", integer=False)
                ax.set_title(f"Mean {y} by {x}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_bibar_{x}_{y}")
                if st.button("Add plot to bundle", key=f"eda_add_bibar_{x}_{y}"):
                    added = add_plot_to_session(fig, title=(save_name or f"bar_{x}_{y}"), page="EDA", kind="biv_barplot")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Boxplot - 2 column (categorical x, numeric y)
        with sub[2]:
            st.subheader("Boxplot - 2 column")
            cat_cols = df_view.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not cat_cols or not num_cols:
                st.info("Need at least one categorical and one numeric column for this plot.")
            else:
                x = st.selectbox("Categorical X (box)", options=cat_cols, key="bi_box_x")
                y = st.selectbox("Numeric Y (box)", options=num_cols, key="bi_box_y")
                fig, ax = plt.subplots()
                try:
                    sns.boxplot(x=x, y=y, data=df_view, ax=ax)
                except Exception:
                    df_view.boxplot(column=y, by=x, ax=ax)
                ax.set_title(f"{y} by {x}")
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_bibox_{x}_{y}")
                if st.button("Add plot to bundle", key=f"eda_add_bibox_{x}_{y}"):
                    added = add_plot_to_session(fig, title=(save_name or f"box_{x}_vs_{y}"), page="EDA", kind="biv_boxplot")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")

        # Heatmap (pairwise) - reuse existing dual multiselect behaviour
        with sub[3]:
            st.subheader("Heatmap")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for heatmap.")
            else:
                if "biv_corr_x" not in st.session_state:
                    st.session_state["biv_corr_x"] = []
                if "biv_corr_y" not in st.session_state:
                    st.session_state["biv_corr_y"] = []

                colx, coly = st.columns(2)
                with colx:
                    if st.button("Select All X", key="bi_select_all_x"):
                        st.session_state["biv_corr_x"] = num_cols
                with coly:
                    if st.button("Select All Y", key="bi_select_all_y"):
                        st.session_state["biv_corr_y"] = num_cols

                selected_x = st.multiselect("X-axis features", options=num_cols, default=st.session_state.get("biv_corr_x", []), key="biv_corr_x_select")
                selected_y = st.multiselect("Y-axis features", options=num_cols, default=st.session_state.get("biv_corr_y", []), key="biv_corr_y_select")

                if selected_x and selected_y:
                    with st.spinner("Computing correlation matrix..."):
                        try:
                            upload_hash = st.session_state.get("upload_hash")

                            @st.cache_data
                            def _cached_corr(_upload_hash, x_cols, y_cols):
                                df_local = st.session_state.get("df")
                                corr = df_local[x_cols + y_cols].corr()
                                return corr.loc[x_cols, y_cols]

                            heatmap_data = _cached_corr(upload_hash, selected_x, selected_y)
                            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(selected_y)), max(6, 0.5 * len(selected_x))))
                            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                            ax.set_title("Correlation between selected features")
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            save_name = st.text_input("Filename (optional)", value="", key=f"eda_fname_corr_{len(selected_x)}_{len(selected_y)}")
                            if st.button("Add plot to bundle", key=f"eda_add_corr_{len(selected_x)}_{len(selected_y)}"):
                                added = add_plot_to_session(fig, title=(save_name or "correlation_heatmap"), page="EDA", kind="heatmap")
                                if added:
                                    st.success(f"Added plot as {added}")
                                else:
                                    st.info("Plot already added or failed to save.")
                        except Exception as e:
                            st.error(f"Error computing heatmap: {e}")
                else:
                    st.info("Select at least one feature for both X and Y to display heatmap.")

    # --- Custom: place previous multi-column EDA content here to avoid vertical sprawl ---
    with top_tabs[2]:
        custom_tabs = st.tabs(["❗ Missing Values", "📈 Histograms (Numerical)", "📦 Box Plots (vs Target)", "📊 Bar Charts (Categorical)", "🌡️ Correlation Heatmap"])

        # Missing Values
        with custom_tabs[0]:
            st.subheader("Missing Values")
            missing_counts = df_view.isnull().sum().sort_values(ascending=False)
            if missing_counts.sum() == 0:
                st.success("No missing values detected.")
            else:
                fig, ax = plt.subplots(figsize=(max(6, len(missing_counts) * 0.3), 4))
                missing_counts.plot(kind="bar", ax=ax)
                annotate_bar_values(ax, integer=True)
                ax.set_ylabel("Missing Count")
                ax.set_title("Missing Values per Column")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                save_name = st.text_input("Filename (optional)", value="", key="eda_fname_missing_univariate")
                if st.button("Add plot to bundle", key="eda_add_missing_univariate"):
                    added = add_plot_to_session(fig, title=(save_name or "missing_values"), page="EDA", kind="missing_values")
                    if added:
                        st.success(f"Added plot as {added}")
                    else:
                        st.info("Plot already added or failed to save.")
                st.session_state["eda_fig_missing_univariate"] = fig

        # Histograms (batch)
        with custom_tabs[1]:
            st.subheader("Histograms (Numerical Features)")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            selected_nums = st.multiselect("Select numerical columns", options=num_cols, default=num_cols[:3], key="hist_cols")
            if selected_nums:
                with st.spinner("Rendering histograms..."):
                    n = len(selected_nums)
                    cols = 2 if n > 1 else 1
                    rows = math.ceil(n / cols)
                    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
                    axes_flat = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
                    for i, col in enumerate(selected_nums):
                        ax = axes_flat[i]
                        df_view[col].dropna().hist(ax=ax, bins=30)
                        ax.set_title(col)
                    for j in range(i + 1, len(axes_flat)):
                        fig.delaxes(axes_flat[j])
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    save_name = st.text_input("Filename (optional)", value="", key="eda_fname_histbatch")
                    if st.button("Add plot to bundle", key="eda_add_histbatch"):
                        added = add_plot_to_session(fig, title=(save_name or "histograms_batch"), page="EDA", kind="histogram_batch")
                        if added:
                            st.success(f"Added plot as {added}")
                        else:
                            st.info("Plot already added or failed to save.")

        # Box Plots vs Target
        with custom_tabs[2]:
            st.subheader("Box Plots (Numerical vs Target)")
            target_col = st.session_state.get("target_column")
            task_type = st.session_state.get("task_type")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            selected_box_cols = st.multiselect("Select numerical columns for boxplots", options=num_cols, default=num_cols[:2], key="box_cols")
            if target_col is None or target_col not in df_view.columns:
                st.info("Target column not set. Please set the target on the Data page.")
            else:
                if selected_box_cols:
                    with st.spinner("Rendering box plots..."):
                        if task_type == "Classification":
                            df_view[target_col] = df_view[target_col].astype("category")
                            n = len(selected_box_cols)
                            cols = 1 if n <= 2 else 2
                            rows = math.ceil(n / cols)
                            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
                            axes_arr = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
                            for i, col in enumerate(selected_box_cols):
                                ax = axes_arr[i]
                                try:
                                    sns.boxplot(x=target_col, y=col, data=df_view, ax=ax)
                                except Exception:
                                    df_view.boxplot(column=col, by=target_col, ax=ax)
                                ax.set_title(f"{col} by {target_col}")
                            for j in range(i + 1, len(axes_arr)):
                                fig.delaxes(axes_arr[j])
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            save_name = st.text_input("Filename (optional)", value="", key="eda_fname_box_vs_target")
                            if st.button("Add plot to bundle", key="eda_add_box_vs_target"):
                                added = add_plot_to_session(fig, title=(save_name or "boxplots_vs_target"), page="EDA", kind="boxplots_vs_target")
                                if added:
                                    st.success(f"Added plot as {added}")
                                else:
                                    st.info("Plot already added or failed to save.")
                        else:
                            try:
                                df_view["_target_bin"] = pd.qcut(df_view[target_col], q=5, duplicates="drop")
                                n = len(selected_box_cols)
                                cols = 1 if n <= 2 else 2
                                rows = math.ceil(n / cols)
                                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
                                axes_arr = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
                                for i, col in enumerate(selected_box_cols):
                                    ax = axes_arr[i]
                                    try:
                                        sns.boxplot(x="_target_bin", y=col, data=df_view, ax=ax)
                                    except Exception:
                                        df_view.boxplot(column=col, by="_target_bin", ax=ax)
                                    ax.set_title(f"{col} by target bins")
                                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                                for j in range(i + 1, len(axes_arr)):
                                    fig.delaxes(axes_arr[j])
                                plt.tight_layout()
                                st.pyplot(fig, clear_figure=True)
                                df_view.drop(columns=["_target_bin"], inplace=True, errors="ignore")
                            except Exception as e:
                                st.error(f"Error creating binned plots: {e}")

        # Bar Charts (Categorical Features)
        with custom_tabs[3]:
            st.subheader("Bar Charts (Categorical Features)")
            cat_cols = df_view.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
            selected_cat = st.multiselect("Select categorical columns", options=cat_cols, default=cat_cols[:2], key="cat_cols")
            if selected_cat:
                with st.spinner("Rendering categorical bar charts..."):
                    n = len(selected_cat)
                    cols = 2 if n > 1 else 1
                    rows = math.ceil(n / cols)
                    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
                    axes_flat = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
                    for i, col in enumerate(selected_cat):
                        ax = axes_flat[i]
                        vc = df_view[col].value_counts().nlargest(20)
                        vc.plot(kind="bar", ax=ax)
                        annotate_bar_values(ax, integer=True)
                        ax.set_title(col)
                        ax.set_ylabel("Count")
                    for j in range(i + 1, len(axes_flat)):
                        fig.delaxes(axes_flat[j])
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    save_name = st.text_input("Filename (optional)", value="", key="eda_fname_catbars")
                    if st.button("Add plot to bundle", key="eda_add_catbars"):
                        added = add_plot_to_session(fig, title=(save_name or "categorical_bar_charts"), page="EDA", kind="cat_barplots")
                        if added:
                            st.success(f"Added plot as {added}")
                        else:
                            st.info("Plot already added or failed to save.")

        # Correlation Heatmap
        with custom_tabs[4]:
            st.subheader("Correlation Heatmap")
            num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for heatmap.")
            else:
                if "cust_corr_x" not in st.session_state:
                    st.session_state["cust_corr_x"] = []
                if "cust_corr_y" not in st.session_state:
                    st.session_state["cust_corr_y"] = []

                colx, coly = st.columns(2)
                with colx:
                    if st.button("Select All X", key="cust_select_all_x"):
                        st.session_state["cust_corr_x"] = num_cols
                with coly:
                    if st.button("Select All Y", key="cust_select_all_y"):
                        st.session_state["cust_corr_y"] = num_cols

                selected_x = st.multiselect("X-axis features", options=num_cols, default=st.session_state.get("cust_corr_x", []), key="cust_corr_x_select")
                selected_y = st.multiselect("Y-axis features", options=num_cols, default=st.session_state.get("cust_corr_y", []), key="cust_corr_y_select")

                if selected_x and selected_y:
                    with st.spinner("Computing correlation matrix..."):
                        try:
                            upload_hash = st.session_state.get("upload_hash")

                            @st.cache_data
                            def _cached_corr(_upload_hash, x_cols, y_cols):
                                df_local = st.session_state.get("df")
                                corr = df_local[x_cols + y_cols].corr()
                                return corr.loc[x_cols, y_cols]

                            heatmap_data = _cached_corr(upload_hash, selected_x, selected_y)
                            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(selected_y)), max(6, 0.5 * len(selected_x))))
                            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                            ax.set_title("Correlation between selected features")
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            save_name = st.text_input("Filename (optional)", value="", key="eda_fname_cust_corr")
                            if st.button("Add plot to bundle", key="eda_add_cust_corr"):
                                added = add_plot_to_session(fig, title=(save_name or "custom_correlation_heatmap"), page="EDA", kind="heatmap")
                                if added:
                                    st.success(f"Added plot as {added}")
                                else:
                                    st.info("Plot already added or failed to save.")
                        except Exception as e:
                            st.error(f"Error computing heatmap: {e}")

    render_info_panel("EDA")
