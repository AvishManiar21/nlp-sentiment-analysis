"""Model Performance tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from utils.theme import apply_chart_theme
from utils.cache import OUTPUT_DIR, RESULTS_DIR


def render_performance_tab(df: pd.DataFrame, eval_results: dict):
    """Render the Model Performance tab content."""
    st.subheader("Model Performance Comparison")
    
    if not eval_results:
        st.info(
            "Model evaluation results not found. "
            "Run `python main.py` to train models and generate evaluation metrics."
        )
        return
    
    if "comparison" in eval_results:
        comparison_df = eval_results["comparison"]
        
        st.markdown("### Performance Metrics")
        
        metric_cols = ["Accuracy", "F1 (weighted)", "F1 (macro)", "Precision", "Recall"]
        available_cols = ["Model"] + [c for c in metric_cols if c in comparison_df.columns]
        
        st.dataframe(
            comparison_df[available_cols].style.format({
                col: "{:.4f}" for col in available_cols if col != "Model"
            }).background_gradient(
                cmap="RdYlGn",
                subset=[c for c in available_cols if c != "Model"]
            ),
            use_container_width=True,
        )
        
        st.markdown("### Accuracy Comparison")
        
        fig = px.bar(
            comparison_df,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            color_continuous_scale="RdYlGn",
            title="Model Accuracy",
        )
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### F1 Score Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "F1 (weighted)" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y="F1 (weighted)",
                    color="F1 (weighted)",
                    color_continuous_scale="RdYlGn",
                    title="Weighted F1 Score",
                )
                apply_chart_theme(fig, height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "F1 (macro)" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y="F1 (macro)",
                    color="F1 (macro)",
                    color_continuous_scale="RdYlGn",
                    title="Macro F1 Score",
                )
                apply_chart_theme(fig, height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        
        if "best_models" in summary:
            st.markdown("### Best Models by Metric")
            
            best_df = pd.DataFrame([
                {"Metric": metric, "Best Model": info["model"], "Score": info["value"]}
                for metric, info in summary["best_models"].items()
            ])
            
            st.dataframe(
                best_df.style.format({"Score": "{:.4f}"}),
                use_container_width=True,
            )
    
    st.markdown("### Confusion Matrices")
    
    # Try to display confusion matrix images from outputs/
    cm_images = sorted(Path(OUTPUT_DIR).glob("confusion_matrix_*.png"))
    
    if cm_images:
        cols = st.columns(2)
        for i, img_path in enumerate(cm_images):
            model_name = img_path.stem.replace("confusion_matrix_", "").replace("_", " ").title()
            with cols[i % 2]:
                st.image(str(img_path), caption=f"Confusion Matrix: {model_name}", use_container_width=True)
    else:
        # Fallback: try to render from CSV in results/
        cm_csvs = sorted(Path(RESULTS_DIR).glob("evaluation_confusion_matrix_*.csv"))
        if cm_csvs:
            cols = st.columns(2)
            for i, csv_path in enumerate(cm_csvs):
                model_name = csv_path.stem.replace("evaluation_confusion_matrix_", "").replace("_", " ").title()
                cm_df = pd.read_csv(csv_path, index_col=0)
                fig = px.imshow(
                    cm_df,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title=f"Confusion Matrix: {model_name}",
                )
                apply_chart_theme(fig, height=350)
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Confusion matrices will appear here after you run the full pipeline. "
                "Run `python main.py` to train models and generate visualizations."
            )
