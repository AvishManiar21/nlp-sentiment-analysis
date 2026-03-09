"""Model Performance tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.theme import apply_chart_theme


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
    st.info(
        "Confusion matrix images are saved in the `outputs/` directory. "
        "Check `confusion_matrix_*.png` files."
    )
