"""Model Performance tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils.theme import apply_chart_theme
from utils.cache import OUTPUT_DIR, RESULTS_DIR, check_dl_models_available


def render_performance_tab(df: pd.DataFrame, eval_results: dict):
    """Render the Model Performance tab content."""
    st.subheader("Model Performance Comparison")

    # Check for deep learning models
    dl_models = check_dl_models_available()

    if dl_models:
        st.markdown("### 🚀 Deep Learning Models Detected")

        dl_cols = st.columns(len(dl_models) if len(dl_models) <= 4 else 4)

        for i, model in enumerate(dl_models):
            with dl_cols[i % len(dl_cols)]:
                st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 8px; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #4CAF50;">✓ {model['name']}</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        <strong>Framework:</strong> {model['framework']}<br>
                        <strong>Architecture:</strong> {model['type']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Show training command for DL models
        with st.expander("💡 How to train Deep Learning models"):
            st.code("""
# Train CNN with TensorFlow and PyTorch
python main.py --train-dl --dl-framework both --dl-model-type cnn

# Train with pre-trained GloVe embeddings for better accuracy
python main.py --train-dl --use-embeddings --embedding-name glove-wiki-gigaword-100

# Train LSTM model
python main.py --train-dl --dl-framework pytorch --dl-model-type lstm

# Train everything with custom parameters
python main.py --train-dl --dl-framework both --dl-model-type both \\
  --use-embeddings --dl-epochs 20 --dl-batch-size 64
            """, language="bash")

    else:
        st.info(
            "🔬 **Deep Learning models not yet trained.** "
            "Run `python main.py --train-dl` to train state-of-the-art CNN and BiLSTM models "
            "with TensorFlow and PyTorch!"
        )

    if not eval_results:
        st.info(
            "Model evaluation results not found. "
            "Run `python main.py` to train models and generate evaluation metrics."
        )
        return
    
    if "comparison" in eval_results:
        comparison_df = eval_results["comparison"]

        st.markdown("### Performance Metrics")

        # Categorize models
        def categorize_model(model_name):
            model_lower = model_name.lower()
            if any(x in model_lower for x in ['cnn', 'lstm', 'bilstm']):
                return '🧠 Deep Learning'
            elif any(x in model_lower for x in ['distilbert', 'bert', 'transformer']):
                return '🤖 Transformers'
            elif any(x in model_lower for x in ['logistic', 'naive', 'svm', 'forest']):
                return '📊 Classical ML'
            elif any(x in model_lower for x in ['vader', 'textblob', 'ensemble']):
                return '📝 Rule-based'
            else:
                return '🔧 Other'

        # Add category column
        comparison_df_display = comparison_df.copy()
        comparison_df_display['Category'] = comparison_df_display['Model'].apply(categorize_model)

        # Reorder columns to show category first
        metric_cols = ["Accuracy", "F1 (weighted)", "F1 (macro)", "Precision", "Recall"]
        available_cols = ["Category", "Model"] + [c for c in metric_cols if c in comparison_df.columns]

        st.dataframe(
            comparison_df_display[available_cols].style.format({
                col: "{:.4f}" for col in available_cols if col not in ["Model", "Category"]
            }).background_gradient(
                cmap="RdYlGn",
                subset=[c for c in available_cols if c not in ["Model", "Category"]]
            ),
            use_container_width=True,
        )
        
        st.markdown("### Accuracy Comparison")

        # Create enhanced bar chart with categories
        fig = px.bar(
            comparison_df_display.sort_values("Accuracy", ascending=False),
            x="Model",
            y="Accuracy",
            color="Category",
            title="Model Accuracy by Category",
            text="Accuracy",
            hover_data={"Accuracy": ":.4f", "Category": True}
        )

        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Add side-by-side comparison of model types
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Top 3 Models")
            top_3 = comparison_df_display.nlargest(3, "Accuracy")[["Model", "Category", "Accuracy", "F1 (weighted)"]]
            for idx, row in top_3.iterrows():
                st.markdown(f"""
                **{row['Model']}** ({row['Category']})
                - Accuracy: `{row['Accuracy']:.4f}`
                - F1 Score: `{row['F1 (weighted)']:.4f}`
                """)

        with col2:
            st.markdown("#### 📊 Model Type Breakdown")
            category_stats = comparison_df_display.groupby("Category").agg({
                "Accuracy": "mean",
                "Model": "count"
            }).round(4)
            category_stats.columns = ["Avg Accuracy", "Count"]
            st.dataframe(category_stats, use_container_width=True)
        
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
