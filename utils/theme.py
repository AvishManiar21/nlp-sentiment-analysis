"""Theme configuration for the NLP Sentiment Analysis dashboard."""

import streamlit as st

COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#94a3b8",
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "background": "#ffffff",
    "surface": "#f8fafc",
    "text": "#334155",
    "text_secondary": "#64748b",
    "border": "#e2e8f0",
}

COLORS_DARK = {
    "positive": "#4ade80",
    "negative": "#f87171",
    "neutral": "#94a3b8",
    "primary": "#60a5fa",
    "secondary": "#a78bfa",
    "background": "#0f172a",
    "surface": "#1e293b",
    "text": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "border": "#334155",
}

CATEGORY_PALETTE = ["#3b82f6", "#8b5cf6", "#22c55e", "#f59e0b", "#ec4899", "#06b6d4"]
MODEL_PALETTE = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]


def get_current_theme():
    """Get current theme from session state."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    return "dark" if st.session_state.dark_mode else "light"


def toggle_theme():
    """Toggle between light and dark theme."""
    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)


def get_colors():
    """Get colors based on current theme."""
    return COLORS_DARK if get_current_theme() == "dark" else COLORS


def get_plotly_theme():
    """Get Plotly theme configuration based on current theme."""
    colors = get_colors()
    is_dark = get_current_theme() == "dark"
    
    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=12,
            color=colors["text"]
        ),
        title_font=dict(size=16, color=colors["text"]),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["surface"],
        margin=dict(t=60, b=50, l=50, r=30),
        hoverlabel=dict(
            bgcolor=colors["surface"],
            font_size=12,
            font_family="Inter"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=f"rgba({_hex_to_rgb(colors['background'])}, 0.8)",
            bordercolor=colors["border"],
        ),
        xaxis=dict(showgrid=True, gridcolor=colors["border"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=colors["border"], zeroline=False),
        colorway=CATEGORY_PALETTE,
    )


def _hex_to_rgb(hex_color):
    """Convert hex color to RGB string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


def apply_chart_theme(fig, height=420, **overrides):
    """Apply unified theme to a Plotly figure. Overrides merge with theme."""
    theme = get_plotly_theme()
    layout = {**theme, "height": height, **overrides}
    fig.update_layout(**layout)
    return fig


def get_sentiment_colors():
    """Get sentiment-specific colors based on current theme."""
    colors = get_colors()
    return {
        "positive": colors["positive"],
        "negative": colors["negative"],
        "neutral": colors["neutral"],
    }


def inject_custom_css():
    """Inject custom CSS for enhanced styling."""
    colors = get_colors()
    is_dark = get_current_theme() == "dark"
    
    st.markdown(f"""
    <style>
        /* Card styling */
        .stMetric {{
            background-color: {colors["surface"]};
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid {colors["border"]};
        }}
        
        /* Sentiment badges */
        .sentiment-positive {{
            background-color: {colors["positive"]}20;
            color: {colors["positive"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 500;
        }}
        .sentiment-negative {{
            background-color: {colors["negative"]}20;
            color: {colors["negative"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 500;
        }}
        .sentiment-neutral {{
            background-color: {colors["neutral"]}20;
            color: {colors["neutral"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 500;
        }}
        
        /* Alert cards */
        .alert-card {{
            background-color: {colors["surface"]};
            border-left: 4px solid {colors["primary"]};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 0.5rem 0.5rem 0;
        }}
        .alert-card-warning {{
            border-left-color: #f59e0b;
        }}
        .alert-card-danger {{
            border-left-color: {colors["negative"]};
        }}
        .alert-card-success {{
            border-left-color: {colors["positive"]};
        }}
        
        /* Section headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .section-header h3 {{
            margin: 0;
        }}
        
        /* Loading skeleton */
        .skeleton {{
            background: linear-gradient(
                90deg,
                {colors["surface"]} 25%,
                {colors["border"]} 50%,
                {colors["surface"]} 75%
            );
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 0.25rem;
        }}
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
    </style>
    """, unsafe_allow_html=True)
