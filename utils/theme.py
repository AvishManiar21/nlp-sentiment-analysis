"""
Theme configuration for the NLP Sentiment Analysis dashboard.

Uses semantic tokens for consistent theming across light and dark modes.
All colors should be accessed through get_theme_tokens() or helper functions.
"""

import streamlit as st

# Semantic theme tokens - these are the ONLY colors that should be used
THEME_LIGHT = {
    # Backgrounds
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8fafc",
    "bg_elevated": "#ffffff",
    "bg_muted": "#f1f5f9",
    
    # Text
    "text_primary": "#1e293b",
    "text_secondary": "#475569",
    "text_muted": "#64748b",
    "text_inverse": "#ffffff",
    
    # Borders
    "border_default": "#e2e8f0",
    "border_muted": "#f1f5f9",
    
    # Semantic colors
    "positive": "#16a34a",
    "negative": "#dc2626",
    "neutral": "#6b7280",
    "warning": "#d97706",
    "info": "#2563eb",
    
    # Accents
    "accent_primary": "#3b82f6",
    "accent_secondary": "#8b5cf6",
    
    # Charts
    "chart_bg": "#ffffff",
    "chart_grid": "#e5e7eb",
}

THEME_DARK = {
    # Backgrounds
    "bg_primary": "#0f172a",
    "bg_secondary": "#1e293b",
    "bg_elevated": "#334155",
    "bg_muted": "#1e293b",
    
    # Text
    "text_primary": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "text_muted": "#94a3b8",
    "text_inverse": "#0f172a",
    
    # Borders
    "border_default": "#334155",
    "border_muted": "#1e293b",
    
    # Semantic colors (slightly brighter for dark mode)
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#9ca3af",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    
    # Accents
    "accent_primary": "#60a5fa",
    "accent_secondary": "#a78bfa",
    
    # Charts
    "chart_bg": "#1e293b",
    "chart_grid": "#374151",
}

# Category colors that work in both themes
CATEGORY_PALETTE = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]
MODEL_PALETTE = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]


def get_current_theme() -> str:
    """Get current theme from session state. Returns 'light' or 'dark'."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    return "dark" if st.session_state.dark_mode else "light"


def is_dark_mode() -> bool:
    """Check if dark mode is currently active."""
    return get_current_theme() == "dark"


def toggle_theme():
    """Toggle between light and dark theme."""
    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)


def get_theme_tokens() -> dict:
    """Get the current theme's color tokens."""
    return THEME_DARK if is_dark_mode() else THEME_LIGHT


def get_sentiment_colors() -> dict:
    """Get sentiment-specific colors for the current theme."""
    tokens = get_theme_tokens()
    return {
        "positive": tokens["positive"],
        "negative": tokens["negative"],
        "neutral": tokens["neutral"],
    }


def get_plotly_theme() -> dict:
    """Get Plotly theme configuration based on current theme."""
    tokens = get_theme_tokens()
    dark = is_dark_mode()
    
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=12,
            color=tokens["text_primary"]
        ),
        title_font=dict(size=16, color=tokens["text_primary"]),
        paper_bgcolor=tokens["chart_bg"],
        plot_bgcolor=tokens["chart_bg"],
        margin=dict(t=60, b=50, l=50, r=30),
        hoverlabel=dict(
            bgcolor=tokens["bg_elevated"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color=tokens["text_primary"],
            bordercolor=tokens["border_default"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=tokens["text_secondary"]),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=tokens["chart_grid"],
            zeroline=False,
            tickfont=dict(color=tokens["text_secondary"]),
            title_font=dict(color=tokens["text_secondary"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=tokens["chart_grid"],
            zeroline=False,
            tickfont=dict(color=tokens["text_secondary"]),
            title_font=dict(color=tokens["text_secondary"]),
        ),
        colorway=CATEGORY_PALETTE,
    )


def apply_chart_theme(fig, height: int = 420, **overrides):
    """Apply unified theme to a Plotly figure. Overrides merge with theme."""
    theme = get_plotly_theme()
    layout = {**theme, "height": height, **overrides}
    fig.update_layout(**layout)
    return fig


def get_wordcloud_colors() -> dict:
    """Get colors for matplotlib word clouds based on current theme."""
    tokens = get_theme_tokens()
    return {
        "background": tokens["bg_secondary"],
        "positive_colormap": "Greens",
        "negative_colormap": "Reds",
    }


def inject_custom_css():
    """Inject custom CSS for enhanced styling that respects the current theme."""
    tokens = get_theme_tokens()
    dark = is_dark_mode()
    
    css = f"""
    <style>
        /* Base app styling */
        .stApp {{
            background-color: {tokens["bg_primary"]};
        }}
        
        /* Metric cards */
        [data-testid="stMetric"] {{
            background-color: {tokens["bg_secondary"]};
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid {tokens["border_default"]};
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {tokens["bg_secondary"]};
        }}
        
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stDateInput label {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Paragraphs and text */
        p, span, li {{
            color: {tokens["text_secondary"]};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {tokens["bg_secondary"]};
            border-radius: 0.5rem;
            padding: 0.25rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {tokens["text_secondary"]};
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {tokens["text_primary"]} !important;
            background-color: {tokens["bg_elevated"]} !important;
            border-radius: 0.375rem;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {tokens["bg_secondary"]};
            color: {tokens["text_primary"]} !important;
            border-radius: 0.5rem;
        }}
        
        .streamlit-expanderContent {{
            background-color: {tokens["bg_muted"]};
            border: 1px solid {tokens["border_default"]};
            border-top: none;
            border-radius: 0 0 0.5rem 0.5rem;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {tokens["accent_primary"]};
            color: {tokens["text_inverse"]};
            border: none;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {tokens["accent_secondary"]};
            transform: translateY(-1px);
        }}
        
        /* Download buttons */
        .stDownloadButton > button {{
            background-color: {tokens["bg_secondary"]};
            color: {tokens["text_primary"]};
            border: 1px solid {tokens["border_default"]};
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {tokens["bg_elevated"]};
            border-color: {tokens["accent_primary"]};
        }}
        
        /* Data frames */
        .stDataFrame {{
            background-color: {tokens["bg_secondary"]};
            border-radius: 0.5rem;
        }}
        
        /* Dividers */
        hr {{
            border-color: {tokens["border_default"]};
        }}
        
        /* Sentiment badges */
        .sentiment-positive {{
            background-color: {tokens["positive"]}20;
            color: {tokens["positive"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .sentiment-negative {{
            background-color: {tokens["negative"]}20;
            color: {tokens["negative"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .sentiment-neutral {{
            background-color: {tokens["neutral"]}20;
            color: {tokens["neutral"]};
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        /* Alert cards */
        .alert-card {{
            background-color: {tokens["bg_secondary"]};
            border-left: 4px solid {tokens["info"]};
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            border-radius: 0 0.5rem 0.5rem 0;
            color: {tokens["text_primary"]};
        }}
        
        .alert-card strong {{
            color: {tokens["text_primary"]};
        }}
        
        .alert-card span {{
            color: {tokens["text_secondary"]};
        }}
        
        .alert-card-success {{
            border-left-color: {tokens["positive"]};
            background-color: {tokens["positive"]}10;
        }}
        
        .alert-card-warning {{
            border-left-color: {tokens["warning"]};
            background-color: {tokens["warning"]}10;
        }}
        
        .alert-card-danger {{
            border-left-color: {tokens["negative"]};
            background-color: {tokens["negative"]}10;
        }}
        
        /* Info boxes */
        .stAlert {{
            background-color: {tokens["bg_secondary"]};
            color: {tokens["text_primary"]};
            border-radius: 0.5rem;
        }}
        
        /* Select boxes */
        .stSelectbox [data-baseweb="select"] {{
            background-color: {tokens["bg_secondary"]};
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {tokens["bg_secondary"]};
            border-color: {tokens["border_default"]};
            color: {tokens["text_primary"]};
        }}
        
        /* Multiselect */
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {tokens["accent_primary"]};
            color: {tokens["text_inverse"]};
        }}
        
        /* Progress bars */
        .stProgress > div > div {{
            background-color: {tokens["accent_primary"]};
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            background-color: transparent;
        }}
        
        .stRadio label {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Spacing utilities */
        .section-spacing {{
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        /* Card container */
        .card {{
            background-color: {tokens["bg_secondary"]};
            border: 1px solid {tokens["border_default"]};
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        
        /* Loading skeleton */
        .skeleton {{
            background: linear-gradient(
                90deg,
                {tokens["bg_secondary"]} 25%,
                {tokens["bg_elevated"]} 50%,
                {tokens["bg_secondary"]} 75%
            );
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 0.375rem;
        }}
        
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        /* Winner badges */
        .winner-badge {{
            background-color: {tokens["positive"]}20;
            color: {tokens["positive"]};
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            text-align: center;
        }}
        
        /* Plotly charts - ensure proper background */
        .js-plotly-plot .plotly {{
            background-color: {tokens["chart_bg"]} !important;
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# Legacy compatibility - keep old names working
COLORS = THEME_LIGHT
COLORS_DARK = THEME_DARK


def get_colors():
    """Legacy function - use get_theme_tokens() instead."""
    return get_theme_tokens()
