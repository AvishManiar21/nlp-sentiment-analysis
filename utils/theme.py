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
    """
    Get current theme name.

    Dark mode has been removed from the app, so we always return 'light'.
    """
    return "light"


def is_dark_mode() -> bool:
    """
    Check if dark mode is currently active.

    Dark mode support has been removed, so this always returns False.
    """
    return False


def toggle_theme():
    """
    Toggle between light and dark theme.

    Kept for backwards compatibility, but now a no-op since the app
    is locked to the light theme.
    """
    return None


def get_theme_tokens() -> dict:
    """Get the current theme's color tokens (light theme only)."""
    return THEME_LIGHT


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
        /* ===== BASE APP STYLING ===== */
        .stApp {{
            background-color: {tokens["bg_primary"]};
        }}
        
        /* ===== TYPOGRAPHY - COMPREHENSIVE COVERAGE ===== */
        /* All headings - both direct and inside markdown containers */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Subheaders rendered by st.subheader */
        .stSubheader, [data-testid="stSubheader"] {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Paragraphs, lists, and general text */
        p, li,
        .stMarkdown p, .stMarkdown li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Span elements (used in many Streamlit components) */
        span {{
            color: inherit;
        }}
        
        /* Bold text - ensure it uses primary color for emphasis */
        strong, b,
        .stMarkdown strong, .stMarkdown b,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Caption/small text */
        .stCaption, [data-testid="stCaption"],
        small, .stMarkdown small,
        figcaption {{
            color: {tokens["text_muted"]} !important;
        }}
        
        /* Code blocks and inline code */
        code, pre,
        .stMarkdown code, .stMarkdown pre,
        [data-testid="stMarkdownContainer"] code {{
            background-color: {tokens["bg_muted"]} !important;
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Blockquotes */
        blockquote,
        .stMarkdown blockquote {{
            border-left-color: {tokens["border_default"]} !important;
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Links */
        a, .stMarkdown a {{
            color: {tokens["accent_primary"]} !important;
        }}
        
        a:hover, .stMarkdown a:hover {{
            color: {tokens["accent_secondary"]} !important;
        }}
        
        /* ===== METRIC CARDS ===== */
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
        
        [data-testid="stMetricDelta"] {{
            opacity: 0.9;
        }}
        
        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background-color: {tokens["bg_secondary"]};
        }}
        
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stRadio label {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {tokens["bg_secondary"]};
            border-radius: 0.5rem;
            padding: 0.25rem;
            gap: 0.25rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {tokens["text_secondary"]} !important;
            background-color: transparent;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {tokens["text_primary"]} !important;
            background-color: {tokens["bg_elevated"]} !important;
            border-radius: 0.375rem;
        }}
        
        /* Tab panel content */
        .stTabs [data-baseweb="tab-panel"] {{
            color: {tokens["text_secondary"]};
        }}
        
        /* ===== EXPANDERS ===== */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary {{
            background-color: {tokens["bg_secondary"]} !important;
            color: {tokens["text_primary"]} !important;
            border-radius: 0.5rem;
        }}
        
        .streamlit-expanderContent,
        [data-testid="stExpander"] > div {{
            background-color: {tokens["bg_muted"]};
            border: 1px solid {tokens["border_default"]};
            border-top: none;
            border-radius: 0 0 0.5rem 0.5rem;
        }}
        
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* ===== BUTTONS ===== */
        .stButton > button {{
            background-color: {tokens["accent_primary"]};
            color: {tokens["text_inverse"]} !important;
            border: none;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {tokens["accent_secondary"]};
            transform: translateY(-1px);
        }}
        
        .stDownloadButton > button {{
            background-color: {tokens["bg_secondary"]};
            color: {tokens["text_primary"]} !important;
            border: 1px solid {tokens["border_default"]};
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {tokens["bg_elevated"]};
            border-color: {tokens["accent_primary"]};
        }}
        
        /* ===== DATA FRAMES & TABLES ===== */
        .stDataFrame {{
            background-color: {tokens["bg_secondary"]};
            border-radius: 0.5rem;
        }}
        
        .stDataFrame [data-testid="stDataFrameResizable"],
        .stDataFrame table {{
            color: {tokens["text_primary"]} !important;
        }}
        
        .stDataFrame th {{
            background-color: {tokens["bg_muted"]} !important;
            color: {tokens["text_primary"]} !important;
        }}
        
        .stDataFrame td {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Pandas Styler rendered tables */
        .dataframe, .dataframe th, .dataframe td {{
            color: {tokens["text_primary"]} !important;
            border-color: {tokens["border_default"]} !important;
        }}
        
        /* ===== DIVIDERS ===== */
        hr, [data-testid="stDivider"] {{
            border-color: {tokens["border_default"]} !important;
        }}
        
        /* ===== SENTIMENT BADGES ===== */
        .sentiment-positive {{
            background-color: {tokens["positive"]}20;
            color: {tokens["positive"]} !important;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .sentiment-negative {{
            background-color: {tokens["negative"]}20;
            color: {tokens["negative"]} !important;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .sentiment-neutral {{
            background-color: {tokens["neutral"]}20;
            color: {tokens["neutral"]} !important;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        /* ===== ALERT CARDS ===== */
        .alert-card {{
            background-color: {tokens["bg_secondary"]};
            border-left: 4px solid {tokens["info"]};
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            border-radius: 0 0.5rem 0.5rem 0;
            color: {tokens["text_primary"]};
        }}
        
        .alert-card strong {{
            color: {tokens["text_primary"]} !important;
        }}
        
        .alert-card span {{
            color: {tokens["text_secondary"]} !important;
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
        
        /* ===== STREAMLIT ALERTS (st.info, st.success, etc.) ===== */
        .stAlert, [data-testid="stAlert"] {{
            background-color: {tokens["bg_secondary"]} !important;
            color: {tokens["text_primary"]} !important;
            border-radius: 0.5rem;
        }}
        
        .stAlert p, [data-testid="stAlert"] p {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* ===== FORM INPUTS ===== */
        /* Text inputs */
        .stTextInput input, .stTextArea textarea {{
            background-color: {tokens["bg_secondary"]} !important;
            color: {tokens["text_primary"]} !important;
            border-color: {tokens["border_default"]} !important;
        }}
        
        .stTextInput label, .stTextArea label {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Select boxes */
        .stSelectbox [data-baseweb="select"] {{
            background-color: {tokens["bg_secondary"]};
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {tokens["bg_secondary"]} !important;
            border-color: {tokens["border_default"]} !important;
            color: {tokens["text_primary"]} !important;
        }}
        
        .stSelectbox label {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        /* Multiselect */
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {tokens["accent_primary"]} !important;
            color: {tokens["text_inverse"]} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] > div {{
            background-color: {tokens["bg_secondary"]} !important;
            border-color: {tokens["border_default"]} !important;
        }}
        
        /* Sliders */
        .stSlider label {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        .stSlider [data-baseweb="slider"] div {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* Radio buttons and checkboxes */
        .stRadio > div {{
            background-color: transparent;
        }}
        
        .stRadio label, .stCheckbox label {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* ===== PROGRESS BARS ===== */
        .stProgress > div > div {{
            background-color: {tokens["accent_primary"]};
        }}
        
        /* ===== UTILITY CLASSES ===== */
        .section-spacing {{
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .card {{
            background-color: {tokens["bg_secondary"]};
            border: 1px solid {tokens["border_default"]};
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        
        .card p, .card span {{
            color: {tokens["text_secondary"]} !important;
        }}
        
        .card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {{
            color: {tokens["text_primary"]} !important;
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
            color: {tokens["positive"]} !important;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            text-align: center;
        }}
        
        /* ===== PLOTLY CHARTS ===== */
        .js-plotly-plot .plotly {{
            background-color: {tokens["chart_bg"]} !important;
        }}
        
        /* Ensure Plotly tooltips are readable */
        .plotly .hoverlayer text {{
            fill: {tokens["text_primary"]} !important;
        }}
        
        /* ===== MATPLOTLIB FIGURES ===== */
        .stImage img, [data-testid="stImage"] img {{
            border-radius: 0.5rem;
        }}
        
        /* ===== TOOLTIP/POPOVER ===== */
        [data-baseweb="popover"] {{
            background-color: {tokens["bg_elevated"]} !important;
        }}
        
        [data-baseweb="popover"] * {{
            color: {tokens["text_primary"]} !important;
        }}
        
        /* ===== JSON VIEWER (for debugging) ===== */
        .stJson {{
            background-color: {tokens["bg_muted"]} !important;
        }}
        
        .stJson * {{
            color: {tokens["text_secondary"]} !important;
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
