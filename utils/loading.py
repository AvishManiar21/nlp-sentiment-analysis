"""Loading state utilities and skeleton components."""

import streamlit as st
import time


def render_skeleton_card(height: int = 100, width: str = "100%"):
    """Render a skeleton loading card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #f0f2f6 25%, #e8eaed 50%, #f0f2f6 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 0.5rem;
        height: {height}px;
        width: {width};
    "></div>
    <style>
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
    </style>
    """, unsafe_allow_html=True)


def render_skeleton_chart(height: int = 400):
    """Render a skeleton for chart loading."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #f0f2f6 25%, #e8eaed 50%, #f0f2f6 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 0.5rem;
        height: {height}px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <span style="color: #94a3b8; font-size: 14px;">Loading chart...</span>
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_metrics(count: int = 6):
    """Render skeleton metrics row."""
    cols = st.columns(count)
    for col in cols:
        with col:
            render_skeleton_card(height=80)


def render_skeleton_table(rows: int = 5, cols: int = 4):
    """Render a skeleton table."""
    st.markdown("""
    <div style="
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
    ">
    """, unsafe_allow_html=True)
    
    header_cols = st.columns(cols)
    for col in header_cols:
        with col:
            render_skeleton_card(height=30)
    
    for _ in range(rows):
        row_cols = st.columns(cols)
        for col in row_cols:
            with col:
                render_skeleton_card(height=25)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_progress_steps(steps: list, current_step: int):
    """
    Render a progress stepper.
    
    Args:
        steps: List of step labels
        current_step: Index of current step (0-based)
    """
    st.markdown("""
    <style>
        .progress-container {
            display: flex;
            justify-content: space-between;
            margin: 1rem 0;
        }
        .progress-step {
            flex: 1;
            text-align: center;
            position: relative;
        }
        .progress-step::before {
            content: '';
            position: absolute;
            top: 15px;
            left: 50%;
            width: 100%;
            height: 2px;
            background: #e2e8f0;
            z-index: 0;
        }
        .progress-step:last-child::before {
            display: none;
        }
        .progress-circle {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            position: relative;
            z-index: 1;
        }
        .progress-circle.completed {
            background: #22c55e;
            color: white;
        }
        .progress-circle.current {
            background: #3b82f6;
            color: white;
            animation: pulse 1.5s infinite;
        }
        .progress-circle.pending {
            background: #e2e8f0;
            color: #94a3b8;
        }
        .progress-label {
            font-size: 12px;
            margin-top: 0.5rem;
            color: #64748b;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    steps_html = '<div class="progress-container">'
    for i, step in enumerate(steps):
        if i < current_step:
            status = "completed"
            icon = "✓"
        elif i == current_step:
            status = "current"
            icon = str(i + 1)
        else:
            status = "pending"
            icon = str(i + 1)
        
        steps_html += f'''
        <div class="progress-step">
            <div class="progress-circle {status}">{icon}</div>
            <div class="progress-label">{step}</div>
        </div>
        '''
    steps_html += '</div>'
    
    st.markdown(steps_html, unsafe_allow_html=True)


def with_loading(func):
    """Decorator to add loading state to a function."""
    def wrapper(*args, **kwargs):
        placeholder = st.empty()
        with placeholder:
            with st.spinner("Loading..."):
                result = func(*args, **kwargs)
        return result
    return wrapper


class LoadingContext:
    """Context manager for loading states with progress."""
    
    def __init__(self, message: str = "Loading...", show_progress: bool = True):
        self.message = message
        self.show_progress = show_progress
        self.placeholder = None
        self.progress_bar = None
    
    def __enter__(self):
        self.placeholder = st.empty()
        with self.placeholder.container():
            st.info(f"⏳ {self.message}")
            if self.show_progress:
                self.progress_bar = st.progress(0)
        return self
    
    def update(self, progress: float, message: str = None):
        """Update progress (0.0 to 1.0) and optionally message."""
        if self.progress_bar:
            self.progress_bar.progress(progress)
        if message and self.placeholder:
            with self.placeholder.container():
                st.info(f"⏳ {message}")
                if self.show_progress:
                    self.progress_bar = st.progress(progress)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.placeholder:
            self.placeholder.empty()
        return False


def render_data_generation_progress():
    """Render progress for data generation on cloud deployment."""
    steps = [
        "Download Reviews",
        "Preprocess Text",
        "Sentiment Analysis",
        "Train Models",
        "Generate Results",
    ]
    
    if "data_gen_step" not in st.session_state:
        st.session_state.data_gen_step = 0
    
    render_progress_steps(steps, st.session_state.data_gen_step)
    
    return st.session_state.data_gen_step


def increment_data_gen_step():
    """Increment the data generation step."""
    if "data_gen_step" not in st.session_state:
        st.session_state.data_gen_step = 0
    st.session_state.data_gen_step += 1
