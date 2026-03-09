"""Tab components for the dashboard."""

from components.tabs.overview import render_overview_tab
from components.tabs.categories import render_categories_tab
from components.tabs.aspects import render_aspects_tab
from components.tabs.trends import render_trends_tab
from components.tabs.performance import render_performance_tab
from components.tabs.deep_dive import render_deep_dive_tab
from components.tabs.insights import render_insights_tab
from components.tabs.compare import render_compare_tab

__all__ = [
    "render_overview_tab",
    "render_categories_tab",
    "render_aspects_tab",
    "render_trends_tab",
    "render_performance_tab",
    "render_deep_dive_tab",
    "render_insights_tab",
    "render_compare_tab",
]
