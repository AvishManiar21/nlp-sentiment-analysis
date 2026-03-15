"""Business Insights tab component with actionable intelligence."""

import streamlit as st
import pandas as pd
from utils.theme import get_sentiment_colors, get_theme_tokens


def _calculate_insights(df: pd.DataFrame) -> dict:
    """Calculate business insights from the data."""
    insights = {
        "alerts": [],
        "top_issues": [],
        "recommendations": [],
        "trends": {},
    }
    
    if len(df) < 10:
        return insights
    
    total = len(df)
    neg_pct = (df["sentiment_label"] == "negative").mean() * 100
    pos_pct = (df["sentiment_label"] == "positive").mean() * 100
    
    if neg_pct > 20:
        insights["alerts"].append({
            "type": "danger",
            "title": "High Negative Sentiment",
            "message": f"{neg_pct:.1f}% of reviews are negative - above 20% threshold",
            "priority": "high",
        })
    
    if pos_pct > 80:
        insights["alerts"].append({
            "type": "success",
            "title": "Excellent Customer Satisfaction",
            "message": f"{pos_pct:.1f}% positive reviews - great performance!",
            "priority": "low",
        })
    
    if "category" in df.columns:
        cat_sentiment = df.groupby("category").agg(
            neg_pct=("sentiment_label", lambda x: (x == "negative").mean() * 100),
            count=("sentiment_label", "count"),
        ).reset_index()
        
        problem_cats = cat_sentiment[
            (cat_sentiment["neg_pct"] > 25) & (cat_sentiment["count"] >= 20)
        ]
        
        for _, row in problem_cats.iterrows():
            insights["alerts"].append({
                "type": "warning",
                "title": f"Category Alert: {row['category']}",
                "message": f"{row['neg_pct']:.1f}% negative sentiment ({row['count']} reviews)",
                "priority": "medium",
            })
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    negative_reviews = df[df["sentiment_label"] == "negative"]
    
    if len(negative_reviews) > 0 and text_col in df.columns:
        issue_keywords = [
            ("shipping", "delivery", "arrived"),
            ("quality", "defect", "broken", "damaged"),
            ("price", "expensive", "overpriced", "cheap"),
            ("service", "support", "response", "help"),
            ("size", "fit", "small", "large"),
        ]
        
        issue_labels = ["Shipping/Delivery", "Quality Issues", "Pricing Concerns", "Customer Service", "Size/Fit"]
        
        neg_text = " ".join(negative_reviews[text_col].dropna().astype(str).str.lower())
        
        issue_counts = []
        for keywords in issue_keywords:
            count = sum(neg_text.count(kw) for kw in keywords)
            issue_counts.append(count)
        
        sorted_issues = sorted(zip(issue_labels, issue_counts), key=lambda x: x[1], reverse=True)
        
        for label, count in sorted_issues[:5]:
            if count > 0:
                pct = (count / len(negative_reviews)) * 100
                insights["top_issues"].append({
                    "issue": label,
                    "mentions": count,
                    "percentage": min(pct, 100),
                })
    
    if insights["top_issues"]:
        top_issue = insights["top_issues"][0]
        insights["recommendations"].append({
            "priority": "high",
            "action": f"Focus on {top_issue['issue']}",
            "reason": f"This is mentioned in {top_issue['mentions']} negative reviews",
            "impact": "Could reduce negative sentiment by up to 20%",
        })
    
    if neg_pct > 15:
        insights["recommendations"].append({
            "priority": "medium",
            "action": "Implement proactive outreach",
            "reason": "Contact customers with 1-2 star ratings within 24 hours",
            "impact": "Can convert 10-15% of detractors to promoters",
        })
    
    if "category" in df.columns:
        cat_sentiment = df.groupby("category").agg(
            avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in df.columns else ("rating", "mean"),
        ).reset_index()
        
        for _, row in cat_sentiment.iterrows():
            insights["trends"][row["category"]] = {
                "sentiment": row["avg_sentiment"] if "avg_sentiment" in row else 0,
                "direction": "up" if row.get("avg_sentiment", 0) > 0 else "down",
            }
    
    return insights


def _render_alert_card(alert: dict):
    """Render an alert card using theme-aware colors."""
    tokens = get_theme_tokens()
    
    type_colors = {
        "danger": tokens["negative"],
        "warning": tokens["warning"],
        "success": tokens["positive"],
        "info": tokens["info"],
    }
    
    type_icons = {
        "danger": "🔴",
        "warning": "🟡",
        "success": "🟢",
        "info": "🔵",
    }
    
    color = type_colors.get(alert["type"], tokens["info"])
    icon = type_icons.get(alert["type"], "ℹ️")
    
    st.markdown(f"""
    <div class="alert-card alert-card-{alert['type']}" style="
        background-color: {color}15;
        border-left: 4px solid {color};
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    ">
        <strong style="color: {tokens['text_primary']};">{icon} {alert['title']}</strong><br>
        <span style="color: {tokens['text_secondary']};">{alert['message']}</span>
    </div>
    """, unsafe_allow_html=True)


def render_insights_tab(df: pd.DataFrame):
    """Render the Business Insights tab content."""
    st.subheader("Business Insights & Recommendations")
    
    insights = _calculate_insights(df)
    
    st.markdown("### Alerts & Notifications")
    
    if insights["alerts"]:
        for alert in sorted(insights["alerts"], key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 3)):
            _render_alert_card(alert)
    else:
        st.success("No critical alerts - all metrics within normal ranges!")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Issues in Negative Reviews")
        
        if insights["top_issues"]:
            for i, issue in enumerate(insights["top_issues"], 1):
                st.markdown(f"""
                **{i}. {issue['issue']}**  
                📊 {issue['mentions']} mentions ({issue['percentage']:.1f}% of negative reviews)
                """)
                st.progress(min(issue['percentage'] / 100, 1.0))
        else:
            st.info("Not enough negative reviews to identify patterns.")
    
    with col2:
        st.markdown("### Actionable Recommendations")
        
        if insights["recommendations"]:
            for rec in insights["recommendations"]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
                
                with st.expander(f"{priority_emoji} {rec['action']}", expanded=rec["priority"] == "high"):
                    st.write(f"**Why:** {rec['reason']}")
                    st.write(f"**Expected Impact:** {rec['impact']}")
        else:
            st.success("Great job! No urgent actions needed.")
    
    st.divider()
    
    st.markdown("### Category Health Overview")
    
    if insights["trends"]:
        cols = st.columns(min(len(insights["trends"]), 4))
        
        for i, (category, trend) in enumerate(insights["trends"].items()):
            with cols[i % len(cols)]:
                delta = trend["sentiment"]
                delta_color = "normal" if delta >= 0 else "inverse"
                
                st.metric(
                    label=category,
                    value=f"{delta:.3f}",
                    delta=f"{'↑' if delta >= 0 else '↓'} {'Positive' if delta >= 0 else 'Negative'}",
                    delta_color=delta_color,
                )
    else:
        st.info("No category data available for trend analysis.")
    
    st.divider()
    
    st.markdown("### Quick Stats Summary")
    
    total = len(df)
    if total == 0:
        st.info("No data for quick stats. Adjust filters to see insights.")
        return
    
    pos_count = (df["sentiment_label"] == "positive").sum()
    neg_count = (df["sentiment_label"] == "negative").sum()
    neu_count = (df["sentiment_label"] == "neutral").sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", f"{total:,}")
    with col2:
        st.metric("Positive", f"{pos_count:,}", delta=f"{(pos_count/total*100):.1f}%")
    with col3:
        st.metric("Neutral", f"{neu_count:,}", delta=f"{(neu_count/total*100):.1f}%", delta_color="off")
    with col4:
        st.metric("Negative", f"{neg_count:,}", delta=f"-{(neg_count/total*100):.1f}%", delta_color="inverse")
