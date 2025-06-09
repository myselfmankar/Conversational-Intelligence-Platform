import streamlit as st
import pandas as pd
from visuals import charts
import plotly.graph_objects as go

def create_metric_card(title, value, sparkline_data=None, help_text=""):
    """Helper function to create a visually appealing metric card."""
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"## {value}")
        if sparkline_data is not None and not sparkline_data.empty:
            # Create a simple, clean sparkline
            spark_fig = go.Figure(go.Scatter(
                x=sparkline_data.index,
                y=sparkline_data.values,
                mode='lines',
                line=dict(width=2, color='#00BFFF'), # Deep Sky Blue
                fill='tozeroy',
            ))
            spark_fig.update_layout(
                height=50,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(spark_fig, use_container_width=True)
        st.caption(help_text)

def render_brand_intelligence_tab(df_display: pd.DataFrame):
    """
    Renders the revamped 'Brand Intelligence' tab with topic suggestions
    and a dynamic, dashboard-like layout.
    """
    st.header("💡 Brand & Topic Intelligence")
    st.info("Analyze sentiment and activity around specific keywords. Use our suggestions or enter your own.")

    # --- Step 1: Intelligent Topic Suggestions ---
    suggested_topics = charts.get_suggested_topics(df_display)
    
    if suggested_topics:
        st.markdown("**Suggested Topics (from NER):**")
        # Use st.expander to make this section neat
        with st.expander("Click to see suggested topics to track"):
            # Display suggestions as clickable buttons that can populate the text input
            if st.button(suggested_topics[0], type="secondary"):
                st.session_state.topics_str = suggested_topics[0]
            if len(suggested_topics) > 1 and st.button(suggested_topics[1], type="secondary"):
                st.session_state.topics_str = suggested_topics[1]
            if len(suggested_topics) > 2 and st.button(suggested_topics[2], type="secondary"):
                st.session_state.topics_str = suggested_topics[2]
            st.caption("Clicking a suggestion will add it to the tracking input below.")

    # --- Step 2: Topic Input ---
    # Use session_state to remember the user's input
    if 'topics_str' not in st.session_state:
        st.session_state.topics_str = "price, battery, bug"

    topics_str = st.text_input(
        "**Topics to Track (comma-separated):**",
        key="topics_str",
        help="Enter keywords you want to analyze."
    )
    
    if not topics_str:
        st.warning("Please enter at least one topic to analyze.")
        st.stop()

    topics = [topic.strip().lower() for topic in topics_str.split(',') if topic.strip()]

    st.markdown("---")

    # --- Step 3: Revamped Dynamic Dashboard Layout ---
    for topic in topics:
        st.markdown(f"### Dashboard for: `{topic}`")
        
        topic_df = df_display[df_display['message'].str.contains(topic, case=False, na=False)].copy()

        if topic_df.empty:
            st.warning(f"No mentions found for '{topic}'. Try another keyword.")
            continue

        # Prepare data for sparkline (mentions over time)
        topic_df['date'] = pd.to_datetime(topic_df['datetime']).dt.date
        mentions_over_time = topic_df.groupby('date').size()

        # Create a 3-column layout for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Total Mentions",
                topic_df.shape[0],
                mentions_over_time,
                "Number of times this topic was mentioned."
            )

        with col2:
            metrics = charts.get_topic_metrics(topic_df, topic)
            pos_ratio = metrics.get('positive_ratio', 0)
            neg_ratio = metrics.get('negative_ratio', 0)
            net_sentiment = pos_ratio - neg_ratio
            
            create_metric_card(
                "Net Sentiment",
                f"{net_sentiment:.1f}%",
                help_text=f"({pos_ratio:.1f}% Positive vs. {neg_ratio:.1f}% Negative)"
            )

        with col3:
            avg_toxicity = metrics.get('avg_toxicity', 0)
            create_metric_card(
                "Avg. Toxicity",
                f"{avg_toxicity:.3f}",
                help_text="Average toxicity score for messages mentioning this topic."
            )
        
        st.markdown("---") # Separator for the next topic