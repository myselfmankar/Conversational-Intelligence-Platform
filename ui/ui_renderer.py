import streamlit as st
import pandas as pd

from .tab_overview import render_overview_tab
from .tab_sentiment import render_sentiment_tab
from .tab_brand import render_brand_intelligence_tab
from .tab_ner import render_ner_tab
from .tab_dynamics import render_dynamics_tab
from .tab_health import render_health_tab
from .tab_download import render_download_tab

def render_dashboard(df_processed: pd.DataFrame):
    """
    Renders the entire dashboard, including sidebar filters and all tabs,
    based on the processed DataFrame.
    """
    # --- Sidebar Filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Filters")
    
    # Create a filterable copy of the data
    df_display = df_processed.copy()

    # Author Filter
    unique_authors = sorted(list(df_display[~df_display['is_system']]['author'].astype(str).unique()))
    selected_author = st.sidebar.selectbox("Filter by Author:", ["All Authors"] + unique_authors)
    if selected_author != "All Authors":
        df_display = df_display[df_display['author'] == selected_author]

    # Keyword Filter
    keyword = st.sidebar.text_input("Filter by Keyword (case-insensitive):")
    if keyword:
        df_display = df_display[df_display['message'].astype(str).str.contains(keyword, case=False, na=False)]
        
    # --- Main Dashboard Area ---
    st.header("Analysis Dashboard")
    st.caption(f"Displaying results for: **{st.session_state.get('current_file_name', 'your chat')}**")

    if df_display.empty:
        st.warning("No messages match the current filter criteria.")
        return

    # --- Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages (filtered)", f"{len(df_display)}")
    col2.metric("Active Participants (filtered)", f"{df_display[~df_display['is_system']]['author'].nunique()}")
    col3.metric("Media Messages (filtered)", f"{len(df_display[df_display['message_type'] == 'media'])}")

    # --- Tabs ---
    tab_titles = ["📊 Overview", "😊 Sentiment", "💡 Brand Intelligence", "📝 NER", "🌐 Dynamics", "🛡️ Health", "💾 Download"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        render_overview_tab(df_display)
    with tabs[1]:
        render_sentiment_tab(df_display)
    with tabs[2]:
        render_brand_intelligence_tab(df_display)
    with tabs[3]:
        render_ner_tab(df_display)
    with tabs[4]:
        render_dynamics_tab(df_display)
    with tabs[5]:
        render_health_tab(df_display)
    with tabs[6]:
        # Download tab should have access to the full, unfiltered data
        render_download_tab(df_processed) 