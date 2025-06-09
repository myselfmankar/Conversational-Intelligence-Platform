# WhatsApp Chat NLP Analyzer
import streamlit as st
import pandas as pd
import re

# --- Import Custom Modules ---
from utils.parser import parse_whatsapp_chat
from utils.file_handler import handle_uploaded_file
from nlp.enrich import enrich_df_with_nlp
from visuals import charts
from ui import tab_brand, tab_overview, tab_sentiment, tab_ner, tab_dynamics, tab_health, tab_download

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="WhatsApp Chat NLP Analyzer")
st.title("WhatsApp Chat NLP Analyzer 💬")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload your WhatsApp chat export (.txt file)", 
    type="txt"
)

# Initialize session state
if 'df_chat_raw' not in st.session_state:
    st.session_state.df_chat_raw = pd.DataFrame()
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = pd.DataFrame()
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False
if 'current_file_name' not in st.session_state: 
    st.session_state.current_file_name = None


if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        # Reset state if a new file is uploaded
        st.session_state.df_chat_raw = pd.DataFrame()
        st.session_state.df_processed = pd.DataFrame()
        st.session_state.analysis_triggered = False
        st.session_state.current_file_name = uploaded_file.name
        st.info(f"New file '{uploaded_file.name}' selected. Click 'Analyze Chat' to process.")


    if st.sidebar.button("Analyze Chat", key="analyze_button"):
        st.session_state.analysis_triggered = True 
        chat_content = handle_uploaded_file(uploaded_file)
        
        if chat_content:
            with st.spinner("Parsing chat file..."):
                st.session_state.df_chat_raw = parse_whatsapp_chat(chat_content)
            
            if st.session_state.df_chat_raw.empty or 'datetime' not in st.session_state.df_chat_raw.columns:
                st.error("Could not parse the chat file. Please check format or content.")
                st.session_state.analysis_triggered = False 
            else:
                st.success(f"Chat parsed: {len(st.session_state.df_chat_raw)} messages.")
                
                with st.spinner("Performing NLP analysis (Summarization & Sentiment)..."):
                    st.session_state.df_processed = enrich_df_with_nlp(st.session_state.df_chat_raw)
                
                if st.session_state.df_processed.empty:
                     st.warning("NLP processing resulted in an empty dataset.")
                     st.session_state.analysis_triggered = False
                else:
                    st.success("NLP analysis complete!")
        else:
            st.error("Failed to read or process the uploaded file.")
            st.session_state.analysis_triggered = False


# --- Display Results if DataFrame is processed and analysis was triggered ---
if st.session_state.analysis_triggered and not st.session_state.df_processed.empty:
    df_display = st.session_state.df_processed.copy()

    st.header("Chat Analysis Dashboard")

    # --- Filters in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Create a unique list of authors excluding "System" for the filter
    unique_authors_for_filter = sorted(list(df_display[~df_display['is_system']]['author'].astype(str).unique()))
    all_authors_option = ["All Authors"] + unique_authors_for_filter
    selected_author = st.sidebar.selectbox("Filter by Author:", all_authors_option, key="author_filter")
    if selected_author != "All Authors":
        df_display = df_display[df_display['author'] == selected_author]

    keyword = st.sidebar.text_input("Filter by Keyword (in original message, case-insensitive):", key="keyword_filter")
    if keyword:
        df_display = df_display[df_display['message'].astype(str).str.contains(keyword, case=False, na=False, regex=False)] 

    # Ensure 'sentiment_label' column exists before trying to get unique values
    if 'sentiment_label' in df_display.columns:
        unique_sentiments = sorted(list(df_display['sentiment_label'].astype(str).unique()))
        all_sentiments_option = ["All Sentiments"] + unique_sentiments
        selected_sentiment = st.sidebar.selectbox("Filter by Sentiment:", all_sentiments_option, key="sentiment_filter")
        if selected_sentiment != "All Sentiments":
            df_display = df_display[df_display['sentiment_label'] == selected_sentiment]
    else:
        st.sidebar.caption("Sentiment data not available for filtering.")


    if df_display.empty and (selected_author != "All Authors" or keyword or (('sentiment_label' in df_display.columns and selected_sentiment != "All Sentiments"))):
        st.warning("No messages match the current filter criteria.")
    elif df_display.empty and uploaded_file: # This means df_processed was empty or became empty before filters
         st.info("The chat appears to be empty or contained no processable messages after initial processing.")
    elif not df_display.empty:
        tab_titles = [
            "📊 Overview & Activity",
            "😊 Sentiment Analysis",
            "💡 Brand Intelligence",
            "📝 Content & Topics (NER)",
            "🌐 Group Dynamics",
            "🛡️ Community Health",
            "💾 Download Data"
        ]
        tab_overview_ui, tab_sentiment_ui, tab_brand_ui, tab_ner_ui, tab_dynamics_ui, tab_health_ui, tab_download_ui = st.tabs(tab_titles)

        with tab_overview_ui:
            tab_overview.render_overview_tab(df_display)
        with tab_sentiment_ui:
            tab_sentiment.render_sentiment_tab(df_display)
        with tab_brand_ui:
            tab_brand.render_brand_intelligence_tab(df_display)
        with tab_ner_ui:
            tab_ner.render_ner_tab(df_display)
        with tab_dynamics_ui:
            tab_dynamics.render_dynamics_tab(df_display)
        with tab_health_ui:
            tab_health.render_health_tab(df_display)
        with tab_download_ui:
            tab_download.render_download_tab(df_display)
elif not st.session_state.analysis_triggered and uploaded_file:
    st.info("☝️ Click the 'Analyze Chat' button in the sidebar to process the uploaded file.")

elif not uploaded_file:
     st.info(
         """
         👋 Welcome to the WhatsApp Chat Analyzer!
         
         **How to use:**
         1. Export a chat from WhatsApp as a `.txt` file.
         2. Upload it using the sidebar on the left.
         3. Click 'Analyze Chat'!
         """
     )

st.sidebar.markdown("---")
st.sidebar.info("Built with Python, Streamlit, Pandas, and Hugging Face Transformers.")