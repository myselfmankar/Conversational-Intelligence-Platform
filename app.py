# Conversational Intelligence Platform
import streamlit as st
import pandas as pd

# --- Import Custom Modules ---
from utils.parser import parse_whatsapp_chat
from nlp.enrich import enrich_df_with_nlp
from ui.ui_renderer import render_dashboard  

# --- App Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Conversational Intelligence Platform",
    page_icon="💡"
)

# --- State Management ---
def initialize_state():
    """Initialize or reset session state variables for a new analysis."""
    st.session_state.analysis_triggered = False
    st.session_state.df_processed = pd.DataFrame()
    st.session_state.current_file_name = None

# --- Core Processing Logic ---
def run_analysis(uploaded_file):
    """
    Orchestrates the backend workflow: parsing, NLP enrichment,
    and storing the final result in the session state.
    """
    st.session_state.analysis_triggered = True
    st.session_state.current_file_name = uploaded_file.name

    try:
        # Step 1: Parse chat file
        chat_content = uploaded_file.getvalue().decode("utf-8")
        with st.spinner("Parsing chat file..."):
            parsed_df = parse_whatsapp_chat(chat_content)

        if parsed_df.empty:
            st.error("Failed to parse the chat file. Please ensure it is a valid WhatsApp export.")
            st.session_state.analysis_triggered = False
            return

        # Step 2: NLP Enrichment
        st.session_state.df_processed = enrich_df_with_nlp(parsed_df)

        if st.session_state.df_processed.empty:
            st.warning("Analysis complete, but no processable text messages were found.")
        else:
            st.success("Analysis complete! The dashboard is ready.")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.session_state.analysis_triggered = False

# --- Main Application UI ---
st.title("💡 Conversational Intelligence Platform")

# Initialize session state if not present
if 'analysis_triggered' not in st.session_state:
    initialize_state()

# --- Sidebar ---
st.sidebar.header("Upload & Analyze")
uploaded_file = st.sidebar.file_uploader(
    "Upload your WhatsApp chat export (.txt file)",
    type="txt",
    on_change=initialize_state
)

if uploaded_file:
    if st.sidebar.button("Analyze Chat", type="primary", use_container_width=True):
        run_analysis(uploaded_file)
else:
    st.info(
        """
        **Welcome! Unlock insights from your conversations.**

        1.  **Export a chat** from WhatsApp as a `.txt` file.
        2.  **Upload it** using the sidebar.
        3.  **Click 'Analyze Chat'** to generate your dashboard.
        """
    )

# --- Dashboard Rendering ---
if st.session_state.analysis_triggered and not st.session_state.df_processed.empty:
    render_dashboard(st.session_state.df_processed)
elif st.session_state.analysis_triggered:
    st.warning("Analysis was triggered, but there is no data to display. Please check your file or upload a new one.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("A project demonstrating end-to-end NLP application development.")

if st.sidebar.button("Clear All Caches & Reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
