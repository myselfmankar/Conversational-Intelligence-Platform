# WhatsApp Chat NLP Analyzer
import streamlit as st
import pandas as pd
import re 

# --- Import Custom Modules ---
from utils.parser import parse_whatsapp_chat
from utils.file_handler import handle_uploaded_file
from nlp.enrich import enrich_df_with_nlp # Assumes this file no longer does NER
from visuals import charts 

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
if 'current_file_name' not in st.session_state: # To reset if new file is uploaded
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
        # --- Key Metrics ---
        col1, col2, col3 = st.columns(3)
        total_messages = len(df_display)
        num_participants = df_display[~df_display['is_system']]['author'].nunique()
        media_messages = len(df_display[df_display['message_type'] == 'media'])
        
        col1.metric("Total Messages (filtered)", f"{total_messages}")
        col2.metric("Active Participants (filtered)", f"{num_participants}")
        col3.metric("Media Messages (filtered)", f"{media_messages}")

        # --- Tabs for different visualizations ---
        # Adjusted tab titles as NER is disabled for now
        tab_titles = ["📊 Overview & Activity", "😊 Sentiment Analysis", "📝 Content & Topics (NER)", "🌐 Group Dynamics", "💾 Download Data"]
        tab_overview, tab_sentiment, tab_ner, tab_dynamics, tab_download = st.tabs(tab_titles)

        with tab_overview:
            st.subheader("Message Volume")
            fig_daily = charts.plot_message_activity_timeline(df_display)
            if fig_daily: st.plotly_chart(fig_daily, use_container_width=True)

            fig_hourly = charts.plot_hourly_activity(df_display)
            if fig_hourly: st.plotly_chart(fig_hourly, use_container_width=True)
            
            st.subheader("Top Contributors")
            fig_auth_act = charts.plot_author_activity(df_display, top_n=10)
            if fig_auth_act: st.plotly_chart(fig_auth_act, use_container_width=True)

            st.subheader("Author Activity Ranking")
            ranked_activity_df = charts.get_ranked_author_activity_df(df_display, top_n=10)
            if not ranked_activity_df.empty: st.dataframe(ranked_activity_df, use_container_width=True)


        with tab_sentiment:
            st.subheader("Sentiment Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = charts.plot_sentiment_distribution_pie(df_display)
                if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("No sentiment data to plot with current filters.")
            
            with col2:
                fig_author_sent = charts.plot_sentiment_per_author(df_display)
                if fig_author_sent: st.plotly_chart(fig_author_sent, use_container_width=True)
                else: st.info("No sentiment data per author to plot with current filters.")
        
        with tab_ner:
            st.subheader("Named Entity Recognition (NER)")
            st.info("This section identifies the key people (PER), organizations (ORG), and locations (LOC) mentioned in the chat.")
            
            # Create columns for a clean layout
            col1, col2 = st.columns(2)
            with col1:
                # Plot for People (PER)
                fig_people = charts.plot_frequent_named_entities(df_display, top_n=15, entity_types=['PER'])
                if fig_people:
                    st.plotly_chart(fig_people, use_container_width=True)
                else:
                    st.caption("No 'Person' entities found in the current selection.")

            with col2:
                # Plot for Orgs and Locations (ORG, LOC)
                fig_orgs_locs = charts.plot_frequent_named_entities(df_display, top_n=15, entity_types=['ORG', 'LOC'])
                if fig_orgs_locs:
                    st.plotly_chart(fig_orgs_locs, use_container_width=True)
                else:
                    st.caption("No 'Organization' or 'Location' entities found.")

        with tab_dynamics:
            st.subheader("Social Network Analysis")
            st.info("This graph visualizes communication patterns. An arrow from User A to User B means A often sent a message right before B. Larger nodes represent more central users.")
            
            fig_network = charts.create_interaction_network_graph(df_display)
            if fig_network:
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.warning("Could not generate a network graph. The chat may be too short or have too few interactions in the current selection.")
        
        with tab_download:
            st.subheader("Explore and Download Your Data")
            st.info("The tables below are fully interactive. You can sort, filter, and download the data as a CSV using the button in the top-right corner of each table.")

            st.markdown("---")
            
            # --- Section 1: Raw Parsed Data ---
            st.markdown("#### 1. Parsed Chat Data (Before NLP)")
            st.caption("This is the direct output from the chat parser, containing the essential message information.")
            
            # Define the columns for the simple, parsed view
            parsed_cols = ['datetime', 'author', 'message', 'message_type', 'is_system']
            # Select only the columns that actually exist in the final DataFrame
            cols_to_show_parsed = [col for col in df_display.columns if col in parsed_cols]
            
            st.dataframe(df_display[cols_to_show_parsed], use_container_width=True)

            st.markdown("---")

            # --- Section 2: Fully Analyzed Data ---
            st.markdown("#### 2. Fully Analyzed Data (With NLP Insights)")
            st.caption("This table includes the results of all NLP operations, including sentiment scores and extracted entities.")
            
            # Define the columns for the full, analyzed view
            analyzed_cols = ['datetime', 'author', 'message', 'sentiment_label', 'sentiment_score', 'entities', 'message_for_nlp']
            # Select only the columns that actually exist
            cols_to_show_analyzed = [col for col in df_display.columns if col in analyzed_cols]

            # The 'entities' column contains lists of dicts, which can be messy. Let's format it.
            df_for_display = df_display[cols_to_show_analyzed].copy()
            if 'entities' in df_for_display.columns:
                # Convert the list of entity dicts into a clean, readable string
                df_for_display['entities'] = df_for_display['entities'].apply(
                    lambda entity_list: ', '.join([entity['word'] for entity in entity_list]) if isinstance(entity_list, list) and entity_list else ''
                )

            st.dataframe(df_for_display, use_container_width=True)
    
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