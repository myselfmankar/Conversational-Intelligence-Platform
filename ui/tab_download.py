import streamlit as st 

def render_download_tab(df_display):
    st.subheader("Explore and Download Your Data")
    st.info("The tables below are fully interactive. You can sort, filter, and download the data as a CSV using the button in the top-right corner of each table.")

    st.markdown("---")
    
    # --- Section 1: Raw Parsed Data ---
    st.markdown("#### 1. Parsed Chat Data (Before NLP)")
    st.caption("This is the direct output from the chat parser, containing the essential message information.")
    
    parsed_cols = ['datetime', 'author', 'message', 'message_type', 'is_system']
    cols_to_show_parsed = [col for col in df_display.columns if col in parsed_cols]
    
    st.dataframe(df_display[cols_to_show_parsed], use_container_width=True)

    st.markdown("---")

    # --- Section 2: Fully Analyzed Data ---
    st.markdown("#### 2. Fully Analyzed Data (With NLP Insights)")
    st.caption("This table includes the results of all NLP operations, including sentiment scores and extracted entities.")

    df_for_display = df_display.copy()
    if 'entities' in df_for_display.columns:
        df_for_display['entities'] = df_for_display['entities'].apply(
            lambda entity_list: ', '.join([entity['word'] for entity in entity_list]) if isinstance(entity_list, list) and entity_list else ''
        )

    st.dataframe(df_for_display, use_container_width=True)