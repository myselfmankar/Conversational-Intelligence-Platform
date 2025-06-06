import pandas as pd
import streamlit as st
import traceback
from .models import get_sentiment_pipeline, get_ner_pipeline, get_summarization_pipeline

# --- Configuration for NLP Tasks ---
MAX_TEXT_LENGTH_FOR_NLP = 500  # Words; messages longer than this will be summarized for NLP
SUMMARIZATION_MIN_LENGTH = 30
SUMMARIZATION_MAX_LENGTH = 120 # Desired length of summary
NLP_BATCH_SIZE = 16 # Adjust based on your VRAM/RAM

def summarize_text_if_long(text: str, summarizer, max_original_len: int, min_summary: int, max_summary: int) -> str:
    """Summarizes text if it's longer than max_original_len words."""
    if not isinstance(text, str) or not text.strip() or summarizer is None:
        return text
    
    if len(text.split()) > max_original_len:
        try:
            summary_result = summarizer(text, min_length=min_summary, max_length=max_summary, truncation=True)
            return summary_result[0]['summary_text']
        except Exception:
            return text
    return text


@st.cache_data(show_spinner="Running NLP analysis on chat messages...") # CHANGED: Better spinner message
def enrich_df_with_nlp(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment, NER, and summarized text to the DataFrame using batch processing.
    This function is cached, so it only runs when the input DataFrame changes.
    """
    if df_input.empty:
        return df_input

    df = df_input.copy()

    # --- Load NLP Models ---
    summarizer = get_summarization_pipeline()
    sentiment_analyzer = get_sentiment_pipeline()
    ner_recognizer = get_ner_pipeline()

    if any(model is None for model in [summarizer, sentiment_analyzer, ner_recognizer]):
        st.error("One or more NLP models failed to load. Aborting NLP enrichment.")
        return df

    # --- Prepare for NLP ---
    nlp_applicable_mask = (~df['is_system']) & (df['message_type'] == 'text') & (df['message'].notna())
    df['message_for_nlp'] = df['message']
    df['sentiment_label'] = "NEUTRAL"
    df['sentiment_score'] = 0.0
    df['entities'] = [[] for _ in range(len(df))]

    # --- 1. Summarization for long messages ---
    if nlp_applicable_mask.any():
        with st.spinner("Step 1/3: Summarizing long messages..."):
            df.loc[nlp_applicable_mask, 'message_for_nlp'] = df.loc[nlp_applicable_mask, 'message'].apply(
                lambda txt: summarize_text_if_long(
                    txt, summarizer, MAX_TEXT_LENGTH_FOR_NLP, SUMMARIZATION_MIN_LENGTH, SUMMARIZATION_MAX_LENGTH
                )
            )

    texts_to_process = df.loc[nlp_applicable_mask, 'message_for_nlp'].fillna("").tolist()

    if not texts_to_process:
        st.success("NLP enrichment complete (no text messages to analyze).")
        return df

    # --- 2. Batch Sentiment Analysis ---
    with st.spinner("Step 2/3: Performing Sentiment Analysis..."):
        sentiment_results = []
        try:
            for i in range(0, len(texts_to_process), NLP_BATCH_SIZE):
                batch = texts_to_process[i:i+NLP_BATCH_SIZE]
                sentiment_results.extend(sentiment_analyzer(batch))
        except Exception as e:
            st.error(f"Error during batch sentiment analysis: {e}")
            st.text_area("Sentiment Analysis Error Traceback", traceback.format_exc(), height=200)

        if sentiment_results and len(sentiment_results) == len(texts_to_process):
            labels = [res.get('label', 'NEUTRAL').upper() for res in sentiment_results]
            scores = [res.get('score', 0.0) for res in sentiment_results]
            df.loc[nlp_applicable_mask, 'sentiment_label'] = labels
            df.loc[nlp_applicable_mask, 'sentiment_score'] = scores

    # --- 3. Batch Named Entity Recognition (NER) ---
    with st.spinner("Step 3/3: Performing Named Entity Recognition..."):
        all_ner_results = []
        # CHANGED: Using the summarized text for NER as well for consistency.
        texts_for_ner = df.loc[nlp_applicable_mask, 'message_for_nlp'].fillna("").tolist()
        
        try:
            # ADDED: Crucial batching loop for NER.
            for i in range(0, len(texts_for_ner), NLP_BATCH_SIZE):
                batch = texts_for_ner[i:i+NLP_BATCH_SIZE]
                all_ner_results.extend(ner_recognizer(batch))
        except Exception as e:
            # ADDED: Crucial error handling for NER.
            st.error(f"Error during batch NER: {e}")
            st.text_area("NER Error Traceback", traceback.format_exc(), height=200)
            
        if all_ner_results and len(all_ner_results) == len(texts_for_ner):
            # This mapping method is clever and robust, keeping it.
            entities_series = pd.Series(all_ner_results, index=df.index[nlp_applicable_mask])
            df['entities'] = entities_series
            df['entities'] = df['entities'].apply(lambda x: x if isinstance(x, list) else [])

    return df