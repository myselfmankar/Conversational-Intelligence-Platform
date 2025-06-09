import streamlit as st
from transformers import pipeline 
import torch 

# We use @st.cache_resource so Streamlit doesn't have to reload these big models
# every single time we change a filter or something.

@st.cache_resource
def get_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    st.write(f"Loading sentiment model: '{model_name}'... This might take a moment the first time!")
    print(f"Attempting to load sentiment model: {model_name}")

    try:
        device_to_use = -1
        if torch.cuda.is_available():
            device_to_use = 0 
        sentiment_model_pipeline = pipeline(
            "sentiment-analysis",  
            model=model_name,      
            device=device_to_use   
        )
        
        print(f"Sentiment model '{model_name}' loaded.")
        return sentiment_model_pipeline
        
    except Exception as e:
        st.error(f"Could not load sentiment model '{model_name}'. Error: {e}")
        return None 

@st.cache_resource 
def get_ner_pipeline():
    model_name = "dslim/bert-base-NER"
    print(f"Attempting to load NER model: {model_name}")
    try:
        device_to_use = -1 
        if torch.cuda.is_available():
            device_to_use = 0

        ner_model_pipeline = pipeline(
            "ner",                 
            model=model_name,
            grouped_entities=True, # like "New" and "York" into "New York".
            device=device_to_use
        )
        
        print(f"NER model '{model_name}' loaded.")
        return ner_model_pipeline
        
    except Exception as e:
        print(f"Could not load NER model '{model_name}'. Error: {e}")
        return None

@st.cache_resource 
def get_summarization_pipeline():
    model_name = "sshleifer/distilbart-cnn-6-6"
    try:
        device_to_use = -1
        if torch.cuda.is_available():
            device_to_use = 0

        summarization_model_pipeline = pipeline(
            "summarization",       
            model=model_name,
            device=device_to_use
        )

        print(f"Summarization model '{model_name}' loaded.")
        return summarization_model_pipeline
    except Exception as e:
        print(f"ERROR loading summarization model '{model_name}': {e}")
        return None

@st.cache_resource
def get_toxicity_pipeline():
    model_name = "unitary/toxic-bert"
    try:
        device_to_use = -1
        if torch.cuda.is_available():
            device_to_use = 0

        toxicity_pipeline = pipeline(
            "text-classification", 
            model="unitary/unbiased-toxic-roberta",
            tokenizer="unitary/unbiased-toxic-roberta",
            device=device_to_use
        )
        
        print(f"Toxicity model '{model_name}' loaded.")
        return toxicity_pipeline
    except Exception as e:
        print(f"ERROR loading toxicity model '{model_name}': {e}")
        return None