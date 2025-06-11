import streamlit as st
from transformers import pipeline 
import torch 

# We use @st.cache_resource so Streamlit doesn't have to reload these big models
# every single time we change a filter or something.

HF_TOKEN = st.secrets.get("HUGGING_FACE_TOKEN")

@st.cache_resource
def get_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    try:
        device_to_use = -1
        if torch.cuda.is_available():
            device_to_use = 0 
        sentiment_model_pipeline = pipeline(
            "sentiment-analysis",  
            model=model_name,      
            device=device_to_use,
            max_length=512,
            truncation=True,
            token=HF_TOKEN     
        )
        
        return sentiment_model_pipeline
        
    except Exception as e:
        print(f"Could not load sentiment model '{model_name}'. Error: {e}")
        return None 

@st.cache_resource 
def get_ner_pipeline():
    # model_name = "dslim/bert-base-NER"
    model_name = "xlm-roberta-large-finetuned-conll03-english" # Using a  MULTILINGUAL NER model
    try:
        device_to_use = -1 
        if torch.cuda.is_available():
            device_to_use = 0

        ner_model_pipeline = pipeline(
            "ner",                 
            model=model_name,
            grouped_entities=True, 
            device=device_to_use,
            token=HF_TOKEN
        )
        
        return ner_model_pipeline
        
    except Exception as e:
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
            device=device_to_use,
            token=HF_TOKEN
        )

        return summarization_model_pipeline
    except Exception as e:
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
            device=device_to_use,
            max_length=512,
            token=HF_TOKEN
        )
        
        return toxicity_pipeline
    except Exception as e:
        return None