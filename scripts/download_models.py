from transformers import pipeline

MODEL_LIST = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",  
    "xlm-roberta-large-finetuned-conll03-english",                                
    "sshleifer/distilbart-cnn-6-6",                       
    "unitary/unbiased-toxic-roberta"                      
]

def download_all_models():
    for model_name in MODEL_LIST:
        try:
            pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            print(f"--- FAILED to download {model_name}. Error: {e} ---")
    

if __name__ == "__main__":
    download_all_models()