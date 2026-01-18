import pandas as pd
import os
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, pipeline

local_cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(local_cache_dir, exist_ok=True)

def run_sentiment_analysis():
    file_path = r"Data\bluesky posts with locations\bluesky_merged_cleaned_FIXED_FINAL.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    print(f"--- Loading Data ---")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")

    print("\n--- Initializing XLM-RoBERTa (Safe Mode) ---")
    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    try:
        
        print("Downloading/Loading model to local project folder...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(
            model_id, 
            use_fast=False, 
            cache_dir=local_cache_dir
        )
        
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_id, 
            cache_dir=local_cache_dir
        )
        
        device = 0 if torch.cuda.is_available() else -1
        sentiment_task = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        print(f"SUCCESS: Model loaded in local folder: {local_cache_dir}")
        
    except Exception as e:
        print(f"AI Loading Error: {e}")
        import traceback
        traceback.print_exc() # troubleshooting
        return

    # Running Analysis
    print("\n--- Analyzing Sentiment ---")
    
    def get_sentiment(text):
        if pd.isna(text) or str(text).strip() == "":
            return "neutral"
        try:
            # XLM-RoBERTa handles Korean, Japanese, French, and English automatically
            result = sentiment_task(str(text)[:512])[0]
            return result['label'].lower()
        except:
            return "neutral"

    # Process in chunks
    df['sentiment'] = df['content'].apply(get_sentiment)

    # Save Final Results
    output_path = r"Data\bluesky posts with locations\bluesky_final_sentiment_results.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nFINISHED!")
    print(f"Final analyzed file: {output_path}")

if __name__ == "__main__":
    run_sentiment_analysis()