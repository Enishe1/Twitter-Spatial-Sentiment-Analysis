import pandas as pd
import os
import torch
import ftfy  # The global fix for Korean/Japanese mojibake
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig

# Prevents Windows permission errors with AI models
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def fix_all_mojibake(text):
    if not isinstance(text, str): return text
    # ftfy automatically detects if it's Korean, Japanese, or Latin mojibake and fixes it
    return ftfy.fix_text(text)

def run_processor(file_path):
    print(f"--- Global Fixer & AI Analysis ---")
    
    # Load and Fix Text
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(f)
        
        print(f"  [Y] Fixing Multi-language Mojibake (Korean/Japanese/Latin)...")
        df['content'] = df['content'].apply(fix_all_mojibake)
        
        # Immediate save so you can verify the Korean text
        fixed_path = file_path.replace(".csv", "_FIXED_FINAL.csv")
        df.to_csv(fixed_path, index=False, encoding='utf-8-sig')
        print(f"  [Y] Verified text saved to: {fixed_path}")
    except Exception as e:
        print(f"  [X] Load Error: {e}")
        return

    # Force-Load AI (Fixes the 'NoneType' error)
    print("\n[3/4] Loading AI Model (Force-Load Mode)...")
    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    try:
        # We load the config and model separately to bypass the 'endswith' bug
        config = AutoConfig.from_pretrained(model_id, local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
        
        device = 0 if torch.cuda.is_available() else -1
        sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        print(f"  [Y] AI Ready.")

        # Analyze
        print(f"  Analyzing {len(df)} rows...")
        df['sentiment'] = df['content'].apply(lambda x: sentiment_task(str(x)[:512])[0]['label'].lower())
        
        final_path = file_path.replace(".csv", "_RESULTS.csv")
        df.to_csv(final_path, index=False, encoding='utf-8-sig')
        print(f"\n[4/4] SUCCESS! Saved to: {final_path}")

    except Exception as e:
        print(f"  [X] AI Still Erroring: {e}")

if __name__ == "__main__":
    target = r"Data\bluesky posts with locations\bluesky_merged_cleaned.csv"
    run_processor(target)