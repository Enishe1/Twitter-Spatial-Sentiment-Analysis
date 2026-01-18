import pandas as pd
import os
import sys


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline

def process_bluesky_data():
    print("--- Bluesky Data Sentiment Processor ---")
    
    # Path Setup
    data_folder = os.path.join("Data", "bluesky posts with locations")
    
    if not os.path.exists(data_folder):
        print(f"Error: The folder '{data_folder}' does not exist!")
        return

    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                 if f.endswith('.csv') and "posts with location" not in f.lower()]
    
    if not all_files:
        print("No input .csv files found.")
        return

    # Loading Files with Universal Fix
    print(f"\n[1/4] Loading {len(all_files)} files...")
    df_list = []
    for file_path in all_files:
        try:
            with open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
                df = pd.read_csv(f)
            df_list.append(df)
            print(f"  [Y] Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  [X] Error reading {file_path}: {e}")

    if not df_list: return
    combined_df = pd.concat(df_list, ignore_index=True)

    # Data Cleaning
    print("\n[2/4] Cleaning data and saving checkpoint...")
    combined_df = combined_df.dropna(subset=['location'])
    combined_df = combined_df[combined_df['location'].astype(str).str.strip() != ""]
    
    # SAVE MERGED FILE NOW (In case AI crashes later)
    checkpoint_path = os.path.join(data_folder, "bluesky_merged_cleaned.csv")
    combined_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
    print(f"  [Y] Merged file saved to: {checkpoint_path}")

    # Multilingual Sentiment Analysis
    print("\n[3/4] Initializing Multilingual XLM-RoBERTa Model...")
    try:
        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        # Explicitly passing both model and tokenizer prevents 'NoneType' errors
        sentiment_task = pipeline(
            "sentiment-analysis", 
            model=model_id,
            tokenizer=model_id,
            device=-1 
        )
    except Exception as e:
        print(f"  [X] AI Model Error: {e}")
        print("Note: Your merged file was still saved above.")
        return

    def get_sentiment(text):
        if pd.isna(text) or str(text).strip() == "":
            return "neutral"
        try:
            # Analyze text (XLM handles Japanese/Italian/French directly)
            result = sentiment_task(str(text)[:512])[0]
            return result['label'].lower()
        except:
            return "neutral"

    print("  Analyzing sentiment (processing 1492 rows)...")
    combined_df['sentiment'] = combined_df['content'].apply(get_sentiment)

    # Final Save
    output_path = os.path.join(data_folder, "bluesky posts with location.csv")
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[4/4] SUCCESS! Final file saved to: {output_path}")

if __name__ == "__main__":
    process_bluesky_data()