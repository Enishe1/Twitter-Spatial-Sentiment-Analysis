import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import folium
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import json
from datetime import datetime
# Geocoding libraries
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from pathlib import Path

# --- 1. CONFIGURATION AND DEVICE SETUP ---

# Check for CUDA availability and set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running Analysis on Device: {DEVICE} ---")

# Define model and file paths
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
INPUT_CSV_PATH = r'C:\Users\muhov\Desktop\Twitter-Spatial-Sentiment-Analysis-main\Twitter-Spatial-Sentiment-Analysis\Data\covid19_tweets.csv' 
OUTPUT_MAP_PATH = 'maps/covid_sentiment_map.html'
OUTPUT_IMAGE_PATH = 'images/sentiment_summary_table.jpg'
OUTPUT_CSV_PATH = 'data/covid19_tweets_with_sentiment.csv'  # NEW: CSV output path
GEOCODING_CACHE_PATH = 'data/geocoding_cache.pkl'
PROCESSED_DATA_PATH = 'data/processed_data.pkl'

# Initialize Geocoder with rate limiting
geolocator = Nominatim(user_agent="spatial_sentiment_analyzer_v2", timeout=10)

# --- 2. DIRECTORY CREATION ---

def create_directories():
    """Create all necessary directories."""
    directories = ['data', 'maps', 'images']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")

# --- 3. GEOCODING CACHE MANAGEMENT ---

def load_geocoding_cache(cache_path=GEOCODING_CACHE_PATH):
    """Load existing geocoding cache from file."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached locations from {cache_path}")
                return cache
    except Exception as e:
        print(f"Warning: Could not load cache file: {e}")
    return {}

def save_geocoding_cache(cache, cache_path=GEOCODING_CACHE_PATH):
    """Save geocoding cache to file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved {len(cache)} locations to cache file: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

def save_processed_data(df, output_path=PROCESSED_DATA_PATH):
    """Save processed data to avoid reprocessing."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save only necessary columns to reduce file size
        cols_to_save = ['text', 'location_str', 'latitude', 'longitude', 
                       'sentiment', 'neg_score', 'neu_score', 'pos_score']
        # Ensure all columns exist
        available_cols = [col for col in cols_to_save if col in df.columns]
        df_subset = df[available_cols].copy()
        df_subset.to_pickle(output_path)
        print(f"Saved processed data to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save processed data: {e}")

def load_processed_data(input_path=PROCESSED_DATA_PATH):
    """Load previously processed data."""
    try:
        if os.path.exists(input_path):
            df = pd.read_pickle(input_path)
            print(f"Loaded processed data from {input_path}")
            return df
    except Exception as e:
        print(f"Warning: Could not load processed data: {e}")
    return None

# --- 4. NEW: CSV EXPORT FUNCTION ---

def export_to_csv(df, output_path=OUTPUT_CSV_PATH, include_original_columns=True):
    """
    Export the analyzed data to a CSV file with sentiment columns.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    - output_path: Path to save the CSV file
    - include_original_columns: If True, include all original columns from input CSV
    """
    print(f"\nExporting results to CSV: {output_path}")
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a copy to avoid modifying the original
        export_df = df.copy()
        
        # Round sentiment scores for better readability
        export_df['neg_score'] = export_df['neg_score'].round(4)
        export_df['neu_score'] = export_df['neu_score'].round(4)
        export_df['pos_score'] = export_df['pos_score'].round(4)
        
        # Calculate sentiment strength (difference between positive and negative)
        export_df['sentiment_strength'] = (export_df['pos_score'] - export_df['neg_score']).round(4)
        
        # Add a confidence score (max probability among the three classes)
        export_df['confidence'] = export_df[['neg_score', 'neu_score', 'pos_score']].max(axis=1).round(4)
        
        # Add analysis timestamp
        export_df['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Reorder columns for better readability
        column_order = [
            'text', 'location_str', 'latitude', 'longitude',
            'sentiment', 'sentiment_strength', 'confidence',
            'neg_score', 'neu_score', 'pos_score',
            'analysis_timestamp'
        ]
        
        # Keep only columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in export_df.columns]
        
        # Add any remaining columns (original columns from input CSV)
        if include_original_columns:
            remaining_columns = [col for col in export_df.columns if col not in existing_columns]
            # Move id-like columns to the beginning if they exist
            id_columns = [col for col in ['id', 'tweet_id', 'user_id'] if col in remaining_columns]
            for id_col in id_columns:
                existing_columns.insert(0, id_col)
                remaining_columns.remove(id_col)
            
            # Add date/time columns next if they exist
            date_columns = [col for col in ['created_at', 'date', 'timestamp', 'tweet_date'] 
                          if col in remaining_columns]
            for date_col in date_columns:
                existing_columns.append(date_col)
                remaining_columns.remove(date_col)
            
            # Add remaining columns
            existing_columns.extend(remaining_columns)
        
        # Reorder the dataframe
        export_df = export_df[existing_columns]
        
        # Save to CSV
        export_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Print summary
        print(f"✓ Successfully exported {len(export_df)} rows to CSV")
        print(f"✓ File saved: {output_path}")
        print(f"✓ File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"✓ Columns exported: {len(export_df.columns)}")
        print(f"✓ Sample columns: {list(export_df.columns)[:10]}...")
        
        # Show a small preview
        print("\nFirst few rows preview:")
        print(export_df[['text', 'location_str', 'sentiment', 'sentiment_strength']].head(3).to_string())
        
        return export_df
        
    except Exception as e:
        print(f"✗ Error exporting to CSV: {e}")
        # Try a simpler export as fallback
        try:
            simple_path = output_path.replace('.csv', '_simple.csv')
            df.to_csv(simple_path, index=False, encoding='utf-8')
            print(f"✓ Saved simplified version to: {simple_path}")
            return df
        except Exception as e2:
            print(f"✗ Failed to save simplified version: {e2}")
            return None

# --- 5. IMPROVED GEOCODING FUNCTION WITH CACHE ---

def geocode_location_with_cache(location_str, cache):
    """Converts a location string into (latitude, longitude) with caching and rate limiting."""
    if not location_str or pd.isna(location_str) or location_str.strip() == "":
        return None, None
    
    # Normalize the location string for better caching
    location_str_norm = str(location_str).strip().lower()
    
    # Check cache first
    if location_str_norm in cache:
        return cache[location_str_norm]
    
    # Rate limiting: 1 request per second for Nominatim
    time.sleep(1.1)  # Slightly more than 1 second to be safe
    
    # Geocoding attempt
    max_retries = 3
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(location_str, exactly_one=True, timeout=10)
            if location:
                lat, lon = location.latitude, location.longitude
                cache[location_str_norm] = (lat, lon)
                return lat, lon
            else:
                cache[location_str_norm] = (None, None)
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == max_retries - 1:
                print(f"Geocoding error for '{location_str}': {e}")
                return None, None
            time.sleep(2)  # Wait longer before retry
        except Exception as e:
            print(f"Unexpected error during geocoding for '{location_str}': {e}")
            cache[location_str_norm] = (None, None)
            return None, None
    
    return None, None

# --- 6. DATA LOADING AND PROCESSING ---

def process_data(file_path, use_cache=True, max_rows=5000):
    """
    Loads and processes data with geocoding, using cache when available.
    """
    # Create directories first
    create_directories()
    
    # Check if processed data already exists
    if use_cache:
        processed_df = load_processed_data()
        if processed_df is not None and not processed_df.empty:
            print("Using previously processed data from cache.")
            return processed_df
    
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Check path.")
        return pd.DataFrame({'text': [], 'latitude': [], 'longitude': []})
    
    print(f"Loading data from {file_path}...")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=',', engine='python', 
                                on_bad_lines='skip', low_memory=False)
                print(f"CSV loaded successfully using {encoding} encoding.")
                break
            except:
                continue
        
        if df is None:
            print("FATAL ERROR: Could not read CSV file with any encoding.")
            return pd.DataFrame({'text': [], 'latitude': [], 'longitude': []})
            
    except Exception as e:
        print(f"FATAL ERROR: Could not read CSV file: {e}")
        return pd.DataFrame({'text': [], 'latitude': [], 'longitude': []})
    
    # Store original column names for reference
    original_columns = df.columns.tolist()
    print(f"Original columns in CSV: {original_columns}")
    
    # Check and rename columns
    print("Checking and standardizing column names...")
    column_mapping = {
        'tweet_content': 'text',
        'user_location': 'location_str',
        'text': 'text',  # If already named text
        'location': 'location_str',
        'user_location_str': 'location_str',
        'tweet': 'text',
        'content': 'text'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            print(f"  Renamed '{old_name}' to '{new_name}'")
    
    # Check if required columns exist
    required_cols = ['text', 'location_str']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"FATAL ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame({'text': [], 'latitude': [], 'longitude': []})
    
    # Clean data
    initial_count = len(df)
    df = df.dropna(subset=['location_str', 'text'])
    df = df[df['text'].str.strip() != '']
    df = df[df['location_str'].str.strip() != '']
    cleaned_count = len(df)
    print(f"Data cleaning: {initial_count - cleaned_count} rows removed, {cleaned_count} rows remaining.")
    
    # Sample if needed
    if len(df) > max_rows:
        print(f"Sampling down to {max_rows} rows...")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    
    # Load geocoding cache
    location_cache = load_geocoding_cache()
    
    # Get unique locations
    unique_locations = df['location_str'].unique()
    print(f"Geocoding {len(unique_locations)} unique locations...")
    
    # Geocode unique locations
    for location_str in tqdm(unique_locations, desc="Geocoding Progress"):
        geocode_location_with_cache(location_str, location_cache)
    
    # Save updated cache
    save_geocoding_cache(location_cache)
    
    # Apply geocoded results
    df['latitude'] = df['location_str'].apply(
        lambda x: location_cache.get(str(x).strip().lower(), (None, None))[0]
    )
    df['longitude'] = df['location_str'].apply(
        lambda x: location_cache.get(str(x).strip().lower(), (None, None))[1]
    )
    
    # Filter rows with valid coordinates
    before_filter = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    after_filter = len(df)
    
    # Count valid geocodes
    valid_geocodes = sum(1 for v in location_cache.values() if v != (None, None))
    print(f"Successfully geocoded {valid_geocodes} unique locations.")
    print(f"Final dataset size: {after_filter} rows with valid coordinates and text.")
    print(f"  ({before_filter - after_filter} rows removed due to geocoding failures)")
    
    # Initialize sentiment columns if they don't exist
    if 'sentiment' not in df.columns:
        df['sentiment'] = None
    if 'neg_score' not in df.columns:
        df['neg_score'] = 0.0
    if 'neu_score' not in df.columns:
        df['neu_score'] = 0.0
    if 'pos_score' not in df.columns:
        df['pos_score'] = 0.0
    
    return df

# --- 7. SENTIMENT ANALYSIS WITH GPU OPTIMIZATION ---

def perform_sentiment_analysis(df):
    """Perform sentiment analysis with GPU acceleration."""
    
    print(f"Loading RoBERTa model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Move model to GPU if available
    model.to(DEVICE)
    model.eval()
    
    # Optimize for GPU
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        batch_size = 64
        print("Using GPU acceleration with batch size 64")
    else:
        batch_size = 32
        print(f"Using CPU with batch size {batch_size}")
    
    results = []
    print("Starting sentiment analysis...")
    
    # Pre-process texts
    texts = df['text'].astype(str).str.replace('\n', ' ').str.strip().tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing Batches"):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Tokenize
            encoded_input = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=128  # Reduced for speed, still good for tweets
            )
            
            # Move to device
            encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
            
            # Inference
            with torch.no_grad():
                output = model(**encoded_input)
                scores = output.logits.detach().cpu().numpy()
            
            # Get probabilities
            probabilities = softmax(scores, axis=1)
            
            # Get predictions
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            predictions = np.argmax(probabilities, axis=1)
            
            for j in range(len(batch_texts)):
                pred_idx = predictions[j]
                results.append({
                    'sentiment': sentiment_labels[pred_idx],
                    'neg_score': float(probabilities[j][0]),
                    'neu_score': float(probabilities[j][1]),
                    'pos_score': float(probabilities[j][2]),
                })
                
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {e}")
            # Fill with neutral sentiment for failed batch
            for _ in range(len(batch_texts)):
                results.append({
                    'sentiment': 'Neutral',
                    'neg_score': 0.33,
                    'neu_score': 0.34,
                    'pos_score': 0.33,
                })
    
    # Update dataframe
    results_df = pd.DataFrame(results)
    df.loc[df.index, ['sentiment', 'neg_score', 'neu_score', 'pos_score']] = results_df.values
    
    return df

# --- 8. FIXED MAP GENERATION ---

def create_sentiment_map(df, output_path):
    """Generate interactive Folium map with fixed template handling."""
    print("Generating interactive sentiment map...")
    
    # Filter data
    df_filtered = df.dropna(subset=['latitude', 'longitude'])
    
    if df_filtered.empty:
        print("Warning: No geo-data available to plot.")
        return
    
    # Create color mapping
    color_map = {
        'Positive': '#2ecc71',  # Green
        'Negative': '#e74c3c',  # Red
        'Neutral': '#3498db'    # Blue
    }
    
    # Calculate center
    center_lat = df_filtered['latitude'].median()
    center_lon = df_filtered['longitude'].median()
    
    # Create map with simpler template
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=2,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Add layer control
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Add markers (limit to 1000 for performance)
    sample_size = min(1000, len(df_filtered))
    sample_df = df_filtered.sample(n=sample_size, random_state=42) if len(df_filtered) > 1000 else df_filtered
    
    print(f"Adding {len(sample_df)} markers to map...")
    
    for idx, row in sample_df.iterrows():
        sentiment = row['sentiment']
        color = color_map.get(sentiment, 'gray')
        
        # Create safe popup content
        location_str = str(row.get('location_str', 'N/A'))[:50]
        tweet_preview = str(row['text'])[:100].replace('"', "'").replace('\n', ' ')
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 250px;">
            <div style="background-color: {color}; padding: 5px; border-radius: 3px; margin-bottom: 5px;">
                <strong style="color: white;">{sentiment} Sentiment</strong>
            </div>
            <p><strong>Location:</strong> {location_str}</p>
            <p><strong>Sentiment Scores:</strong><br>
            Negative: {row['neg_score']:.3f}<br>
            Neutral: {row['neu_score']:.3f}<br>
            Positive: {row['pos_score']:.3f}</p>
            <p><strong>Tweet Preview:</strong><br>
            "{tweet_preview}..."</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + (row['pos_score'] - row['neg_score']) * 3,  # Size based on sentiment strength
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{sentiment} sentiment"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <strong>Sentiment Legend</strong><br>
    <i class="fa fa-circle" style="color:#2ecc71"></i> Positive<br>
    <i class="fa fa-circle" style="color:#e74c3c"></i> Negative<br>
    <i class="fa fa-circle" style="color:#3498db"></i> Neutral<br>
    <br>
    <em>Marker size indicates<br>sentiment strength</em>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the map
    try:
        m.save(output_path)
        print(f"Successfully generated map: {output_path}")
    except Exception as e:
        print(f"Error saving map: {e}")
        # Try simple save as fallback
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(m._repr_html_())
            print(f"Map saved using fallback method: {output_path}")
        except Exception as e2:
            print(f"Failed to save map: {e2}")

# --- 9. FIXED IMAGE GENERATION ---

def create_sentiment_image(df, output_path):
    """Generate sentiment distribution visualization."""
    print("Generating sentiment summary image...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Filter out any errors or None values
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    order = ['Positive', 'Neutral', 'Negative']  # Reordered for better visual
    
    # Prepare data for plotting
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for sentiment in order:
        if sentiment in sentiment_counts:
            count = sentiment_counts[sentiment]
            plot_data.append(count)
            plot_labels.append(sentiment)
            plot_colors.append(colors.get(sentiment, 'gray'))
    
    if plot_data:
        # Bar chart
        bars = ax1.bar(plot_labels, plot_data, color=plot_colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Tweet Sentiment Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment Category', fontsize=12)
        ax1.set_ylabel('Number of Tweets', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, plot_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(plot_data)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentages
        total = sum(plot_data)
        for i, (label, value) in enumerate(zip(plot_labels, plot_data)):
            percentage = (value / total) * 100
            ax1.text(i, value/2, f'{percentage:.1f}%', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=11)
        
        # Plot 2: Pie chart
        wedges, texts, autotexts = ax2.pie(plot_data, labels=plot_labels, colors=plot_colors,
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Sentiment Proportion', fontsize=14, fontweight='bold')
        
        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add overall statistics text
        stats_text = f"""
        Overall Statistics:
        Total Tweets Analyzed: {total:,}
        Average Positive Score: {df['pos_score'].mean():.3f}
        Average Negative Score: {df['neg_score'].mean():.3f}
        Average Neutral Score: {df['neu_score'].mean():.3f}
        """
        
        fig.text(0.02, 0.98, stats_text.strip(), 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.98, 0.02, f"Generated: {timestamp}", 
                fontsize=9, ha='right', alpha=0.7)
    
    plt.suptitle('COVID-19 Twitter Sentiment Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Successfully generated image: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        # Try alternative save location
        alt_path = 'sentiment_summary.jpg'
        plt.savefig(alt_path, dpi=150, bbox_inches='tight')
        print(f"Saved image to alternative location: {alt_path}")
    finally:
        plt.close()

# --- 10. MAIN EXECUTION ---

if __name__ == "__main__":
    print("=" * 60)
    print("COVID-19 SPATIAL SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Create all necessary directories first
    create_directories()
    
    # Process data (with caching)
    print("\n1. DATA PROCESSING")
    data_df = process_data(INPUT_CSV_PATH, use_cache=True, max_rows=5000)
    
    if data_df.empty or len(data_df) < 10:
        print("Error: Insufficient data for analysis.")
        print(f"DataFrame shape: {data_df.shape}")
        exit(1)
    
    # Perform sentiment analysis if not already done
    sentiment_cols_exist = all(col in data_df.columns for col in ['sentiment', 'neg_score', 'neu_score', 'pos_score'])
    sentiment_data_exists = sentiment_cols_exist and not data_df['sentiment'].isna().all()
    
    if not sentiment_data_exists:
        print("\n2. SENTIMENT ANALYSIS")
        data_df = perform_sentiment_analysis(data_df)
        
        # Save processed data
        save_processed_data(data_df)
    else:
        print("\n2. SENTIMENT ANALYSIS (Already processed - using cached results)")
    
    # Export to CSV (NEW FEATURE)
    print("\n3. EXPORTING RESULTS TO CSV")
    export_to_csv(data_df, OUTPUT_CSV_PATH, include_original_columns=True)
    
    # Generate visualizations
    print("\n4. GENERATING VISUALIZATIONS")
    create_sentiment_map(data_df, OUTPUT_MAP_PATH)
    create_sentiment_image(data_df, OUTPUT_IMAGE_PATH)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Device used: {DEVICE}")
    print(f"Total tweets analyzed: {len(data_df):,}")
    print(f"\nSentiment distribution:")
    
    sentiment_stats = data_df['sentiment'].value_counts()
    total = len(data_df)
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in sentiment_stats:
            count = sentiment_stats[sentiment]
            percentage = (count / total) * 100
            print(f"  {sentiment:8s}: {count:5d} tweets ({percentage:5.1f}%)")
    
    print(f"\nAverage sentiment scores:")
    print(f"  Positive: {data_df['pos_score'].mean():.3f}")
    print(f"  Neutral:  {data_df['neu_score'].mean():.3f}")
    print(f"  Negative: {data_df['neg_score'].mean():.3f}")
    
    print(f"\nOutput files:")
    print(f"  📄 Results CSV:         {OUTPUT_CSV_PATH}")
    print(f"  📍 Interactive map:     {OUTPUT_MAP_PATH}")
    print(f"  📊 Summary image:      {OUTPUT_IMAGE_PATH}")
    print(f"  💾 Cached geocoding:   {GEOCODING_CACHE_PATH}")
    print(f"  💾 Processed data:     {PROCESSED_DATA_PATH}")
    print("=" * 60)