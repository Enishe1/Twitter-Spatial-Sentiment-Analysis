import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from wordcloud import WordCloud, STOPWORDS

# Setup paths
input_file = r"Data/bluesky posts with locations/bluesky_final_sentiment_results.csv"
cleaned_csv_output = r"Data/bluesky posts with locations/bluesky_sentiment_countries_only.csv"
output_folder = "plots"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(cleaned_csv_output), exist_ok=True)

def generate_all_plots():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # --- CLEAN AND CREATE NEW CSV (CASE-INSENSITIVE) ---
    print("--- Cleaning Locations (Fixing Case Sensitivity) ---")
    raw_df = pd.read_csv(input_file)
    
    def get_country(loc):
        if pd.isna(loc): return "Unknown"
        
        # Take the last part (City, Country -> Country) 
        # Strip spaces and force to UPPERCASE for matching
        country_raw = str(loc).split(',')[-1].strip().upper()
        
        # Comprehensive Mapping (Keys are UPPERCASE)
        mapping = {
            "USA": "United States",
            "US": "United States",
            "UK": "United Kingdom",
            "UNITED KINGDOM": "United Kingdom",
            "GB": "United Kingdom",
            "JAPAN": "Japan",
            "UKRAINE": "Ukraine",
            "SOUTH KOREA": "South Korea",
            "BRASIL": "Brazil",
            "BRAZIL": "Brazil"
        }
        
         
        if country_raw in mapping:
            return mapping[country_raw]
        return country_raw.title()

    raw_df['location'] = raw_df['location'].apply(get_country)
    
    print(f"Saving newly fixed CSV to: {cleaned_csv_output}")
    raw_df.to_csv(cleaned_csv_output, index=False)

    # --- DO MAPPING ON THE NEWLY CREATED CSV ---
    print(f"--- Loading {cleaned_csv_output} for Mapping ---")
    df = pd.read_csv(cleaned_csv_output)
    
    # Print verification to terminal
    print("\nFINAL VERIFIED COUNTS (Japan & Ukraine fixed):")
    counts = df['location'].value_counts()
    print(f"Japan: {len(df[df['location'] == 'Japan'])}")
    print(f"Ukraine: {len(df[df['location'] == 'Ukraine'])}")
    print("\nTop 10 Global Volume:")
    print(counts.head(10))

    # Prepare sentiment scores and dates
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['created_at'])

    # --- 3. SPATIAL ANALYSIS (The Map) ---
    print("\nGenerating Fixed Map...")
    geo_df = df.groupby('location').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        total_samples=('location', 'count')
    ).reset_index()

    fig_map = px.choropleth(geo_df, 
                            locations="location", 
                            locationmode='country names',
                            color="avg_sentiment",
                            hover_name="location",
                            hover_data={
                                'location': False, 
                                'avg_sentiment': ':.2f', 
                                'total_samples': True 
                            },
                            color_continuous_scale="RdYlGn",
                            range_color=[-1, 1],
                            labels={'avg_sentiment': 'Avg Sentiment', 'total_samples': 'Total Posts'},
                            title="Global AI Sentiment (Case-Insensitive Nationwide Aggregation)")
    
    fig_map.write_html(os.path.join(output_folder, "2_sentiment_choropleth.html"))

    # --- POST VOLUME (Bar Chart) ---
    plt.figure(figsize=(10, 8))
    counts.head(15).sort_values().plot(kind='barh', color='teal')
    plt.title("Post Volume by Nation (All Casing Fixed)")
    plt.xlabel("Total Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "1_post_volume_country.png"))

    # --- SENTIMENT DISTRIBUTION ---
    plt.figure(figsize=(8, 8))
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['gray', 'green', 'red'])
    plt.title("Overall AI Sentiment Distribution")
    plt.savefig(os.path.join(output_folder, "3_sentiment_pie.png"))

    # --- WORD CLOUDS ---
    for sent in ['positive', 'negative']:
        text = " ".join(df[df['sentiment'] == sent]['content'].astype(str))
        if len(text) > 10:
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Topics in {sent.capitalize()} Posts")
            plt.savefig(os.path.join(output_folder, f"7_wordcloud_{sent}.png"))

    print(f"\nSUCCESS!")

if __name__ == "__main__":
    generate_all_plots()