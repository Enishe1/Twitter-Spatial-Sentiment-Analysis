import pandas as pd
from .twitter_client import TwitterClient
from utils.sentiment import SentimentAnalyzer
import os
from datetime import datetime

def analyze_keyword(keyword, max_results=50, model_name=None):
    """
    Main pipeline to analyze tweets for a keyword
    
    Args:
        keyword: Search term or hashtag
        max_results: Number of tweets to fetch
        model_name: Specific model to use ('roberta_twitter', 'vader', 'bert')
                   If None, uses the best model from previous tests
    
    Returns:
        Dictionary with analysis results
    """
    client = TwitterClient()
    analyzer = SentimentAnalyzer(model_name=model_name)
    
    print(f"Searching for tweets about: {keyword}")
    print(f"Using model: {analyzer.model_name}")
    
    # Fetch tweets
    tweets_df = client.fetch_tweets(keyword, max_results)
    
    if tweets_df.empty:
        print("X No tweets found with location data")
        return empty_results()
    
    print(f"Y Found {len(tweets_df)} tweets with location data")
    
    # Do sentiment analysis
    print("Sentiment analysis in progress...")
    tweets_df['sentiment'] = tweets_df['text'].apply(analyzer.analyze_sentiment)
    
    # Add color mapping for visualization

    tweets_df['color'] = tweets_df['sentiment'].map({
        'positive': '#00FF00',  # Positive is Green
        'negative': '#FF0000',  # Negative is Red
        'neutral': '#FFFF00'    # Neutral is Yellow
    })
    # hexadecimal color values


    # Calculate metrics
    sentiment_counts = tweets_df['sentiment'].value_counts()
    total = len(tweets_df)
    
    result = {
        'total_tweets': total,
        'positive_count': sentiment_counts.get('positive', 0),
        'negative_count': sentiment_counts.get('negative', 0),
        'neutral_count': sentiment_counts.get('neutral', 0),
        'positive_pct': (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0,
        'negative_pct': (sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0,
        'neutral_pct': (sentiment_counts.get('neutral', 0) / total * 100) if total > 0 else 0,
        'tweets_df': tweets_df,
        'model_used': analyzer.model_name,
        'keyword': keyword,
        'timestamp': datetime.now().isoformat()
    }
    
    print("Y Analysis complete!")
    print(f"   Positive: {result['positive_count']} ({result['positive_pct']:.1f}%)")
    print(f"   Negative: {result['negative_count']} ({result['negative_pct']:.1f}%)")
    print(f"   Neutral: {result['neutral_count']} ({result['neutral_pct']:.1f}%)")
    
    return result

def empty_results():
    """Return empty results structure"""
    return {
        'total_tweets': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0,
        'positive_pct': 0,
        'negative_pct': 0,
        'neutral_pct': 0,
        'tweets_df': pd.DataFrame(),
        'model_used': 'none',
        'keyword': '',
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Test the pipeline
    results = analyze_keyword("python", 10)
    print(f"Test results: {results['total_tweets']} tweets analyzed")