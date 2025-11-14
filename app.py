import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import analyze_keyword
from src.model_comparison import test_three_models
import json

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Map",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive { color: #00FF00; }
    .negative { color: #FF0000; }
    .neutral { color: #FFFF00; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header"> Twitter Sentiment Map</h1>', unsafe_allow_html=True)
st.markdown("Analyze real-time Twitter sentiment for any keyword!")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    try:
        with open("models/model_metrics.json", "r") as f:
            metrics = json.load(f)
            best_model = metrics.get('best_model', 'roberta_twitter')
        st.info(f"Auto-selected: **{best_model}**")
    except:
        best_model = 'roberta_twitter'
        st.info("Using default: **RoBERTa-Twitter**")
    
    st.markdown("---")
    
    # Search settings
    st.subheader("Search Settings")
    keyword = st.text_input("Enter keyword or hashtag:", "artificial intelligence")
    tweet_count = st.slider("Number of tweets:", 10, 100, 30)
    
    st.markdown("---")
    st.markdown("### Legend")
    st.markdown('- <span class="positive"> Positive</span>', unsafe_allow_html=True)
    st.markdown('- <span class="negative"> Negative</span>', unsafe_allow_html=True)
    st.markdown('- <span class="neutral"> Neutral</span>', unsafe_allow_html=True)

# Main analysis
if st.button("Analyze Tweets") or keyword:
    with st.spinner(f"Analyzing '{keyword}' with {best_model}..."):
        results = analyze_keyword(keyword, tweet_count, model_name=best_model)
        
        if results['total_tweets'] == 0:
            st.error("X No geotagged tweets found. Try a different keyword or check if tweets have location data.")
        else:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                st.metric("Total Tweets", results['total_tweets'])
            with col2:
                st.metric("Positive", results['positive_count'], delta=f"{results['positive_pct']:.1f}%")
            with col3:
                st.metric("Negative", results['negative_count'], delta=f"{results['negative_pct']:.1f}%", delta_color="inverse")
            with col4:
                st.metric("Neutral", results['neutral_count'], delta=f"{results['neutral_pct']:.1f}%")
            
            # Map and charts
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("Sentiment Map")
                # Display the map with colored markers
                if not results['tweets_df'].empty:
                    st.map(results['tweets_df'])
                else:
                    st.warning("No tweets with location data to display on map.")
            
            with col_right:
                st.subheader("Sentiment Distribution")
                
                # Pie chart
                fig = px.pie(
                    values=[results['positive_count'], results['negative_count'], results['neutral_count']],
                    names=['Positive', 'Negative', 'Neutral'],
                    color=['Positive', 'Negative', 'Neutral'],
                    color_discrete_map={'Positive': '#00FF00', 'Negative': '#FF0000', 'Neutral': '#FFFF00'}
                )
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment breakdown
                st.markdown("### Breakdown")
                st.write(f"**Positive:** {results['positive_count']} tweets ({results['positive_pct']:.1f}%)")
                st.write(f"**Negative:** {results['negative_count']} tweets ({results['negative_pct']:.1f}%)")
                st.write(f"**Neutral:** {results['neutral_count']} tweets ({results['neutral_pct']:.1f}%)")
            
            # Show raw data
            with st.expander(" View Tweet Data"):
                if not results['tweets_df'].empty:
                    display_df = results['tweets_df'][['text', 'sentiment', 'created_at']].copy()
                    display_df['text_preview'] = display_df['text'].str[:80] + '...'
                    st.dataframe(display_df[['text_preview', 'sentiment', 'created_at']])
                    
                    # Download option
                    csv = results['tweets_df'].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"tweets_{keyword.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No tweet data available for download.")

# Footer
st.markdown("---")
st.markdown("*Note: Only tweets with location data are displayed. Analysis based on last 24 hours.*")
st.markdown("**Model Info:** Using RoBERTa-Twitter fine-tuned for Twitter sentiment analysis")