import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

class TwitterClient:
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not self.bearer_token:
            raise ValueError("X Please set TWITTER_BEARER_TOKEN in your .env file")
        
        print("Y Twitter client initialized")
        self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
    
    def fetch_tweets(self, keyword, max_results=50):
        """
        Fetch recent tweets with location data
        
        Args:
            keyword: Search term or hashtag
            max_results: Maximum number of tweets to return
        
        Returns:
            DataFrame with tweet data including text and location
        """
        try:
            # clean search query
            search_query = keyword.strip()
            if search_query.startswith('#'):
                search_query = f"#{search_query[1:]}"
            
            # Build query - exclude retweets, require geo data, English only
            query = f"{search_query} -is:retweet has:geo lang:en"
            
            print(f"Searching Twitter for: {query}")
            
            # Fetch tweets from last 24 hours
            start_time = (datetime.now() - timedelta(hours=24)).isoformat() + "Z"
            
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                start_time=start_time,
                tweet_fields=['created_at', 'geo', 'text', 'author_id', 'public_metrics'],
                expansions=['geo.place_id', 'author_id'],
                place_fields=['contained_within', 'country', 'country_code', 'full_name', 'geo', 'id', 'name', 'place_type'],
                user_fields=['location', 'name', 'username']
            )
            
            if not tweets.data:
                print("X No tweets found matching criteria")
                return pd.DataFrame()
            
            print(f"Y Found {len(tweets.data)} tweets, processing locations...")
            
            # Process tweets with location data
            tweet_data = []
            for tweet in tweets.data:
                if tweet.geo:
                    location_info = self.extract_location_info(tweet, tweets.includes)
                    if location_info:
                        tweet_data.append({
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'author_id': tweet.author_id,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'latitude': location_info['latitude'],
                            'longitude': location_info['longitude'],
                            'location_type': location_info['type'],
                            'place_name': location_info.get('place_name', ''),
                            'country': location_info.get('country', '')
                        })
            
            df = pd.DataFrame(tweet_data)
            print(f" {len(df)} tweets with usable location data")
            return df
            
        except tweepy.TooManyRequests:
            print("Rate limit exceeded, waiting...")
            time.sleep(60)
            return self.fetch_tweets(keyword, max_results)
            
        except Exception as e:
            print(f"X Error fetching tweets: {e}")
            return pd.DataFrame()
    
    def extract_location_info(self, tweet, includes):
        """
        Extract location coordinates from tweet geo data
        
        Args:
            tweet: Tweet object
            includes: Includes from API response
        
        Returns:
            Dictionary with location info or None
        """
        try:
            if tweet.geo.get('coordinates'):
                # Point coordinates (most precise)
                coords = tweet.geo['coordinates']
                return {
                    'latitude': coords['coordinates'][1],
                    'longitude': coords['coordinates'][0],
                    'type': 'coordinates'
                }
            elif tweet.geo.get('place_id') and includes.get('places'):
                # Place-based coordinates (less precise)
                place_id = tweet.geo['place_id']
                place = next((p for p in includes['places'] if p.id == place_id), None)
                
                if place and place.geo and place.geo.get('bbox'):
                    bbox = place.geo['bbox']
                    # Use center of bounding box
                    # A bounding box is typically a list of 4 values:
                    # [min_lon, min_lat, max_lon, max_lat]
                    lat = (bbox[1] + bbox[3]) / 2
                    lon = (bbox[0] + bbox[2]) / 2
                    # This gives us the latitude and longitude of the center of the geographic area defined by the bounding box

                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'type': 'place_center',
                        'place_name': place.full_name,
                        'country': place.country_code
                    }
            
            return None
            
        except Exception as e:
            print(f"X Error extracting location: {e}")
            return None
    
    def get_user_location(self, author_id, includes):
        """Extract location from user profile (fallback)"""
        try:
            if includes.get('users'):
                user = next((u for u in includes['users'] if u.id == author_id), None)
                if user and user.location:
                    return user.location
        except:
            pass
        return None

if __name__ == "__main__":
    # Test the Twitter client
    client = TwitterClient()
    test_df = client.fetch_tweets("python", 10)
    print(f"Test fetched {len(test_df)} tweets")