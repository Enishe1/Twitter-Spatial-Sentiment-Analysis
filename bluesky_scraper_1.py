﻿"""
Bluesky AI Posts Collector 
Simple, reliable, with location extraction
"""
from atproto import Client, models
import pandas as pd
import time
from datetime import datetime
import re
import os
from typing import List, Dict

class BlueskyAICollector:
    def __init__(self, handle: str, app_password: str):
        """
        Initialize with YOUR Bluesky credentials
        handle: yourusername.bsky.social
        app_password: 16-char code from app passwords page
        """
        print("Initializing Bluesky collector...")
        
        try:
            self.client = Client()
            # This is the ONLY authentication that matters
            self.client.login(handle, app_password)
            print(f"Authenticated as: {handle}")
            self.authenticated = True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("\n COMMON FIXES:")
            print("1. Get app password from: https://bsky.app/settings/app-passwords")
            print("2. Use the 16-char CODE, not the app NAME")
            print("3. Handle must include .bsky.social")
            print("   Example: username.bsky.social")
            print("4. App passwords expire if you change main password")
            self.authenticated = False
            return
        
        # Create Data folder
        os.makedirs('Data', exist_ok=True)
        
        # AI search terms
        self.search_terms = [
            "AI",
            "artificial intelligence", 
            "machine learning",
            "deep learning",
            "neural network",
            "ChatGPT",
            "GPT",
            "LLM",
            "data science"
        ]
    
    def extract_simple_location(self, description: str, handle: str) -> str:
        """
        Extract location from profile description or handle
        Returns location string or empty if not found
        """
        if not description:
            description = ""
        
        # Method 1: Look for location in description
        location_patterns = [
            r'📍\s*(.+)',
            r'Location:\s*(.+)',
            r'location:\s*(.+)',
            r'Based in\s*(.+)',
            r'based in\s*(.+)',
            r'From\s*(.+)',
            r'from\s*(.+)',
            r'🏠\s*(.+)',
            r'🌍\s*(.+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                location = match.group(1) if match.lastindex else match.group(0)
                location = location.strip()
                # Take first part before comma or newline
                location = location.split(',')[0].strip()
                location = location.split('\n')[0].strip()
                if location and len(location) > 1:
                    return location[:50]  # Reasonable length limit
        
        # Method 2: Check TLD in handle for country
        tld_to_country = {
            '.us': 'USA',
            '.uk': 'UK',
            '.ca': 'Canada',
            '.au': 'Australia',
            '.de': 'Germany',
            '.fr': 'France',
            '.jp': 'Japan',
            '.in': 'India',
            '.cn': 'China',
            '.br': 'Brazil',
            '.mx': 'Mexico',
            '.ru': 'Russia',
            '.kr': 'South Korea',
            '.sg': 'Singapore',
            '.it': 'Italy',
            '.es': 'Spain',
            '.nl': 'Netherlands',
        }
        
        handle_lower = handle.lower()
        for tld, country in tld_to_country.items():
            if handle_lower.endswith(tld):
                return country
        
        # Method 3: Look for country names in description
        countries = [
            'USA', 'United States', 'UK', 'United Kingdom', 'Canada', 'Australia',
            'Germany', 'France', 'Japan', 'India', 'China', 'Brazil', 'Mexico',
            'Russia', 'South Korea', 'Italy', 'Spain', 'Netherlands', 'Sweden',
            'Norway', 'Denmark', 'Finland', 'Switzerland', 'Austria', 'Poland',
        ]
        
        for country in countries:
            if country.lower() in description.lower():
                return country
        
        # Method 4: Look for major cities
        major_cities = [
            'New York', 'NYC', 'Los Angeles', 'LA', 'Chicago', 'London',
            'Berlin', 'Paris', 'Tokyo', 'Sydney', 'Toronto', 'Singapore',
            'San Francisco', 'SF', 'Seattle', 'Boston', 'Austin', 'Miami',
            'Amsterdam', 'Stockholm', 'Oslo', 'Copenhagen', 'Zurich', 'Vienna',
        ]
        
        for city in major_cities:
            if city.lower() in description.lower():
                return city
        
        return ""  # No location found
    
    def search_ai_posts(self, search_term: str, limit: int = 100) -> List[Dict]:
        """Search for posts with a given term"""
        posts_data = []
        
        try:
            print(f"  Searching: '{search_term}'")
            
            # Create search parameters
            params = models.AppBskyFeedSearchPosts.Params(
                q=search_term,
                limit=min(limit, 100)
            )
            
            # Make the API call
            response = self.client.app.bsky.feed.search_posts(params)
            
            # Process each post
            for i, post in enumerate(response.posts):
                try:
                    # Extract basic post info
                    author = post.author
                    record = post.record
                    
                    # Get user profile for location
                    location = ""
                    try:
                        # Try to get user profile
                        profile = self.client.get_profile(author.handle)
                        if hasattr(profile, 'description'):
                            location = self.extract_simple_location(
                                profile.description, 
                                author.handle
                            )
                    except:
                        pass  # Skip if profile inaccessible
                    
                    # Create post data
                    post_data = {
                        'link': f"https://bsky.app/profile/{author.handle}/post/{post.uri.split('/')[-1]}",
                        'content': record.text if hasattr(record, 'text') else "",
                        'sentiment': "",  # Empty for now
                        'location': location,
                        'username': author.handle,
                        'display_name': author.display_name if hasattr(author, 'display_name') else "",
                        'created_at': record.created_at if hasattr(record, 'created_at') else "",
                        'like_count': post.like_count if hasattr(post, 'like_count') else 0,
                        'reply_count': post.reply_count if hasattr(post, 'reply_count') else 0,
                        'repost_count': post.repost_count if hasattr(post, 'repost_count') else 0,
                        'post_id': post.uri.split('/')[-1],
                        'search_term': search_term,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    posts_data.append(post_data)
                    
                    # Progress update
                    if (i + 1) % 20 == 0:
                        print(f"    Processed {i + 1} posts")
                    
                except Exception as e:
                    # Skip individual post errors
                    continue
            
            print(f"    Found {len(posts_data)} AI posts")
            
            # Respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"    Search error: {e}")
        
        return posts_data
    
    def collect_posts(self, total_posts: int = 1000) -> str:
        """
        Main collection function
        Returns path to saved CSV file
        """
        if not self.authenticated:
            print("Cannot collect: Authentication failed")
            return ""
        
        print("\n" + "=" * 60)
        print(f"COLLECTING {total_posts} AI POSTS FROM BLUESKY")
        print("=" * 60)
        
        all_posts = []
        start_time = time.time()
        
        # Try each search term until we reach target
        for term in self.search_terms:
            if len(all_posts) >= total_posts:
                break
            
            remaining = total_posts - len(all_posts)
            posts_from_term = self.search_ai_posts(term, min(200, remaining))
            
            # Remove duplicates by post ID
            existing_ids = {p['post_id'] for p in all_posts}
            new_posts = [p for p in posts_from_term if p['post_id'] not in existing_ids]
            
            all_posts.extend(new_posts)
            
            print(f"  After '{term}': {len(all_posts)} total posts")
            
            # Stop if we're not getting new posts
            if len(new_posts) < 10:
                print(f"  Few new posts, trying next term...")
        
        # Trim to exact target
        all_posts = all_posts[:total_posts]
        
        return self.save_to_csv(all_posts, start_time)
    
    def save_to_csv(self, posts: List[Dict], start_time: float) -> str:
        """Save posts to CSV with summary"""
        if not posts:
            print("No posts collected")
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Ensure required columns exist
        required_columns = ['link', 'content', 'sentiment', 'location']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns (required first)
        other_columns = [c for c in df.columns if c not in required_columns]
        df = df[required_columns + other_columns]
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Data/bluesky_ai_{len(df)}_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Print summary
        self.print_summary(df, filename, start_time)
        
        return filename
    
    def print_summary(self, df: pd.DataFrame, filename: str, start_time: float):
        """Print collection summary"""
        print("\n" + "=" * 60)
        print(" COLLECTION COMPLETE!")
        print("=" * 60)
        
        total_posts = len(df)
        posts_with_location = df[df['location'] != ''].shape[0]
        location_percentage = (posts_with_location / total_posts * 100) if total_posts > 0 else 0
        
        print(f" STATISTICS:")
        print(f"   Total posts collected: {total_posts:,}")
        print(f"   Posts with location: {posts_with_location:,} ({location_percentage:.1f}%)")
        print(f"   Blank locations: {total_posts - posts_with_location:,}")
        
        print(f"\n LOCATION DISTRIBUTION:")
        if posts_with_location > 0:
            location_counts = df['location'].value_counts()
            top_locations = location_counts.head(10)
            
            for location, count in top_locations.items():
                if location:  # Skip empty
                    percentage = (count / total_posts) * 100
                    print(f"   {location}: {count:,} posts ({percentage:.1f}%)")
        
        print(f"\n  PERFORMANCE:")
        collection_time = time.time() - start_time
        print(f"   Collection time: {collection_time:.1f} seconds")
        print(f"   Posts per second: {total_posts/collection_time:.1f}" if collection_time > 0 else "")
        
        print(f"\n OUTPUT:")
        print(f"   File saved: {filename}")
        print(f"   File size: {os.path.getsize(filename)/1024:.1f} KB")
        
        print(f"\n COLUMNS IN CSV:")
        print("   1. link - URL to Bluesky post")
        print("   2. content - Post text (for sentiment analysis)")
        print("   3. sentiment - EMPTY (for your model to fill)")
        print("   4. location - Extracted location (or blank)")
        print("   ... plus metadata columns")
        
        print("\n" + "=" * 60)
        print(" NEXT STEP: Run your sentiment analysis model")
        print("   on the 'content' column!")
        print("=" * 60)

def main():
    """Main function - edit YOUR credentials here"""
    print("=" * 60)
    print("BLUESKY AI POSTS COLLECTOR")
    print("=" * 60)
    
    # EDIT EDIT EDIT
    # EDIT EDIT EDIT
    # EDIT EDIT EDIT
    # EDIT THESE TWO LINES WITH YOUR CREDENTIALS 
    YOUR_HANDLE = ""  # CHANGE THIS
    YOUR_APP_PASSWORD = ""  # CHANGE THIS
    #  EDIT THESE TWO LINES WITH YOUR CREDENTIALS 
    # EDIT EDIT EDIT
    # EDIT EDIT EDIT
    # EDIT EDIT EDIT
    
    print(f"\nUsing handle: {YOUR_HANDLE}")
    print("Using app password: [hidden]")
    
    if YOUR_HANDLE == "" or YOUR_APP_PASSWORD == "":
        print("\n ERROR: You must edit the code with YOUR credentials!")
        print("\n HOW TO GET CREDENTIALS:")
        print("1. Go to: https://bsky.app/settings/app-passwords")
        print("2. Click 'Add App Password'")
        print("3. Copy the 16-CHARACTER CODE (looks like: abcd-efgh-ijkl-mnop)")
        print("4. Paste it in YOUR_APP_PASSWORD above")
        print("5. Use your full Bluesky handle (with .bsky.social)")
        return
    
    # Get target post count
    try:
        target_input = input(f"\nHow many AI posts to collect? (default: 500): ").strip()
        if target_input:
            target_posts = int(target_input)
        else:
            target_posts = 500
    except:
        target_posts = 500
        print(f"Using default: {target_posts} posts")
    
    print(f"\nStarting collection of {target_posts} AI posts...")
    print("This may take 2-5 minutes")
    print("-" * 60)
    
    # Create collector and collect
    collector = BlueskyAICollector(YOUR_HANDLE, YOUR_APP_PASSWORD)
    
    if not collector.authenticated:
        return
    
    try:
        output_file = collector.collect_posts(target_posts)
        
        if output_file:
            print(f"\n Data ready for sentiment analysis!")
            print(f"   File: {output_file}")
            
            # Show sample
            import pandas as pd
            df_sample = pd.read_csv(output_file, nrows=3)
            print(f"\n SAMPLE DATA (first 3 rows):")
            print(df_sample[['link', 'content', 'location']].to_string())
        
    except KeyboardInterrupt:
        print("\n Collection interrupted by user")
    except Exception as e:
        print(f"\n Collection error: {e}")

if __name__ == "__main__":
    # First install: pip install atproto pandas
    main()