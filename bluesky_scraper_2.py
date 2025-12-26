"""
Bluesky AI Posts Collector - FIXED VERSION
1. Fixed authentication (no @ symbol)
2. Prevents multiple posts from same user
3. Better European content
"""
from atproto import Client, models
import pandas as pd
import time
from datetime import datetime
import re
import os
from typing import List, Dict, Set

class BlueskyAICollector:
    def __init__(self, handle: str, app_password: str):
        """
        Initialize with YOUR Bluesky credentials
        username.bsky.social  (NO @ symbol!)
        """
        print("Initializing Bluesky collector...")
        
        Remove @ symbol if present
        handle = handle.replace('@', '').strip()
        
        try:
            self.client = Client()
            self.client.login(handle, app_password)
            print(f"Authenticated as: {handle}")
            self.authenticated = True
            
        except Exception as e:
            print(f"Authentication failed: {str(e)[:100]}")
            print(f"\n Your handle was: '{handle}'")
            print("   Remove @ symbol and ensure format: username.bsky.social")
            print("   Get NEW app password: https://bsky.app/settings/app-passwords")
            self.authenticated = False
            return
        
        os.makedirs('Data', exist_ok=True)
        
        # More diverse search terms
        self.search_terms = [
            "AI", "artificial intelligence", "machine learning",
            # European terms
            "EU AI", "European AI", "künstliche Intelligenz",
            "intelligence artificielle", "UK AI", "London tech",
            "Berlin KI", "Paris IA", "Amsterdam AI",
            # Global terms
            "data science", "deep learning", "neural network",
            "ChatGPT", "GPT", "LLM", "AGI"
        ]
        
        # Track already collected users
        self.collected_users: Set[str] = set()
        self.max_posts_per_user = 3  # Don't collect more than 3 posts from same user
    
    def extract_simple_location(self, description: str, handle: str) -> str:
        """Extract location with better European detection"""
        if not description:
            description = ""
        
        # Enhanced European detection
        european_patterns = [
            r'📍\s*(.+)',
            r'Location:\s*(.+)',
            r'Standort:\s*(.+)',  # German
            r'Localisation:\s*(.+)',  # French
            r'Ubicación:\s*(.+)',  # Spanish
            r'Posizione:\s*(.+)',  # Italian
            r'Locatie:\s*(.+)',  # Dutch
        ]
        
        for pattern in european_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                location = match.group(1) if match.lastindex else match.group(0)
                location = location.strip()
                location = location.split(',')[0].strip()
                location = location.split('\n')[0].strip()
                if location and len(location) > 1:
                    return location[:50]
        
        # European TLDs (priority)
        european_tlds = {
            '.uk': 'UK', '.de': 'Germany', '.fr': 'France',
            '.it': 'Italy', '.es': 'Spain', '.nl': 'Netherlands',
            '.se': 'Sweden', '.no': 'Norway', '.dk': 'Denmark',
            '.fi': 'Finland', '.pl': 'Poland', '.ch': 'Switzerland',
            '.at': 'Austria', '.be': 'Belgium', '.ie': 'Ireland',
            '.pt': 'Portugal', '.gr': 'Greece'
        }
        
        handle_lower = handle.lower()
        for tld, country in european_tlds.items():
            if handle_lower.endswith(tld):
                return country
        
        # US/other TLDs (lower priority)
        other_tlds = {
            '.us': 'USA', '.ca': 'Canada', '.au': 'Australia',
            '.jp': 'Japan', '.in': 'India', '.cn': 'China',
            '.br': 'Brazil', '.mx': 'Mexico', '.ru': 'Russia',
        }
        
        for tld, country in other_tlds.items():
            if handle_lower.endswith(tld):
                return country
        
        return ""
    
    def search_ai_posts(self, search_term: str, limit: int = 100) -> List[Dict]:
        """Search for posts with user diversity control"""
        posts_data = []
        user_post_counts = {}  # Track posts per user in this search
        
        try:
            print(f" Searching: '{search_term}'")
            
            params = models.AppBskyFeedSearchPosts.Params(
                q=search_term,
                limit=min(limit, 100)
            )
            
            response = self.client.app.bsky.feed.search_posts(params)
            
            for i, post in enumerate(response.posts):
                try:
                    author = post.author
                    username = author.handle
                    
                    # FIX 3: Limit posts from same user
                    user_post_counts[username] = user_post_counts.get(username, 0) + 1
                    
                    # Skip if user already contributed too many posts
                    if (username in self.collected_users and 
                        user_post_counts[username] > self.max_posts_per_user):
                        continue
                    
                    # Skip if user has too many posts in this search
                    if user_post_counts[username] > 2:  # Max 2 posts per search
                        continue
                    
                    # Get profile and location
                    location = ""
                    try:
                        profile = self.client.get_profile(username)
                        if hasattr(profile, 'description'):
                            location = self.extract_simple_location(
                                profile.description, 
                                username
                            )
                    except:
                        pass
                    
                    # Create post data
                    post_data = {
                        'link': f"https://bsky.app/profile/{username}/post/{post.uri.split('/')[-1]}",
                        'content': post.record.text if hasattr(post.record, 'text') else "",
                        'sentiment': "",
                        'location': location,
                        'username': username,
                        'display_name': author.display_name if hasattr(author, 'display_name') else "",
                        'created_at': post.record.created_at if hasattr(post.record, 'created_at') else "",
                        'like_count': post.like_count if hasattr(post, 'like_count') else 0,
                        'reply_count': post.reply_count if hasattr(post, 'reply_count') else 0,
                        'repost_count': post.repost_count if hasattr(post, 'repost_count') else 0,
                        'post_id': post.uri.split('/')[-1],
                        'search_term': search_term,
                        'collected_at': datetime.now().isoformat(),
                        'user_post_number': user_post_counts[username]  # Track which post this is for user
                    }
                    
                    posts_data.append(post_data)
                    self.collected_users.add(username)  # Mark user as collected
                    
                    if (i + 1) % 20 == 0:
                        print(f"    Processed {i + 1} posts, {len(posts_data)} unique")
                    
                except Exception:
                    continue
            
            print(f"    Found {len(posts_data)} unique AI posts (from {len(set(p['username'] for p in posts_data))} users)")
            time.sleep(1)
            
        except Exception as e:
            print(f"    Search error: {e}")
        
        return posts_data
    
    def collect_posts(self, total_posts: int = 1000) -> str:
        """Main collection with user diversity"""
        if not self.authenticated:
            return ""
        
        print(f"\n COLLECTING {total_posts} DIVERSE AI POSTS")
        print("=" * 60)
        print(f"Settings: Max {self.max_posts_per_user} posts per user")
        print("Searching European & global content")
        print("=" * 60)
        
        all_posts = []
        start_time = time.time()
        
        # Track progress by unique users
        unique_users = set()
        
        for term_index, term in enumerate(self.search_terms):
            if len(all_posts) >= total_posts:
                break
            
            # Progress update
            elapsed = time.time() - start_time
            print(f"\n[{term_index + 1}/{len(self.search_terms)}] '{term}'")
            print(f"Progress: {len(all_posts)}/{total_posts} posts")
            print(f"Unique users: {len(unique_users)}")
            print(f"Elapsed: {elapsed/60:.1f} minutes")
            
            remaining = total_posts - len(all_posts)
            posts_from_term = self.search_ai_posts(term, min(150, remaining))
            
            # Deduplicate by post ID AND username
            existing_ids = {p['post_id'] for p in all_posts}
            new_posts = []
            
            for post in posts_from_term:
                if (post['post_id'] not in existing_ids and 
                    post['username'] not in unique_users):
                    new_posts.append(post)
                    unique_users.add(post['username'])
            
            all_posts.extend(new_posts)
            
            print(f"  Added {len(new_posts)} new posts from {len(set(p['username'] for p in new_posts))} users")
            
            if len(new_posts) < 5:
                print(f"  Few results, moving on...")
        
        # Final trim
        all_posts = all_posts[:total_posts]
        
        return self.save_to_csv(all_posts, start_time)
    
    def save_to_csv(self, posts: List[Dict], start_time: float) -> str:
        """Save with diversity statistics"""
        if not posts:
            print("No posts collected")
            return ""
        
        df = pd.DataFrame(posts)
        
        # Ensure required columns
        required = ['link', 'content', 'sentiment', 'location']
        for col in required:
            if col not in df.columns:
                df[col] = ''
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Data/bluesky_diverse_{len(df)}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Enhanced statistics
        total_posts = len(df)
        unique_users = df['username'].nunique()
        posts_with_location = df[df['location'] != ''].shape[0]
        
        print(f"\n" + "=" * 60)
        print(f" DIVERSITY REPORT")
        print("=" * 60)
        print(f"Total posts: {total_posts}")
        print(f"Unique users: {unique_users}")
        print(f"Posts per user (avg): {total_posts/unique_users:.2f}")
        print(f"Posts with location: {posts_with_location} ({posts_with_location/total_posts*100:.1f}%)")
        
        # European vs US breakdown
        european_locations = ['UK', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
                             'Sweden', 'Norway', 'Denmark', 'Finland', 'Switzerland']
        us_locations = ['USA', 'United States']
        
        eu_posts = df[df['location'].isin(european_locations)].shape[0]
        us_posts = df[df['location'].isin(us_locations)].shape[0]
        
        print(f"\n GEOGRAPHIC DISTRIBUTION:")
        print(f"  European posts: {eu_posts} ({eu_posts/total_posts*100:.1f}%)")
        print(f"  US posts: {us_posts} ({us_posts/total_posts*100:.1f}%)")
        
        # Top locations
        print(f"\n TOP LOCATIONS:")
        for loc, count in df['location'].value_counts().head(10).items():
            if loc:
                print(f"  {loc}: {count} posts")
        
        print(f"\n  Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f" Saved: {filename}")
        
        return filename

def main():
    """Main function with interactive credentials"""
    print("=" * 60)
    print(" DIVERSE BLUESKY AI COLLECTOR")
    print("=" * 60)
    print("Features:")
    print("- Limits posts per user (no spam)")
    print("- European-focused search terms")
    print("- Better location extraction")
    print("=" * 60)
    
    # Get credentials interactively (safer)
    handle = input("Bluesky handle (username.bsky.social): ").strip()
    app_password = input("App password (16 chars): ").strip()
    
    if not handle or not app_password:
        print("Credentials required")
        return
    
    try:
        target = input("Posts to collect (default: 500): ").strip()
        target_posts = int(target) if target else 500
    except:
        target_posts = 500
    
    print(f"\nStarting collection of {target_posts} diverse posts...")
    
    collector = BlueskyAICollector(handle, app_password)
    
    if not collector.authenticated:
        return
    
    try:
        output_file = collector.collect_posts(target_posts)
        if output_file:
            print(f"\n Success! File: {output_file}")
    except KeyboardInterrupt:
        print("\n  Stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")

if __name__ == "__main__":
    # pip install atproto pandas
    main()