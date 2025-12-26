"""
Bluesky AI Posts Collector - CLEAN VERSION
Uses unambiguous AI search terms only
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
        CORRECT: username.bsky.social  (NO @ symbol!)
        """
        print("Initializing Bluesky collector...")
        
        # Remove @ symbol if present
        handle = handle.replace('@', '').strip()
        
        try:
            self.client = Client()
            self.client.login(handle, app_password)
            print(f" Authenticated as: {handle}")
            self.authenticated = True
            
        except Exception as e:
            print(f" Authentication failed: {str(e)[:100]}")
            print(f"\n Your handle was: '{handle}'")
            print("   Remove @ symbol and ensure format: username.bsky.social")
            print("   Get NEW app password: https://bsky.app/settings/app-passwords")
            self.authenticated = False
            return
        
        os.makedirs('Data', exist_ok=True)
        
        # UNAMBIGUOUS AI SEARCH TERMS ONLY
        # No short terms like "IA" or "AI" alone that could match other languages
        self.search_terms = [
            # English - full phrases with context
            "artificial intelligence",
            "machine learning", 
            "deep learning",
            "neural network",
            "ChatGPT", 
            "AI assistant",
            "GPT",
            "large language model",
            "machine learning",
            "AGI", 
            "artificial general intelligence",
            "AI research paper",
            
            # Spanish - full phrases only
            "inteligencia artificial",
            "aprendizaje automático",
            "red neuronal modelo",
            
            # German - full phrases only
            "künstliche Intelligenz",
            "maschinelles Lernen",
            
            # French - full phrases only
            "intelligence artificielle",
            "apprentissage automatique",
            
            # Italian - full phrases only
            "intelligenza artificiale ricerca",
            
            # European context
            "European AI research",
            "UK artificial intelligence",
            "Germany machine learning",
            "intelligence artificielle",
            "inteligencia artificial",
            "IA"
            
            # Tech companies/context
            "OpenAI ChatGPT",
            "Google AI research",
            "Microsoft AI development",
            "Meta artificial intelligence",
        ]
        
        # Track already collected users
        self.collected_users: Set[str] = set()
        self.max_posts_per_user = 2  # Even stricter limit
        
        # Language detection
        self.language_indicators = {
            'english': ['the ', ' and ', ' for ', ' with ', ' this ', ' that '],
            'spanish': [' el ', ' la ', ' y ', ' en ', ' de ', ' que '],
            'german': [' der ', ' die ', ' das ', ' und ', ' für ', ' von '],
            'french': [' le ', ' la ', ' les ', ' et ', ' à ', ' dans '],
            'italian': [' il ', ' la ', ' lo ', ' e ', ' di ', ' che ']
        }
    
    def extract_location(self, description: str, handle: str) -> str:
        """
        Extract location from profile
        Returns location string or empty if not found
        """
        if not description:
            description = ""
        
        # Universal patterns
        patterns = [
            r'📍\s*(.+)',
            r'Location:\s*(.+)',
            r'Standort:\s*(.+)',
            r'Ubicación:\s*(.+)',
            r'Localisation:\s*(.+)',
            r'Località:\s*(.+)',
            r'Based in\s*(.+)',
            r'living in\s*(.+)',
            r'from\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                location = match.group(1) if match.lastindex else match.group(0)
                location = location.strip()
                location = location.split(',')[0].strip()
                location = location.split('\n')[0].strip()
                if location and len(location) > 1:
                    # Clean common false positives
                    false_positives = ['http', 'https', 'www.', '.com', 'AI', 'ML', 'tech']
                    if not any(fp in location.lower() for fp in false_positives):
                        return location[:50]
        
        # Check country domains
        tld_to_country = {
            # Europe
            '.uk': 'United Kingdom', '.de': 'Germany', '.fr': 'France',
            '.it': 'Italy', '.es': 'Spain', '.nl': 'Netherlands',
            '.se': 'Sweden', '.no': 'Norway', '.dk': 'Denmark',
            '.fi': 'Finland', '.pl': 'Poland', '.ch': 'Switzerland',
            '.at': 'Austria', '.be': 'Belgium', '.ie': 'Ireland',
            '.pt': 'Portugal', '.gr': 'Greece',
            
            # Americas
            '.us': 'United States', '.ca': 'Canada', '.mx': 'Mexico',
            '.br': 'Brazil', '.ar': 'Argentina', '.cl': 'Chile',
            
            # Asia/Oceania
            '.jp': 'Japan', '.in': 'India', '.cn': 'China',
            '.kr': 'South Korea', '.sg': 'Singapore', '.au': 'Australia',
        }
        
        handle_lower = handle.lower()
        for tld, country in tld_to_country.items():
            if handle_lower.endswith(tld):
                return country
        
        return ""
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not text or len(text) < 20:
            return 'unknown'
        
        text_lower = f" {text.lower()} "
        scores = {}
        
        for lang, indicators in self.language_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 1
            scores[lang] = score
        
        # Special characters boost
        if 'ñ' in text_lower or '¿' in text or '¡' in text:
            scores['spanish'] += 3
        if 'ä' in text_lower or 'ö' in text_lower or 'ü' in text_lower or 'ß' in text_lower:
            scores['german'] += 3
        if 'é' in text_lower or 'è' in text_lower or 'ê' in text_lower or 'à' in text_lower:
            scores['french'] += 2
        
        max_lang = max(scores, key=scores.get)
        return max_lang if scores[max_lang] > 2 else 'english'
    
    def search_ai_posts(self, search_term: str, limit: int = 100) -> List[Dict]:
        """Search for AI posts with user diversity control"""
        posts_data = []
        user_post_counts = {}
        
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
                    
                    # Skip if user already at limit
                    if username in self.collected_users:
                        user_post_counts[username] = user_post_counts.get(username, 0) + 1
                        if user_post_counts[username] > self.max_posts_per_user:
                            continue
                    else:
                        user_post_counts[username] = 1
                    
                    # Get content
                    content = post.record.text if hasattr(post.record, 'text') else ""
                    
                    # Skip if content doesn't contain AI-related keywords
                    ai_keywords = ['artificial intelligence', 'machine learning', 'deep learning',
                                 'neural network', 'AI', 'ChatGPT', 'GPT', 'LLM', 'inteligencia artificial',
                                 'künstliche Intelligenz', 'intelligence artificielle']
                    if not any(keyword.lower() in content.lower() for keyword in ai_keywords):
                        continue  # Not actually about AI
                    
                    # Get location
                    location = ""
                    try:
                        profile = self.client.get_profile(username)
                        if hasattr(profile, 'description'):
                            location = self.extract_location(profile.description, username)
                    except:
                        pass
                    
                    # Detect language
                    language = self.detect_language(content)
                    
                    post_data = {
                        'link': f"https://bsky.app/profile/{username}/post/{post.uri.split('/')[-1]}",
                        'content': content[:10000],  # Limit length
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
                        'language': language,
                        'collected_at': datetime.now().isoformat(),
                        'user_post_count': user_post_counts[username]
                    }
                    
                    posts_data.append(post_data)
                    self.collected_users.add(username)
                    
                    if (i + 1) % 25 == 0:
                        unique = len(set(p['username'] for p in posts_data))
                        print(f"    Processed {i + 1} posts, {unique} unique users")
                    
                except Exception:
                    continue
            
            unique_users = len(set(p['username'] for p in posts_data))
            print(f"    Found {len(posts_data)} AI posts from {unique_users} users")
            
            time.sleep(1.5)  # Respect rate limits
            
        except Exception as e:
            print(f"    Search error: {e}")
        
        return posts_data
    
    def collect_posts(self, total_posts: int = 1000) -> str:
        """Main collection function"""
        if not self.authenticated:
            return ""
        
        print(f"\n COLLECTING {total_posts} AI POSTS")
        print("=" * 60)
        print(f"Settings: Max {self.max_posts_per_user} posts per user")
        print("Using unambiguous AI search terms only")
        print("=" * 60)
        
        all_posts = []
        start_time = time.time()
        collected_terms = set()
        
        for term_index, term in enumerate(self.search_terms):
            if len(all_posts) >= total_posts:
                break
            
            # Skip if we already searched similar term
            words = term.split()
            if len(words) > 1:
                base_word = words[0]
                if any(base_word in t for t in collected_terms):
                    print(f"  Skipping similar term: '{term}'")
                    continue
            
            collected_terms.add(term)
            
            # Progress update
            elapsed = (time.time() - start_time) / 60
            print(f"\n[{term_index + 1}/{len(self.search_terms)}] '{term}'")
            print(f"Progress: {len(all_posts)}/{total_posts} posts")
            print(f"Unique users: {len(self.collected_users)}")
            print(f"Elapsed: {elapsed:.1f} minutes")
            
            remaining = total_posts - len(all_posts)
            posts_from_term = self.search_ai_posts(term, min(120, remaining))
            
            # Deduplicate
            existing_ids = {p['post_id'] for p in all_posts}
            new_posts = [p for p in posts_from_term if p['post_id'] not in existing_ids]
            
            all_posts.extend(new_posts)
            
            print(f"  Added {len(new_posts)} new posts")
            
            # Stop if we're not getting results
            if len(new_posts) < 3:
                print(f"  Few results, moving to next term...")
        
        # Final trim
        all_posts = all_posts[:total_posts]
        
        return self.save_results(all_posts, start_time)
    
    def save_results(self, posts: List[Dict], start_time: float) -> str:
        """Save results with comprehensive statistics"""
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
        filename = f"Data/bluesky_ai_{len(df)}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Statistics
        total_posts = len(df)
        unique_users = df['username'].nunique()
        posts_with_location = df[df['location'] != ''].shape[0]
        
        print(f"\n" + "=" * 60)
        print(f" COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total posts: {total_posts}")
        print(f"Unique users: {unique_users}")
        print(f"Avg posts per user: {total_posts/unique_users:.2f}")
        print(f"Posts with location: {posts_with_location} ({posts_with_location/total_posts*100:.1f}%)")
        
        # Language distribution
        if 'language' in df.columns:
            print(f"\n  LANGUAGE DISTRIBUTION:")
            for lang, count in df['language'].value_counts().items():
                percentage = (count / total_posts) * 100
                print(f"  {lang.title()}: {count} posts ({percentage:.1f}%)")
        
        # Geographic distribution
        print(f"\n GEOGRAPHIC DISTRIBUTION:")
        
        # Count European vs non-European
        european_countries = ['United Kingdom', 'Germany', 'France', 'Italy', 'Spain',
                             'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland',
                             'Poland', 'Switzerland', 'Austria', 'Belgium', 'Ireland',
                             'Portugal', 'Greece']
        
        european_posts = df[df['location'].isin(european_countries)].shape[0]
        us_posts = df[df['location'].isin(['United States', 'USA', 'US'])].shape[0]
        
        print(f"  European posts: {european_posts} ({european_posts/total_posts*100:.1f}%)")
        print(f"  US posts: {us_posts} ({us_posts/total_posts*100:.1f}%)")
        
        # Top locations
        print(f"\n TOP LOCATIONS:")
        top_locations = df['location'].value_counts().head(10)
        for loc, count in top_locations.items():
            if loc:
                percentage = (count / total_posts) * 100
                print(f"  {loc}: {count} posts ({percentage:.1f}%)")
        
        print(f"\n  Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f" Saved: {filename}")
        print(f"\n Ready for sentiment analysis!")
        print("   The 'sentiment' column is empty for your model")
        
        return filename

def main():
    """Main function"""
    print("=" * 60)
    print(" BLUESKY AI RELATED POSTS COLLECTOR")
    print("=" * 60)
    print("Features:")
    print("- Unambiguous AI search terms only")
    print("- No 'IA' or short ambiguous terms")
    print("- Limits posts per user (diversity)")
    print("- Multilingual support")
    print("=" * 60)
    
    # Get credentials
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
    
    print(f"\nStarting collection of {target_posts} AI posts...")
    
    collector = BlueskyAICollector(handle, app_password)
    
    if not collector.authenticated:
        return
    
    try:
        output_file = collector.collect_posts(target_posts)
        if output_file:
            print(f"\n Success! File: {output_file}")
            
            # Show quick sample
            df = pd.read_csv(output_file, nrows=3)
            print(f"\n SAMPLE (first 3 posts):")
            for _, row in df.iterrows():
                print(f"\nPost: {row['link'][:60]}...")
                print(f"Content: {row['content'][:80]}..." if len(str(row['content'])) > 80 else f"Content: {row['content']}")
                print(f"Location: {row['location']}")
                print("-" * 40)
                
    except KeyboardInterrupt:
        print("\n  Stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")

if __name__ == "__main__":
    # Install: pip install atproto pandas
    main()