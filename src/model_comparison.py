import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import os

class ModelTester:
    def __init__(self):
        self.models = {
            'roberta_twitter': self.roberta_twitter_predict,
            'vader': self.vader_predict,  
            'naive_bayes': self.naive_bayes_predict
        }
        self.loaded_models = {}
    
    def load_airline_dataset(self):
        """Load and prepare Twitter Airline Sentiment dataset"""
        # https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment?resource=download
        try:
            # Load the dataset
            df = pd.read_csv('data/Tweets.csv')
            
            # Map sentiment to our format
            sentiment_map = {
                'positive': 'positive',
                'negative': 'negative', 
                'neutral': 'neutral'
            }
            
            # Prepare data
            df = df[['text', 'airline_sentiment']].dropna()
            df['sentiment'] = df['airline_sentiment'].map(sentiment_map)
            df = df[df['sentiment'].notna()]
            
            print(f"Loaded {len(df)} tweets from airline dataset")
            return df
            
        except Exception as e:
            print(f"X Error loading airline dataset: {e}")
            return None
    
    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        try:
            from transformers import pipeline
            
            if model_name == 'roberta_twitter':
                self.loaded_models['roberta_twitter'] = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            return self.loaded_models.get(model_name)
            
        except Exception as e:
            print(f"X Error loading model {model_name}: {e}")
            return None
    
    def roberta_twitter_predict(self, text):
        try:
            model = self.load_model('roberta_twitter')
            if not model: return 'neutral'
            result = model(text[:512])[0]
            return result['label'].lower()
        except:
            return 'neutral'
    
    def vader_predict(self, text):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            if 'vader_analyzer' not in self.loaded_models:
                self.loaded_models['vader_analyzer'] = SentimentIntensityAnalyzer()
            score = self.loaded_models['vader_analyzer'].polarity_scores(text)['compound']
            if score >= 0.05: return 'positive'
            elif score <= -0.05: return 'negative'
            else: return 'neutral'
        except:
            return 'neutral'
    
    def naive_bayes_predict(self, text):
        try:
            if 'naive_bayes' not in self.loaded_models:
                print("Training Naive Bayes on airline dataset...")
                
                # Load and prepare training data
                train_data = self.load_airline_dataset()
                if train_data is None:
                    raise Exception("Could not load airline dataset")
                
                # Split data (use 80% for training)
                train_size = int(0.8 * len(train_data))
                train_texts = train_data['text'].values[:train_size]
                train_labels = train_data['sentiment'].values[:train_size]
                
                print(f"Training on {len(train_texts)} tweets...")
                
                # Create and train pipeline
                self.loaded_models['naive_bayes'] = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        max_features=5000,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )),
                    ('nb', MultinomialNB(alpha=0.1))
                ])
                
                self.loaded_models['naive_bayes'].fit(train_texts, train_labels)
                print("Y Naive Bayes trained on airline data")
            
            return self.loaded_models['naive_bayes'].predict([text])[0]
            
        except Exception as e:
            print(f"X Naive Bayes error: {e}")
            return 'neutral'
    
    def evaluate_models(self, test_data, text_col='text', label_col='sentiment'):
        results = {}
        
        print("Testing 3 models...")
        
        for model_name, predict_func in self.models.items():
            print(f"Evaluating {model_name}...")
            
            predictions = []
            for i, text in enumerate(test_data[text_col]):
                if i % 100 == 0 and i > 0:
                    print(f"  {model_name}: {i}/{len(test_data)}")
                predictions.append(predict_func(text))
            
            accuracy = accuracy_score(test_data[label_col], predictions)
            f1 = f1_score(test_data[label_col], predictions, average='weighted')
            report = classification_report(test_data[label_col], predictions, output_dict=True)
            
            results[model_name] = {
                'accuracy': round(accuracy, 4),
                'f1_score': round(f1, 4),
                'precision': round(report['weighted avg']['precision'], 4),
                'recall': round(report['weighted avg']['recall'], 4),
                'support': len(test_data)
            }
            
            print(f"   {model_name} - F1: {f1:.4f}")
        
        return results
    
    def select_best_model(self, results):
        return max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    
    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports/model_evaluation", exist_ok=True)
        
        best_model = self.select_best_model(results)
        
        with open("models/model_metrics.json", "w") as f:
            json.dump({
                'timestamp': timestamp,
                'best_model': best_model,
                'results': results
            }, f, indent=2)
        
        metrics_df = pd.DataFrame({
            model: {k: v for k, v in metrics.items()} 
            for model, metrics in results.items()
        }).T
        metrics_df.to_csv(f"reports/model_evaluation/metrics_{timestamp}.csv")
        print(f"Results saved. The best model is : {best_model}")


def test_three_models():
    tester = ModelTester()
    
    try:
        # Use airline dataset for testing
        test_data = tester.load_airline_dataset()
        if test_data is None:
            raise Exception("No dataset available")
        
        # Use last 20% for testing
        test_size = len(test_data)
        test_data = test_data.tail(test_size // 5)  # 20% for testing
        
        print(f"Testing on {len(test_data)} airline tweets...")
        results = tester.evaluate_models(test_data)
        tester.save_results(results)
        best_model = tester.select_best_model(results)
        
        print(f"\nBest model: {best_model}")
        for model, metrics in results.items():
            print(f"   {model:15} → F1: {metrics['f1_score']:.3f}")
        
        return results, best_model
        
    except Exception as e:
        print(f"X Error: {e}")
        return None, None


if __name__ == "__main__":
    print("Testing models on Twitter Airline dataset...")
    results, best_model = test_three_models()