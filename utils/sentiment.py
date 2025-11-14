from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import json
import os

class SentimentAnalyzer:
    def __init__(self, model_name=None):
        self.model_name = self.load_best_model() if model_name is None else model_name
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.loaded_models = {}
    
    def load_best_model(self):
        """Load the best performing model from tests"""
        try:
            with open("models/model_metrics.json", "r") as f:
                metrics = json.load(f)
                return metrics.get('best_model', 'roberta_twitter')
        except:
            return 'roberta_twitter'  # most likely the best based on first test runs
    
    def clean_text(self, text):
        """Clean tweet text"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using the selected model"""
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 3:
            return 'neutral'
        
        if self.model_name == 'roberta_twitter':
            return self._roberta_predict(cleaned)
        elif self.model_name == 'vader':
            return self._vader_predict(cleaned)
        elif self.model_name == 'naive_bayes':
            return self._naive_bayes_predict(cleaned)
        else:
            return self._roberta_predict(cleaned)  # Default fallback
    
    def _roberta_predict(self, text):
        try:
            if 'roberta_twitter' not in self.loaded_models:
                from transformers import pipeline
                self.loaded_models['roberta_twitter'] = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            
            result = self.loaded_models['roberta_twitter'](text[:512])[0]
            return result['label'].lower()
        except Exception as e:
            print(f"RoBERTa prediction error: {e}")
            return 'neutral'
    
    def _vader_predict(self, text):
        try:
            score = self.vader_analyzer.polarity_scores(text)['compound']
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"VADER prediction error: {e}")
            return 'neutral'
    
    def _naive_bayes_predict(self, text):
        try:
            if 'naive_bayes' not in self.loaded_models:
                # Load the trained model
                import joblib
                self.loaded_models['naive_bayes'] = joblib.load('models/naive_bayes_model.pkl')
            
            return self.loaded_models['naive_bayes'].predict([text])[0]
        except Exception as e:
            print(f"Naive Bayes prediction error: {e}")
            return 'neutral'
    
    def batch_analyze(self, texts):
        """Analyze multiple texts at once"""
        return [self.analyze_sentiment(text) for text in texts]