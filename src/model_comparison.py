import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

class ModelTester:
    def __init__(self):
        self.models = {
            'roberta_twitter': self.roberta_twitter_predict,  # Transformer-based (modern deep learning)
            'vader': self.vader_predict,   # Lexicon-based (rule-based sentiment dictionary)
            'naive_bayes': self.naive_bayes_predict #  Traditional ML (statistical approach)
        }
        self.loaded_models = {}
        # Register the plots directory
        os.makedirs("plots", exist_ok=True)
    
    def load_airline_dataset(self):
        """Load and prepare Twitter Airline Sentiment dataset"""
        try:
            df = pd.read_csv('data/Tweets.csv')
            
            sentiment_map = {
                'positive': 'positive',
                'negative': 'negative', 
                'neutral': 'neutral'
            }
            
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
                
                train_data = self.load_airline_dataset()
                if train_data is None:
                    raise Exception("Could not load airline dataset")
                
                train_size = int(0.8 * len(train_data))
                train_texts = train_data['text'].values[:train_size]
                train_labels = train_data['sentiment'].values[:train_size]
                
                print(f"Training on {len(train_texts)} tweets...")
                
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
    
    def create_comparison_plots(self, results, all_predictions, test_data, label_col='sentiment'):
        """Create comprehensive comparison plots"""
        
        #  plot 1: Performance Metrics Comparison Bar Plot
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            plt.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Model Performance Comparison\n(Transformer vs Lexicon vs Traditional ML)')
        plt.xticks(x + width*1.5, ['RoBERTa Twitter\n(Transformer)', 'VADER\n(Lexicon)', 'Naive Bayes\n(Traditional ML)'])
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # plot 2: Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sentiments = ['negative', 'neutral', 'positive']
        
        for idx, (model_name, predictions) in enumerate(all_predictions.items()):
            cm = confusion_matrix(test_data[label_col], predictions, labels=sentiments)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sentiments, 
                       yticklabels=sentiments, ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # plot 3: F1-Score Comparison by Class
        plt.figure(figsize=(10, 6))
        
        for model_name in model_names:
            predictions = all_predictions[model_name]
            report = classification_report(test_data[label_col], predictions, 
                                         labels=sentiments, output_dict=True)
            f1_scores = [report[sentiment]['f1-score'] for sentiment in sentiments]
            plt.plot(sentiments, f1_scores, marker='o', linewidth=2, label=model_name)
        
        plt.xlabel('Sentiment Class')
        plt.ylabel('F1-Score')
        plt.title('F1-Score by Sentiment Class\n(Comparing Model Strengths per Class)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('plots/f1_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # plot 4: Radar Chart for Model Comparison
        self.create_radar_chart(results, model_names)
        
        # plot 5: Training vs Inference Time (simulated)
        self.create_training_inference_plot(model_names)
        
        print("All comparison plots saved to 'plots/' folder")
    
    def create_radar_chart(self, results, model_names):
        """Create radar chart comparing multiple metrics"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in model_names:
            values = [results[model_name][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart\n(Multi-Metric Comparison)', size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig('plots/performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_inference_plot(self, model_names):
        """Create plot comparing training vs inference characteristics"""
        # Simulated data for demonstration
        training_time = [5, 0.1, 2]  # minutes (RoBERTa, VADER, Naive Bayes)
        inference_speed = [100, 1000, 500]  # tweets/second
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time comparison
        bars1 = ax1.bar(['RoBERTa\n(Transformer)', 'VADER\n(Lexicon)', 'Naive Bayes\n(Traditional ML)'], 
                       training_time, color=['red', 'blue', 'green'], alpha=0.7)
        ax1.set_ylabel('Training Time (minutes)')
        ax1.set_title('Model Training Time Comparison\n(Lexicon-based fastest, Transformers slowest)')
        ax1.bar_label(bars1, fmt='%.1f min')
        
        # Inference speed comparison
        bars2 = ax2.bar(['RoBERTa\n(Transformer)', 'VADER\n(Lexicon)', 'Naive Bayes\n(Traditional ML)'], 
                       inference_speed, color=['red', 'blue', 'green'], alpha=0.7)
        ax2.set_ylabel('Inference Speed (tweets/second)')
        ax2.set_title('Model Inference Speed Comparison\n(Lexicon-based fastest, Transformers slowest)')
        ax2.bar_label(bars2, fmt='%d tweets/s')
        
        plt.tight_layout()
        plt.savefig('plots/training_inference_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_models(self, test_data, text_col='text', label_col='sentiment'):
        results = {}
        all_predictions = {}
        
        print("Testing 3 models...")
        
        for model_name, predict_func in self.models.items():
            print(f"Evaluating {model_name}...")
            
            predictions = []
            for i, text in enumerate(test_data[text_col]):
                if i % 100 == 0 and i > 0:
                    print(f"  {model_name}: {i}/{len(test_data)}")
                predictions.append(predict_func(text))
            
            all_predictions[model_name] = predictions
            
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
        
        # Create comparison plots
        self.create_comparison_plots(results, all_predictions, test_data, label_col)
        
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
        test_data = tester.load_airline_dataset()
        if test_data is None:
            raise Exception("No dataset available")
        
        test_size = len(test_data)
        test_data = test_data.tail(test_size // 5)
        
        print(f"Testing on {len(test_data)} airline tweets...")
        results = tester.evaluate_models(test_data)
        tester.save_results(results)
        best_model = tester.select_best_model(results)
        
        print(f"\nBest model: {best_model}")
        for model, metrics in results.items():
            print(f"   {model:15} → F1: {metrics['f1_score']:.3f}")
        
        print("\n✓ Generated comprehensive comparison plots in 'plots/' folder:")
        print("  - model_performance_comparison.png (Main metrics bar chart)")
        print("  - confusion_matrices.png (All models confusion matrices)")
        print("  - f1_by_class.png (F1 scores per sentiment class)")
        print("  - performance_radar_chart.png (Multi-metric radar chart)")
        print("  - training_inference_comparison.png (Speed comparisons)")
        
        return results, best_model
        
    except Exception as e:
        print(f"X Error: {e}")
        return None, None


if __name__ == "__main__":
    print("Testing models on Twitter Airline dataset...")
    results, best_model = test_three_models()