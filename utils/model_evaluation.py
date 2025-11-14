import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.express as px
import os

def plot_model_comparison(results):
    """Create comparison plots of model performance"""
    metrics_df = pd.DataFrame({
        model: {k: v for k, v in metrics.items() if k != 'detailed_report'} 
        for model, metrics in results.items()
    }).T
    
    # Create bar chart comparing models
    fig = px.bar(
        metrics_df.reset_index().melt(id_vars=['index'], value_vars=['accuracy', 'f1_score', 'precision', 'recall']),
        x='index',
        y='value',
        color='variable',
        barmode='group',
        title='Model Performance Comparison'
    )
    
    os.makedirs('plots/model_performance', exist_ok=True)
    fig.write_html('plots/model_performance/model_comparison.html')
    
    return 'plots/model_performance/model_comparison.html'

def generate_model_report(results):
    """Generate detailed model evaluation report"""
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    
    report = f"""
    # Model Evaluation Report
    
    ## Best Performing Model: {best_model[0]}
    - F1 Score: {best_model[1]['f1_score']:.3f}
    - Accuracy: {best_model[1]['accuracy']:.3f}
    - Precision: {best_model[1]['precision']:.3f}
    - Recall: {best_model[1]['recall']:.3f}
    
    ## All Models Performance:
    """
    
    for model_name, metrics in results.items():
        report += f"""
    ### {model_name}
    - F1 Score: {metrics['f1_score']:.3f}
    - Accuracy: {metrics['accuracy']:.3f}
    - Precision: {metrics['precision']:.3f}
    - Recall: {metrics['recall']:.3f}
    """
    
    # Save report
    os.makedirs('reports/model_evaluation', exist_ok=True)
    with open('reports/model_evaluation/latest_report.md', 'w') as f:
        f.write(report)
    
    return report