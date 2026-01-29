import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceMetrics:
    def __init__(self, data):
        self.data = data
        self.metrics = {}
    
    def calculate_intent_accuracy(self):
        if 'true_intent' not in self.data.columns or 'predicted_intent' not in self.data.columns:
            raise ValueError("Data must contain 'true_intent' and 'predicted_intent' columns")
        
        accuracy = accuracy_score(self.data['true_intent'], self.data['predicted_intent'])
        self.metrics['intent_accuracy'] = accuracy
        return accuracy
    
    def generate_confusion_matrix(self, normalize=None):
        y_true = self.data['true_intent']
        y_pred = self.data['predicted_intent']
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
        
        return cm, labels
    
    def plot_confusion_matrix(self, save_path=None, figsize=(12, 10)):
        cm, labels = self.generate_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Intent Recognition', fontsize=16, fontweight='bold')
        plt.ylabel('True Intent', fontsize=12)
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm, labels
    
    def plot_normalized_confusion_matrix(self, save_path=None, figsize=(12, 10)):
        cm, labels = self.generate_confusion_matrix(normalize='true')
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='RdYlGn', 
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix - Intent Recognition', fontsize=16, fontweight='bold')
        plt.ylabel('True Intent', fontsize=12)
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return cm, labels
    
    def get_classification_report(self):
        y_true = self.data['true_intent']
        y_pred = self.data['predicted_intent']
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        return report_df
    
    def calculate_response_time_metrics(self):
        if 'response_time_ms' not in self.data.columns:
            return None
        
        metrics = {
            'mean_response_time': self.data['response_time_ms'].mean(),
            'median_response_time': self.data['response_time_ms'].median(),
            'std_response_time': self.data['response_time_ms'].std(),
            'min_response_time': self.data['response_time_ms'].min(),
            'max_response_time': self.data['response_time_ms'].max(),
            'p95_response_time': self.data['response_time_ms'].quantile(0.95),
            'p99_response_time': self.data['response_time_ms'].quantile(0.99)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_time_to_first_response(self):
        if 'time_to_first_response_ms' not in self.data.columns:
            return None
        
        metrics = {
            'mean_ttfr': self.data['time_to_first_response_ms'].mean(),
            'median_ttfr': self.data['time_to_first_response_ms'].median(),
            'p95_ttfr': self.data['time_to_first_response_ms'].quantile(0.95)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_completion_rate(self):
        if 'completed' not in self.data.columns:
            return None
        
        completion_rate = self.data['completed'].mean()
        self.metrics['completion_rate'] = completion_rate
        return completion_rate
    
    def calculate_conversion_rate(self):
        if 'converted' not in self.data.columns:
            return None
        
        conversion_rate = self.data['converted'].mean()
        self.metrics['conversion_rate'] = conversion_rate
        return conversion_rate
    
    def calculate_csat(self):
        if 'csat_score' not in self.data.columns:
            return None
        
        csat_metrics = {
            'mean_csat': self.data['csat_score'].mean(),
            'median_csat': self.data['csat_score'].median(),
            'csat_distribution': self.data['csat_score'].value_counts(normalize=True).to_dict(),
            'satisfied_rate': (self.data['csat_score'] >= 4).mean()
        }
        
        self.metrics.update(csat_metrics)
        return csat_metrics
    
    def calculate_nps(self):
        if 'nps_score' not in self.data.columns:
            return None
        
        promoters = (self.data['nps_score'] >= 9).sum()
        detractors = (self.data['nps_score'] <= 6).sum()
        total = len(self.data)
        
        nps = ((promoters - detractors) / total) * 100
        
        nps_metrics = {
            'nps': nps,
            'promoters_pct': (promoters / total) * 100,
            'passives_pct': ((self.data['nps_score'].between(7, 8)).sum() / total) * 100,
            'detractors_pct': (detractors / total) * 100
        }
        
        self.metrics.update(nps_metrics)
        return nps_metrics
    
    def calculate_fallback_rate(self):
        if 'fallback_triggered' not in self.data.columns:
            return None
        
        fallback_rate = self.data['fallback_triggered'].mean()
        self.metrics['fallback_rate'] = fallback_rate
        return fallback_rate
    
    def calculate_all_metrics(self):
        self.calculate_intent_accuracy()
        self.calculate_response_time_metrics()
        self.calculate_time_to_first_response()
        self.calculate_completion_rate()
        self.calculate_conversion_rate()
        self.calculate_csat()
        self.calculate_nps()
        self.calculate_fallback_rate()
        self.calculate_confidence_metrics()
        
        return self.metrics
    
    def get_metrics_summary(self):
        if not self.metrics:
            self.calculate_all_metrics()
        
        summary_df = pd.DataFrame([self.metrics]).T
        summary_df.columns = ['Value']
        return summary_df
    
    def calculate_confidence_metrics(self):
        if 'confidence' not in self.data.columns:
            return None
        
        confidence_metrics = {
            'mean_confidence': self.data['confidence'].mean(),
            'median_confidence': self.data['confidence'].median(),
            'low_confidence_rate': (self.data['confidence'] < 0.6).mean()
        }
        
        self.metrics.update(confidence_metrics)
        return confidence_metrics
