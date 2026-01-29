import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class OptimizationStrategies:
    def __init__(self, data):
        self.data = data
        self.recommendations = []
    
    def analyze_fallback_patterns(self):
        fallback_data = self.data[self.data['fallback_triggered'] == 1]
        
        fallback_by_intent = fallback_data.groupby('true_intent').size().sort_values(ascending=False)
        fallback_by_confidence = fallback_data.groupby('confidence_category').size()
        fallback_by_channel = fallback_data.groupby('channel').size()
        
        recommendations = {
            'high_fallback_intents': fallback_by_intent.head(5).to_dict(),
            'fallback_confidence_pattern': fallback_by_confidence.to_dict(),
            'fallback_channel_pattern': fallback_by_channel.to_dict()
        }
        
        if len(fallback_by_intent) > 0:
            top_fallback_intent = fallback_by_intent.index[0]
            self.recommendations.append({
                'category': 'Intent Training',
                'priority': 'High',
                'issue': f'High fallback rate for intent: {top_fallback_intent}',
                'recommendation': f'Retrain NLU model with more examples for {top_fallback_intent} intent',
                'expected_impact': 'Reduce fallback rate by 15-20%'
            })
        
        return recommendations
    
    def identify_low_confidence_patterns(self):
        low_conf_data = self.data[self.data['confidence'] < 0.7]
        
        if len(low_conf_data) > 0:
            low_conf_intents = low_conf_data['true_intent'].value_counts().head(5)
            low_conf_channels = low_conf_data['channel'].value_counts()
            
            for intent in low_conf_intents.index[:3]:
                self.recommendations.append({
                    'category': 'Model Confidence',
                    'priority': 'Medium',
                    'issue': f'Low confidence predictions for intent: {intent}',
                    'recommendation': f'Add more training examples and improve feature extraction for {intent}',
                    'expected_impact': 'Improve accuracy by 10-15%'
                })
        
        return low_conf_data
    
    def analyze_response_time_bottlenecks(self):
        slow_responses = self.data[self.data['response_time_ms'] > 
                                   self.data['response_time_ms'].quantile(0.90)]
        
        slow_by_intent = slow_responses.groupby('true_intent')['response_time_ms'].mean().sort_values(ascending=False)
        slow_by_channel = slow_responses.groupby('channel')['response_time_ms'].mean().sort_values(ascending=False)
        
        if len(slow_by_intent) > 0:
            slowest_intent = slow_by_intent.index[0]
            self.recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'issue': f'Slow response time for intent: {slowest_intent} ({slow_by_intent.iloc[0]:.0f}ms)',
                'recommendation': 'Optimize backend processing, add caching, or implement async responses',
                'expected_impact': 'Reduce response time by 30-40%'
            })
        
        return slow_by_intent, slow_by_channel
    
    def conversion_optimization_analysis(self):
        converted_data = self.data[self.data['converted'] == 1]
        not_converted_data = self.data[self.data['converted'] == 0]
        
        conversion_factors = {
            'avg_confidence_converted': converted_data['confidence'].mean(),
            'avg_confidence_not_converted': not_converted_data['confidence'].mean(),
            'avg_session_length_converted': converted_data['session_length'].mean(),
            'avg_session_length_not_converted': not_converted_data['session_length'].mean(),
            'avg_response_time_converted': converted_data['response_time_ms'].mean(),
            'avg_response_time_not_converted': not_converted_data['response_time_ms'].mean()
        }
        
        if conversion_factors['avg_response_time_not_converted'] > conversion_factors['avg_response_time_converted'] * 1.2:
            self.recommendations.append({
                'category': 'Conversion Optimization',
                'priority': 'High',
                'issue': 'Slow response times correlate with lower conversion',
                'recommendation': 'Prioritize response time optimization for conversion-critical intents',
                'expected_impact': 'Increase conversion rate by 5-10%'
            })
        
        return conversion_factors
    
    def csat_improvement_analysis(self):
        low_csat = self.data[self.data['csat_score'] <= 2]
        high_csat = self.data[self.data['csat_score'] >= 4]
        
        if len(low_csat) > 0:
            low_csat_intents = low_csat['true_intent'].value_counts().head(3)
            low_csat_channels = low_csat['channel'].value_counts()
            
            for intent in low_csat_intents.index:
                self.recommendations.append({
                    'category': 'User Satisfaction',
                    'priority': 'Medium',
                    'issue': f'Low CSAT scores for intent: {intent}',
                    'recommendation': f'Review and improve responses for {intent}, add more personalization',
                    'expected_impact': 'Improve CSAT by 0.5-1.0 points'
                })
        
        csat_factors = {
            'low_csat_avg_confidence': low_csat['confidence'].mean() if len(low_csat) > 0 else 0,
            'high_csat_avg_confidence': high_csat['confidence'].mean() if len(high_csat) > 0 else 0,
            'low_csat_avg_response_time': low_csat['response_time_ms'].mean() if len(low_csat) > 0 else 0,
            'high_csat_avg_response_time': high_csat['response_time_ms'].mean() if len(high_csat) > 0 else 0
        }
        
        return csat_factors
    
    def channel_optimization(self):
        channel_performance = self.data.groupby('channel').agg({
            'converted': 'mean',
            'csat_score': 'mean',
            'response_time_ms': 'mean',
            'completed': 'mean',
            'user_id': 'count'
        }).round(3)
        
        best_channel = channel_performance['converted'].idxmax()
        worst_channel = channel_performance['converted'].idxmin()
        
        self.recommendations.append({
            'category': 'Channel Optimization',
            'priority': 'Medium',
            'issue': f'Performance gap between channels (best: {best_channel}, worst: {worst_channel})',
            'recommendation': f'Study {best_channel} implementation and apply learnings to {worst_channel}',
            'expected_impact': 'Improve overall conversion by 3-7%'
        })
        
        return channel_performance
    
    def intent_accuracy_improvement(self):
        from sklearn.metrics import classification_report
        
        y_true = self.data['true_intent']
        y_pred = self.data['predicted_intent']
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        low_performing_intents = []
        for intent, metrics in report.items():
            if intent not in ['accuracy', 'macro avg', 'weighted avg']:
                if metrics['f1-score'] < 0.7:
                    low_performing_intents.append((intent, metrics['f1-score']))
        
        low_performing_intents.sort(key=lambda x: x[1])
        
        for intent, score in low_performing_intents[:3]:
            self.recommendations.append({
                'category': 'Intent Recognition',
                'priority': 'High',
                'issue': f'Low F1-score for intent: {intent} ({score:.2f})',
                'recommendation': f'Collect more training data and add more varied examples for {intent}',
                'expected_impact': 'Improve F1-score by 0.15-0.25'
            })
        
        return low_performing_intents
    
    def personalization_opportunities(self):
        user_segments = self.data.groupby('user_id').agg({
            'true_intent': lambda x: list(x.value_counts().index[:3]),
            'session_length': 'mean',
            'channel': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
            'converted': 'mean'
        }).reset_index()
        
        high_value_users = user_segments[user_segments['converted'] > 0.5]
        
        if len(high_value_users) > 0:
            self.recommendations.append({
                'category': 'Personalization',
                'priority': 'Medium',
                'issue': f'{len(high_value_users)} high-value users identified',
                'recommendation': 'Implement personalized greetings and priority routing for high-value users',
                'expected_impact': 'Increase retention by 10-15% and CSAT by 0.3-0.5 points'
            })
        
        return user_segments
    
    def accessibility_improvements(self):
        accessibility_metrics = {
            'fallback_rate': self.data['fallback_triggered'].mean(),
            'low_confidence_rate': (self.data['confidence'] < 0.6).mean(),
            'slow_response_rate': (self.data['response_time_ms'] > 1000).mean()
        }
        
        if accessibility_metrics['fallback_rate'] > 0.15:
            self.recommendations.append({
                'category': 'Accessibility',
                'priority': 'High',
                'issue': f'High fallback rate: {accessibility_metrics["fallback_rate"]*100:.1f}%',
                'recommendation': 'Add fallback intent handlers, improve error messages, provide suggested actions',
                'expected_impact': 'Reduce fallback rate by 30-40%'
            })
        
        return accessibility_metrics
    
    def generate_optimization_report(self):
        self.analyze_fallback_patterns()
        self.identify_low_confidence_patterns()
        self.analyze_response_time_bottlenecks()
        self.conversion_optimization_analysis()
        self.csat_improvement_analysis()
        self.channel_optimization()
        self.intent_accuracy_improvement()
        self.personalization_opportunities()
        self.accessibility_improvements()
        
        recommendations_df = pd.DataFrame(self.recommendations)
        
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations_df['priority_rank'] = recommendations_df['priority'].map(priority_order)
        recommendations_df = recommendations_df.sort_values('priority_rank').drop('priority_rank', axis=1)
        
        return recommendations_df
    
    def predict_conversion_factors(self):
        features = ['confidence', 'response_time_ms', 'session_length', 
                   'fallback_triggered', 'csat_score']
        
        available_features = [f for f in features if f in self.data.columns]
        
        X = self.data[available_features].fillna(0)
        y = self.data['converted']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def visualize_recommendations(self, save_path=None):
        if not self.recommendations:
            self.generate_optimization_report()
        
        rec_df = pd.DataFrame(self.recommendations)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        category_counts = rec_df['category'].value_counts()
        axes[0].barh(category_counts.index, category_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Number of Recommendations', fontsize=12)
        axes[0].set_title('Optimization Recommendations by Category', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        priority_counts = rec_df['priority'].value_counts()
        colors_priority = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        axes[1].bar(priority_counts.index, priority_counts.values,
                   color=[colors_priority.get(x, 'blue') for x in priority_counts.index],
                   alpha=0.7)
        axes[1].set_ylabel('Number of Recommendations', fontsize=12)
        axes[1].set_title('Optimization Recommendations by Priority', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
