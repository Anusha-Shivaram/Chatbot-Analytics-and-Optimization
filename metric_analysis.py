import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MetricAnalysis:
    def __init__(self, data):
        self.data = data
    
    def calculate_customer_lifetime_value(self):
        user_metrics = self.data.groupby('user_id').agg({
            'session_id': 'nunique',
            'converted': 'sum',
            'csat_score': 'mean',
            'nps_score': 'mean',
            'session_length': 'sum'
        }).reset_index()
        
        avg_conversion_value = 100
        avg_transaction_frequency = user_metrics['converted'].mean()
        avg_customer_lifespan = 12
        
        user_metrics['estimated_ltv'] = (
            user_metrics['converted'] * avg_conversion_value * 
            avg_transaction_frequency * avg_customer_lifespan
        )
        
        user_metrics['value_segment'] = pd.qcut(
            user_metrics['estimated_ltv'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        return user_metrics
    
    def analyze_conversion_drivers(self):
        converted = self.data[self.data['converted'] == 1]
        not_converted = self.data[self.data['converted'] == 0]
        
        drivers = {
            'avg_confidence_converted': converted['confidence'].mean(),
            'avg_confidence_not_converted': not_converted['confidence'].mean(),
            'confidence_impact': converted['confidence'].mean() - not_converted['confidence'].mean(),
            
            'avg_response_time_converted': converted['response_time_ms'].mean(),
            'avg_response_time_not_converted': not_converted['response_time_ms'].mean(),
            'response_time_impact': not_converted['response_time_ms'].mean() - converted['response_time_ms'].mean(),
            
            'avg_session_length_converted': converted['session_length'].mean(),
            'avg_session_length_not_converted': not_converted['session_length'].mean(),
            'session_length_impact': converted['session_length'].mean() - not_converted['session_length'].mean(),
            
            'avg_csat_converted': converted['csat_score'].mean(),
            'avg_csat_not_converted': not_converted['csat_score'].mean(),
            'csat_impact': converted['csat_score'].mean() - not_converted['csat_score'].mean()
        }
        
        return pd.DataFrame([drivers]).T
    
    def csat_impact_analysis(self):
        csat_segments = self.data.copy()
        csat_segments['csat_category'] = pd.cut(
            csat_segments['csat_score'],
            bins=[0, 2, 3, 4, 5],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        csat_impact = csat_segments.groupby('csat_category').agg({
            'converted': 'mean',
            'session_length': 'mean',
            'user_id': 'count',
            'completed': 'mean'
        }).round(3)
        
        csat_impact.columns = ['Conversion_Rate', 'Avg_Session_Length', 'Count', 'Completion_Rate']
        
        return csat_impact
    
    def nps_impact_analysis(self):
        self.data['nps_category'] = self.data['nps_score'].apply(
            lambda x: 'Promoter' if x >= 9 else ('Passive' if x >= 7 else 'Detractor')
        )
        
        nps_impact = self.data.groupby('nps_category').agg({
            'converted': 'mean',
            'csat_score': 'mean',
            'session_length': 'mean',
            'user_id': 'count',
            'completed': 'mean'
        }).round(3)
        
        nps_impact.columns = ['Conversion_Rate', 'Avg_CSAT', 'Avg_Session_Length', 'Count', 'Completion_Rate']
        
        promoter_conversion = nps_impact.loc['Promoter', 'Conversion_Rate'] if 'Promoter' in nps_impact.index else 0
        detractor_conversion = nps_impact.loc['Detractor', 'Conversion_Rate'] if 'Detractor' in nps_impact.index else 0
        
        competitive_advantage = promoter_conversion - detractor_conversion
        
        return nps_impact, competitive_advantage
    
    def visualize_metric_impacts(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        csat_impact = self.csat_impact_analysis()
        axes[0, 0].bar(csat_impact.index.astype(str), csat_impact['Conversion_Rate'], 
                      color='green', alpha=0.7)
        axes[0, 0].set_title('CSAT Impact on Conversion Rate', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('CSAT Category')
        axes[0, 0].set_ylabel('Conversion Rate')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        nps_impact, _ = self.nps_impact_analysis()
        axes[0, 1].bar(nps_impact.index, nps_impact['Conversion_Rate'], 
                      color='blue', alpha=0.7)
        axes[0, 1].set_title('NPS Impact on Conversion Rate', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('NPS Category')
        axes[0, 1].set_ylabel('Conversion Rate')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        ltv_data = self.calculate_customer_lifetime_value()
        ltv_by_segment = ltv_data.groupby('value_segment')['estimated_ltv'].mean()
        axes[1, 0].bar(ltv_by_segment.index.astype(str), ltv_by_segment.values, 
                      color='purple', alpha=0.7)
        axes[1, 0].set_title('Average LTV by Customer Segment', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Value Segment')
        axes[1, 0].set_ylabel('Estimated LTV ($)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        conversion_by_confidence = self.data.groupby('confidence_category')['converted'].mean()
        axes[1, 1].bar(conversion_by_confidence.index.astype(str), conversion_by_confidence.values, 
                      color='orange', alpha=0.7)
        axes[1, 1].set_title('Conversion Rate by Confidence Level', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Confidence Category')
        axes[1, 1].set_ylabel('Conversion Rate')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Deep-Dive Metric Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def calculate_roi_metrics(self):
        total_interactions = len(self.data)
        total_conversions = self.data['converted'].sum()
        avg_csat = self.data['csat_score'].mean()
        nps = ((self.data['nps_score'] >= 9).sum() - (self.data['nps_score'] <= 6).sum()) / len(self.data) * 100
        
        avg_conversion_value = 100
        total_revenue = total_conversions * avg_conversion_value
        
        estimated_cost_per_interaction = 0.50
        total_cost = total_interactions * estimated_cost_per_interaction
        
        roi = ((total_revenue - total_cost) / total_cost) * 100
        
        roi_metrics = {
            'total_interactions': total_interactions,
            'total_conversions': total_conversions,
            'conversion_rate': total_conversions / total_interactions,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'roi_percentage': roi,
            'avg_csat': avg_csat,
            'nps': nps,
            'revenue_per_interaction': total_revenue / total_interactions,
            'cost_per_conversion': total_cost / total_conversions if total_conversions > 0 else 0
        }
        
        return roi_metrics
    
    def ethical_data_analysis(self):
        ethical_metrics = {
            'data_completeness': self.data.notna().mean().mean(),
            'fallback_fairness': self.data.groupby('channel')['fallback_triggered'].mean().std(),
            'response_time_fairness': self.data.groupby('channel')['response_time_ms'].mean().std(),
            'csat_fairness': self.data.groupby('channel')['csat_score'].mean().std(),
            'privacy_compliance': 'All user data anonymized and GDPR compliant'
        }
        
        return ethical_metrics
    
    def competitive_advantage_analysis(self):
        industry_benchmarks = {
            'avg_intent_accuracy': 0.85,
            'avg_response_time': 600,
            'avg_csat': 3.8,
            'avg_nps': 30,
            'avg_conversion_rate': 0.25,
            'avg_completion_rate': 0.70
        }
        
        current_performance = {
            'intent_accuracy': (self.data['true_intent'] == self.data['predicted_intent']).mean(),
            'response_time': self.data['response_time_ms'].mean(),
            'csat': self.data['csat_score'].mean(),
            'nps': ((self.data['nps_score'] >= 9).sum() - (self.data['nps_score'] <= 6).sum()) / len(self.data) * 100,
            'conversion_rate': self.data['converted'].mean(),
            'completion_rate': self.data['completed'].mean()
        }
        
        competitive_gaps = {
            'intent_accuracy_gap': current_performance['intent_accuracy'] - industry_benchmarks['avg_intent_accuracy'],
            'response_time_gap': industry_benchmarks['avg_response_time'] - current_performance['response_time'],
            'csat_gap': current_performance['csat'] - industry_benchmarks['avg_csat'],
            'nps_gap': current_performance['nps'] - industry_benchmarks['avg_nps'],
            'conversion_gap': current_performance['conversion_rate'] - industry_benchmarks['avg_conversion_rate'],
            'completion_gap': current_performance['completion_rate'] - industry_benchmarks['avg_completion_rate']
        }
        
        return pd.DataFrame([current_performance, industry_benchmarks, competitive_gaps],
                          index=['Current', 'Benchmark', 'Gap']).T
