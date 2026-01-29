import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

class ExploratoryDataAnalysis:
    def __init__(self, data):
        self.data = data
        self.insights = {}
    
    def get_basic_statistics(self):
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        stats_df = self.data[numerical_cols].describe()
        return stats_df
    
    def analyze_intent_distribution(self, save_path=None):
        if 'true_intent' not in self.data.columns:
            return None
        
        intent_counts = self.data['true_intent'].value_counts()
        intent_pct = self.data['true_intent'].value_counts(normalize=True) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].bar(intent_counts.index, intent_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Intent', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Intent Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        axes[1].bar(intent_pct.index, intent_pct.values, color='coral', alpha=0.7)
        axes[1].set_xlabel('Intent', fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].set_title('Intent Distribution (Percentage)', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.insights['intent_distribution'] = intent_counts.to_dict()
        return intent_counts
    
    def analyze_channel_performance(self, save_path=None):
        if 'channel' not in self.data.columns:
            return None
        
        channel_metrics = self.data.groupby('channel').agg({
            'user_id': 'count',
            'converted': 'mean',
            'csat_score': 'mean',
            'response_time_ms': 'mean',
            'completed': 'mean'
        }).round(3)
        
        channel_metrics.columns = ['Total_Interactions', 'Conversion_Rate', 
                                   'Avg_CSAT', 'Avg_Response_Time', 'Completion_Rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].bar(channel_metrics.index, channel_metrics['Total_Interactions'], 
                       color='purple', alpha=0.7)
        axes[0, 0].set_title('Interactions by Channel', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].bar(channel_metrics.index, channel_metrics['Conversion_Rate'], 
                       color='green', alpha=0.7)
        axes[0, 1].set_title('Conversion Rate by Channel', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Rate')
        
        axes[1, 0].bar(channel_metrics.index, channel_metrics['Avg_CSAT'], 
                       color='orange', alpha=0.7)
        axes[1, 0].set_title('Average CSAT by Channel', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Score')
        
        axes[1, 1].bar(channel_metrics.index, channel_metrics['Avg_Response_Time'], 
                       color='red', alpha=0.7)
        axes[1, 1].set_title('Average Response Time by Channel', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Milliseconds')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.insights['channel_performance'] = channel_metrics.to_dict()
        return channel_metrics
    
    def analyze_temporal_patterns(self, save_path=None):
        if 'timestamp' not in self.data.columns:
            return None
        
        hourly_data = self.data.groupby('hour').agg({
            'user_id': 'count',
            'converted': 'mean',
            'response_time_ms': 'mean'
        }).reset_index()
        
        daily_data = self.data.groupby('day_of_week').agg({
            'user_id': 'count',
            'converted': 'mean',
            'csat_score': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        axes[0].plot(hourly_data['hour'], hourly_data['user_id'], 
                     marker='o', linewidth=2, color='blue')
        axes[0].fill_between(hourly_data['hour'], hourly_data['user_id'], alpha=0.3)
        axes[0].set_xlabel('Hour of Day', fontsize=12)
        axes[0].set_ylabel('Number of Interactions', fontsize=12)
        axes[0].set_title('Hourly Interaction Pattern', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1].bar(daily_data['day_of_week'], daily_data['user_id'], 
                    color='teal', alpha=0.7)
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Number of Interactions', fontsize=12)
        axes[1].set_title('Daily Interaction Pattern', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_names)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return hourly_data, daily_data
    
    def analyze_sentiment_distribution(self, save_path=None):
        if 'sentiment' not in self.data.columns:
            return None
        
        sentiment_counts = self.data['sentiment'].value_counts()
        sentiment_by_intent = pd.crosstab(self.data['true_intent'], 
                                         self.data['sentiment'], 
                                         normalize='index') * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                    autopct='%1.1f%%', startangle=90,
                    colors=[colors.get(x, 'blue') for x in sentiment_counts.index])
        axes[0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
        
        sentiment_by_intent.plot(kind='bar', stacked=True, ax=axes[1], 
                                color=[colors.get(x, 'blue') for x in sentiment_by_intent.columns])
        axes[1].set_xlabel('Intent', fontsize=12)
        axes[1].set_ylabel('Percentage', fontsize=12)
        axes[1].set_title('Sentiment Distribution by Intent', fontsize=14, fontweight='bold')
        axes[1].legend(title='Sentiment')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return sentiment_counts, sentiment_by_intent
    
    def analyze_response_time_distribution(self, save_path=None):
        if 'response_time_ms' not in self.data.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].hist(self.data['response_time_ms'], bins=50, 
                     color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(self.data['response_time_ms'].mean(), 
                       color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(self.data['response_time_ms'].median(), 
                       color='green', linestyle='--', linewidth=2, label='Median')
        axes[0].set_xlabel('Response Time (ms)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        axes[1].boxplot([self.data.groupby('channel')['response_time_ms'].apply(list).values[i] 
                        for i in range(len(self.data['channel'].unique()))],
                       labels=self.data['channel'].unique())
        axes[1].set_xlabel('Channel', fontsize=12)
        axes[1].set_ylabel('Response Time (ms)', fontsize=12)
        axes[1].set_title('Response Time by Channel', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def correlation_analysis(self, save_path=None):
        numerical_cols = ['confidence', 'response_time_ms', 'session_length', 
                         'converted', 'csat_score', 'nps_score', 'completed',
                         'time_to_first_response_ms', 'fallback_triggered']
        
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        corr_matrix = self.data[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f')
        plt.title('Correlation Matrix - Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def funnel_analysis(self, save_path=None):
        total_sessions = self.data['session_id'].nunique()
        completed_sessions = self.data[self.data['completed'] == 1]['session_id'].nunique()
        converted_sessions = self.data[self.data['converted'] == 1]['session_id'].nunique()
        
        funnel_data = {
            'Stage': ['Total Sessions', 'Completed', 'Converted'],
            'Count': [total_sessions, completed_sessions, converted_sessions],
            'Percentage': [100, 
                          (completed_sessions/total_sessions)*100, 
                          (converted_sessions/total_sessions)*100]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#2ecc71', '#f39c12']
        
        for i, row in funnel_df.iterrows():
            width = row['Percentage'] / 100
            ax.barh(i, width, color=colors[i], alpha=0.7, height=0.6)
            ax.text(width/2, i, f"{row['Stage']}\n{row['Count']} ({row['Percentage']:.1f}%)", 
                   ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(funnel_df)-0.5)
        ax.axis('off')
        ax.set_title('Conversion Funnel Analysis', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return funnel_df
    
    def ab_test_simulation(self, metric='converted'):
        if metric not in self.data.columns:
            return None
        
        self.data['variant'] = np.random.choice(['A', 'B'], size=len(self.data))
        
        group_a = self.data[self.data['variant'] == 'A'][metric]
        group_b = self.data[self.data['variant'] == 'B'][metric]
        
        t_stat, p_value = ttest_ind(group_a, group_b)
        
        u_stat, p_value_mw = mannwhitneyu(group_a, group_b)
        
        results = {
            'metric': metric,
            'group_a_mean': group_a.mean(),
            'group_b_mean': group_b.mean(),
            'group_a_std': group_a.std(),
            'group_b_std': group_b.std(),
            't_statistic': t_stat,
            'p_value_ttest': p_value,
            'u_statistic': u_stat,
            'p_value_mannwhitney': p_value_mw,
            'significant': p_value < 0.05
        }
        
        return results
    
    def cohort_analysis(self):
        if 'date' not in self.data.columns:
            return None
        
        self.data['cohort'] = self.data.groupby('user_id')['date'].transform('min')
        self.data['period'] = (pd.to_datetime(self.data['date']) - 
                               pd.to_datetime(self.data['cohort'])).dt.days
        
        cohort_data = self.data.groupby(['cohort', 'period']).agg({
            'user_id': 'nunique',
            'converted': 'mean'
        }).reset_index()
        
        return cohort_data
