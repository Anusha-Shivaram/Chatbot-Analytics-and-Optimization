import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class UserSegmentation:
    def __init__(self, data):
        self.data = data
        self.segments = None
        self.scaler = StandardScaler()
    
    def prepare_features(self):
        user_features = self.data.groupby('user_id').agg({
            'session_id': 'nunique',
            'true_intent': lambda x: x.value_counts().index[0],
            'confidence': 'mean',
            'response_time_ms': 'mean',
            'session_length': 'mean',
            'converted': 'sum',
            'csat_score': 'mean',
            'nps_score': 'mean',
            'fallback_triggered': 'sum',
            'completed': 'mean',
            'channel': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        user_features.columns = ['user_id', 'num_sessions', 'primary_intent', 
                                'avg_confidence', 'avg_response_time', 'avg_session_length',
                                'total_conversions', 'avg_csat', 'avg_nps', 
                                'total_fallbacks', 'completion_rate', 'primary_channel']
        
        return user_features
    
    def perform_clustering(self, n_clusters=4):
        user_features = self.prepare_features()
        
        numerical_features = ['num_sessions', 'avg_confidence', 'avg_response_time',
                            'avg_session_length', 'total_conversions', 'avg_csat',
                            'avg_nps', 'completion_rate']
        
        X = user_features[numerical_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_features['segment'] = kmeans.fit_predict(X_scaled)
        
        self.segments = user_features
        return user_features
    
    def visualize_segments(self, save_path=None):
        if self.segments is None:
            self.perform_clustering()
        
        segment_summary = self.segments.groupby('segment').agg({
            'user_id': 'count',
            'num_sessions': 'mean',
            'avg_confidence': 'mean',
            'total_conversions': 'mean',
            'avg_csat': 'mean',
            'completion_rate': 'mean'
        }).round(2)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].bar(segment_summary.index, segment_summary['user_id'], 
                      color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Users per Segment', fontweight='bold')
        axes[0, 0].set_xlabel('Segment')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].bar(segment_summary.index, segment_summary['num_sessions'], 
                      color='coral', alpha=0.7)
        axes[0, 1].set_title('Avg Sessions per Segment', fontweight='bold')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Sessions')
        
        axes[0, 2].bar(segment_summary.index, segment_summary['avg_confidence'], 
                      color='green', alpha=0.7)
        axes[0, 2].set_title('Avg Confidence per Segment', fontweight='bold')
        axes[0, 2].set_xlabel('Segment')
        axes[0, 2].set_ylabel('Confidence')
        
        axes[1, 0].bar(segment_summary.index, segment_summary['total_conversions'], 
                      color='purple', alpha=0.7)
        axes[1, 0].set_title('Avg Conversions per Segment', fontweight='bold')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Conversions')
        
        axes[1, 1].bar(segment_summary.index, segment_summary['avg_csat'], 
                      color='orange', alpha=0.7)
        axes[1, 1].set_title('Avg CSAT per Segment', fontweight='bold')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('CSAT Score')
        
        axes[1, 2].bar(segment_summary.index, segment_summary['completion_rate'], 
                      color='teal', alpha=0.7)
        axes[1, 2].set_title('Completion Rate per Segment', fontweight='bold')
        axes[1, 2].set_xlabel('Segment')
        axes[1, 2].set_ylabel('Rate')
        
        plt.suptitle('User Segment Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return segment_summary
    
    def segment_profiles(self):
        if self.segments is None:
            self.perform_clustering()
        
        profiles = []
        for segment_id in self.segments['segment'].unique():
            segment_data = self.segments[self.segments['segment'] == segment_id]
            
            profile = {
                'segment_id': segment_id,
                'size': len(segment_data),
                'avg_sessions': segment_data['num_sessions'].mean(),
                'primary_intent': segment_data['primary_intent'].mode()[0] if len(segment_data['primary_intent'].mode()) > 0 else 'unknown',
                'avg_confidence': segment_data['avg_confidence'].mean(),
                'conversion_rate': segment_data['total_conversions'].sum() / len(segment_data),
                'avg_csat': segment_data['avg_csat'].mean(),
                'avg_nps': segment_data['avg_nps'].mean(),
                'completion_rate': segment_data['completion_rate'].mean(),
                'primary_channel': segment_data['primary_channel'].mode()[0] if len(segment_data['primary_channel'].mode()) > 0 else 'unknown'
            }
            profiles.append(profile)
        
        profile_df = pd.DataFrame(profiles)
        return profile_df
    
    def intent_based_segmentation(self):
        intent_segments = self.data.groupby('user_id').agg({
            'true_intent': lambda x: list(x.value_counts().index[:3]),
            'session_id': 'nunique',
            'converted': 'mean',
            'csat_score': 'mean'
        }).reset_index()
        
        intent_segments.columns = ['user_id', 'top_intents', 'num_sessions', 
                                  'conversion_rate', 'avg_csat']
        
        return intent_segments
    
    def channel_preference_analysis(self):
        channel_users = self.data.groupby(['user_id', 'channel']).size().reset_index(name='count')
        channel_pivot = channel_users.pivot(index='user_id', columns='channel', values='count').fillna(0)
        
        dominant_channel = channel_pivot.idxmax(axis=1)
        
        channel_segments = self.data.groupby('user_id').agg({
            'converted': 'mean',
            'csat_score': 'mean',
            'response_time_ms': 'mean'
        }).reset_index()
        
        channel_segments['dominant_channel'] = dominant_channel
        
        return channel_segments
    
    def engagement_scoring(self):
        user_engagement = self.data.groupby('user_id').agg({
            'session_id': 'nunique',
            'session_length': 'mean',
            'completed': 'mean',
            'converted': 'sum'
        }).reset_index()
        
        user_engagement['engagement_score'] = (
            user_engagement['session_id'] * 0.3 +
            user_engagement['session_length'] * 0.2 +
            user_engagement['completed'] * 30 +
            user_engagement['converted'] * 20
        )
        
        user_engagement['engagement_level'] = pd.qcut(
            user_engagement['engagement_score'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return user_engagement
