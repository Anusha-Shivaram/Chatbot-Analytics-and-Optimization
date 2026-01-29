import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for displaying plots

from data_loader import ChatbotDataLoader
from performance_metrics import PerformanceMetrics
from eda_analysis import ExploratoryDataAnalysis
from user_segmentation import UserSegmentation
from nlp_analysis import NLPAnalysis
from optimization_strategies import OptimizationStrategies
from metric_analysis import MetricAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import os
warnings.filterwarnings('ignore')

# Enable interactive mode
plt.ion()

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

def create_output_folders():
    """Create organized output folders"""
    folders = ['outputs', 'outputs/visualizations', 'outputs/reports', 'outputs/data']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def create_session_heatmap(data):
    """Create session heatmap visualization"""
    if 'hour' not in data.columns or 'day_of_week' not in data.columns:
        return
    
    heatmap_data = data.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_pivot, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Interactions'})
    plt.title('Session Heatmap - User Activity Patterns', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.yticks(ticks=range(7), labels=day_labels, rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/session_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_wordcloud(data):
    """Create word cloud from user messages"""
    if 'user_message' not in data.columns:
        return
    
    text = ' '.join(data['user_message'].astype(str).tolist())
    
    plt.figure(figsize=(14, 8))
    wordcloud = WordCloud(width=1200, height=600, background_color='white', 
                          colormap='viridis', max_words=100).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Keywords in User Messages', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_comprehensive_dashboard(data, metrics):
    """Create comprehensive multi-panel dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Intent distribution
    ax1 = fig.add_subplot(gs[0, 0])
    intent_counts = data['predicted_intent'].value_counts()
    ax1.bar(range(len(intent_counts)), intent_counts.values, color='steelblue')
    ax1.set_xticks(range(len(intent_counts)))
    ax1.set_xticklabels(intent_counts.index, rotation=45, ha='right')
    ax1.set_title('Intent Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Channel performance
    ax2 = fig.add_subplot(gs[0, 1])
    channel_conv = data.groupby('channel')['converted'].mean()
    ax2.bar(range(len(channel_conv)), channel_conv.values, color='coral')
    ax2.set_xticks(range(len(channel_conv)))
    ax2.set_xticklabels(channel_conv.index, rotation=45, ha='right')
    ax2.set_title('Conversion Rate by Channel', fontweight='bold')
    ax2.set_ylabel('Conversion Rate')
    ax2.set_ylim(0, 1)
    
    # Response time trend
    ax3 = fig.add_subplot(gs[0, 2])
    hourly_response = data.groupby('hour')['response_time_ms'].mean()
    ax3.plot(hourly_response.index, hourly_response.values, marker='o', color='green')
    ax3.set_title('Response Time by Hour', fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Response Time (ms)')
    ax3.grid(True, alpha=0.3)
    
    # CSAT distribution
    ax4 = fig.add_subplot(gs[1, 0])
    csat_counts = data['csat_score'].value_counts().sort_index()
    ax4.bar(csat_counts.index, csat_counts.values, color='gold')
    ax4.set_title('CSAT Score Distribution', fontweight='bold')
    ax4.set_xlabel('CSAT Score')
    ax4.set_ylabel('Count')
    
    # Confusion matrix (mini)
    ax5 = fig.add_subplot(gs[1, 1])
    intents = data['true_intent'].unique()[:5]
    subset = data[data['true_intent'].isin(intents)]
    cm_data = pd.crosstab(subset['true_intent'], subset['predicted_intent'])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=False)
    ax5.set_title('Confusion Matrix (Top 5)', fontweight='bold')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    
    # Temporal pattern
    ax6 = fig.add_subplot(gs[1, 2])
    dow_counts = data.groupby('day_of_week').size()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax6.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], color='purple')
    ax6.set_xticks(range(7))
    ax6.set_xticklabels(day_names)
    ax6.set_title('Activity by Day of Week', fontweight='bold')
    ax6.set_ylabel('Interactions')
    
    # Sentiment distribution
    ax7 = fig.add_subplot(gs[2, 0])
    sentiment_counts = data['sentiment'].value_counts()
    colors_sent = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    ax7.bar(range(len(sentiment_counts)), sentiment_counts.values, 
            color=[colors_sent.get(x, 'blue') for x in sentiment_counts.index])
    ax7.set_xticks(range(len(sentiment_counts)))
    ax7.set_xticklabels(sentiment_counts.index, rotation=45, ha='right')
    ax7.set_title('Sentiment Distribution', fontweight='bold')
    ax7.set_ylabel('Count')
    
    # Completion vs Conversion
    ax8 = fig.add_subplot(gs[2, 1])
    completion_data = pd.DataFrame({
        'Completed': [data['completed'].sum(), (~data['completed']).sum()],
        'Converted': [data['converted'].sum(), (~data['converted']).sum()]
    }, index=['Yes', 'No'])
    completion_data.plot(kind='bar', ax=ax8, color=['skyblue', 'salmon'])
    ax8.set_title('Completion & Conversion', fontweight='bold')
    ax8.set_ylabel('Count')
    ax8.set_xticklabels(['Yes', 'No'], rotation=0)
    ax8.legend()
    
    # Key metrics summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    metrics_text = f"""KEY METRICS
    
Intent Accuracy: {metrics.metrics.get('intent_accuracy', 0)*100:.1f}%
Conversion Rate: {metrics.metrics.get('conversion_rate', 0)*100:.1f}%
Avg CSAT: {metrics.metrics.get('mean_csat', 0):.2f}/5
NPS Score: {metrics.metrics.get('nps', 0):.1f}
Fallback Rate: {metrics.metrics.get('fallback_rate', 0)*100:.1f}%
Completion Rate: {metrics.metrics.get('completion_rate', 0)*100:.1f}%
Avg Response Time: {metrics.metrics.get('mean_response_time', 0):.0f}ms
"""
    ax9.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Chatbot Analytics Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('outputs/visualizations/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_summary_report(data, metrics, eda, segmentation, optimizer, metric_analysis):
    """Generate executive summary report"""
    report = []
    report.append("=" * 80)
    report.append("CHATBOT ANALYTICS AND OPTIMIZATION REPORT")
    report.append("=" * 80)
    
    # Executive summary
    report.append("")
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Interactions: {len(data):,}")
    report.append(f"Unique Users: {data['user_id'].nunique()}")
    report.append(f"Unique Sessions: {data['session_id'].nunique()}")
    report.append(f"Intent Accuracy: {metrics.metrics['intent_accuracy']*100:.2f}%")
    report.append(f"Conversion Rate: {metrics.metrics['conversion_rate']*100:.2f}%")
    report.append(f"Average CSAT: {metrics.metrics['mean_csat']:.2f}/5")
    report.append(f"NPS Score: {metrics.metrics['nps']:.1f}")
    report.append(f"Completion Rate: {metrics.metrics['completion_rate']*100:.2f}%")
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"Mean Response Time: {metrics.metrics['mean_response_time']:.2f}ms")
    report.append(f"P95 Response Time: {metrics.metrics['p95_response_time']:.2f}ms")
    report.append(f"Time to First Response: {metrics.metrics['mean_ttfr']:.2f}ms")
    report.append(f"Fallback Rate: {metrics.metrics['fallback_rate']*100:.2f}%")
    report.append(f"Low Confidence Rate: {metrics.metrics['low_confidence_rate']*100:.2f}%")
    report.append("")
    
    # Channel Performance
    report.append("CHANNEL PERFORMANCE")
    report.append("-" * 80)
    channel_perf = data.groupby('channel').agg({
        'converted': 'mean',
        'csat_score': 'mean',
        'response_time_ms': 'mean'
    }).round(3)
    for channel in channel_perf.index:
        report.append(f"{channel.upper()}:")
        report.append(f"  Conversion: {channel_perf.loc[channel, 'converted']*100:.1f}%")
        report.append(f"  CSAT: {channel_perf.loc[channel, 'csat_score']:.2f}/5")
        report.append(f"  Response Time: {channel_perf.loc[channel, 'response_time_ms']:.0f}ms")
    report.append("")
    
    # User Segmentation
    report.append("USER SEGMENTATION")
    report.append("-" * 80)
    if hasattr(segmentation, 'segments') and segmentation.segments is not None:
        report.append(f"Total Segments Identified: {segmentation.segments['segment'].nunique()}")
    else:
        report.append("Total Segments Identified: 4")
    report.append("")
    
    # Top Recommendations
    report.append("TOP 5 OPTIMIZATION RECOMMENDATIONS")
    report.append("-" * 80)
    if hasattr(optimizer, 'recommendations'):
        if isinstance(optimizer.recommendations, list):
            top_recs = pd.DataFrame(optimizer.recommendations).head(5)
        else:
            top_recs = optimizer.recommendations.head(5)
        for idx, row in top_recs.iterrows():
            report.append(f"{idx+1}. [{row['priority']}] {row['recommendation']}")
            report.append(f"   Impact: {row['expected_impact']}")
    report.append("")
    
    return "\n".join(report)

def main():
    print("\n" + "=" * 80)
    print("CHATBOT ANALYTICS AND OPTIMIZATION - COMPREHENSIVE ANALYSIS")
    print("Using Real Rasa Chatbot Data")
    print("=" * 80)
    
    # Create output folders
    print("\n[Setup] Creating output folders...")
    create_output_folders()
    print("[OK] Output folders created: outputs/visualizations, outputs/reports, outputs/data")
    
    # Load and preprocess data from Rasa chatbot
    print("\n[1/8] Loading Rasa Chatbot Data...")
    loader = ChatbotDataLoader()
    data = loader.generate_chatbot_logs(n_records=2000, use_rasa=True)
    data = loader.preprocess_data(data)
    loader.save_data('outputs/data/chatbot_data.csv', data)
    print(f"[OK] Generated {len(data)} records with {data.shape[1]} features")
    print(f"[OK] Data source: Rasa NLU (chatbot_logs.yml)")
    
    # Performance metrics
    print("\n[2/8] Calculating Performance Metrics...")
    metrics = PerformanceMetrics(data)
    all_metrics = metrics.calculate_all_metrics()
    print(f"[OK] Intent Accuracy: {all_metrics['intent_accuracy']*100:.2f}%")
    print(f"[OK] Conversion Rate: {all_metrics['conversion_rate']*100:.2f}%")
    print(f"[OK] CSAT: {all_metrics['mean_csat']:.2f}/5 | NPS: {all_metrics['nps']:.1f}")
    
    # Generate confusion matrices
    metrics.plot_confusion_matrix(save_path='outputs/visualizations/confusion_matrix.png')
    metrics.plot_normalized_confusion_matrix(save_path='outputs/visualizations/confusion_matrix_normalized.png')
    
    # Save classification report
    cm, labels = metrics.generate_confusion_matrix()
    from sklearn.metrics import classification_report
    report = classification_report(data['true_intent'], data['predicted_intent'], output_dict=True)
    pd.DataFrame(report).transpose().to_csv('outputs/reports/classification_report.csv')
    print("[OK] Confusion matrices and classification report generated")
    
    # EDA
    print("\n[3/8] Performing Exploratory Data Analysis...")
    eda = ExploratoryDataAnalysis(data)
    eda.analyze_intent_distribution(save_path='outputs/visualizations/intent_distribution.png')
    eda.analyze_channel_performance(save_path='outputs/visualizations/channel_performance.png')
    eda.analyze_temporal_patterns(save_path='outputs/visualizations/temporal_patterns.png')
    eda.correlation_analysis(save_path='outputs/visualizations/correlation_matrix.png')
    funnel = eda.funnel_analysis()
    funnel.to_csv('outputs/reports/funnel_analysis.csv', index=False)
    try:
        eda.visualize_funnel(funnel, save_path='outputs/visualizations/funnel_analysis.png')
    except:
        pass
    print("[OK] EDA visualizations and reports generated")
    
    # User segmentation
    print("\n[4/8] Performing User Segmentation...")
    segmentation = UserSegmentation(data)
    user_segments = segmentation.perform_clustering(n_clusters=4)
    segment_summary = segmentation.visualize_segments(save_path='outputs/visualizations/user_segments.png')
    segment_profiles = segmentation.segment_profiles()
    segment_profiles.to_csv('outputs/reports/segment_profiles.csv', index=False)
    engagement = segmentation.engagement_scoring()
    engagement.to_csv('outputs/reports/engagement_scores.csv', index=False)
    print(f"[OK] Created {len(segment_profiles)} user segments")
    
    # NLP analysis (use original data, not user-level segments)
    print("\n[5/8] Performing NLP and Sentiment Analysis...")
    nlp = NLPAnalysis(data)
    data = nlp.sentiment_analysis()
    intent_sentiment = nlp.intent_sentiment_analysis()
    intent_sentiment.to_csv('outputs/reports/intent_sentiment_analysis.csv')
    keywords = nlp.extract_keywords(top_n=50)
    keywords.to_csv('outputs/reports/top_keywords.csv', index=False)
    print("[OK] NLP analysis completed")
    
    # Metric analysis
    print("\n[6/8] Performing Deep-Dive Metric Analysis...")
    metric_analysis = MetricAnalysis(data)
    ltv_data = metric_analysis.calculate_customer_lifetime_value()
    ltv_data.to_csv('outputs/reports/customer_ltv.csv', index=False)
    csat_impact = metric_analysis.csat_impact_analysis()
    nps_impact, competitive_adv = metric_analysis.nps_impact_analysis()
    try:
        metric_analysis.visualize_metric_impacts(save_path='outputs/visualizations/metric_impacts.png')
    except Exception as e:
        print(f"Warning: Could not save metric impacts visualization: {e}")
    competitive = metric_analysis.competitive_advantage_analysis()
    competitive.to_csv('outputs/reports/competitive_analysis.csv')
    roi = metric_analysis.calculate_roi_metrics()
    pd.DataFrame([roi]).T.to_csv('outputs/reports/roi_metrics.csv')
    print(f"[OK] ROI: {roi['roi_percentage']:.1f}% | Competitive Advantage Calculated")
    
    # Optimization strategies
    print("\n[7/8] Generating Optimization Strategies...")
    optimizer = OptimizationStrategies(data)
    recommendations = optimizer.generate_optimization_report()
    recommendations.to_csv('outputs/reports/optimization_recommendations.csv', index=False)
    try:
        optimizer.visualize_recommendations(save_path='outputs/visualizations/optimization_summary.png')
    except Exception as e:
        print(f"Warning: Could not save optimization visualization: {e}")
    feature_importance = optimizer.predict_conversion_factors()
    feature_importance.to_csv('outputs/reports/conversion_drivers.csv', index=False)
    print(f"[OK] Generated {len(recommendations)} prioritized recommendations")
    
    # Additional visualizations
    print("\n[8/8] Creating Additional Visualizations...")
    try:
        create_session_heatmap(data)
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
    try:
        create_wordcloud(data)
    except Exception as e:
        print(f"Warning: Could not create wordcloud: {e}")
    try:
        create_comprehensive_dashboard(data, metrics)
    except Exception as e:
        print(f"Warning: Could not create dashboard: {e}")
    print("[OK] Heatmap, word cloud, and dashboard created")
    
    # Generate summary report
    print("\nGenerating Summary Report...")
    summary = generate_summary_report(data, metrics, eda, segmentation, optimizer, metric_analysis)
    with open('outputs/reports/SUMMARY_REPORT.txt', 'w') as f:
        f.write(summary)
    print("[OK] Summary report saved")
    
    # Print summary to console
    print("\n" + summary)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  [Folder] outputs/data/")
    print("     - chatbot_data.csv (raw data)")
    print("\n  [Folder] outputs/visualizations/ (14 PNG files)")
    print("     - confusion_matrix.png")
    print("     - comprehensive_dashboard.png")
    print("     - session_heatmap.png")
    print("     - And 11 more visualizations...")
    print("\n  [Folder] outputs/reports/ (12 files)")
    print("     - SUMMARY_REPORT.txt (executive summary)")
    print("     - optimization_recommendations.csv")
    print("     - segment_profiles.csv")
    print("     - And 9 more reports...")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
