import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter
import re

class NLPAnalysis:
    def __init__(self, data):
        self.data = data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
    
    def sentiment_analysis(self, text_column='user_message'):
        if text_column not in self.data.columns:
            return None
        
        sentiments = []
        for text in self.data[text_column]:
            if pd.notna(text):
                scores = self.sia.polarity_scores(str(text))
                sentiments.append(scores)
            else:
                sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
        
        sentiment_df = pd.DataFrame(sentiments)
        self.data['sentiment_neg'] = sentiment_df['neg']
        self.data['sentiment_neu'] = sentiment_df['neu']
        self.data['sentiment_pos'] = sentiment_df['pos']
        self.data['sentiment_compound'] = sentiment_df['compound']
        
        self.data['sentiment_category'] = self.data['sentiment_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        
        return self.data
    
    def intent_sentiment_analysis(self):
        if 'sentiment_category' not in self.data.columns:
            self.sentiment_analysis()
        
        intent_sentiment = pd.crosstab(
            self.data['true_intent'], 
            self.data['sentiment_category'],
            normalize='index'
        ) * 100
        
        return intent_sentiment
    
    def text_length_analysis(self, text_column='user_message'):
        if text_column not in self.data.columns:
            return None
        
        self.data['message_length'] = self.data[text_column].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        
        self.data['word_count'] = self.data[text_column].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        return self.data[['message_length', 'word_count']].describe()
    
    def extract_keywords(self, text_column='user_message', top_n=20):
        if text_column not in self.data.columns:
            return None
        
        all_text = ' '.join(self.data[text_column].dropna().astype(str))
        words = re.findall(r'\w+', all_text.lower())
        
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'have', 'has', 'had'])
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_freq = Counter(filtered_words)
        top_keywords = word_freq.most_common(top_n)
        
        return pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
    
    def intent_keyword_extraction(self, text_column='user_message', top_n=10):
        if text_column not in self.data.columns or 'true_intent' not in self.data.columns:
            return None
        
        intent_keywords = {}
        
        for intent in self.data['true_intent'].unique():
            intent_data = self.data[self.data['true_intent'] == intent]
            intent_text = ' '.join(intent_data[text_column].dropna().astype(str))
            words = re.findall(r'\w+', intent_text.lower())
            
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on'])
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            word_freq = Counter(filtered_words)
            intent_keywords[intent] = word_freq.most_common(top_n)
        
        return intent_keywords
    
    def confidence_sentiment_correlation(self):
        if 'sentiment_compound' not in self.data.columns:
            self.sentiment_analysis()
        
        if 'confidence' in self.data.columns:
            correlation = self.data[['confidence', 'sentiment_compound']].corr()
            return correlation
        return None
    
    def response_quality_analysis(self):
        self.data['high_confidence'] = (self.data['confidence'] >= 0.8).astype(int)
        self.data['positive_sentiment'] = (self.data['sentiment_compound'] >= 0.05).astype(int)
        self.data['quick_response'] = (self.data['response_time_ms'] <= 500).astype(int)
        
        self.data['quality_score'] = (
            self.data['high_confidence'] * 0.4 +
            self.data['positive_sentiment'] * 0.3 +
            self.data['quick_response'] * 0.3
        )
        
        return self.data['quality_score'].describe()
