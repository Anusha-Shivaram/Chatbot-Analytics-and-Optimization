import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta

class ChatbotDataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None
        self.rasa_intents = []
        self.rasa_examples = {}
    
    def load_rasa_nlu(self, yml_path='chatbot_logs.yml'):
        """Load intents from Rasa NLU YAML file"""
        try:
            with open(yml_path, 'r', encoding='utf-8') as f:
                rasa_data = yaml.safe_load(f)
            
            if 'nlu' in rasa_data:
                for item in rasa_data['nlu']:
                    intent = item.get('intent')
                    examples = item.get('examples', '')
                    
                    if intent:
                        self.rasa_intents.append(intent)
                        # Parse examples (remove markdown formatting)
                        example_list = [line.strip('- ').split('[')[0].strip() 
                                       for line in examples.split('\n') 
                                       if line.strip() and line.strip().startswith('-')]
                        self.rasa_examples[intent] = example_list
            
            print(f"[OK] Loaded {len(self.rasa_intents)} intents from Rasa chatbot: {', '.join(self.rasa_intents)}")
            return self.rasa_intents
        except Exception as e:
            print(f"Warning: Could not load Rasa data: {e}")
            # Fallback to default intents
            self.rasa_intents = ['greet', 'goodbye', 'bot_challenge', 'query_knowledge_base']
            return self.rasa_intents
    
    def generate_chatbot_logs(self, n_records=2000, use_rasa=True):
        """Generate realistic chatbot conversation logs"""
        np.random.seed(42)
        
        # Load Rasa intents if requested
        if use_rasa and not self.rasa_intents:
            self.load_rasa_nlu()
        
        # Use Rasa intents or fallback
        if self.rasa_intents:
            intents = self.rasa_intents
        else:
            intents = ['greet', 'goodbye', 'affirm', 'deny', 'help', 'inform', 'thankyou', 
                       'request_info', 'complaint', 'feedback', 'booking', 'cancel', 'modify']
        
        channels = ['web', 'mobile', 'whatsapp', 'telegram', 'facebook']
        
        # Generate timestamps (last 30 days)
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(8, 22),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        ) for _ in range(n_records)]
        
        # Generate realistic chatbot interaction data
        data = {
            'timestamp': timestamps,
            'session_id': [f'sess_{i//10:04d}' for i in range(n_records)],
            'user_id': [f'user_{np.random.randint(1, 200):03d}' for _ in range(n_records)],
            'user_message': [],
            'true_intent': np.random.choice(intents, n_records, 
                                           p=self._generate_intent_distribution(len(intents))),
            'predicted_intent': [],
            'confidence': [],
            'channel': np.random.choice(channels, n_records),
            'response_time_ms': np.random.gamma(2, 200, n_records).astype(int),
            'time_to_first_response_ms': np.random.gamma(1.5, 150, n_records).astype(int),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], n_records, p=[0.4, 0.4, 0.2]),
            'csat_score': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.10, 0.20, 0.40, 0.25]),
            'nps_score': np.random.choice(list(range(0, 11)), n_records),
            'completed': np.random.choice([True, False], n_records, p=[0.75, 0.25]),
            'converted': np.random.choice([True, False], n_records, p=[0.30, 0.70]),
            'fallback_triggered': np.random.choice([True, False], n_records, p=[0.12, 0.88]),
        }
        
        # Generate user messages from Rasa examples
        for true_intent in data['true_intent']:
            if true_intent in self.rasa_examples and self.rasa_examples[true_intent]:
                msg = np.random.choice(self.rasa_examples[true_intent])
            else:
                msg = f"User message for {true_intent}"
            data['user_message'].append(msg)
        
        # Generate predicted intents (with some errors for realism)
        for true_intent in data['true_intent']:
            if np.random.random() < 0.85:  # 85% accuracy
                predicted = true_intent
                conf = np.random.uniform(0.75, 0.99)
            else:
                predicted = np.random.choice([i for i in intents if i != true_intent])
                conf = np.random.uniform(0.40, 0.75)
            
            data['predicted_intent'].append(predicted)
            data['confidence'].append(conf)
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['session_length'] = df.groupby('session_id')['session_id'].transform('count')
        df['user_total_interactions'] = df.groupby('user_id')['user_id'].transform('count')
        
        # Calculate revenue for converted users
        df['revenue'] = df['converted'].apply(lambda x: np.random.uniform(20, 150) if x else 0)
        
        # Add confidence category
        df['confidence_category'] = pd.cut(df['confidence'], 
                                           bins=[0, 0.5, 0.7, 0.9, 1.0],
                                           labels=['Low', 'Medium', 'High', 'Very High'])
        
        self.data = df
        print(f"[OK] Generated {n_records} chatbot log records")
        print(f"[OK] Intents: {len(intents)} ({', '.join(intents[:5])}...)")
        print(f"[OK] Users: {df['user_id'].nunique()}, Sessions: {df['session_id'].nunique()}")
        print(f"[OK] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def _generate_intent_distribution(self, n_intents):
        """Generate realistic intent distribution (some intents more common)"""
        probs = np.random.dirichlet(np.ones(n_intents) * 2)
        return probs
    
    def load_data(self, file_path=None):
        """Load data from CSV/JSON/Excel file"""
        if file_path:
            self.file_path = file_path
        
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            with open(self.file_path, 'r') as f:
                json_data = json.load(f)
            self.data = pd.DataFrame(json_data)
        elif self.file_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.file_path)
        
        print(f"[OK] Loaded {len(self.data)} records from {self.file_path}")
        return self.data
    
    def preprocess_data(self, df=None):
        """Preprocess and validate chatbot data"""
        if df is None:
            df = self.data
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle missing values
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].fillna(df['confidence'].median())
        
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].fillna('neutral')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Validate numeric ranges
        if 'csat_score' in df.columns:
            df['csat_score'] = df['csat_score'].clip(1, 5)
        
        if 'nps_score' in df.columns:
            df['nps_score'] = df['nps_score'].clip(0, 10)
        
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].clip(0, 1)
        
        self.data = df
        print(f"[OK] Preprocessed data: {len(df)} records, {len(df.columns)} features")
        return df
    
    def save_data(self, output_path, df=None):
        """Save processed data to file"""
        if df is None:
            df = self.data
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        
        print(f"[OK] Saved data to {output_path}")
        return output_path
