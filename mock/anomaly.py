import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
import os
import datetime

class AuthLogAnomalyDetector:
    """
    Anomaly detector for authentication API logs using Isolation Forest
    """
    
    def __init__(self, contamination=0.1):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in the dataset (0-0.5)
        """
        self.contamination = contamination
        self.model = None
        self.preprocessor = None
        self.numerical_features = [
            'time_ms',
            'status_code',
            'operation_success',
            'rate_limit_remaining_pct'
        ]
        
        self.categorical_features = [
            'status_category',
            'operation_type',
            'request_path',
            'request_client_type',
            'environment',
            'region'
        ]
    
    def load_logs(self, file_path):
        """
        Load auth logs from a JSON file
        
        Args:
            file_path: Path to the JSON file containing auth logs
            
        Returns:
            List of log entries
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def preprocess_logs(self, logs):
        """
        Transform raw log entries into a feature DataFrame
        
        Args:
            logs: List of log entries
            
        Returns:
            pandas DataFrame with features for anomaly detection
        """
        # Extract features from each log entry
        features = []
        
        for log in logs:
            # Extract the key features for anomaly detection
            status_code = log['response']['status_code']
            
            # Define status category
            if 200 <= status_code < 300:
                status_category = 'success'
            elif 400 <= status_code < 500:
                status_category = 'client_error'
            elif 500 <= status_code < 600:
                status_category = 'server_error'
            else:
                status_category = 'other'
            
            # Calculate rate limit remaining percentage
            rate_limit = log['security']['rate_limit']
            rate_limit_remaining_pct = rate_limit['remaining'] / rate_limit['limit']
            
            feature_dict = {
                # Numerical features
                'time_ms': log['response']['time_ms'],
                'status_code': status_code,
                'operation_success': 1 if log['operation']['success'] else 0,
                'rate_limit_remaining_pct': rate_limit_remaining_pct,
                
                # Categorical features
                'status_category': status_category,
                'operation_type': log['operation']['type'],
                'request_path': log['request']['path'],
                'request_client_type': log['request']['client_type'],
                'environment': log['auth_service']['environment'],
                'region': log['auth_service']['region'],
                
                # Metadata (not used for detection, only for reference)
                'timestamp': log['timestamp'],
                'tracing_id': log['tracing']['correlation_id'],
                'is_anomalous_flag': log.get('is_anomalous', False)  # Reference only
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def build_preprocessor(self):
        """
        Build a preprocessing pipeline for the features
        
        Returns:
            sklearn ColumnTransformer pipeline
        """
        # Preprocessor for numerical features - standardize them
        numerical_transformer = StandardScaler()
        
        # Preprocessor for categorical features - one-hot encode them
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop non-feature columns
        )
        
        return preprocessor
    
    def train(self, features_df):
        """
        Train the anomaly detection model
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Trained model
        """
        # Build the preprocessing pipeline
        self.preprocessor = self.build_preprocessor()
        
        # Create and train the isolation forest model
        model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('isolation_forest', IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            ))
        ])
        
        # Select only feature columns for training
        features_for_training = features_df[self.numerical_features + self.categorical_features]
        
        # Fit the model
        model.fit(features_for_training)
        
        self.model = model
        return self.model
    
    def predict(self, features_df):
        """
        Predict anomalies in the features
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            DataFrame with anomaly predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Select only feature columns for prediction
        features_for_prediction = features_df[self.numerical_features + self.categorical_features]
        
        # Make predictions
        # Isolation Forest returns: 1 for normal, -1 for anomalies
        predictions = self.model.predict(features_for_prediction)
        
        # Add predictions to the DataFrame
        result_df = features_df.copy()
        result_df['anomaly_score'] = predictions
        result_df['predicted_anomaly'] = result_df['anomaly_score'].apply(lambda x: True if x == -1 else False)
        
        # Apply explicit rule-based checks for extreme anomalies
        # Compute statistics for the time_ms field
        time_mean = result_df['time_ms'].mean()
        time_std = result_df['time_ms'].std()
        time_median = result_df['time_ms'].median()
        
        # Flag extremely high response times (more than 10x median or over 10 seconds)
        result_df.loc[(result_df['time_ms'] > max(10 * time_median, 10000)), 'predicted_anomaly'] = True
        
        # Flag extremely low response times (less than 10% of median and below 10ms)
        result_df.loc[(result_df['time_ms'] < 0.1 * time_median) & (result_df['time_ms'] < 10), 'predicted_anomaly'] = True
        
        # Flag server errors (5xx status codes)
        result_df.loc[result_df['status_code'] >= 500, 'predicted_anomaly'] = True
        
        # Flag client errors with unusual response times
        result_df.loc[(result_df['status_code'] >= 400) & (result_df['status_code'] < 500) & 
                    (result_df['time_ms'] > 3 * time_median), 'predicted_anomaly'] = True
        
        # Update anomaly score for rule-based detections
        result_df.loc[result_df['predicted_anomaly'] == True, 'anomaly_score'] = -1
        
        return result_df
    
    def detect_anomalies(self, logs, return_latest=False):
        """
        End-to-end anomaly detection on logs
        
        Args:
            logs: List of log entries
            return_latest: If True, only return the latest log entry with prediction
            
        Returns:
            DataFrame with anomaly predictions
        """
        # Preprocess logs
        features_df = self.preprocess_logs(logs)
        
        # Train model
        self.train(features_df)
        
        # Make predictions
        result_df = self.predict(features_df)
        
        # Return results
        if return_latest:
            # Convert timestamp to datetime for sorting
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            # Sort by timestamp and get the latest entry
            result_df = result_df.sort_values('timestamp', ascending=False).head(1)
        
        return result_df
    
    def evaluate_accuracy(self, result_df):
        """
        Evaluate model accuracy against the 'is_anomalous_flag' field
        Note: This is for reference only, not for actual detection
        
        Args:
            result_df: DataFrame with predictions and is_anomalous_flag
            
        Returns:
            Dictionary with accuracy metrics
        """
        if 'is_anomalous_flag' not in result_df.columns:
            return None
        
        # Calculate metrics
        total = len(result_df)
        correct = sum(result_df['predicted_anomaly'] == result_df['is_anomalous_flag'])
        accuracy = correct / total
        
        # Calculate precision, recall, and F1 score for anomaly class
        true_pos = sum((result_df['predicted_anomaly'] == True) & (result_df['is_anomalous_flag'] == True))
        false_pos = sum((result_df['predicted_anomaly'] == True) & (result_df['is_anomalous_flag'] == False))
        false_neg = sum((result_df['predicted_anomaly'] == False) & (result_df['is_anomalous_flag'] == True))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def analyze_anomalies(self, result_df):
        """
        Analyze the detected anomalies to gain insights
        
        Args:
            result_df: DataFrame with anomaly predictions
            
        Returns:
            Dictionary with analysis results
        """
        anomalies = result_df[result_df['predicted_anomaly'] == True].copy()
        normal = result_df[result_df['predicted_anomaly'] == False].copy()
        
        # Prepare analysis results
        analysis = {
            'total_logs': len(result_df),
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(result_df) * 100 if len(result_df) > 0 else 0,
            'response_time': {
                'normal_avg': normal['time_ms'].mean() if len(normal) > 0 else 0,
                'anomalous_avg': anomalies['time_ms'].mean() if len(anomalies) > 0 else 0
            },
            'status_codes': anomalies['status_code'].value_counts().to_dict() if len(anomalies) > 0 else {},
            'operation_types': anomalies['operation_type'].value_counts().to_dict() if len(anomalies) > 0 else {}
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Auth Log Anomaly Detection')
    parser.add_argument('--input', '-i', required=True, help='Path to the auth logs JSON file')
    parser.add_argument('--contamination', '-c', type=float, default=0.1, help='Expected proportion of anomalies (0.0-0.5)')
    parser.add_argument('--latest', '-l', action='store_true', help='Only check the latest log entry')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate accuracy against is_anomalous field')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AuthLogAnomalyDetector(contamination=args.contamination)
    
    # Load logs
    print(f"Loading logs from {args.input}...")
    logs = detector.load_logs(args.input)
    print(f"Loaded {len(logs)} log entries.")
    
    # Run detection
    print("Running anomaly detection...")
    result_df = detector.detect_anomalies(logs, return_latest=args.latest)
    
    # Print results
    if args.latest:
        latest_log = result_df.iloc[0]
        print("\n=== Latest Log Analysis ===")
        print(f"Timestamp: {latest_log['timestamp']}")
        print(f"Operation: {latest_log['operation_type']}")
        print(f"Path: {latest_log['request_path']}")
        print(f"Status Code: {latest_log['status_code']}")
        print(f"Response Time: {latest_log['time_ms']} ms")
        print(f"Predicted as Anomalous: {latest_log['predicted_anomaly']}")
        if 'is_anomalous_flag' in latest_log:
            print(f"Is Anomalous Flag (reference): {latest_log['is_anomalous_flag']}")
    else:
        # Analyze all anomalies
        analysis = detector.analyze_anomalies(result_df)
        
        print("\n=== Anomaly Detection Results ===")
        print(f"Total logs analyzed: {analysis['total_logs']}")
        print(f"Anomalies detected: {analysis['anomalies_detected']} ({analysis['anomaly_percentage']:.2f}%)")
        print(f"Normal logs: {analysis['total_logs'] - analysis['anomalies_detected']}")
        
        print("\nResponse Time Analysis:")
        print(f"  Normal logs avg: {analysis['response_time']['normal_avg']:.2f} ms")
        print(f"  Anomalous logs avg: {analysis['response_time']['anomalous_avg']:.2f} ms")
        
        print("\nStatus Code Distribution in Anomalies:")
        for status, count in analysis['status_codes'].items():
            print(f"  HTTP {status}: {count}")
        
        print("\nOperation Types in Anomalies:")
        for op_type, count in analysis['operation_types'].items():
            print(f"  {op_type}: {count}")
    
    # Evaluate accuracy if requested
    if args.evaluate:
        metrics = detector.evaluate_accuracy(result_df)
        if metrics:
            print("\n=== Model Evaluation ===")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
        else:
            print("\nNo 'is_anomalous_flag' field found for evaluation.")

if __name__ == "__main__":
    main()