import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import os
import datetime

class ResponseTimeAnomalyDetector:
    """
    Anomaly detector focusing only on response time (time_ms) using Isolation Forest
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
    
    def load_logs(self, file_path):
        """
        Load logs from a JSON file
        
        Args:
            file_path: Path to the JSON file containing logs
            
        Returns:
            List of log entries
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def preprocess_logs(self, logs):
        """
        Transform raw log entries into a feature DataFrame with only time_ms
        
        Args:
            logs: List of log entries
            
        Returns:
            pandas DataFrame with only time_ms feature for anomaly detection
        """
        # Extract only the time_ms feature from each log entry
        features = []
        
        for log in logs:
            # Extract response time and minimal metadata
            feature_dict = {
                # The only feature we care about for detection
                'time_ms': log['response']['time_ms'],
                
                # Metadata (not used for detection, only for reference)
                'timestamp': log['timestamp'],
                'operation_type': log['operation']['type'] if 'operation' in log else 
                                 log.get('search_service', {}).get('type') if 'search_service' in log else
                                 log.get('payment_service', {}).get('operation') if 'payment_service' in log else
                                 log.get('booking_service', {}).get('operation') if 'booking_service' in log else
                                 log.get('feedback_service', {}).get('operation') if 'feedback_service' in log else
                                 'unknown',
                'status_code': log['response']['status_code'],
                'request_path': log['request']['path'],
                'tracing_id': log['tracing']['correlation_id'],
                'is_anomalous_flag': log.get('is_anomalous', False)  # Reference only
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def build_preprocessor(self):
        """
        Build a preprocessing pipeline for the time_ms feature
        
        Returns:
            sklearn StandardScaler
        """
        # Preprocessor for time_ms - just standardize it
        return StandardScaler()
    
    def train(self, features_df):
        """
        Train the anomaly detection model
        
        Args:
            features_df: DataFrame with time_ms feature
            
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
        
        # Select only time_ms feature for training
        features_for_training = features_df[['time_ms']]
        
        # Fit the model
        model.fit(features_for_training)
        
        self.model = model
        return self.model
    
    def predict(self, features_df):
        """
        Predict anomalies based solely on time_ms
        
        Args:
            features_df: DataFrame with time_ms feature
            
        Returns:
            DataFrame with anomaly predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Select only time_ms feature for prediction
        features_for_prediction = features_df[['time_ms']]
        
        # Make predictions
        # Isolation Forest returns: 1 for normal, -1 for anomalies
        predictions = self.model.predict(features_for_prediction)
        
        # Add predictions to the DataFrame
        result_df = features_df.copy()
        result_df['anomaly_score'] = predictions
        result_df['predicted_anomaly'] = result_df['anomaly_score'].apply(lambda x: True if x == -1 else False)
        
        # Calculate statistics for more information
        time_median = result_df['time_ms'].median()
        
        # Calculate how many times higher than median the time is
        result_df['times_median'] = result_df['time_ms'] / time_median
        
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
        Analyze the detected response time anomalies to gain insights
        
        Args:
            result_df: DataFrame with anomaly predictions
            
        Returns:
            Dictionary with analysis results
        """
        anomalies = result_df[result_df['predicted_anomaly'] == True].copy()
        normal = result_df[result_df['predicted_anomaly'] == False].copy()
        
        # Calculate time statistics
        time_stats = {
            'min': result_df['time_ms'].min(),
            'max': result_df['time_ms'].max(),
            'mean': result_df['time_ms'].mean(),
            'median': result_df['time_ms'].median(),
            'std': result_df['time_ms'].std(),
            'p95': result_df['time_ms'].quantile(0.95),
            'p99': result_df['time_ms'].quantile(0.99)
        }
        
        # Calculate anomaly threshold
        if len(anomalies) > 0 and len(normal) > 0:
            # Find the minimum response time that's classified as anomalous
            min_anomaly_time = anomalies['time_ms'].min()
            # Calculate this as a multiple of the median normal response time
            median_normal_time = normal['time_ms'].median()
            threshold_multiple = min_anomaly_time / median_normal_time if median_normal_time > 0 else float('inf')
        else:
            min_anomaly_time = None
            threshold_multiple = None
        
        # Get operation types for anomalies if available
        if 'operation_type' in anomalies.columns:
            operation_types = anomalies['operation_type'].value_counts().to_dict()
        else:
            operation_types = {}
        
        # Prepare analysis results
        analysis = {
            'total_logs': len(result_df),
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(result_df) * 100 if len(result_df) > 0 else 0,
            'response_time': {
                'normal_avg': normal['time_ms'].mean() if len(normal) > 0 else 0,
                'normal_median': normal['time_ms'].median() if len(normal) > 0 else 0,
                'anomalous_avg': anomalies['time_ms'].mean() if len(anomalies) > 0 else 0,
                'anomalous_median': anomalies['time_ms'].median() if len(anomalies) > 0 else 0,
                'min_anomaly_time': min_anomaly_time,
                'threshold_multiple': threshold_multiple
            },
            'time_stats': time_stats,
            'operation_types': operation_types
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Response Time Anomaly Detection')
    parser.add_argument('--input', '-i', required=True, help='Path to the logs JSON file')
    parser.add_argument('--contamination', '-c', type=float, default=0.1, help='Expected proportion of anomalies (0.0-0.5)')
    parser.add_argument('--latest', '-l', action='store_true', help='Only check the latest log entry')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate accuracy against is_anomalous field')
    parser.add_argument('--output', '-o', help='Path to save anomaly detection results (CSV)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ResponseTimeAnomalyDetector(contamination=args.contamination)
    
    # Load logs
    print(f"Loading logs from {args.input}...")
    logs = detector.load_logs(args.input)
    print(f"Loaded {len(logs)} log entries.")
    
    # Run detection
    print("Running response time anomaly detection...")
    result_df = detector.detect_anomalies(logs, return_latest=args.latest)
    
    # Save results if requested
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    
    # Print results
    if args.latest:
        latest_log = result_df.iloc[0]
        print("\n=== Latest Log Analysis ===")
        print(f"Timestamp: {latest_log['timestamp']}")
        print(f"Operation: {latest_log['operation_type']}")
        print(f"Path: {latest_log['request_path']}")
        print(f"Status Code: {latest_log['status_code']}")
        print(f"Response Time: {latest_log['time_ms']} ms")
        print(f"Times Median: {latest_log['times_median']:.2f}x")
        print(f"Predicted as Anomalous: {latest_log['predicted_anomaly']}")
        if 'is_anomalous_flag' in latest_log:
            print(f"Is Anomalous Flag (reference): {latest_log['is_anomalous_flag']}")
    else:
        # Analyze all anomalies
        analysis = detector.analyze_anomalies(result_df)
        
        print("\n=== Response Time Anomaly Detection Results ===")
        print(f"Total logs analyzed: {analysis['total_logs']}")
        print(f"Anomalies detected: {analysis['anomalies_detected']} ({analysis['anomaly_percentage']:.2f}%)")
        print(f"Normal logs: {analysis['total_logs'] - analysis['anomalies_detected']}")
        
        print("\nResponse Time Statistics:")
        print(f"  Min: {analysis['time_stats']['min']:.2f} ms")
        print(f"  Max: {analysis['time_stats']['max']:.2f} ms")
        print(f"  Mean: {analysis['time_stats']['mean']:.2f} ms")
        print(f"  Median: {analysis['time_stats']['median']:.2f} ms")
        print(f"  Std Dev: {analysis['time_stats']['std']:.2f} ms")
        print(f"  95th Percentile: {analysis['time_stats']['p95']:.2f} ms")
        print(f"  99th Percentile: {analysis['time_stats']['p99']:.2f} ms")
        
        print("\nNormal vs Anomalous Response Times:")
        print(f"  Normal avg: {analysis['response_time']['normal_avg']:.2f} ms")
        print(f"  Normal median: {analysis['response_time']['normal_median']:.2f} ms")
        print(f"  Anomalous avg: {analysis['response_time']['anomalous_avg']:.2f} ms")
        print(f"  Anomalous median: {analysis['response_time']['anomalous_median']:.2f} ms")
        
        if analysis['response_time']['threshold_multiple'] is not None:
            print(f"\nApproximate Anomaly Threshold: {analysis['response_time']['threshold_multiple']:.2f}x median response time")
        
        if analysis['operation_types']:
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