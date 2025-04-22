import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import datetime
import glob
from dateutil import parser

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
    
    def detect_anomalies(self, logs):
        """
        End-to-end anomaly detection on logs
        
        Args:
            logs: List of log enties
            
        Returns:
            DataFrame with anomaly predictions
        """
        # Preprocess logs
        features_df = self.preprocess_logs(logs)
        
        # Train model
        self.train(features_df)
        
        # Make predictions
        result_df = self.predict(features_df)
        
        return result_df

def add_response_anomaly_to_logs(input_file, result_df, output_folder="combined_anomalous"):
    """
    Creates a copy of the log file with response_anomaly flags added in NDJSON format
    (one log entry per line with empty lines between entries)
    
    Args:
        input_file: Path to the original log file
        result_df: DataFrame with anomaly detection results
        output_folder: Path to the folder where output files will be stored
        
    Returns:
        Path to the created/updated file
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get just the filename without path
    file_name = os.path.basename(input_file)
    name, ext = os.path.splitext(file_name)
    output_file = os.path.join(output_folder, f"{name}_with_anomalies{ext}")
    
    # Load the original logs
    with open(input_file, 'r') as f:
        content = f.read()
        # Handle both array-formatted and newline-delimited JSON
        if content.strip().startswith('[') and content.strip().endswith(']'):
            # Array-formatted JSON
            logs = json.loads(content)
        else:
            # Newline-delimited JSON or other format
            logs = []
            for line in content.strip().split('\n\n'):
                if line.strip():
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    
    # Create a dictionary from result_df for fast lookup
    anomaly_map = {}
    for _, row in result_df.iterrows():
        timestamp = row['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.isoformat()
        anomaly_map[timestamp] = row['predicted_anomaly']
    
    # Track statistics for discrepancies
    total_logs = len(logs)
    response_anomaly_count = 0
    original_anomaly_count = 0
    discrepancy_count = 0
    
    # Update each log entry with anomaly information
    updated_logs = []
    for log in logs:
        timestamp = log.get('timestamp')
        original_anomalous = log.get('is_anomalous', False)
        
        # Count original anomalies
        if original_anomalous:
            original_anomaly_count += 1
        
        # Add the anomaly flag
        log_copy = log.copy()
        response_anomaly = False
        
        if timestamp in anomaly_map:
            response_anomaly = bool(anomaly_map[timestamp])
            log_copy['response_anomaly'] = response_anomaly
        else:
            # If not found in results, assume not anomalous
            log_copy['response_anomaly'] = False
        
        # Count response anomalies and discrepancies
        if response_anomaly:
            response_anomaly_count += 1
            
        if original_anomalous != response_anomaly:
            discrepancy_count += 1
        
        updated_logs.append(log_copy)
    
    # Write the updated logs to the output file in NDJSON format
    # (one log per line with an empty line between entries)
    with open(output_file, 'w') as f:
        for i, log in enumerate(updated_logs):
            json_line = json.dumps(log)
            f.write(json_line)
            # Add empty line between entries (but not after the last one)
            if i < len(updated_logs) - 1:
                f.write('\n\n')
    
    # Print analytics about discrepancies
    print(f"\nAnalytics for {file_name}:")
    print(f"  Total logs: {total_logs}")
    print(f"  Original anomalies: {original_anomaly_count} ({original_anomaly_count/total_logs*100:.2f}%)")
    print(f"  Response time anomalies: {response_anomaly_count} ({response_anomaly_count/total_logs*100:.2f}%)")
    print(f"  Discrepancies: {discrepancy_count} ({discrepancy_count/total_logs*100:.2f}%)")
    
    if discrepancy_count > 0:
        print(f"  Note: Some logs marked as anomalous in the original data don't have anomalous response times.")
    
    return output_file

def analyze_log_file(input_file, output_folder, contamination=0.1):
    
    print(f"\nProcessing {input_file}...")
    
    # Check if file exists and has content
    if not os.path.exists(input_file):
        print(f"  Error: File not found: {input_file}")
        return None
    
    # Initialize detector
    detector = ResponseTimeAnomalyDetector(contamination=contamination)
    
    try:
        # Load logs
        logs = detector.load_logs(input_file)
        print(f"  Loaded {len(logs)} log entries.")
        
        if not logs:
            print(f"  Warning: No logs found in {input_file}")
            return None
        
        # Run detection
        print(f"  Running response time anomaly detection...")
        result_df = detector.detect_anomalies(logs)
        
        # Save the logs with added anomaly information
        output_file = add_response_anomaly_to_logs(input_file, result_df, output_folder)
        
        # Get some statistics
        anomalies = result_df[result_df['predicted_anomaly'] == True]
        normal = result_df[result_df['predicted_anomaly'] == False]
        
        print(f"  Total logs analyzed: {len(result_df)}")
        print(f"  Anomalies detected: {len(anomalies)} ({len(anomalies)/len(result_df)*100:.2f}%)")
        print(f"  Normal logs: {len(normal)}")
        print(f"  Normal avg time: {normal['time_ms'].mean():.2f} ms")
        print(f"  Anomalous avg time: {anomalies['time_ms'].mean():.2f} ms")
        print(f"  Results saved to: {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"  Error processing {input_file}: {str(e)}")
        return None

def main():
    # Define the log file patterns to process
    output_folder = "combine_logs"
    log_files = [
        "auth_interactions.json",
        "booking_interactions.json",
        "feedback_interactions.json",
        "payment_interactions.json",
        "search_interactions.json"
    ]
    
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process each log file
    print("Starting automated response time anomaly detection...")
    
    files_processed = 0
    
    # First try with full paths
    for log_file in log_files:
        full_path = os.path.join(current_dir, log_file)
        if os.path.exists(full_path):
            analyze_log_file(full_path, output_folder)
            files_processed += 1
    
    # If using glob to search subdirectories
    if files_processed == 0:
        for log_file in log_files:
            pattern = os.path.join(current_dir, "**", log_file)
            matches = glob.glob(pattern, recursive=True)
            
            for match in matches:
                analyze_log_file(match, output_folder)
                files_processed += 1
    
    if files_processed == 0:
        print("\nNo log files found. Please make sure the log files exist in the current directory or its subdirectories.")
    else:
        print(f"\nProcessed {files_processed} log files successfully.")

if __name__ == "__main__":
    main()