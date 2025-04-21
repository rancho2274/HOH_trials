import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
from datetime import datetime
import os
import time
import sys
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class LogFileHandler(FileSystemEventHandler):
    """Handler for log file changes."""
    
    def __init__(self, file_path, anomaly_detector):
        self.file_path = os.path.abspath(file_path)
        self.anomaly_detector = anomaly_detector
        self.last_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.file_path:
            current_size = os.path.getsize(self.file_path)
            if current_size > self.last_size:  # File has grown
                self.anomaly_detector.check_new_logs()
                self.last_size = current_size


class APIAnomalyDetector:
    """Real-time anomaly detector for API logs."""
    
    def __init__(self, file_path, contamination=0.1, retrain_interval=20, min_logs=10):
        self.file_path = file_path
        self.contamination = contamination
        self.retrain_interval = retrain_interval  # Retrain model after this many new logs
        self.min_logs = min_logs  # Minimum logs required for initial training
        self.model = None
        self.pipeline = None
        self.numerical_features = []
        self.categorical_features = []
        self.logs_processed = 0
        self.new_logs_since_retrain = 0
        self.last_position = 0
        
        # Initialize log storage
        self.all_logs = []
        self.log_dataframe = None
        
        # Load initial logs if available
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            self.load_all_logs()
    
    def load_all_logs(self):
        """Load all logs from the file."""
        try:
            with open(self.file_path, 'r') as f:
                self.all_logs = json.load(f)
            
            self.logs_processed = len(self.all_logs)
            print(f"Loaded {self.logs_processed} logs from {self.file_path}")
            
            # Update last position
            self.last_position = os.path.getsize(self.file_path)
            
            # If we have enough logs, train the initial model
            if self.logs_processed >= self.min_logs:
                self.train_model()
        except Exception as e:
            print(f"Error loading logs: {e}")
            self.all_logs = []
    
    def load_new_logs(self):
        """Load only new logs appended since last check."""
        new_logs = []
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
                if content.strip():  # Ensure content is not empty
                    all_logs = json.loads(content)
                    if len(all_logs) > self.logs_processed:
                        new_logs = all_logs[self.logs_processed:]
                        self.all_logs.extend(new_logs)
                        self.logs_processed = len(self.all_logs)
                        self.new_logs_since_retrain += len(new_logs)
            return new_logs
        except json.JSONDecodeError:
            print("Warning: File appears to be incomplete or malformed. Waiting for complete write.")
            return []
        except Exception as e:
            print(f"Error loading new logs: {e}")
            return []
    
    def flatten_json(self, nested_json, prefix=''):
        """Flatten a nested JSON structure."""
        flattened = {}
        for key, value in nested_json.items():
            if isinstance(value, dict) and key not in ['request_headers', 'request_body', 'response']:
                flattened.update(self.flatten_json(value, f"{prefix}{key}_"))
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened
    
    def extract_features(self, logs):
        """Extract and prepare features from log entries."""
        # Flatten each log entry
        flat_logs = [self.flatten_json(log) for log in logs]
        
        # Convert to DataFrame
        df = pd.DataFrame(flat_logs)
        
        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Extract response time in milliseconds 
        if 'response_time_ms' in df.columns:
            df['response_time_ms'] = pd.to_numeric(df['response_time_ms'], errors='coerce')
        elif 'response_body_time_ms' in df.columns:
            df['response_time_ms'] = pd.to_numeric(df['response_body_time_ms'], errors='coerce')
        
        # Extract status code
        if 'response_status_code' in df.columns:
            df['status_code'] = df['response_status_code']
        
        # Create status code category
        if 'status_code' in df.columns:
            df['status_category'] = df['status_code'].apply(
                lambda x: 'success' if 200 <= x < 300 else
                        'client_error' if 400 <= x < 500 else
                        'server_error' if 500 <= x < 600 else 'other'
            )
        
        # Identify the service type from filename
        service_name = os.path.basename(self.file_path).split('_')[0]
        
        # Select features for anomaly detection
        features = []
        
        # Numerical features
        self.numerical_features = []
        for feature in ['response_time_ms', 'hour', 'minute', 'day_of_week']:
            if feature in df.columns:
                self.numerical_features.append(feature)
                features.append(feature)
        
        # Categorical features
        self.categorical_features = []
        service_specific_features = {
            'auth': ['auth_service_type', 'operation_type', 'request_method', 'status_category'],
            'search': ['search_service_type', 'request_method', 'status_category'],
            'booking': ['booking_service_type', 'booking_service_operation', 'request_method', 'status_category'],
            'payment': ['payment_service_operation', 'request_method', 'status_category'],
            'feedback': ['feedback_service_operation', 'request_method', 'status_category']
        }
        
        for feature in service_specific_features.get(service_name, []):
            if feature in df.columns:
                self.categorical_features.append(feature)
                features.append(feature)
        
        # Create a copy with selected features
        features_df = df[features].copy()
        
        # Fill missing values
        for feature in self.numerical_features:
            features_df[feature] = features_df[feature].fillna(features_df[feature].median())
        
        for feature in self.categorical_features:
            features_df[feature] = features_df[feature].fillna('unknown')
        
        return df, features_df
    
    def create_preprocessing_pipeline(self):
        """Create a preprocessing pipeline for the features."""
        # Define preprocessing for numerical features
        numerical_transformer = StandardScaler()
        
        # Define preprocessing for categorical features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessing steps
        transformers = []
        
        if self.numerical_features:
            transformers.append(('num', numerical_transformer, self.numerical_features))
        
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0  # Return dense array
        )
        
        # Create a pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        return pipeline
    
    def train_model(self):
        """Train the Isolation Forest model on all logs."""
        if len(self.all_logs) < self.min_logs:
            print(f"Not enough logs to train model (need {self.min_logs}, have {len(self.all_logs)})")
            return False
        
        print(f"Training model on {len(self.all_logs)} logs...")
        
        # Extract features
        self.log_dataframe, features_df = self.extract_features(self.all_logs)
        
        # Create preprocessing pipeline
        self.pipeline = self.create_preprocessing_pipeline()
        
        # Transform features
        X = self.pipeline.fit_transform(features_df)
        
        # Train Isolation Forest model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.model.fit(X)
        
        # Reset counter for new logs since retrain
        self.new_logs_since_retrain = 0
        
        print("Model training complete.")
        return True
    
    def predict_anomalies(self, logs):
        """Predict anomalies using the trained model."""
        # Check if we have a trained model
        if self.model is None or self.pipeline is None:
            print("Model not trained yet.")
            return None, None
        
        # Extract features from logs
        df, features_df = self.extract_features(logs)
        
        # Transform features
        X = self.pipeline.transform(features_df)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal points)
        predictions = self.model.predict(X)
        
        # Get decision function values (lower values indicate more anomalous points)
        scores = self.model.decision_function(X)
        
        # Add predictions to the dataframe
        df['anomaly_score'] = scores
        df['predicted_anomaly'] = (predictions == -1).astype(int)
        
        return df, predictions
    
    def check_new_logs(self):
        """Check for new logs and detect anomalies."""
        # Load new logs
        new_logs = self.load_new_logs()
        
        if not new_logs:
            return
        
        print(f"Found {len(new_logs)} new log entries.")
        
        # Check if we need to train or retrain the model
        if self.model is None:
            if len(self.all_logs) >= self.min_logs:
                self.train_model()
            else:
                print(f"Waiting for more logs to train model ({len(self.all_logs)}/{self.min_logs})...")
                return
        elif self.new_logs_since_retrain >= self.retrain_interval:
            print(f"Retraining model after {self.new_logs_since_retrain} new logs...")
            self.train_model()
        
        # Now predict anomalies for the new logs
        df, predictions = self.predict_anomalies(new_logs)
        
        if df is not None and not df.empty:
            # Print results for each new log entry
            for i, (_, row) in enumerate(df.iterrows()):
                timestamp = row.get('timestamp', 'N/A')
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                log_index = self.logs_processed - len(new_logs) + i
                
                if row['predicted_anomaly'] == 1:
                    print(f"\n[{timestamp}] ANOMALY DETECTED in log #{log_index}:")
                    # Print key fields based on what's available
                    for field in ['request_method', 'request_path', 'status_code', 'response_time_ms']:
                        if field in row:
                            print(f"  {field}: {row[field]}")
                    
                    print(f"  Anomaly score: {row['anomaly_score']:.4f}")
                    
                    # Compare with ground truth if available (for testing)
                    if 'is_anomalous' in row:
                        print(f"  Ground truth (for reference): {row['is_anomalous']}")
                else:
                    # Just a simple notification for normal logs
                    print(f"[{timestamp}] Log #{log_index} is normal (score: {row['anomaly_score']:.4f})")


def monitor_logs(file_path, contamination=0.1, retrain_interval=20, min_logs=10):
    """Monitor log file for new entries and detect anomalies."""
    print(f"Starting API log anomaly detection for {file_path}")
    print(f"Configuration: contamination={contamination}, retrain_interval={retrain_interval}, min_logs={min_logs}")
    
    # Make sure the file exists
    if not os.path.exists(file_path):
        print(f"Error: Log file {file_path} not found.")
        return
    
    # Create the anomaly detector
    detector = APIAnomalyDetector(
        file_path=file_path,
        contamination=contamination,
        retrain_interval=retrain_interval,
        min_logs=min_logs
    )
    
    # Set up file watchdog
    event_handler = LogFileHandler(file_path, detector)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    observer.start()
    
    print(f"Monitoring {file_path} for changes... (Press Ctrl+C to exit)")
    
    try:
        # Initial check for any existing logs
        detector.check_new_logs()
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping monitoring...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-Time API Log Anomaly Detection')
    parser.add_argument('--file', '-f', default='auth_interactions.json', help='Path to the log file to monitor')
    parser.add_argument('--contamination', '-c', type=float, default=0.1, help='Expected proportion of anomalies (0.0-0.5)')
    parser.add_argument('--retrain', '-r', type=int, default=20, help='Number of new logs before retraining the model')
    parser.add_argument('--min-logs', '-m', type=int, default=10, help='Minimum logs required for initial training')
    
    args = parser.parse_args()
    monitor_logs(args.file, args.contamination, args.retrain, args.min_logs)