import os
import sys
from flask import Flask, render_template, redirect, url_for, jsonify, request
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Import the anomaly detection functionality directly
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
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    return json.loads(content)
                else:
                    # Handle newline-delimited JSON
                    logs = []
                    for line in content.split('\n\n'):
                        if line.strip():
                            try:
                                logs.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse line as JSON: {line[:50]}...")
                    return logs
        except Exception as e:
            print(f"Error loading logs from {file_path}: {str(e)}")
            return []
    
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
            if 'response' not in log or 'time_ms' not in log['response']:
                continue
                
            # Extract response time and minimal metadata
            feature_dict = {
                # The only feature we care about for detection
                'time_ms': log['response']['time_ms'],
                
                # Metadata (not used for detection, only for reference)
                'timestamp': log['timestamp'],
                'operation_type': log.get('operation', {}).get('type') if 'operation' in log else 
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
            logs: List of log entries
            
        Returns:
            DataFrame with anomaly predictions
        """
        if not logs:
            return pd.DataFrame()
            
        # Preprocess logs
        features_df = self.preprocess_logs(logs)
        
        if len(features_df) == 0:
            return pd.DataFrame()
        
        # Train model
        self.train(features_df)
        
        # Make predictions
        result_df = self.predict(features_df)
        
        return result_df

def add_response_anomaly_to_logs(input_file, result_df, output_folder=None):
    """
    Creates a copy of the log file with response_anomaly flags added and saves it
    
    Args:
        input_file: Path to the original log file
        result_df: DataFrame with anomaly detection results
        output_folder: Path to the folder where output files will be stored (if None, original file is updated)
        
    Returns:
        Path to the created/updated file
    """
    if result_df.empty:
        print(f"No anomaly results for {input_file}. Skipping...")
        return input_file
        
    # Load the original logs
    try:
        with open(input_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                logs = json.loads(content)
            else:
                # Handle newline-delimited JSON
                logs = []
                for line in content.split('\n\n'):
                    if line.strip():
                        try:
                            logs.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    except Exception as e:
        print(f"Error loading logs from {input_file}: {str(e)}")
        return input_file
    
    # Create a dictionary from result_df for fast lookup
    anomaly_map = {}
    for _, row in result_df.iterrows():
        timestamp = row['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.isoformat()
        anomaly_map[timestamp] = row['predicted_anomaly']
    
    # Update each log entry with anomaly information
    for log in logs:
        timestamp = log.get('timestamp')
        
        # Add the anomaly flag
        if timestamp in anomaly_map:
            log['response_anomaly'] = bool(anomaly_map[timestamp])
        else:
            # If not found in results, assume not anomalous
            log['response_anomaly'] = False
    
    # Determine output file path
    if output_folder:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get just the filename without path
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, file_name)
    else:
        # Overwrite original file
        output_file = input_file
    
    # Write the updated logs back to the file
    with open(output_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Updated logs with response_anomaly flags: {output_file}")
    return output_file

def process_log_files():
    """
    Process all log files, detect anomalies, and update logs with response_anomaly flags
    
    Returns:
        Dictionary with statistics for each API
    """
    # Create an instance of the anomaly detector
    detector = ResponseTimeAnomalyDetector()
    
    # Define the log files to process
    log_files = {
        "auth": "auth_interactions.json",
        "search": "search_interactions.json",
        "booking": "booking_interactions.json",
        "payment": "payment_interactions.json",
        "feedback": "feedback_interactions.json"
    }
    
    # Get the travel_app directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    travel_app_dir = os.path.dirname(current_dir)  # Go up one level to reach travel_app
    
    # Gather overall statistics
    stats = {
        "total_logs": 0,
        "anomalies": 0,
        "anomaly_percent": 0,
        "normal_logs": 0,
        "normal_avg": 0,
        "anomalous_avg": 0,
        "updated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Prepare stats for each API
    api_stats = {
        "auth": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "search": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "booking": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "payment": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "feedback": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0}
    }
    
    # Process each log file
    for api_name, log_file in log_files.items():
        file_path = os.path.join(travel_app_dir, log_file)
        if os.path.exists(file_path):
            try:
                # Load logs
                logs = detector.load_logs(file_path)
                if logs:
                    # Process logs and get results
                    result_df = detector.detect_anomalies(logs)
                    
                    # Update logs with response_anomaly flags
                    add_response_anomaly_to_logs(file_path, result_df)
                    
                    if not result_df.empty:
                        # Count anomalies
                        anomalies = result_df[result_df['predicted_anomaly'] == True]
                        normal = result_df[result_df['predicted_anomaly'] == False]
                        
                        # Update overall statistics
                        stats["total_logs"] += len(result_df)
                        stats["anomalies"] += len(anomalies)
                        stats["normal_logs"] += len(normal)
                        
                        # Update API-specific statistics
                        api_stats[api_name]["total_logs"] = len(result_df)
                        api_stats[api_name]["anomalies"] = len(anomalies)
                        
                        # Calculate API-specific anomaly percentage
                        if len(result_df) > 0:
                            api_stats[api_name]["anomaly_percent"] = round(len(anomalies) / len(result_df) * 100, 1)
                        
                        # Calculate API-specific response time averages
                        if len(normal) > 0:
                            api_stats[api_name]["normal_avg"] = round(normal['time_ms'].mean(), 2)
                        
                        if len(anomalies) > 0:
                            api_stats[api_name]["anomalous_avg"] = round(anomalies['time_ms'].mean(), 2)
                        
                        # Update overall response time averages (weighted by log count)
                        if len(normal) > 0:
                            normal_avg = normal['time_ms'].mean()
                            stats["normal_avg"] = round((stats["normal_avg"] * (stats["normal_logs"] - len(normal)) + 
                                              normal_avg * len(normal)) / stats["normal_logs"], 2)
                        
                        if len(anomalies) > 0:
                            anomalous_avg = anomalies['time_ms'].mean()
                            if stats["anomalies"] > 0:
                                stats["anomalous_avg"] = round((stats["anomalous_avg"] * (stats["anomalies"] - len(anomalies)) + 
                                                anomalous_avg * len(anomalies)) / stats["anomalies"], 2)
                            else:
                                stats["anomalous_avg"] = round(anomalous_avg, 2)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Calculate overall anomaly percentage
    if stats["total_logs"] > 0:
        stats["anomaly_percent"] = round(stats["anomalies"] / stats["total_logs"] * 100, 1)
    
    # Save stats to a file for future reference
    stats_file = os.path.join(current_dir, 'static', 'dashboard_stats.json')
    with open(stats_file, 'w') as f:
        combined_stats = {
            "overall": stats,
            "api_stats": api_stats
        }
        json.dump(combined_stats, f, indent=2)
    
    return stats, api_stats

@app.route('/')
def dashboard():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    
    return render_template('dashboard.html', stats=stats, api_stats=api_stats)

@app.route('/refresh')
def refresh():
    # Process log files and get updated statistics
    stats, api_stats = process_log_files()
    
    # If an AJAX request, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "stats": stats,
            "api_stats": api_stats
        })
    
    # Otherwise, refresh the page
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, port=5050)