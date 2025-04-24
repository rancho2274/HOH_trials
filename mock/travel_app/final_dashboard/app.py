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

def add_response_anomaly_to_logs(input_file, result_df, output_folder, api_name):
    """
    Creates a log file with response_anomaly flags in the combine_logs folder
    
    Args:
        input_file: Path to the original log file
        result_df: DataFrame with anomaly detection results
        output_folder: Path to the combine_logs folder
        api_name: Name of the API (auth, search, etc.)
        
    Returns:
        Path to the created output file
    """
    if result_df.empty:
        print(f"No anomaly results for {input_file}. Skipping...")
        return None
        
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
        return None
    
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
    for log in logs:
        timestamp = log.get('timestamp')
        original_anomalous = log.get('is_anomalous', False)
        
        # Count original anomalies
        if original_anomalous:
            original_anomaly_count += 1
        
        # Add the anomaly flag
        if timestamp in anomaly_map:
            response_anomaly = bool(anomaly_map[timestamp])
            log['response_anomaly'] = response_anomaly
        else:
            # If not found in results, assume not anomalous
            log['response_anomaly'] = False
        
        # Count response anomalies and discrepancies
        if log['response_anomaly']:
            response_anomaly_count += 1
            
        if original_anomalous != log['response_anomaly']:
            discrepancy_count += 1
    
    # Create the output file name according to the format: {api_name}_interactions_with_anomalies.json
    output_file = os.path.join(output_folder, f"{api_name}_interactions_with_anomalies.json")
    
    # Write logs to the output file in NDJSON format (each log entry on its own line with a blank line between)
    with open(output_file, 'w') as f:
        for i, log in enumerate(logs):
            f.write(json.dumps(log))
            if i < len(logs) - 1:
                f.write('\n\n')  # Add empty line between log entries
    
    print(f"Updated logs with response_anomaly flags saved to: {output_file}")
    
    # Print analytics about anomalies
    print(f"\nAnalytics for {api_name} logs:")
    print(f"  Total logs: {total_logs}")
    percentage_original = (original_anomaly_count / total_logs * 100) if total_logs > 0 else 0
    percentage_response = (response_anomaly_count / total_logs * 100) if total_logs > 0 else 0
    percentage_discrepancy = (discrepancy_count / total_logs * 100) if total_logs > 0 else 0
    
    print(f"  Original anomalies: {original_anomaly_count} ({percentage_original:.2f}%)")
    print(f"  Response time anomalies: {response_anomaly_count} ({percentage_response:.2f}%)")
    print(f"  Discrepancies: {discrepancy_count} ({percentage_discrepancy:.2f}%)")
    
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
    
    # Ensure combine_logs directory exists
    combine_logs_dir = os.path.join(travel_app_dir, "combine_logs")
    if not os.path.exists(combine_logs_dir):
        os.makedirs(combine_logs_dir)
        print(f"Created combine_logs directory: {combine_logs_dir}")
    
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
    
    # Debug info
    print("\n==== Starting Log Processing ====")
    print(f"Looking for logs in: {travel_app_dir}")
    print(f"Saving processed logs to: {combine_logs_dir}")
    
    # Process each log file
    for api_name, log_file in log_files.items():
        file_path = os.path.join(travel_app_dir, log_file)
        print(f"\nChecking for file: {file_path}")
        
        if os.path.exists(file_path):
            print(f"  File exists! Processing {api_name} logs...")
            try:
                # Load logs
                logs = detector.load_logs(file_path)
                print(f"  Loaded {len(logs)} log entries for {api_name}")
                
                if logs:
                    # Process logs and get results
                    result_df = detector.detect_anomalies(logs)
                    
                    # Update logs with response_anomaly flags and save to combine_logs directory
                    add_response_anomaly_to_logs(file_path, result_df, combine_logs_dir, api_name)
                    
                    if not result_df.empty:
                        # Count anomalies
                        anomalies = result_df[result_df['predicted_anomaly'] == True]
                        normal = result_df[result_df['predicted_anomaly'] == False]
                        
                        print(f"  {api_name} stats: {len(result_df)} total logs, {len(anomalies)} anomalies")
                        
                        # Update API-specific statistics
                        api_stats[api_name]["total_logs"] = len(result_df)
                        api_stats[api_name]["anomalies"] = len(anomalies)
                        
                        # Calculate API-specific anomaly percentage
                        if len(result_df) > 0:
                            api_stats[api_name]["anomaly_percent"] = round(len(anomalies) / len(result_df) * 100, 1)
                        
                        # FIXED: Calculate API-specific response time averages
                        if len(normal) > 0:
                            # Simple mean of time_ms column for normal logs
                            api_stats[api_name]["normal_avg"] = round(normal['time_ms'].mean(), 2)
                        
                        if len(anomalies) > 0:
                            # Simple mean of time_ms column for anomalous logs
                            api_stats[api_name]["anomalous_avg"] = round(anomalies['time_ms'].mean(), 2)
                        
                        # Update overall stats
                        stats["total_logs"] += len(result_df)
                        stats["anomalies"] += len(anomalies)
                        stats["normal_logs"] += len(normal)
                        
                        # FIXED: Calculate overall normal and anomalous response time averages
                        if len(normal) > 0:
                            normal_sum = normal['time_ms'].sum()
                            # For the first API being processed
                            if api_name == "auth" or stats["normal_avg"] == 0:
                                stats["normal_avg"] = round(normal_sum / len(normal), 2)
                            else:
                                # Calculate weighted average for subsequent APIs
                                previous_total = stats["normal_avg"] * (stats["normal_logs"] - len(normal))
                                new_avg = (previous_total + normal_sum) / stats["normal_logs"]
                                stats["normal_avg"] = round(new_avg, 2)
                        
                        if len(anomalies) > 0:
                            anomalies_sum = anomalies['time_ms'].sum()
                            # For the first API being processed
                            if api_name == "auth" or stats["anomalous_avg"] == 0:
                                stats["anomalous_avg"] = round(anomalies_sum / len(anomalies), 2)
                            else:
                                # Calculate weighted average for subsequent APIs
                                previous_total = stats["anomalous_avg"] * (stats["anomalies"] - len(anomalies))
                                new_avg = (previous_total + anomalies_sum) / stats["anomalies"]
                                stats["anomalous_avg"] = round(new_avg, 2)
                    else:
                        print(f"  No valid data found in {api_name} logs after processing")
                        
                else:
                    print(f"  No logs found in {api_name} file")
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  File not found: {file_path}")
    
    # Calculate overall anomaly percentage
    if stats["total_logs"] > 0:
        stats["anomaly_percent"] = round(stats["anomalies"] / stats["total_logs"] * 100, 1)
    
    # Print overall stats for debugging
    print("\n==== Overall Statistics ====")
    print(f"Total logs: {stats['total_logs']}")
    print(f"Anomalies: {stats['anomalies']} ({stats['anomaly_percent']}%)")
    print(f"Normal logs: {stats['normal_logs']}")
    print(f"Normal average response time: {stats['normal_avg']} ms")
    print(f"Anomalous average response time: {stats['anomalous_avg']} ms")
    
    # Save stats to a file for future reference
    stats_file = os.path.join(current_dir, 'static', 'dashboard_stats.json')
    try:
        # Ensure the static directory exists
        static_dir = os.path.join(current_dir, 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            
        # Write stats to file
        with open(stats_file, 'w') as f:
            combined_stats = {
                "overall": stats,
                "api_stats": api_stats
            }
            json.dump(combined_stats, f, indent=2)
            print(f"Stats saved to {stats_file}")
    except Exception as e:
        print(f"Error saving stats to {stats_file}: {str(e)}")
    
    return stats, api_stats

@app.route('/')
def dashboard():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('dashboard.html', stats=stats, api_stats=api_stats)

@app.route('/api/system')
def system_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="System", stats=stats)

@app.route('/api/auth')
def auth_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="Authentication", stats=api_stats['auth'])

@app.route('/api/search')
def search_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="Search", stats=api_stats['search'])

@app.route('/api/booking')
def booking_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="Booking", stats=api_stats['booking'])

@app.route('/api/payment')
def payment_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="Payment", stats=api_stats['payment'])

@app.route('/api/feedback')
def feedback_api():
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    return render_template('api_health.html', api_name="Feedback", stats=api_stats['feedback'])

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