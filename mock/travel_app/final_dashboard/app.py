import os
import sys
from flask import Flask, render_template, redirect, url_for, jsonify, request
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
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
    
        if not logs:
            return pd.DataFrame()
            
    # Preprocess logs
        features_df = self.preprocess_logs(logs)
        
        if len(features_df) == 0:
            return pd.DataFrame()
        
        # Calculate adaptive thresholds based on the distribution of response times
        # This helps ensure changes to individual logs are better handled
        times = features_df['time_ms'].values
        
        # Calculate basic statistics
        median_time = np.median(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate the 95th percentile as a reference for extreme values
        percentile_95 = np.percentile(times, 95)
        
        # Set an adaptive threshold that's more robust to changes
        adaptive_threshold = max(1000, median_time * 2, mean_time + 2 * std_time)
        
        print(f"Anomaly detection statistics:")
        print(f"  Median response time: {median_time:.2f}ms")
        print(f"  Mean response time: {mean_time:.2f}ms")
        print(f"  95th percentile: {percentile_95:.2f}ms")
        print(f"  Adaptive threshold: {adaptive_threshold:.2f}ms")
        
        # Train model with custom contamination parameter based on data distribution
        # Use a lower contamination if the data seems to have few outliers
        estimated_contamination = len(times[times > adaptive_threshold]) / len(times)
        contamination = min(max(0.01, estimated_contamination), 0.1)  # Keep between 1% and 10%
        
        print(f"  Estimated contamination: {estimated_contamination:.4f}")
        print(f"  Using contamination: {contamination:.4f}")
        
        # Store the original contamination value
        original_contamination = self.contamination
        
        # Use the adaptive contamination for this run
        self.contamination = contamination
        
        # Train the model
        self.train(features_df)
        
        # Make predictions
        result_df = self.predict(features_df)
        
        # Add an extra pass to properly label manually modified logs
        # If a log has a time below our adaptive threshold, make sure it's not anomalous
        result_df.loc[result_df['time_ms'] < adaptive_threshold, 'force_normal'] = True
        
        # Override predictions for manually modified logs
        manual_corrections = result_df['force_normal'] == True
        if manual_corrections.any():
            num_corrections = manual_corrections.sum()
            print(f"  Applied manual corrections to {num_corrections} logs (likely modified)")
            result_df.loc[manual_corrections, 'predicted_anomaly'] = False
            result_df.loc[manual_corrections, 'anomaly_score'] = 1  # 1 means normal in Isolation Forest
        
        # Restore the original contamination value
        self.contamination = original_contamination
        
        return result_df

def detect_spikes(logs, threshold_multiplier=3.0, min_threshold=1000):
    """
    Simple function to detect response time spikes in logs
    
    Args:
        logs: List of log entries
        threshold_multiplier: Multiple of average response time to consider a spike
        min_threshold: Minimum response time to be considered a spike (ms)
        
    Returns:
        Dictionary with spike information
    """
    if not logs:
        return {"spikes": [], "api_spikes": {}}
    
    # Extract response times by API
    api_response_times = {
        "auth": [],
        "search": [],
        "booking": [],
        "payment": [],
        "feedback": []
    }
    
    # Process each log
    for log in logs:
        if 'response' in log and 'time_ms' in log['response']:
            time_ms = log['response']['time_ms']
            
            # Determine API type
            api_type = "unknown"
            if 'auth_service' in log:
                api_type = "auth"
            elif 'search_service' in log:
                api_type = "search"
            elif 'booking_service' in log:
                api_type = "booking"
            elif 'payment_service' in log:
                api_type = "payment"
            elif 'feedback_service' in log:
                api_type = "feedback"
            
            # Add to the appropriate API list
            if api_type in api_response_times:
                api_response_times[api_type].append({
                    "timestamp": log.get("timestamp"),
                    "response_time": time_ms,
                    "api": api_type
                })
    
    # Detect spikes for each API
    all_spikes = []
    api_spikes = {}
    
    for api, data in api_response_times.items():
        if not data:
            api_spikes[api] = []
            continue
        
        # Get response times
        times = [entry["response_time"] for entry in data]
        
        # Calculate average
        avg_time = sum(times) / len(times)
        
        # Set threshold
        threshold = max(min_threshold, avg_time * threshold_multiplier)
        
        # Find spikes
        spikes = []
        for entry in data:
            if entry["response_time"] >= threshold:
                # Add additional spike info
                spike = entry.copy()
                spike["threshold"] = threshold
                spike["times_avg"] = entry["response_time"] / avg_time
                spikes.append(spike)
        
        # Store spikes for this API
        api_spikes[api] = spikes
        all_spikes.extend(spikes)
    
    return {
        "spikes": all_spikes,
        "api_spikes": api_spikes
    }

def generate_alerts(spike_data):
    """
    Generate alerts from spike data
    
    Args:
        spike_data: Dictionary with spike information
        
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    # Process all spikes
    for spike in spike_data["spikes"]:
        # Determine severity based on how many times above threshold
        times_avg = spike.get("times_avg", 0)
        
        if times_avg >= 5.0:
            severity = "CRITICAL"
        elif times_avg >= 3.0:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
        
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(spike["timestamp"].replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = spike["timestamp"]
        
        # Create alert
        alert = {
            "severity": severity,
            "api": spike["api"].upper(),
            "message": f"{severity}: {spike['api'].upper()} API response time spike detected",
            "details": f"Response time of {spike['response_time']:.0f}ms is {times_avg:.1f}x higher than normal",
            "timestamp": formatted_time
        }
        
        alerts.append(alert)
    
    # Sort alerts by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    alerts.sort(key=lambda x: severity_order.get(x["severity"], 3))
    
    return alerts

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
    # Use both timestamp and response time for more accurate matching
    anomaly_map = {}
    for _, row in result_df.iterrows():
        timestamp = row['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.isoformat()
        
        # Store both the anomaly flag and the response time that was analyzed
        anomaly_map[timestamp] = {
            'predicted_anomaly': row['predicted_anomaly'],
            'analyzed_time_ms': row['time_ms']
        }
    
    # Track statistics for discrepancies
    total_logs = len(logs)
    response_anomaly_count = 0
    original_anomaly_count = 0
    discrepancy_count = 0
    updated_count = 0
    
    # Update each log entry with anomaly information
    for log in logs:
        timestamp = log.get('timestamp')
        original_anomalous = log.get('is_anomalous', False)
        
        # Count original anomalies
        if original_anomalous:
            original_anomaly_count += 1
        
        # Add the anomaly flag
        if timestamp in anomaly_map:
            # Get the analysis results for this timestamp
            analysis_result = anomaly_map[timestamp]
            
            # Get the current response time from the log
            current_time_ms = log['response']['time_ms']
            
            # Check if the response time in the log has been manually modified
            # If the current time is different from what was analyzed, reconsider the anomaly flag
            if abs(current_time_ms - analysis_result['analyzed_time_ms']) > 0.001:  # Allow for floating point differences
                print(f"Log at {timestamp} was modified: analyzed={analysis_result['analyzed_time_ms']}ms, current={current_time_ms}ms")
                
                # If the time was reduced below the threshold, mark as non-anomalous
                # Using a simple heuristic: if the time is less than 1000ms, it's likely normal
                # Adjust this threshold based on your specific requirements
                if current_time_ms < 1000 and analysis_result['predicted_anomaly']:
                    log['response_anomaly'] = False
                    updated_count += 1
                    print(f"  Changed anomaly flag to FALSE due to manual time reduction")
                else:
                    log['response_anomaly'] = analysis_result['predicted_anomaly']
            else:
                # No modification detected, use the analyzed result
                log['response_anomaly'] = analysis_result['predicted_anomaly']
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
    print(f"  Manually updated logs: {updated_count}")
    
    return output_file

def process_log_files_with_spikes():
    """
    Process all log files, detect anomalies, and identify spikes
    
    Returns:
        Dictionary with statistics for each API and spike information
    """
    # Get regular statistics
    stats, api_stats = process_log_files()
    
    # Get the travel_app directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    travel_app_dir = os.path.dirname(current_dir)  # Go up one level to reach travel_app
    
    # Load all logs for spike detection
    all_logs = []
    log_files = {
        "auth": "auth_interactions.json",
        "search": "search_interactions.json",
        "booking": "booking_interactions.json",
        "payment": "payment_interactions.json",
        "feedback": "feedback_interactions.json"
    }
    
    for api_name, log_file in log_files.items():
        file_path = os.path.join(travel_app_dir, log_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
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
                                    pass
                all_logs.extend(logs)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    # Detect spikes
    spike_info = detect_spikes(all_logs)
    
    # Generate alerts from spikes
    alerts = generate_alerts(spike_info)
    
    # Return combined information
    return {
        "overall": stats,
        "api_stats": api_stats,
        "spikes": spike_info["spikes"],
        "api_spikes": spike_info["api_spikes"],
        "alerts": alerts,
        "has_alerts": len(alerts) > 0
    }

def process_log_files():
    """
    Process all log files, detect anomalies, and update logs with response_anomaly flags
    with improved handling of manually modified logs
    
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
    
    # Check if there are any existing processed logs in combine_logs directory
    # We'll check these to detect manual modifications
    existing_processed_files = {}
    for api_name in log_files.keys():
        processed_file = os.path.join(combine_logs_dir, f"{api_name}_interactions_with_anomalies.json")
        if os.path.exists(processed_file):
            existing_processed_files[api_name] = processed_file
    
    if existing_processed_files:
        print(f"Found existing processed logs: {list(existing_processed_files.keys())}")
    
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
                
                # Check if we have previously processed logs to compare against
                previous_logs = []
                if api_name in existing_processed_files:
                    try:
                        previous_logs = detector.load_logs(existing_processed_files[api_name])
                        print(f"  Also loaded {len(previous_logs)} previously processed logs for comparison")
                    except Exception as e:
                        print(f"  Error loading previous logs: {str(e)}")
                
                # Create lookup for previous anomaly flags
                previous_anomaly_map = {}
                if previous_logs:
                    for prev_log in previous_logs:
                        if 'timestamp' in prev_log and 'response' in prev_log and 'time_ms' in prev_log['response']:
                            timestamp = prev_log['timestamp']
                            previous_anomaly_map[timestamp] = {
                                'time_ms': prev_log['response']['time_ms'],
                                'response_anomaly': prev_log.get('response_anomaly', False)
                            }
                
                # Check for manual modifications by comparing current logs with previous processed logs
                manual_modifications = []
                for log in logs:
                    timestamp = log.get('timestamp')
                    if timestamp in previous_anomaly_map:
                        prev_data = previous_anomaly_map[timestamp]
                        current_time = log['response']['time_ms']
                        
                        # If response time has changed significantly and anomaly flag was set
                        if abs(current_time - prev_data['time_ms']) > 0.1 and prev_data['response_anomaly']:
                            manual_modifications.append({
                                'timestamp': timestamp,
                                'previous_time': prev_data['time_ms'],
                                'current_time': current_time,
                                'previous_anomaly': prev_data['response_anomaly']
                            })
                
                if manual_modifications:
                    print(f"  Detected {len(manual_modifications)} manual modifications to response times!")
                    for mod in manual_modifications[:5]:  # Show max 5 examples
                        print(f"    {mod['timestamp']}: {mod['previous_time']}ms → {mod['current_time']}ms")
                    
                    if len(manual_modifications) > 5:
                        print(f"    ... and {len(manual_modifications) - 5} more")
                
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
                        
                        # Calculate API-specific response time averages
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
                        
                        # Calculate overall normal and anomalous response time averages
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
    
    return stats, api_stats

@app.route('/')
def dashboard():
    # Process log files and get statistics with spike detection
    combined_stats = process_log_files_with_spikes()
    
    # Pass all required variables to the template
    return render_template('dashboard.html', 
                          stats=combined_stats['overall'], 
                          api_stats=combined_stats['api_stats'],
                          spikes=combined_stats.get('spikes', []),
                          api_spikes=combined_stats.get('api_spikes', {}),
                          alerts=combined_stats.get('alerts', []),
                          has_alerts=combined_stats.get('has_alerts', False))



@app.route('/refresh')
def refresh():
    # Process log files and get updated statistics with spike detection
    combined_stats = process_log_files_with_spikes()
    
    # If an AJAX request, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(combined_stats)
    
    # Otherwise, refresh the page
    return redirect(url_for('dashboard'))

@app.route('/api/check_spikes')
def check_spikes():
    """API endpoint to check for new spikes"""
    combined_stats = process_log_files_with_spikes()
    return jsonify({
        "spikes": combined_stats.get('spikes', []),
        "api_spikes": combined_stats.get('api_spikes', {}),
        "alerts": combined_stats.get('alerts', []),
        "has_alerts": combined_stats.get('has_alerts', False)
    })
    
# Add these routes to your app.py file

@app.route('/kibana_config')
def kibana_config():
    """Return Kibana configuration settings as JSON"""
    # Define dashboard IDs for each API
    # You would replace these with actual dashboard IDs from your Kibana setup
    kibana_dashboards = {
        "system": "system-overview-dashboard",
        "auth": "auth-api-dashboard",
        "search": "search-api-dashboard",
        "booking": "booking-api-dashboard",
        "payment": "payment-api-dashboard",
        "feedback": "feedback-api-dashboard"
    }
    
    # Return configuration as JSON
    return jsonify({
        "kibana_url": "http://localhost:5601",
        "dashboards": kibana_dashboards
    })

# Modify your existing route handlers to pass Kibana dashboard IDs to templates
@app.route('/api/system')
def system_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="System", 
                          stats=combined_stats['overall'],
                          kibana_dashboard_id="system-overview-dashboard")

@app.route('/api/auth')
def auth_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="Authentication", 
                          stats=combined_stats['api_stats']['auth'],
                          kibana_dashboard_id="auth-api-dashboard")

@app.route('/api/search')
def search_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="Search", 
                          stats=combined_stats['api_stats']['search'],
                          kibana_dashboard_id="search-api-dashboard")

@app.route('/api/booking')
def booking_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="Booking", 
                          stats=combined_stats['api_stats']['booking'],
                          kibana_dashboard_id="booking-api-dashboard")

@app.route('/api/payment')
def payment_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="Payment", 
                          stats=combined_stats['api_stats']['payment'],
                          kibana_dashboard_id="payment-api-dashboard")

@app.route('/api/feedback')
def feedback_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    return render_template('api_health.html', 
                          api_name="Feedback", 
                          stats=combined_stats['api_stats']['feedback'],
                          kibana_dashboard_id="feedback-api-dashboard")

if __name__ == '__main__':
    app.run(debug=True, port=5050)