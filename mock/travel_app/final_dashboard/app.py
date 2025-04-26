import os
import sys
from flask import Flask, render_template, redirect, url_for, jsonify, request
import json
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
        times = features_df['time_ms'].values
        
        # Calculate basic statistics
        median_time = np.median(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate percentiles to better handle skewed distributions
        percentile_75 = np.percentile(times, 75)
        percentile_90 = np.percentile(times, 90)
        
        # IMPROVED: Lower the fixed minimum threshold and use more sensitive multipliers
        # Old version: adaptive_threshold = max(1000, median_time * 2, mean_time + 2 * std_time)
        adaptive_threshold = max(
            750,  # Lower fixed minimum threshold from 1000 to 750ms
            median_time * 1.5,  # Lower multiplier from 2.0 to 1.5
            mean_time + 1.5 * std_time,  # Lower std multiplier from 2.0 to 1.5
            percentile_75 * 1.2  # Add percentile-based threshold
        )
        
        print(f"Anomaly detection statistics:")
        print(f"  Median response time: {median_time:.2f}ms")
        print(f"  Mean response time: {mean_time:.2f}ms")
        print(f"  75th percentile: {percentile_75:.2f}ms")
        print(f"  90th percentile: {percentile_90:.2f}ms")
        print(f"  Improved adaptive threshold: {adaptive_threshold:.2f}ms")
        
        # IMPROVED: More sensitive contamination for better anomaly detection
        # Calculate how many responses would be flagged with our improved threshold
        estimated_contamination = len(times[times > adaptive_threshold]) / len(times)
        # Set minimum contamination to 0.02 (2%) instead of 0.01 (1%)
        # This ensures Isolation Forest will be more sensitive
        contamination = min(max(0.02, estimated_contamination), 0.15)  # Range: 2% to 15%
        
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
        
        # IMPROVED: More intelligent post-processing for edge cases
        # Instead of forcing all responses below threshold to be normal,
        # use a combination of threshold and distance from mean
        
        # Create a series indicating if a response time is extreme (>3000ms)
        extreme_response_times = result_df['time_ms'] >= 3000
        
        # Create a series indicating if a response time is high multiple of mean
        mean_multiple = result_df['time_ms'] / mean_time
        high_mean_multiple = mean_multiple >= 2.0  # 2x or higher than mean
        
        # Only force normal if response time is below half of threshold 
        # AND not extreme AND not a high multiple of the mean
        result_df.loc[(result_df['time_ms'] < adaptive_threshold/2) & 
                    ~extreme_response_times & 
                    ~high_mean_multiple, 'force_normal'] = True
        
        # Force anomaly status for very extreme values, regardless of Isolation Forest
        result_df.loc[extreme_response_times, 'force_anomaly'] = True
        
        # Apply manual corrections
        manual_corrections = result_df['force_normal'] == True
        if manual_corrections.any():
            num_corrections = manual_corrections.sum()
            print(f"  Applied {num_corrections} manual 'normal' corrections")
            result_df.loc[manual_corrections, 'predicted_anomaly'] = False
            result_df.loc[manual_corrections, 'anomaly_score'] = 1  # 1 = normal
        
        # Apply forced anomalies (very high response times)
        forced_anomalies = result_df.get('force_anomaly') == True
        if forced_anomalies.any():
            num_forced = forced_anomalies.sum()
            print(f"  Forced {num_forced} extreme values to be anomalies")
            result_df.loc[forced_anomalies, 'predicted_anomaly'] = True
            result_df.loc[forced_anomalies, 'anomaly_score'] = -1  # -1 = anomaly
        
        # Restore the original contamination value
        self.contamination = original_contamination
        
        # Add a column indicating how many times higher than mean
        result_df['times_mean'] = result_df['time_ms'] / mean_time
        
        # Count and print how many anomalies were detected
        anomalies_count = result_df['predicted_anomaly'].sum()
        normal_count = len(result_df) - anomalies_count
        print(f"  Detected {anomalies_count} anomalies out of {len(result_df)} logs ({anomalies_count/len(result_df)*100:.2f}%)")
        
        return result_df

def detect_spikes(logs, threshold_multiplier=3.0, min_threshold=1000):
    """
    Simple function to detect response time spikes in logs compared to average normal time
    
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
    
    # First, calculate average normal response time for each API
    api_averages = {}
    
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
                    "api": api_type,
                    "operation": log.get('operation', {}).get('type') if 'operation' in log else 
                               log.get('search_service', {}).get('type') if 'search_service' in log else
                               log.get('payment_service', {}).get('operation') if 'payment_service' in log else
                               log.get('booking_service', {}).get('operation') if 'booking_service' in log else
                               log.get('feedback_service', {}).get('operation') if 'feedback_service' in log else
                               'unknown'
                })
    
    # Calculate average for each API (excluding outliers)
    for api, data in api_response_times.items():
        if not data:
            api_averages[api] = 0
            continue
        
        # Get response times
        times = [entry["response_time"] for entry in data]
        
        # Sort times and exclude top 10% to avoid skewing from existing outliers
        sorted_times = sorted(times)
        cutoff = int(len(sorted_times) * 0.9)
        normal_times = sorted_times[:cutoff] if cutoff > 0 else sorted_times
        
        # Calculate average of normal times
        if normal_times:
            api_averages[api] = sum(normal_times) / len(normal_times)
        else:
            api_averages[api] = 0
    
    # Detect spikes for each API
    all_spikes = []
    api_spikes = {}
    
    for api, data in api_response_times.items():
        if not data:
            api_spikes[api] = []
            continue
        
        # Get the average normal response time for this API
        avg_normal_time = api_averages[api]
        if avg_normal_time == 0:
            # Fall back to calculating simple average if no normal time available
            times = [entry["response_time"] for entry in data]
            avg_normal_time = sum(times) / len(times) if times else 0
        
        # Find spikes
        spikes = []
        for entry in data:
            response_time = entry["response_time"]
            # Calculate how many times higher than normal
            times_avg = response_time / avg_normal_time if avg_normal_time > 0 else 0
            
            # Consider a spike if response time is significantly higher than average
            if times_avg >= threshold_multiplier:
                # Add additional spike info
                spike = entry.copy()
                spike["normal_avg"] = avg_normal_time
                spike["times_avg"] = times_avg
                spikes.append(spike)
        
        # Store spikes for this API
        api_spikes[api] = spikes
        all_spikes.extend(spikes)
    
    return {
        "spikes": all_spikes,
        "api_spikes": api_spikes,
        "api_averages": api_averages
    }

def generate_alerts(spike_data):
    """
    Generate alerts from spike data with enhanced details
    comparing to average normal response times
    
    Args:
        spike_data: Dictionary with spike information
        
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    # Process all spikes
    for spike in spike_data["spikes"]:
        # Determine severity based on how many times above normal
        times_avg = spike.get("times_avg", 0)
        
        
        if times_avg >= 10.0:
            severity = "CRITICAL"
        elif times_avg >= 6.0:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
        
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(spike["timestamp"].replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = spike["timestamp"]
        
        # Create more descriptive details for each alert
        api_type = spike["api"].upper()
        response_time = spike["response_time"]
        normal_avg = spike.get("normal_avg", 0)
        
        if severity == "CRITICAL":
            details = f"Response time of {response_time:.0f}ms is {times_avg:.1f}x higher than the normal average of {normal_avg:.0f}ms. This indicates a severe performance degradation that requires immediate attention."
        else:
            details = f"Response time of {response_time:.0f}ms is higher than the normal average of {normal_avg:.0f}ms. This may indicate an emerging issue."
        
        # Add operation type if available
        operation = spike.get("operation")
        if operation:
            details += f" The affected operation is '{operation}'."
        
        # Create alert
        alert = {
            "severity": severity,
            "api": api_type,
            "message": f"{severity}: {api_type} API response time spike detected",
            "details": details,
            "timestamp": formatted_time,
            "response_time": response_time,
            "normal_avg": normal_avg,
            "times_avg": times_avg,
            "operation": operation
        }
        
        alerts.append(alert)
        
        # If you've implemented email notification:
        # if severity == "CRITICAL":
        #     send_alert_email(alert)
    
    # Sort alerts by severity and time
    severity_order = {"CRITICAL": 0, "MEDIUM": 1}
    alerts.sort(key=lambda x: (severity_order.get(x["severity"], 2), -x["response_time"]))
    
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

def send_alert_email(alert):
    """
    Send an email notification for critical alerts
    
    Args:
        alert: Dictionary containing alert information
    """
    # Email configuration
    sender_email = "rachitprasad2274@gmail.com"  # Update with your actual email
    receiver_email = "rachitchandawar40@gmail.com"  # Update with recipient email
    app_password = "mnxp ykbw rujv eeaz"  # Use an app password for Gmail, not your regular password
    
    # Create message
    message = MIMEMultipart()
    message["Subject"] = f"CRITICAL ALERT: {alert['api']} API Response Time Spike"
    message["From"] = sender_email
    message["To"] = receiver_email
    
    # Create detailed email body
    body = f"""
    <html>
    <body>
        <h2 style="color: #c00;">CRITICAL ALERT NOTIFICATION</h2>
        
        <p><strong>API:</strong> {alert['api']}</p>
        <p><strong>Time:</strong> {alert['timestamp']}</p>
        <p><strong>Severity:</strong> {alert['severity']}</p>
        
        <p><strong>Details:</strong> {alert['details']}</p>
        
        <p><strong>Response time:</strong> {alert['response_time']}ms</p>
        <p><strong>Normal average:</strong> {alert.get('normal_avg', 0)}ms</p>
        <p><strong>Times above normal:</strong> {alert.get('times_avg', 0):.1f}x</p>
        
        <p style="color: #c00;"><strong>Please investigate this issue immediately.</strong></p>
    </body>
    </html>
    """
    
    # Attach HTML body to email
    message.attach(MIMEText(body, "html"))
    
    try:
        # Create server connection with Gmail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        
        print(f"Attempting to send email alert to {receiver_email}")
        
        # Login using app password (not regular password)
        server.login(sender_email, app_password)
        
        # Send email
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        print(f"Email alert successfully sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {str(e)}")
        # Print full exception details for debugging
        import traceback
        print(traceback.format_exc())
        return False
def process_log_files_with_spikes():
   
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
        "has_alerts": len(alerts) > 0,
        # Changed from response_anomalies to original_anomalies
        "active_issues": stats.get("original_anomalies", 0) 
    }

def process_log_files():
   
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
        "original_anomalies": 0,  # Count based on is_anomalous flag
        "response_anomalies": 0,  # Count based on response_anomaly detection
        "anomaly_percent": 0,
        "normal_logs": 0,
        "normal_avg": 0,
        "anomalous_avg": 0,
        "updated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Prepare stats for each API
    api_stats = {
        "auth": {"total_logs": 0, "original_anomalies": 0, "response_anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0, "anomalies": 0},
        "search": {"total_logs": 0, "original_anomalies": 0, "response_anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0, "anomalies": 0},
        "booking": {"total_logs": 0, "original_anomalies": 0, "response_anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0, "anomalies": 0},
        "payment": {"total_logs": 0, "original_anomalies": 0, "response_anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0, "anomalies": 0},
        "feedback": {"total_logs": 0, "original_anomalies": 0, "response_anomalies": 0, "anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0, "anomalies": 0}
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
                    
                    # Count original anomalies (based on is_anomalous flag)
                    original_anomalies_count = 0
                    for log in logs:
                        if log.get('is_anomalous', False):
                            original_anomalies_count += 1
                    
                    # Update logs with response_anomaly flags and save to combine_logs directory
                    add_response_anomaly_to_logs(file_path, result_df, combine_logs_dir, api_name)
                    
                    if not result_df.empty:
                        # Count response time anomalies
                        response_anomalies = result_df[result_df['predicted_anomaly'] == True]
                        normal = result_df[result_df['predicted_anomaly'] == False]
                        
                        api_stats[api_name]["total_logs"] = len(result_df)
                        api_stats[api_name]["original_anomalies"] = original_anomalies_count
                        api_stats[api_name]["response_anomalies"] = len(response_anomalies)
    
    # Set active issues to original anomalies count (is_anomalous flag)
                        api_stats[api_name]["anomalies"] = original_anomalies_count
                          # Make sure 'anomalies' is set correctly
                        
                        # Calculate API-specific anomaly percentage (based on original anomalies)
                        if len(result_df) > 0:
                            api_stats[api_name]["anomaly_percent"] = round(original_anomalies_count / len(result_df) * 100, 1)
                        
                        # Calculate API-specific response time averages
                        if len(normal) > 0:
                            api_stats[api_name]["normal_avg"] = round(normal['time_ms'].mean(), 2)
                        
                        if len(response_anomalies) > 0:
                            api_stats[api_name]["anomalous_avg"] = round(response_anomalies['time_ms'].mean(), 2)
                        
                        # Update overall stats
                        stats["total_logs"] += len(result_df)
                        stats["original_anomalies"] += original_anomalies_count
                        stats["response_anomalies"] += len(response_anomalies)
                        stats["normal_logs"] += len(normal)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Calculate overall anomaly percentage (based on original anomalies)
    if stats["total_logs"] > 0:
        stats["anomaly_percent"] = round(stats["original_anomalies"] / stats["total_logs"] * 100, 1)
    
    # Calculate overall normal and anomalous response time averages
    total_normal_time = 0
    total_normal_count = 0
    total_anomalous_time = 0
    total_anomalous_count = 0

    for api_name in api_stats:
        # Add up normal response times
        if api_stats[api_name]["normal_avg"] > 0:
            normal_logs = api_stats[api_name]["total_logs"] - api_stats[api_name]["response_anomalies"]
            total_normal_time += api_stats[api_name]["normal_avg"] * normal_logs
            total_normal_count += normal_logs
        
        # Add up anomalous response times
        if api_stats[api_name]["anomalous_avg"] > 0:
            anomalous_logs = api_stats[api_name]["response_anomalies"]
            total_anomalous_time += api_stats[api_name]["anomalous_avg"] * anomalous_logs
            total_anomalous_count += anomalous_logs

    # Calculate averages for the entire system
    stats["normal_avg"] = round(total_normal_time / total_normal_count, 2) if total_normal_count > 0 else 0
    stats["anomalous_avg"] = round(total_anomalous_time / total_anomalous_count, 2) if total_anomalous_count > 0 else 0
    stats["anomalies"] = stats["response_anomalies"]  # Ensure 'anomalies' is set correctly for overall stats
    
    return stats, api_stats

class APIForecastingSystem:
    """Forecasting system for API performance based on recent logs"""
    
    def __init__(self, log_directory, time_window_minutes=None, interval_seconds=900):
        self.log_directory = log_directory
        self.time_window_minutes = time_window_minutes
        self.interval_seconds = interval_seconds
        self.api_names = ["auth", "search", "booking", "payment", "feedback"]
        
        # Storage for time series data
        self.time_series = {
            api: {'timestamps': [], 'error_rates': []}
            for api in self.api_names
        }
        # Add system-wide metrics
        self.time_series['system'] = {'timestamps': [], 'error_rates': []}
        
        # For forecast results
        self.forecasts = {
            api: {'future_timestamps': [], 'predicted_error_rates': []}
            for api in self.api_names + ['system']
        }

    def load_recent_logs(self):
        """Load and process logs from the last time_window_minutes"""
        now = datetime.now()
        if self.time_window_minutes is not None:
            cutoff_time = now - timedelta(minutes=self.time_window_minutes)
        else:
            # Set cutoff_time to a very old date to include all logs
            cutoff_time = datetime(1970, 1, 1)    
        
        # Reset time series data
        for api in self.api_names + ['system']:
            self.time_series[api] = {'timestamps': [], 'error_rates': []}
        
        # Process each API's logs
        all_logs = []
        for api_name in self.api_names:
            file_path = os.path.join(self.log_directory, f"{api_name}_interactions.json")
            if not os.path.exists(file_path):
                print(f"Warning: No log file found for {api_name}")
                continue
                
            logs = self._read_logs(file_path)
            if not logs:
                continue
                
            # Tag each log with its API type
            for log in logs:
                log['api_type'] = api_name
            
            all_logs.extend(logs)
        
        # Sort all logs by timestamp
        all_logs.sort(key=lambda x: x.get('timestamp', ''))
        
        # Filter to recent logs only
        recent_logs = []
        for log in all_logs:
            try:
                log_time = datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
                if log_time >= cutoff_time:
                    recent_logs.append(log)
            except (ValueError, TypeError):
                continue
        
        if recent_logs:
            first_log_time = datetime.fromisoformat(recent_logs[0].get('timestamp', '').replace('Z', '+00:00'))
            start_time = first_log_time - timedelta(
                seconds=first_log_time.timestamp() % self.interval_seconds
            )
        else:
            start_time = cutoff_time        
        
        # Process logs in time intervals
        self._process_logs_by_interval(recent_logs, start_time, now)
        
        return recent_logs
        
    def _read_logs(self, file_path):
        """Read logs from a file, handling both array and newline-delimited formats"""
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
                                pass
                    return logs
        except Exception as e:
            print(f"Error reading logs from {file_path}: {str(e)}")
            return []
            
    def _process_logs_by_interval(self, logs, start_time, end_time):
        """Process logs into time intervals"""
        # Create time buckets
        interval_delta = timedelta(seconds=self.interval_seconds)
        current_time = start_time
        time_buckets = []
        
        while current_time <= end_time:
            time_buckets.append({
                'start': current_time,
                'end': current_time + interval_delta,
                'logs': {api: [] for api in self.api_names},
                'all_logs': []
            })
            current_time += interval_delta
            
        # Assign logs to buckets
        for log in logs:
            try:
                log_time = datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
                api_type = log.get('api_type', 'unknown')
                
                for bucket in time_buckets:
                    if bucket['start'] <= log_time < bucket['end']:
                        if api_type in self.api_names:
                            bucket['logs'][api_type].append(log)
                        bucket['all_logs'].append(log)
                        break
            except (ValueError, TypeError):
                continue
                
        # Calculate metrics for each bucket
        for bucket in time_buckets:
            timestamp = bucket['start']
            
            # Add timestamp to all series
            for api in self.api_names + ['system']:
                self.time_series[api]['timestamps'].append(timestamp)
            
            # Calculate metrics for each API
            for api in self.api_names:
                api_logs = bucket['logs'][api]
                self._calculate_metrics(api, api_logs, timestamp)
            
            # Calculate system-wide metrics
            all_logs = bucket['all_logs']
            self._calculate_metrics('system', all_logs, timestamp)
    
    def _calculate_metrics(self, api_name, logs, timestamp):
        """Calculate error rate for a set of logs"""
        if not logs:
            # No logs in this interval - use 0 for error rate
            self.time_series[api_name]['error_rates'].append(0)
            return
            
        # More nuanced error detection
        total_requests = len(logs)
        
        # Detailed error classification
        def is_error_log(log):
            status_code = log.get('response', {}).get('status_code', 200)
            return (
                # Explicit server error status codes
                status_code in [500, 502, 503, 504] or 
                # Client error status codes that should count as errors
                (400 <= status_code < 600) or 
                # Explicit anomaly flag
                log.get('is_anomalous', False)
            )
        
        # Count errors
        error_count = sum(1 for log in logs if is_error_log(log))
        
        # Calculate error rate as a percentage
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
        
        # Store the error rate as a percentage
        self.time_series[api_name]['error_rates'].append(error_rate)
        
    def generate_forecasts(self, forecast_periods=6):
        """Generate forecasts based on recent trends"""
        now = datetime.now()
        future_timestamps = []
        
        # Generate future timestamps
        for i in range(1, forecast_periods + 1):
            future_timestamps.append(now + timedelta(seconds=i * self.interval_seconds))
        
        # For each API, generate forecasts
        for api in self.api_names + ['system']:
            self._forecast_api_metrics(api, future_timestamps, forecast_periods)
            
        return self.forecasts
    
    def _forecast_api_metrics(self, api_name, future_timestamps, forecast_periods):
        """Generate forecasts for a specific API"""
        # Get historical data
        error_rates = self.time_series[api_name]['error_rates']
        
        # Store future timestamps
        self.forecasts[api_name]['future_timestamps'] = future_timestamps
        
        # Error rate forecast - simple linear extrapolation
        if len(error_rates) > 1:
            # Get trend from the last few intervals
            num_points = min(5, len(error_rates))
            recent_errors = error_rates[-num_points:]
            
            if len(set(recent_errors)) > 1:  # Check if we have varying data
                # Calculate slope of recent trend
                x = np.arange(num_points)
                # Fit a line to the recent data
                slope, intercept = np.polyfit(x, recent_errors, 1)
                
                # Predict future values
                predictions = []
                for i in range(1, forecast_periods + 1):
                    predicted_value = slope * (num_points + i - 1) + intercept
                    # Ensure predictions stay between 0 and 100 percent
                    predicted_value = max(0, min(100, predicted_value))
                    predictions.append(predicted_value)
                
                self.forecasts[api_name]['predicted_error_rates'] = predictions
            else:
                # If all recent values are the same, predict the same value continuing
                self.forecasts[api_name]['predicted_error_rates'] = [recent_errors[-1]] * forecast_periods
        else:
            # Not enough data, use last known value or 0
            last_error_rate = error_rates[-1] if error_rates else 0
            self.forecasts[api_name]['predicted_error_rates'] = [last_error_rate] * forecast_periods
    
    def get_visualization_data(self):
        """Prepare data for visualization"""
        result = {
            'metadata': {
                'error_rate_unit': 'percent'  # Add this to indicate error rates are percentages
            },
            'historical': {
                api: {
                    'timestamps': [t.strftime("%H:%M:%S") for t in self.time_series[api]['timestamps']],
                    'error_rates': self.time_series[api]['error_rates']
                } for api in self.api_names + ['system']
            },
            'forecast': {
                api: {
                    'timestamps': [t.strftime("%H:%M:%S") for t in self.forecasts[api]['future_timestamps']],
                    'error_rates': self.forecasts[api]['predicted_error_rates']
                } for api in self.api_names + ['system']
            }
        }
        return result
@app.route('/alerts')
def alerts():
    """
    Display detailed alerts page with filtering and management options
    """
    # Process log files and get statistics with spike detection
    combined_stats = process_log_files_with_spikes()
    
    # Get all alerts
    alerts = combined_stats.get('alerts', [])
    
    # Count critical alerts
    critical_count = sum(1 for alert in alerts if alert['severity'] == 'CRITICAL')
    
    # Pass all required variables to the template
    return render_template('alerts.html', 
                          alerts=alerts,
                          critical_count=critical_count,
                          stats=combined_stats['overall'], 
                          api_stats=combined_stats['api_stats'])
@app.route('/')
def dashboard():
    # Process log files and get statistics with spike detection
    combined_stats = process_log_files_with_spikes()
    overall_stats = combined_stats['overall']
    
    # Pass all required variables to the template including forecast_data
    return render_template('dashboard.html', 
                          stats=overall_stats, 
                          api_stats=combined_stats['api_stats'],
                          original_anomaly_count=overall_stats.get('original_anomalies', 0),
                          response_anomaly_count=overall_stats.get('response_anomalies', 0),
                          spikes=combined_stats.get('spikes', []),
                          api_spikes=combined_stats.get('api_spikes', {}),
                          alerts=combined_stats.get('alerts', []),
                          has_alerts=combined_stats.get('has_alerts', False),
                          forecast_data=combined_stats.get('forecast_data', None))

@app.route('/api/error_rates')
def error_rates():
    """API endpoint to provide error rate data for charts using real forecast system"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        travel_app_dir = os.path.dirname(current_dir)  # Go up one level to reach travel_app
        
        # Initialize forecasting system
        forecasting_system = APIForecastingSystem(
            log_directory=travel_app_dir, 
            time_window_minutes=None,  # Use all available logs
            interval_seconds=900  # 15-minute intervals
        )
        
        # Load recent logs and generate forecasts
        forecasting_system.load_recent_logs()
        forecasting_system.generate_forecasts(forecast_periods=6)
        
        # Get visualization data
        visualization_data = forecasting_system.get_visualization_data()
        
        return jsonify(visualization_data)
    except Exception as e:
        import traceback
        print(f"Error generating forecast data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def generate_mock_data():
    """Generate mock time series data for charts"""
    import random
    
    # Create a list of timestamps (last 24 hours in hourly intervals)
    timestamps = []
    error_rates = []
    
    # Generate 24 hourly timestamps with random error rates
    for i in range(24):
        hour = str(i).zfill(2)
        timestamps.append(f"{hour}:00:00")
        
        # Generate a random error rate between 0-30%
        error_rate = round(random.random() * 30, 2)
        error_rates.append(error_rate)
    
    return {
        'timestamps': timestamps,
        'error_rates': error_rates
    }
@app.context_processor
def inject_alerts_count():
    """
    Inject the alerts count into all templates
    """
    try:
        # Get alert count from the latest data
        combined_stats = process_log_files_with_spikes()
        alerts = combined_stats.get('alerts', [])
        return {'alerts_count': len(alerts)}
    except Exception as e:
        print(f"Error getting alerts count for context processor: {str(e)}")
        return {'alerts_count': 0}
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
    # Change from response_anomalies to original_anomalies
    active_issues = combined_stats['overall'].get('original_anomalies', 0)
    return render_template('api_health.html', 
                          api_name="System", 
                          stats=combined_stats['overall'],
                          active_issues=active_issues,
                          kibana_dashboard_id="system-overview-dashboard")

@app.route('/api/auth')
def auth_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    stats = combined_stats['api_stats']['auth']
    
    # Add debug print
    print(f"DEBUG - Auth API route - original_anomalies: {stats.get('original_anomalies', 0)}")
    
    # This is already correct, using original_anomalies
    active_issues = stats.get('original_anomalies', 0)
    print(f"DEBUG - Auth API - active_issues to be displayed: {active_issues}")
    
    return render_template('api_health.html', 
                          api_name="Authentication", 
                          stats=stats,
                          active_issues=active_issues,
                          kibana_dashboard_id="auth-api-dashboard")

@app.route('/api/search')
def search_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    # Change from response_anomalies to original_anomalies
    active_issues = combined_stats['api_stats']['search'].get('original_anomalies', 0)
    return render_template('api_health.html', 
                          api_name="Search", 
                          stats=combined_stats['api_stats']['search'],
                          active_issues=active_issues,
                          kibana_dashboard_id="search-api-dashboard")

@app.route('/api/booking')
def booking_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    # Change from response_anomalies to original_anomalies
    active_issues = combined_stats['api_stats']['booking'].get('original_anomalies', 0)
    return render_template('api_health.html', 
                          api_name="Booking", 
                          stats=combined_stats['api_stats']['booking'],
                          active_issues=active_issues,
                          kibana_dashboard_id="booking-api-dashboard")
@app.route('/api/payment')
def payment_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    # Change from response_anomalies to original_anomalies
    active_issues = combined_stats['api_stats']['payment'].get('original_anomalies', 0)
    return render_template('api_health.html', 
                          api_name="Payment", 
                          stats=combined_stats['api_stats']['payment'],
                          active_issues=active_issues,
                          kibana_dashboard_id="payment-api-dashboard")

@app.route('/api/feedback')
def feedback_api():
    # Process log files and get statistics
    combined_stats = process_log_files_with_spikes()
    # Change from response_anomalies to original_anomalies
    active_issues = combined_stats['api_stats']['feedback'].get('original_anomalies', 0)
    return render_template('api_health.html', 
                          api_name="Feedback", 
                          stats=combined_stats['api_stats']['feedback'],
                          active_issues=active_issues,
                          kibana_dashboard_id="feedback-api-dashboard")

if __name__ == '__main__':
    app.run(debug=True, port=5050)