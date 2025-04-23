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
                'is_anomalous': log.get('is_anomalous', False),  # For issue count & system health
                'response_anomaly': log.get('response_anomaly', False)  # For response time analysis
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

def process_log_files():
    """
    Process log files from combine_logs directory to get statistics
    
    Returns:
        Dictionary with statistics for each API
    """
    # Define the log files to process - check both parent and local combine_logs directories
    log_files = {
        "auth": "auth_interactions_with_anomalies.json",
        "search": "search_interactions_with_anomalies.json",
        "booking": "booking_interactions_with_anomalies.json",
        "payment": "payment_interactions_with_anomalies.json",
        "feedback": "feedback_interactions_with_anomalies.json"
    }
    
    # Get the current directory and possible combine_logs directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First try the combine_logs directory in the current directory
    local_combine_logs = os.path.join(current_dir, "combine_logs")
    
    # Then try the parent combine_logs directory
    parent_dir = os.path.dirname(current_dir)
    parent_combine_logs = os.path.join(parent_dir, "combine_logs")
    
    # Print debug info
    print(f"Checking local combine_logs: {local_combine_logs}")
    print(f"Checking parent combine_logs: {parent_combine_logs}")
    
    # Determine which combine_logs to use - prefer parent as it has the actual data
    combine_logs_dir = parent_combine_logs if os.path.exists(parent_combine_logs) else local_combine_logs
    
    # Final log directory
    print(f"Using combine_logs directory: {combine_logs_dir}")
    
    # Verify the files exist
    for api_name, file_name in log_files.items():
        file_path = os.path.join(combine_logs_dir, file_name)
        if os.path.exists(file_path):
            print(f"Found {api_name} log file: {file_path}")
        else:
            print(f"WARNING: File not found: {file_path}")
    
    # Gather overall statistics
    stats = {
        "total_logs": 0,
        "anomalies": 0,  # Based on is_anomalous flag
        "anomaly_percent": 0,  # Based on is_anomalous flag
        "rt_anomalies": 0,  # Based on response_anomaly
        "rt_anomaly_percent": 0,  # Based on response_anomaly
        "normal_logs": 0,
        "normal_avg": 0,
        "anomalous_avg": 0,
        "updated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Prepare stats for each API
    api_stats = {
        "auth": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "rt_anomalies": 0, "rt_anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "search": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "rt_anomalies": 0, "rt_anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "booking": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "rt_anomalies": 0, "rt_anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "payment": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "rt_anomalies": 0, "rt_anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0},
        "feedback": {"total_logs": 0, "anomalies": 0, "anomaly_percent": 0, "rt_anomalies": 0, "rt_anomaly_percent": 0, "normal_avg": 0, "anomalous_avg": 0}
    }
    
    # Process each log file
    for api_name, log_file in log_files.items():
        file_path = os.path.join(combine_logs_dir, log_file)
        
        if os.path.exists(file_path):
            print(f"Processing {api_name} logs from {file_path}")
            try:
                # Load logs
                detector = ResponseTimeAnomalyDetector()
                logs = detector.load_logs(file_path)
                print(f"Loaded {len(logs)} log entries for {api_name}")
                
                if logs:
                    # Process logs
                    features_df = detector.preprocess_logs(logs)
                    
                    if not features_df.empty:
                        # Count logs
                        total_logs = len(features_df)
                        
                        # Count is_anomalous logs for issues and system health
                        anomalous_logs = features_df[features_df['is_anomalous'] == True]
                        num_anomalies = len(anomalous_logs)
                        
                        # Count response_anomaly logs for response time analysis
                        rt_anomalous_logs = features_df[features_df['response_anomaly'] == True]
                        rt_normal_logs = features_df[features_df['response_anomaly'] == False]
                        
                        num_rt_anomalies = len(rt_anomalous_logs)
                        num_rt_normal = len(rt_normal_logs)
                        
                        # Calculate statistics
                        api_stats[api_name]["total_logs"] = total_logs
                        
                        # is_anomalous stats (for issues and system health)
                        api_stats[api_name]["anomalies"] = num_anomalies
                        if total_logs > 0:
                            api_stats[api_name]["anomaly_percent"] = round((num_anomalies / total_logs) * 100, 1)
                        
                        # response_anomaly stats (for response time)
                        api_stats[api_name]["rt_anomalies"] = num_rt_anomalies
                        if total_logs > 0:
                            api_stats[api_name]["rt_anomaly_percent"] = round((num_rt_anomalies / total_logs) * 100, 1)
                        
                        # Response time averages (using response_anomaly)
                        if num_rt_normal > 0:
                            api_stats[api_name]["normal_avg"] = round(rt_normal_logs['time_ms'].mean(), 2)
                        
                        if num_rt_anomalies > 0:
                            api_stats[api_name]["anomalous_avg"] = round(rt_anomalous_logs['time_ms'].mean(), 2)
                        
                        # Update overall statistics
                        stats["total_logs"] += total_logs
                        stats["anomalies"] += num_anomalies  # Using is_anomalous
                        stats["rt_anomalies"] += num_rt_anomalies  # Using response_anomaly
                        stats["normal_logs"] += num_rt_normal
                        
                        # Update weighted averages (using response_anomaly)
                        if num_rt_normal > 0:
                            normal_avg = rt_normal_logs['time_ms'].mean()
                            normal_count = stats["normal_logs"]
                            if normal_count > 0:
                                stats["normal_avg"] = round(
                                    ((stats["normal_avg"] * (normal_count - num_rt_normal)) + 
                                     (normal_avg * num_rt_normal)) / normal_count, 2
                                )
                            else:
                                stats["normal_avg"] = round(normal_avg, 2)
                        
                        if num_rt_anomalies > 0:
                            anomalous_avg = rt_anomalous_logs['time_ms'].mean()
                            anomaly_count = stats["rt_anomalies"]
                            if anomaly_count > 0:
                                stats["anomalous_avg"] = round(
                                    ((stats["anomalous_avg"] * (anomaly_count - num_rt_anomalies)) + 
                                     (anomalous_avg * num_rt_anomalies)) / anomaly_count, 2
                                )
                            else:
                                stats["anomalous_avg"] = round(anomalous_avg, 2)
                        
                        print(f"Processed {api_name} logs: {total_logs} total, {num_anomalies} is_anomalous, {num_rt_anomalies} response_anomaly")
                    else:
                        print(f"No valid data found in {api_name} logs")
                else:
                    print(f"No logs found in {api_name} file")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {file_path}")
    
    # Calculate overall anomaly percentages
    if stats["total_logs"] > 0:
        stats["anomaly_percent"] = round((stats["anomalies"] / stats["total_logs"]) * 100, 1)  # Based on is_anomalous
        stats["rt_anomaly_percent"] = round((stats["rt_anomalies"] / stats["total_logs"]) * 100, 1)  # Based on response_anomaly
    
    # Print overall stats for debugging
    print("\n==== Overall Statistics ====")
    print(f"Total logs: {stats['total_logs']}")
    print(f"is_anomalous: {stats['anomalies']} ({stats['anomaly_percent']}%)")
    print(f"response_anomaly: {stats['rt_anomalies']} ({stats['rt_anomaly_percent']}%)")
    print(f"Normal logs: {stats['normal_logs']}")
    print(f"Normal average response time: {stats['normal_avg']} ms")
    print(f"Anomalous average response time: {stats['anomalous_avg']} ms")
    
    return stats, api_stats

# Add a utility function to prepare dashboard data for templates
def prepare_dashboard_data():
    """Process logs and prepare data for dashboard display"""
    # Process log files and get statistics
    stats, api_stats = process_log_files()
    
    # Calculate UI values
    dashboard_data = {
        "total_apis": stats["total_logs"],
        "active_issues": stats["anomalies"],  # Based on is_anomalous
        "avg_response": stats["anomalous_avg"],  # From response_anomaly
        "system_health": 100 - stats["anomaly_percent"] if stats["total_logs"] > 0 else 100,  # Based on is_anomalous
        "api_stats": api_stats,
        "stats": stats
    }
    
    # Print stats being sent to template (for debugging)
    print("\n==== Dashboard Data ====")
    print(f"Total Logs (APIs): {dashboard_data['total_apis']}")
    print(f"Active Issues (is_anomalous): {dashboard_data['active_issues']}")
    print(f"Avg Response (response_anomaly): {dashboard_data['avg_response']}ms")
    print(f"System Health (based on is_anomalous): {dashboard_data['system_health']}%")
    
    return dashboard_data

@app.route('/')
def dashboard():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('dashboard.html', 
                           title='API Monitoring Dashboard', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/system')
def system():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('sys.html', 
                           title='System Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/auth')
def auth():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('auth.html', 
                           title='Authentication API Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/search')
def search_dashboard():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('srch.html', 
                           title='Search API Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/booking')
def booking_dashboard():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('book.html', 
                           title='Booking API Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/payment')
def payment_dashboard():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('pay.html', 
                           title='Payment API Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

@app.route('/feedback')
def feedback_dashboard():
    # Get dashboard data
    data = prepare_dashboard_data()
    
    return render_template('fedbk.html', 
                           title='Feedback API Health | API Monitoring', 
                           stats=data["stats"], 
                           api_stats=data["api_stats"],
                           dash_data=data)

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
    
    # Otherwise, redirect to the dashboard
    return redirect(url_for('dashboard'))

@app.route('/api/dashboard-data')
def dashboard_data():
    # Process log files and get updated statistics
    stats, api_stats = process_log_files()
    
    # Calculate general metrics based on your definitions
    total_apis = stats["total_logs"]  # Total logs across all APIs
    active_issues = stats["anomalies"]  # Using is_anomalous flag
    avg_response = stats["anomalous_avg"]  # Average time_ms of response_anomaly logs
    
    # Calculate system health (100% - percentage of is_anomalous)
    system_health = 100 - stats["anomaly_percent"] if stats["total_logs"] > 0 else 100
    
    # Calculate increase/decrease trends
    prev_anomalies = active_issues - 2 if active_issues > 2 else 0  # Simulated previous value
    anomaly_change = active_issues - prev_anomalies
    
    # Create data for the dashboard UI
    data = {
        "total_apis": total_apis,
        "total_apis_note": f"+{total_apis // 10} this month",  # Simulated monthly increase
        "active_issues": active_issues,
        "active_issues_note": f"{'+' if anomaly_change >= 0 else ''}{anomaly_change} from yesterday",
        "avg_response": round(avg_response, 2),
        "avg_response_note": f"{round(avg_response * 0.1, 1)}ms {'improvement' if avg_response < 300 else 'slower'}",
        "system_health": round(system_health, 1),
        "system_health_note": f"{'-' if stats['anomaly_percent'] > 0 else '+'}{min(stats['anomaly_percent'], 10)}% this week",
        
        # API specific statuses with response times (using anomalous avg from response_anomaly)
        "auth_status": api_stats["auth"]["anomalous_avg"] if api_stats["auth"]["anomalous_avg"] > 0 else 145,
        "payment_status": api_stats["payment"]["anomalous_avg"] if api_stats["payment"]["anomalous_avg"] > 0 else 235,
        "user_status": 125,  # Not tracked directly in our stats
        "order_status": 410,  # Not tracked directly in our stats
        
        # Create alerts based on actual anomalies (using is_anomalous flag for issues)
        "alerts": [
            {
                "message": get_alert_message(stats, api_stats, 0),
                "time": "5 minutes ago"
            },
            {
                "message": get_alert_message(stats, api_stats, 1),
                "time": "15 minutes ago"
            }
        ]
    }
    
    return jsonify(data)

def get_alert_message(stats, api_stats, index):
    """Generate meaningful alert messages based on anomaly data"""
    # List of possible alert messages (based on is_anomalous)
    alert_messages = []
    
    # Add alerts based on specific API anomalies
    if api_stats["payment"]["anomalies"] > 0:
        alert_messages.append(f"Payment API Issues Detected: {api_stats['payment']['anomalies']} anomalies")
    
    if api_stats["auth"]["anomalies"] > 0:
        alert_messages.append(f"Authentication API Issues: {api_stats['auth']['anomalies']} anomalies")
    
    if api_stats["search"]["anomalies"] > 0:
        alert_messages.append(f"Search API Issues: {api_stats['search']['anomalies']} anomalies")
    
    if api_stats["booking"]["anomalies"] > 0:
        alert_messages.append(f"Booking API Issues: {api_stats['booking']['anomalies']} anomalies")
    
    if api_stats["feedback"]["anomalies"] > 0:
        alert_messages.append(f"Feedback API Issues: {api_stats['feedback']['anomalies']} anomalies")
    
    # Add a general system alert if anomaly percentage is high
    if stats["anomaly_percent"] > 20:
        alert_messages.append(f"High System-wide Anomaly Rate: {stats['anomaly_percent']}%")
    
    # Return appropriate message based on index
    if len(alert_messages) > index:
        return alert_messages[index]
    elif index == 0 and not alert_messages:
        return "No critical issues detected"
    else:
        return "All systems operational"

if __name__ == '__main__':
    app.run(debug=True, port=5050)