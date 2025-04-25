import json
import os
import time
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

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
        now = datetime.datetime.now()
        if self.time_window_minutes is not None:
            cutoff_time = now - datetime.timedelta(minutes=self.time_window_minutes)
        else:
        # Set cutoff_time to a very old date to include all logs
            cutoff_time = datetime.datetime(1970, 1, 1)    
        
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
                log_time = datetime.datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
                if log_time >= cutoff_time:
                    recent_logs.append(log)
            except (ValueError, TypeError):
                continue
        
        if recent_logs:
            first_log_time = datetime.datetime.fromisoformat(recent_logs[0].get('timestamp', '').replace('Z', '+00:00'))
            start_time = first_log_time - datetime.timedelta(
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
        interval_delta = datetime.timedelta(seconds=self.interval_seconds)
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
                log_time = datetime.datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
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
                # Client error status codes
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
        
        # Debug logging
        print(f"API: {api_name}")
        print(f"Total Requests: {total_requests}")
        print(f"Error Count: {error_count}")
        print(f"Error Rate: {error_rate:.2f}%")
        print("Log Details:")
        for log in logs:
            status_code = log.get('response', {}).get('status_code', 200)
            is_anomalous = log.get('is_anomalous', False)
            print(f"  - Status Code: {status_code}, Anomalous: {is_anomalous}")
    
    def generate_forecasts(self, forecast_periods=6):
        """Generate forecasts based on recent trends"""
        now = datetime.datetime.now()
        future_timestamps = []
        
        # Generate future timestamps
        for i in range(1, forecast_periods + 1):
            future_timestamps.append(now + datetime.timedelta(seconds=i * self.interval_seconds))
        
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

@app.route('/')
def index():
    return render_template('forecasting.html')

@app.route('/forecast_data')
def forecast_data():
    """Generate forecast data for API performance"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(_file_))
        
        # Initialize forecasting system
        forecasting_system = APIForecastingSystem(current_dir, 
                                                 time_window_minutes=None, 
                                                 interval_seconds=900)
        
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

if __name__ == '__main__':
    # Set template folder to the current directory
    app.template_folder = os.path.dirname(os.path.abspath(_file_))
    
    app.run(debug=True, port=5050)