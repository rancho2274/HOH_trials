import os
import sys
from flask import Flask, render_template

# Add the travel_app directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
travel_app_dir = os.path.dirname(current_dir)  # Go up one level to reach travel_app
sys.path.append(travel_app_dir)

# Now import the ResponseTimeAnomalyDetector
from f_rt_anomaly import ResponseTimeAnomalyDetector

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Create an instance of the anomaly detector
    detector = ResponseTimeAnomalyDetector()
    
    # Define the log file patterns to process
    log_files = [
        "auth_interactions.json",
        "booking_interactions.json",
        "feedback_interactions.json",
        "payment_interactions.json",
        "search_interactions.json"
    ]
    
    # Gather statistics from all log files
    stats = {
        "total_logs": 0,
        "anomalies": 0,
        "anomaly_percent": 0,
        "normal_logs": 0,
        "normal_avg": 0,
        "anomalous_avg": 0
    }
    
    # Process each log file
    for log_file in log_files:
        file_path = os.path.join(travel_app_dir, log_file)
        if os.path.exists(file_path):
            try:
                # Load logs
                logs = detector.load_logs(file_path)
                if logs:
                    # Process logs and get results
                    result_df = detector.detect_anomalies(logs)
                    
                    # Count anomalies
                    anomalies = result_df[result_df['predicted_anomaly'] == True]
                    normal = result_df[result_df['predicted_anomaly'] == False]
                    
                    # Update statistics
                    stats["total_logs"] += len(result_df)
                    stats["anomalies"] += len(anomalies)
                    stats["normal_logs"] += len(normal)
                    
                    # Update response time averages (weighted by log count)
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
    
    # Calculate anomaly percentage
    if stats["total_logs"] > 0:
        stats["anomaly_percent"] = round(stats["anomalies"] / stats["total_logs"] * 100, 1)
    
    return render_template('dashboard.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True, port=5050)