# spike_detector.py
# Place this file in your final_dashboard directory

import json
import os
import datetime
import numpy as np
from collections import defaultdict

class SpikeDetector:
    """
    Simple spike detector for API response times
    """
    
    def __init__(self, threshold_multiplier=3.0, min_response_time=1000):
        """
        Initialize the spike detector
        
        Args:
            threshold_multiplier: Multiple of average response time to consider a spike
            min_response_time: Minimum response time (ms) to be considered a spike
        """
        self.threshold_multiplier = threshold_multiplier
        self.min_response_time = min_response_time
        
    def load_logs(self, file_path):
        """Load logs from a JSON file"""
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
    
    def detect_spikes(self, logs):
        """
        Detect response time spikes in logs
        
        Args:
            logs: List of log entries
            
        Returns:
            Dictionary with spike information
        """
        if not logs:
            return {"spikes": [], "normal_avg": 0, "spike_threshold": 0}
        
        # Extract response times
        response_times = []
        response_data = []
        
        for log in logs:
            if 'response' in log and 'time_ms' in log['response']:
                time_ms = log['response']['time_ms']
                response_times.append(time_ms)
                
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
                
                # Get operation type
                operation = None
                if api_type == 'auth' and 'operation' in log and 'type' in log['operation']:
                    operation = log['operation']['type']
                elif api_type == 'search' and 'search_service' in log and 'type' in log['search_service']:
                    operation = log['search_service']['type']
                elif api_type == 'booking' and 'booking_service' in log and 'operation' in log['booking_service']:
                    operation = log['booking_service']['operation']
                elif api_type == 'payment' and 'payment_service' in log and 'operation' in log['payment_service']:
                    operation = log['payment_service']['operation']
                elif api_type == 'feedback' and 'feedback_service' in log and 'operation' in log['feedback_service']:
                    operation = log['feedback_service']['operation']
                
                response_data.append({
                    "timestamp": log.get("timestamp"),
                    "response_time": time_ms,
                    "api_type": api_type,
                    "operation": operation,
                    "status_code": log['response']['status_code'],
                    "log": log
                })
        
        if not response_times:
            return {"spikes": [], "normal_avg": 0, "spike_threshold": 0}
        
        # Calculate average response time (excluding top 10% to reduce outlier impact)
        sorted_times = sorted(response_times)
        num_to_exclude = max(1, int(len(sorted_times) * 0.1))
        normal_times = sorted_times[:-num_to_exclude] if len(sorted_times) > 10 else sorted_times
        
        normal_avg = np.mean(normal_times)
        spike_threshold = max(self.min_response_time, normal_avg * self.threshold_multiplier)
        
        # Identify spikes
        spikes = []
        for data in response_data:
            if data["response_time"] >= spike_threshold:
                # Calculate how many times higher than average
                times_avg = data["response_time"] / normal_avg if normal_avg > 0 else 0
                
                # Add spike information
                spikes.append({
                    "timestamp": data["timestamp"],
                    "response_time": data["response_time"],
                    "api_type": data["api_type"],
                    "operation": data["operation"],
                    "status_code": data["status_code"],
                    "threshold": spike_threshold,
                    "times_avg": times_avg,
                    "deviation": data["response_time"] - normal_avg
                })
        
        # Sort spikes by time (most recent first)
        spikes.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "spikes": spikes,
            "normal_avg": normal_avg,
            "spike_threshold": spike_threshold
        }
    
    def detect_all_api_spikes(self, travel_app_dir):
        """
        Detect spikes in all API log files
        
        Args:
            travel_app_dir: Path to the travel_app directory with log files
            
        Returns:
            Dictionary with spike information for all APIs
        """
        # Define log files to process
        log_files = {
            "auth": os.path.join(travel_app_dir, "auth_interactions.json"),
            "search": os.path.join(travel_app_dir, "search_interactions.json"),
            "booking": os.path.join(travel_app_dir, "booking_interactions.json"),
            "payment": os.path.join(travel_app_dir, "payment_interactions.json"),
            "feedback": os.path.join(travel_app_dir, "feedback_interactions.json")
        }
        
        # Process each API
        result = {
            "all_spikes": [],
            "api_spikes": {},
            "api_stats": {}
        }
        
        for api_name, file_path in log_files.items():
            if os.path.exists(file_path):
                # Load logs
                logs = self.load_logs(file_path)
                
                if logs:
                    # Detect spikes
                    spike_data = self.detect_spikes(logs)
                    
                    # Store results
                    result["api_stats"][api_name] = {
                        "normal_avg": spike_data["normal_avg"],
                        "spike_threshold": spike_data["spike_threshold"],
                        "total_logs": len(logs),
                        "spike_count": len(spike_data["spikes"])
                    }
                    
                    # Add API type to each spike
                    for spike in spike_data["spikes"]:
                        spike["api"] = api_name
                    
                    # Store spikes by API
                    result["api_spikes"][api_name] = spike_data["spikes"]
                    
                    # Add to all spikes
                    result["all_spikes"].extend(spike_data["spikes"])
            
        # Sort all spikes by response time (highest first)
        result["all_spikes"].sort(key=lambda x: x["response_time"], reverse=True)
        
        # Calculate overall stats
        all_spikes = result["all_spikes"]
        result["total_spike_count"] = len(all_spikes)
        result["has_spikes"] = len(all_spikes) > 0
        
        if all_spikes:
            result["highest_spike"] = all_spikes[0]["response_time"]
            result["latest_spike"] = sorted(all_spikes, key=lambda x: x["timestamp"], reverse=True)[0]
        else:
            result["highest_spike"] = 0
            result["latest_spike"] = None
        
        return result
    
    def generate_alerts(self, spike_data):
        """
        Generate alerts from spike data
        
        Args:
            spike_data: Dictionary with spike information
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Process all spikes
        for spike in spike_data["all_spikes"]:
            # Determine severity based on how many times above threshold
            times_avg = spike["times_avg"]
            
            if times_avg >= 5.0:
                severity = "CRITICAL"
            elif times_avg >= 3.0:
                severity = "HIGH"
            else:
                severity = "MEDIUM"
            
            # Format timestamp for display
            try:
                dt = datetime.datetime.fromisoformat(spike["timestamp"].replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = spike["timestamp"]
            
            # Create alert
            alert = {
                "severity": severity,
                "api": spike["api"].upper(),
                "message": f"{severity}: {spike['api'].upper()} API response time spike detected",
                "details": f"Response time of {spike['response_time']:.0f}ms is {times_avg:.1f}x higher than normal",
                "response_time": spike["response_time"],
                "threshold": spike["threshold"],
                "timestamp": formatted_time,
                "times_normal": times_avg
            }
            
            alerts.append(alert)
        
        # Sort alerts by severity and time
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["response_time"]))
        
        return alerts