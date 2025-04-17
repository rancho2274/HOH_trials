import json
import pandas as pd
import requests
import time
import uuid
from datetime import datetime
import socket
import sys

class APILogCollector:
    def __init__(self):
        self.logs = []
        self.api_keys = {}  # Stores API keys in memory

    def add_api_key(self, service_name, api_key):
        """Add an API key for a specific service"""
        self.api_keys[service_name] = api_key
        print(f"API key for {service_name} added and stored in memory.")

    def fetch_api_logs(self, endpoint, service_name=None, params=None, headers=None, method="GET", body=None):
        """
        Fetch API logs from an endpoint
        
        Parameters:
        - endpoint: URL to call
        - service_name: Name of the service (used for API key lookup)
        - params: Query parameters
        - headers: HTTP headers
        - method: HTTP method (GET, POST, etc.)
        - body: Request body data
        
        Returns:
        - log_entry: The collected log data
        - response: The response from the API call
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        # Initialize headers if None
        if headers is None:
            headers = {}
            
        # Add default user agent if not present
        if 'User-Agent' not in headers:
            headers['User-Agent'] = f"API-Log-Collector/{sys.version.split()[0]}"
        
        # Add API key to headers if available for this service
        if service_name and service_name in self.api_keys:
            headers['Authorization'] = f"Bearer {self.api_keys[service_name]}"
            
        # Add correlation ID to headers for tracing
        headers['X-Correlation-ID'] = correlation_id
        
        try:
            # Get client IP address
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))  # Connect to Google DNS to get outgoing IP
                client_ip = s.getsockname()[0]
                s.close()
            except:
                client_ip = "127.0.0.1"
            
            # Record request details
            request_time = datetime.now().isoformat()
            
            # Make the API request based on the HTTP method
            if method.upper() == "GET":
                response = requests.get(endpoint, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(endpoint, params=params, headers=headers, json=body)
            elif method.upper() == "PUT":
                response = requests.put(endpoint, params=params, headers=headers, json=body)
            elif method.upper() == "DELETE":
                response = requests.delete(endpoint, params=params, headers=headers, json=body)
            elif method.upper() == "PATCH":
                response = requests.patch(endpoint, params=params, headers=headers, json=body)
            else:
                response = requests.request(method, endpoint, params=params, headers=headers, json=body)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Create comprehensive log entry
            log_entry = {
                "timestamp": request_time,
                "http_method": method.upper(),
                "endpoint_path": endpoint,
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "ip_address": client_ip,
                "user_agent": headers.get("User-Agent", "unknown"),
                "request_headers": headers,
                "request_params": params,
                "request_body": body,
                "response_headers": dict(response.headers),
                "response_size": len(response.content),
                "response_body_preview": response.text[:200] if response.text else None,
                "error_message": None,
                "correlation_id": correlation_id,
                "service_name": service_name,
                "environment": "Python Standalone",
                "is_success": 200 <= response.status_code < 300
            }
            
            # Store the log
            self.logs.append(log_entry)
            
            return log_entry, response
                
        except Exception as e:
            # Create error log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "http_method": method.upper(),
                "endpoint_path": endpoint,
                "status_code": 0,  # 0 indicates connection error
                "response_time": time.time() - start_time,
                "ip_address": client_ip if 'client_ip' in locals() else "127.0.0.1",
                "user_agent": headers.get("User-Agent", "unknown"),
                "request_headers": headers,
                "request_params": params,
                "request_body": body,
                "response_headers": None,
                "response_size": 0,
                "error_message": str(e),
                "correlation_id": correlation_id,
                "service_name": service_name,
                "environment": "Python Standalone",
                "is_success": False
            }
            
            # Store the log
            self.logs.append(log_entry)
            
            return log_entry, None

    def get_logs_dataframe(self):
        """Convert logs to a pandas DataFrame for analysis"""
        if not self.logs:
            return pd.DataFrame()
            
        # Create a DataFrame but handle nested structures
        df = pd.DataFrame(self.logs)
        
        # Convert headers and params to string to make them displayable
        if 'request_headers' in df.columns:
            df['request_headers'] = df['request_headers'].apply(lambda x: json.dumps(x, sort_keys=True) if x else None)
        if 'request_params' in df.columns:
            df['request_params'] = df['request_params'].apply(lambda x: json.dumps(x, sort_keys=True) if x else None)
        if 'response_headers' in df.columns:
            df['response_headers'] = df['response_headers'].apply(lambda x: json.dumps(x, sort_keys=True) if x else None)
        
        return df

    def save_logs(self, filename="api_logs.json"):
        """Save logs to a JSON file"""
        if not self.logs:
            print("No logs to save.")
            return None
            
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to {filename}")
        return filename

    def display_logs(self, n=5):
        """
        Display the most recent n logs
        
        Parameters:
        - n: Number of logs to display
        """
        if not self.logs:
            print("No logs collected yet.")
            return
        
        df = self.get_logs_dataframe().tail(n)
        
        # Display a subset of the most important columns
        columns = [
            'timestamp', 'service_name', 'http_method', 'endpoint_path', 
            'status_code', 'response_time', 'response_size', 
            'correlation_id', 'error_message'
        ]
        # Only include columns that exist in the dataframe
        display_cols = [col for col in columns if col in df.columns]
        
        # Print the dataframe
        print(df[display_cols].to_string())

    def clear_logs(self):
        """Clear all collected logs"""
        self.logs = []
        print("Logs cleared.")

def fetch_multiple_logs(collector, endpoints_with_services, include_bodies=False):
    """
    Fetch logs from multiple endpoints
    
    Parameters:
    - collector: The APILogCollector instance
    - endpoints_with_services: List of tuples (endpoint, service_name, method)
    - include_bodies: Whether to include response bodies in the logs
    
    Returns:
    - DataFrame with the collected logs
    """
    for endpoint_info in endpoints_with_services:
        if len(endpoint_info) == 2:
            endpoint, service = endpoint_info
            method = "GET"
        else:
            endpoint, service, method = endpoint_info
            
        print(f"Fetching from {endpoint} ({service}) using {method}...")
        log_entry, response = collector.fetch_api_logs(endpoint, service_name=service, method=method)
        
        status_emoji = "✅" if log_entry['is_success'] else "❌"
        print(f"{status_emoji} Status: {log_entry['status_code']}, Response time: {log_entry['response_time']}s")
        
        if include_bodies and response:
            try:
                print("Response preview:")
                print(response.text[:150] + ("..." if len(response.text) > 150 else ""))
            except:
                pass
                
        time.sleep(1)  # Be nice to the APIs
    
    print(f"\nTotal logs collected: {len(collector.logs)}")
    return collector.get_logs_dataframe()

def main():
    # Initialize collector
    collector = APILogCollector()
    
    # Define sample public APIs that don't require authentication
    # Format: (endpoint, service_name, method)
    sample_endpoints = [
        ("https://cat-fact.herokuapp.com/facts", "cat-facts"),
        ("https://api.publicapis.org/entries", "public-apis"),
        ("https://www.boredapi.com/api/activity", "bored-api"),
        ("https://api.agify.io/?name=michael", "agify"),
        ("https://jsonplaceholder.typicode.com/posts/1", "jsonplaceholder", "GET"),
        ("https://jsonplaceholder.typicode.com/posts", "jsonplaceholder", "POST"),
        ("https://reqres.in/api/users", "reqres", "POST")
    ]
    
    # Fetch logs
    df = fetch_multiple_logs(collector, sample_endpoints)
    
    # Display collected logs
    print("\nCollected API Logs:")
    print(df.to_string())
    
    # Save logs
    collector.save_logs()
    
    return collector

if __name__ == "__main__":
    main()