import json
import random
import datetime
import uuid
import ipaddress
import time
from faker import Faker
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize Faker for generating realistic data
fake = Faker()

# Define constants for log generation
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
ENDPOINTS = [
    "/api/users", 
    "/api/products", 
    "/api/orders", 
    "/api/auth/login", 
    "/api/auth/logout",
    "/api/payments",
    "/api/shipping",
    "/api/notifications",
    "/api/settings",
    "/api/analytics"
]

# Status codes with their general meanings
STATUS_CODES = {
    # 2xx Success
    200: "OK",
    201: "Created",
    204: "No Content",
    # 4xx Client errors
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    # 5xx Server errors
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout"
}

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "PostmanRuntime/7.28.0",
    "python-requests/2.25.1"
]

# Request patterns
REQUEST_PATTERNS = {
    "login_flow": ["/api/auth/login", "/api/users", "/api/settings"],
    "shopping_flow": ["/api/products", "/api/orders", "/api/payments", "/api/shipping"],
    "admin_flow": ["/api/users", "/api/analytics", "/api/settings"]
}

# Common errors by status code
ERROR_MESSAGES = {
    400: [
        "Invalid request parameters",
        "Missing required field",
        "Validation failed for input data",
        "Malformed JSON in request body"
    ],
    401: [
        "Missing authentication token",
        "Invalid authentication credentials",
        "Token expired",
        "User not authenticated"
    ],
    403: [
        "Insufficient permissions to access resource",
        "Account suspended",
        "IP address blocked",
        "Rate limit exceeded"
    ],
    404: [
        "Resource not found",
        "Endpoint does not exist",
        "User ID not found",
        "Product no longer available"
    ],
    422: [
        "Unable to process request",
        "Data validation failed",
        "Invalid format for field 'email'",
        "Value out of allowed range"
    ],
    429: [
        "Rate limit exceeded",
        "Too many requests",
        "API quota exceeded",
        "Please try again later"
    ],
    500: [
        "Internal server error",
        "Unexpected condition encountered",
        "Database connection failed",
        "Unhandled exception occurred"
    ],
    502: [
        "Bad gateway",
        "Invalid response from upstream server",
        "Proxy error",
        "Upstream server unavailable"
    ],
    503: [
        "Service temporarily unavailable",
        "Server is overloaded",
        "Service is down for maintenance",
        "Please try again later"
    ],
    504: [
        "Gateway timeout",
        "Request timed out",
        "Upstream server did not respond in time",
        "Connection timeout"
    ]
}

def generate_ip():
    """Generate a random IP address"""
    return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

def generate_correlation_id():
    """Generate a correlation ID in UUID format"""
    return str(uuid.uuid4())

def generate_request_headers(auth_required=False):
    """Generate realistic request headers"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": random.choice(USER_AGENTS)
    }
    
    if auth_required:
        headers["Authorization"] = f"Bearer {uuid.uuid4().hex}"
    
    return headers

def generate_request_body(endpoint, is_anomalous=False):
    """Generate a request body based on the endpoint"""
    if endpoint == "/api/users":
        body = {
            "name": fake.name(),
            "email": fake.email(),
            "role": random.choice(["user", "admin", "guest"])
        }
        
        # Introduce anomalies in request body
        if is_anomalous:
            if random.random() < 0.5:
                # Missing required fields
                del body["name"]
            else:
                # Invalid format
                body["email"] = "invalid-email-format"
                
    elif endpoint == "/api/auth/login":
        body = {
            "username": fake.user_name(),
            "password": "password123"  # Simplified for demo
        }
        
        # Introduce anomalies
        if is_anomalous:
            if random.random() < 0.5:
                # Invalid credentials
                body["password"] = "wrong"
            else:
                # Missing field
                del body["password"]
                
    elif endpoint == "/api/products":
        body = {
            "category": random.choice(["electronics", "clothing", "food"]),
            "price_range": f"{random.randint(10, 50)}-{random.randint(51, 1000)}"
        }
        
    elif endpoint == "/api/orders":
        body = {
            "product_id": uuid.uuid4().hex,
            "quantity": random.randint(1, 10),
            "shipping_address": fake.address()
        }
        
        # Introduce anomalies
        if is_anomalous and random.random() < 0.7:
            # Invalid product ID format
            body["product_id"] = "invalid-id"
            # Negative quantity
            body["quantity"] = -5
            
    elif endpoint == "/api/payments":
        body = {
            "order_id": uuid.uuid4().hex,
            "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
            "amount": round(random.uniform(10.0, 1000.0), 2)
        }
        
        # Introduce anomalies
        if is_anomalous and random.random() < 0.7:
            # Invalid amount
            body["amount"] = -100 if random.random() < 0.5 else "not-a-number"
    else:
        # Generic body for other endpoints
        body = {"params": {"query": fake.sentence()}}
        
    return body

def generate_response_size(status_code):
    """Generate appropriate response size based on status code"""
    if status_code >= 200 and status_code < 300:
        # Success responses typically have more content
        return random.randint(512, 8192)
    elif status_code == 204:
        # No content
        return 0
    elif status_code >= 400 and status_code < 500:
        # Client errors usually have smaller error messages
        return random.randint(128, 512)
    else:
        # Server errors
        return random.randint(256, 1024)

def generate_error_message(status_code):
    """Generate an appropriate error message based on status code"""
    if status_code < 400:
        return None
    
    if status_code in ERROR_MESSAGES:
        return random.choice(ERROR_MESSAGES[status_code])
    else:
        return "Unknown error occurred"

def generate_related_requests(correlation_id, flow_type, base_timestamp, include_errors=False):
    """Generate a sequence of related requests with the same correlation ID"""
    related_logs = []
    current_timestamp = base_timestamp
    
    # Get the sequence of endpoints for this flow
    endpoints = REQUEST_PATTERNS.get(flow_type, random.sample(ENDPOINTS, 3))
    
    for i, endpoint in enumerate(endpoints):
        # Add some time between requests
        current_timestamp += datetime.timedelta(milliseconds=random.randint(50, 500))
        
        # Determine if this request should be anomalous
        is_anomalous = include_errors and (i == len(endpoints) - 1 or random.random() < 0.3)
        
        # For anomalous flows, increase the chance of errors in later steps
        if is_anomalous:
            status_code = random.choice(list(ERROR_MESSAGES.keys())) if random.random() < 0.8 else 200
            response_time = random.uniform(0.5, 10.0) if random.random() < 0.7 else random.uniform(0.05, 0.3)
        else:
            status_code = 200
            response_time = random.uniform(0.05, 0.3)
        
        # Generate the log entry
        log_entry = generate_log_entry(
            timestamp=current_timestamp,
            endpoint=endpoint,
            status_code=status_code,
            response_time=response_time,
            correlation_id=correlation_id,
            is_anomalous=is_anomalous
        )
        
        related_logs.append(log_entry)
        
        # If we got an error, we might stop the flow
        if status_code >= 400 and random.random() < 0.7:
            break
            
    return related_logs

def generate_log_entry(timestamp=None, endpoint=None, status_code=None, response_time=None, 
                       correlation_id=None, is_anomalous=False):
    """Generate a single API log entry"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Select random endpoint if not provided
    if endpoint is None:
        endpoint = random.choice(ENDPOINTS)
    
    # Determine HTTP method based on endpoint
    if endpoint == "/api/auth/login":
        method = "POST"
    elif endpoint == "/api/auth/logout":
        method = "POST"
    elif endpoint == "/api/users" and random.random() < 0.7:
        method = "GET"
    elif endpoint == "/api/products":
        method = "GET" if random.random() < 0.8 else "POST"
    elif endpoint == "/api/orders":
        method = random.choice(["GET", "POST", "PUT"])
    else:
        method = random.choice(HTTP_METHODS)
    
    # Determine if authentication is needed
    auth_required = endpoint not in ["/api/auth/login", "/api/products"] or method != "GET"
    
    # Generate a correlation ID if not provided
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    # Determine status code if not provided
    if status_code is None:
        if is_anomalous and random.random() < 0.8:
            # Higher chance of error for anomalous requests
            status_code = random.choice([400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
        else:
            # Normal requests mostly succeed
            weights = [0.95, 0.05]  # 95% success, 5% error
            status_category = random.choices(["success", "error"], weights=weights)[0]
            
            if status_category == "success":
                status_code = random.choice([200, 201, 204])
            else:
                status_code = random.choice([400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
    
    # Generate appropriate response time
    if response_time is None:
        if status_code in [502, 503, 504] or is_anomalous:
            # Timeouts and server errors tend to take longer
            response_time = random.uniform(1.0, 10.0)  # 1-10 seconds
        else:
            response_time = random.uniform(0.05, 0.5)  # 50-500ms
    
    # Generate request body
    request_body = generate_request_body(endpoint, is_anomalous)
    
    # For anomalous cases, sometimes make the request body invalid JSON
    if is_anomalous and random.random() < 0.2:
        request_body = str(request_body)[:-1]  # Remove closing brace to make invalid JSON
    
    # Generate response size
    response_size = generate_response_size(status_code)
    
    # Generate error message if applicable
    error_message = generate_error_message(status_code)
    
    # Create the log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "http_method": method,
        "endpoint_path": endpoint,
        "status_code": status_code,
        "response_time": round(response_time, 3),  # in seconds
        "ip_address": generate_ip(),
        "user_agent": random.choice(USER_AGENTS),
        "request_headers": generate_request_headers(auth_required),
        "request_body": request_body,
        "response_size": response_size,
        "error_message": error_message,
        "correlation_id": correlation_id,
        "is_anomalous": is_anomalous  # Meta field to help with labeling (would not be in real logs)
    }
    
    return log_entry

def generate_api_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of API logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.7)):  # 70% of normal logs are individual
        logs.append(generate_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.5)):  # 50% of anomalous logs are individual
        logs.append(generate_log_entry(is_anomalous=True))
    
    # Generate normal request flows
    num_normal_flows = int(num_normal * 0.3 / 3)  # Approximately 3 logs per flow
    for _ in range(num_normal_flows):
        flow_type = random.choice(list(REQUEST_PATTERNS.keys()))
        correlation_id = generate_correlation_id()
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        flow_logs.extend(generate_related_requests(
            correlation_id=correlation_id,
            flow_type=flow_type,
            base_timestamp=base_timestamp,
            include_errors=False
        ))
    
    # Generate anomalous request flows
    num_anomalous_flows = int(num_anomalous * 0.5 / 3)  # Approximately 3 logs per flow
    for _ in range(num_anomalous_flows):
        flow_type = random.choice(list(REQUEST_PATTERNS.keys()))
        correlation_id = generate_correlation_id()
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        flow_logs.extend(generate_related_requests(
            correlation_id=correlation_id,
            flow_type=flow_type,
            base_timestamp=base_timestamp,
            include_errors=True
        ))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_logs_to_file(logs, format='json', filename='api_logs'):
    """Save logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        df = pd.DataFrame(logs)
        
        # Convert complex columns to strings
        for col in ['request_headers', 'request_body']:
            df[col] = df[col].apply(lambda x: json.dumps(x))
        
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_logs(logs):
    """Print analysis of the generated logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['status_code']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Count by endpoint
    endpoints = {}
    for log in logs:
        endpoint = log['endpoint_path']
        endpoints[endpoint] = endpoints.get(endpoint, 0) + 1
    
    # Count unique correlation IDs
    correlation_ids = set(log['correlation_id'] for log in logs)
    
    print("\n=== Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Unique request flows (correlation IDs): {len(correlation_ids)}")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Endpoint Distribution ===")
    for endpoint, count in sorted(endpoints.items(), key=lambda x: x[1], reverse=True):
        print(f"{endpoint}: {count} logs ({count/total_logs*100:.2f}%)")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate 1000 logs with 20% anomalies
    logs = generate_api_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_logs_to_file(logs, format='json')
    csv_path = save_logs_to_file(logs, format='csv')
    
    print(f"\nLogs have been saved to {json_path} and {csv_path}")
    print("You can use these files for training your Isolation Forest model.")
    
    # Print sample logs (2 normal, 2 anomalous)
    print("\n=== Sample Normal Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2))
    
    print("\n=== Sample Anomalous Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2))