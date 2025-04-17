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
import os

# Initialize Faker for generating realistic data
fake = Faker()

# Define constants for API Gateway log generation
GATEWAY_TYPES = ["kong", "apigee", "aws-gateway", "azure-gateway", "custom-gateway"]
GATEWAY_ENVIRONMENTS = ["dev", "staging", "production", "test"]
GATEWAY_REGIONS = ["us-east", "us-west", "eu-west", "ap-south", "global"]

# HTTP Methods
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]

# Gateway Routes/Endpoints (representing different microservices)
GATEWAY_ROUTES = {
    "user-service": ["/api/v1/users", "/api/v1/users/{id}", "/api/v1/users/search", "/api/v1/auth"],
    "product-service": ["/api/v1/products", "/api/v1/products/{id}", "/api/v1/products/search", "/api/v1/categories"],
    "order-service": ["/api/v1/orders", "/api/v1/orders/{id}", "/api/v1/cart", "/api/v1/checkout"],
    "payment-service": ["/api/v1/payments", "/api/v1/payments/{id}", "/api/v1/refunds", "/api/v1/wallet"],
    "notification-service": ["/api/v1/notifications", "/api/v1/emails", "/api/v1/sms", "/api/v1/push"],
    "analytics-service": ["/api/v1/metrics", "/api/v1/reports", "/api/v1/dashboards", "/api/v1/events"]
}

# Upstream services (actual microservices behind the gateway)
UPSTREAM_SERVICES = {
    "user-service": {"host": "user-service.internal", "port": 8080},
    "product-service": {"host": "product-service.internal", "port": 8081},
    "order-service": {"host": "order-service.internal", "port": 8082},
    "payment-service": {"host": "payment-service.internal", "port": 8083},
    "notification-service": {"host": "notification-service.internal", "port": 8084},
    "analytics-service": {"host": "analytics-service.internal", "port": 8085}
}

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 10; Mobile; rv:88.0) Gecko/88.0 Firefox/88.0",
    "PostmanRuntime/7.28.0",
    "python-requests/2.25.1",
    "axios/0.21.1",
    "curl/7.76.1",
    "okhttp/4.9.1"
]

# Client types/platforms
CLIENT_TYPES = ["web", "ios", "android", "api", "desktop", "iot", "unknown"]

# API versions
API_VERSIONS = ["v1", "v2", "v3", "beta"]

# Gateway plugins commonly used
GATEWAY_PLUGINS = [
    "rate-limiting", 
    "key-auth", 
    "jwt-auth", 
    "oauth2", 
    "cors", 
    "ip-restriction",
    "request-transformer", 
    "response-transformer", 
    "cache", 
    "request-size-limiting",
    "prometheus", 
    "datadog", 
    "http-log", 
    "file-log"
]

# Gateway status reasons
GATEWAY_STATUS_REASONS = {
    200: ["OK", "Success"],
    201: ["Created", "Resource created successfully"],
    204: ["No Content", "Success with no body"],
    301: ["Moved Permanently", "Resource has been permanently moved"],
    302: ["Found", "Resource temporarily moved"],
    400: ["Bad Request", "Invalid request syntax", "Missing required parameter", "Invalid parameter format"],
    401: ["Unauthorized", "Missing credentials", "Invalid credentials", "Token expired"],
    403: ["Forbidden", "Access denied", "Insufficient permissions", "API key lacks access"],
    404: ["Not Found", "Resource not found", "Endpoint does not exist", "API version not supported"],
    405: ["Method Not Allowed", "HTTP method not supported for this resource"],
    408: ["Request Timeout", "Client took too long to send request"],
    429: ["Too Many Requests", "Rate limit exceeded", "Quota exceeded"],
    500: ["Internal Server Error", "Unexpected error", "Service error"],
    502: ["Bad Gateway", "Invalid response from upstream", "Upstream connection error"],
    503: ["Service Unavailable", "Service temporarily offline", "Maintenance mode"],
    504: ["Gateway Timeout", "Upstream service timeout", "Request processing timeout"]
}

# Common client errors
CLIENT_ERRORS = {
    400: ["Missing required fields", "Invalid data format", "Validation error", "Malformed JSON"],
    401: ["Invalid API key", "Expired token", "Invalid token signature", "Missing authentication"],
    403: ["User does not have permission", "Account suspended", "Resource access forbidden"],
    404: ["Resource not found", "Endpoint does not exist", "API version deprecated"]
}

# Common gateway/server errors
SERVER_ERRORS = {
    500: ["Internal server error", "Unhandled exception", "Database error", "Memory overflow"],
    502: ["Bad response from upstream service", "Upstream returned invalid data", "Service integration error"],
    503: ["Service is down for maintenance", "Service overloaded", "Dependency service unavailable"],
    504: ["Upstream service timeout", "Processing timeout", "Database query timeout"]
}

def generate_ip():
    """Generate a random IP address"""
    return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

def generate_gateway_route():
    """Generate a random gateway route"""
    service = random.choice(list(GATEWAY_ROUTES.keys()))
    route = random.choice(GATEWAY_ROUTES[service])
    
    # Replace any {id} placeholders with actual IDs
    if "{id}" in route:
        route = route.replace("{id}", str(uuid.uuid4())[:8])
    
    return service, route

def generate_request_headers(include_auth=True):
    """Generate realistic request headers for gateway logs"""
    headers = {
        "accept": random.choice(["application/json", "application/xml", "*/*", "application/json, text/plain, */*"]),
        "user-agent": random.choice(USER_AGENTS),
        "x-forwarded-for": generate_ip(),
        "x-request-id": str(uuid.uuid4()),
        "x-client-type": random.choice(CLIENT_TYPES),
        "x-api-version": random.choice(API_VERSIONS)
    }
    
    # Add content-type for requests with bodies
    if random.random() < 0.7:
        headers["content-type"] = random.choice([
            "application/json", 
            "application/x-www-form-urlencoded", 
            "multipart/form-data"
        ])
    
    # Add auth headers with some probability
    if include_auth:
        auth_type = random.choice(["api-key", "bearer-token", "basic", "none"])
        
        if auth_type == "api-key":
            headers["x-api-key"] = f"apk_{uuid.uuid4().hex[:16]}"
        elif auth_type == "bearer-token":
            headers["authorization"] = f"Bearer {uuid.uuid4().hex}"
        elif auth_type == "basic":
            # Simulated basic auth
            headers["authorization"] = f"Basic {uuid.uuid4().hex[:16]}"
    
    return headers

def generate_response_headers(status_code):
    """Generate response headers based on status code"""
    headers = {
        "content-type": "application/json",
        "x-response-time": f"{random.uniform(0.01, 2.0):.3f}s",
        "x-request-id": str(uuid.uuid4()),
        "date": (datetime.datetime.now() - datetime.timedelta(
            seconds=random.randint(0, 86400)
        )).strftime("%a, %d %b %Y %H:%M:%S GMT")
    }
    
    # Add cache headers for successful responses
    if 200 <= status_code < 300 and random.random() < 0.5:
        headers["cache-control"] = random.choice([
            "no-cache", 
            "no-store", 
            "max-age=300", 
            "must-revalidate"
        ])
    
    # Add pagination headers for list endpoints
    if status_code == 200 and random.random() < 0.4:
        headers["x-total-count"] = str(random.randint(10, 1000))
        headers["x-page-number"] = str(random.randint(1, 10))
        headers["x-page-size"] = str(random.randint(10, 100))
    
    # Add rate limiting headers
    if random.random() < 0.3 or status_code == 429:
        headers["x-rate-limit-limit"] = str(random.choice([60, 100, 1000]))
        headers["x-rate-limit-remaining"] = str(random.randint(0, 100))
        headers["x-rate-limit-reset"] = str(int(time.time()) + random.randint(1, 3600))
    
    return headers

def calculate_response_time(route, method, status_code, is_anomalous=False):
    """Calculate a realistic response time based on the endpoint and status"""
    # Base response time ranges by endpoint type
    if is_anomalous and random.random() < 0.8:
        # Anomalously slow response
        base_time = random.uniform(1.0, 10.0)  # 1-10 seconds
    else:
        if "search" in route:
            base_time = random.uniform(0.1, 0.8)  # Search operations are slower
        elif method in ["POST", "PUT", "DELETE"]:
            base_time = random.uniform(0.05, 0.5)  # Write operations
        else:
            base_time = random.uniform(0.01, 0.3)  # Read operations
    
    # Adjust based on status code
    if status_code >= 500:
        # Server errors often have higher response times
        multiplier = random.uniform(1.5, 5.0)
    elif status_code == 408 or status_code == 504:
        # Timeout status codes
        multiplier = random.uniform(5.0, 15.0)
    elif status_code == 429:
        # Rate limit status usually returns quickly
        multiplier = random.uniform(0.1, 0.5)
    elif 400 <= status_code < 500:
        # Client errors generally return quickly
        multiplier = random.uniform(0.1, 1.2)
    else:
        # Success codes
        multiplier = 1.0
    
    return base_time * multiplier

def generate_query_string(route):
    """Generate a realistic query string based on the endpoint"""
    query_params = []
    
    if "search" in route:
        # Search parameters
        if random.random() < 0.8:
            query_params.append(f"q={fake.word()}")
        
        if random.random() < 0.6:
            query_params.append(f"page={random.randint(1, 10)}")
            query_params.append(f"limit={random.choice([10, 20, 50, 100])}")
        
        if random.random() < 0.4:
            query_params.append(f"sort={random.choice(['name', 'created_at', 'updated_at'])}")
            query_params.append(f"order={random.choice(['asc', 'desc'])}")
    
    elif "products" in route:
        # Product filters
        if random.random() < 0.5:
            query_params.append(f"category={random.choice(['electronics', 'clothing', 'food'])}")
        
        if random.random() < 0.3:
            query_params.append(f"min_price={random.randint(5, 50)}")
            query_params.append(f"max_price={random.randint(50, 500)}")
    
    elif "orders" in route:
        # Order filters
        if random.random() < 0.6:
            query_params.append(f"status={random.choice(['pending', 'processing', 'shipped', 'delivered'])}")
        
        if random.random() < 0.4:
            # Date range
            days = random.randint(1, 90)
            query_params.append(f"from_date={(datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')}")
            query_params.append(f"to_date={(datetime.datetime.now()).strftime('%Y-%m-%d')}")
    
    # Add generic pagination to any endpoint with some probability
    if not query_params and random.random() < 0.4:
        query_params.append(f"page={random.randint(1, 5)}")
        query_params.append(f"per_page={random.choice([10, 20, 50])}")
    
    # Join all parameters
    return "&".join(query_params)

def generate_gateway_log_entry(timestamp=None, correlation_id=None, is_anomalous=False):
    """Generate a single API gateway log entry"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Generate gateway details
    gateway_type = random.choice(GATEWAY_TYPES)
    gateway_environment = random.choice(GATEWAY_ENVIRONMENTS)
    gateway_region = random.choice(GATEWAY_REGIONS)
    gateway_instance_id = f"{gateway_type}-{gateway_region}-{random.randint(1, 10)}"
    
    # Generate correlation ID if not provided (for tracing requests across services)
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Generate trace ID for distributed tracing
    trace_id = f"trace-{uuid.uuid4().hex[:16]}"
    span_id = f"span-{uuid.uuid4().hex[:8]}"
    
    # Generate route and service
    service, route = generate_gateway_route()
    upstream_service = UPSTREAM_SERVICES[service]
    
    # Determine HTTP method based on endpoint
    if "search" in route:
        method = "GET" if random.random() < 0.9 else "POST"
    elif "{id}" in route:
        method = random.choice(["GET", "PUT", "DELETE"])
    elif "auth" in route:
        method = "POST"
    elif "checkout" in route or "payments" in route:
        method = random.choice(["POST", "PUT"])
    else:
        method = random.choice(HTTP_METHODS)
    
    # Determine status code based on anomaly flag
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous requests
        if random.random() < 0.7:
            # Server errors more common for anomalies
            status_code = random.choice([500, 502, 503, 504])
        else:
            # Client errors
            status_code = random.choice([400, 401, 403, 404, 405, 429])
    else:
        # Normal requests mostly succeed
        weights = [0.95, 0.04, 0.01]  # 95% success, 4% client error, 1% server error
        status_category = random.choices(["success", "client_error", "server_error"], weights=weights)[0]
        
        if status_category == "success":
            status_code = random.choice([200, 201, 204])
        elif status_category == "client_error":
            status_code = random.choice([400, 401, 403, 404, 429])
        else:
            status_code = random.choice([500, 502, 503, 504])
    
    # Generate client IP
    client_ip = generate_ip()
    
    # Generate request headers
    # Auth might fail for 401/403 status codes
    include_auth = not (status_code == 401 or status_code == 403)
    request_headers = generate_request_headers(include_auth)
    
    # Generate response headers
    response_headers = generate_response_headers(status_code)
    
    # Calculate response time
    response_time = calculate_response_time(route, method, status_code, is_anomalous)
    
    # Generate active plugins for this request
    active_plugins = random.sample(GATEWAY_PLUGINS, random.randint(2, 5))
    
    # Generate plugin-specific data
    plugin_data = {}
    if "rate-limiting" in active_plugins:
        plugin_data["rate-limiting"] = {
            "limit": random.choice([60, 100, 1000]),
            "remaining": random.randint(0, 100),
            "reset": int(time.time()) + random.randint(1, 3600)
        }
    
    if "key-auth" in active_plugins or "jwt-auth" in active_plugins:
        plugin_data["auth"] = {
            "consumer_id": f"consumer-{uuid.uuid4().hex[:8]}",
            "credential_id": f"cred-{uuid.uuid4().hex[:8]}"
        }
    
    # Generate error message if applicable
    if status_code >= 400:
        if 400 <= status_code < 500 and status_code in CLIENT_ERRORS:
            error_message = random.choice(CLIENT_ERRORS[status_code])
        elif status_code >= 500 and status_code in SERVER_ERRORS:
            error_message = random.choice(SERVER_ERRORS[status_code])
        else:
            # Get a generic message for the status code
            if status_code in GATEWAY_STATUS_REASONS:
                error_message = random.choice(GATEWAY_STATUS_REASONS[status_code])
            else:
                error_message = "Unknown error"
    else:
        error_message = None
    
    # Generate response size
    if status_code == 204:
        response_size = 0
    elif status_code >= 400:
        response_size = random.randint(100, 1000)  # Error responses are smaller
    else:
        response_size = random.randint(200, 50000)  # Success responses vary widely
    
    # Create the gateway log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "gateway": {
            "type": gateway_type,
            "environment": gateway_environment,
            "region": gateway_region,
            "instance_id": gateway_instance_id
        },
        "request": {
            "id": request_headers.get("x-request-id", str(uuid.uuid4())),
            "method": method,
            "uri": route,
            "query_string": generate_query_string(route) if method == "GET" and random.random() < 0.7 else "",
            "headers": request_headers,
            "client_ip": client_ip,
            "size": random.randint(0, 10000)
        },
        "response": {
            "status": status_code,
            "status_message": random.choice(GATEWAY_STATUS_REASONS.get(status_code, ["Unknown"])),
            "headers": response_headers,
            "size": response_size,
            "error": error_message
        },
        "upstream": {
            "service": service,
            "host": upstream_service["host"],
            "port": upstream_service["port"],
            "latency_ms": round(response_time * 1000, 2)
        },
        "route": {
            "id": f"route-{service}-{uuid.uuid4().hex[:6]}",
            "path": route,
            "protocols": ["http", "https"]
        },
        "plugins": {
            "active": active_plugins,
            "data": plugin_data
        },
        "metrics": {
            "request_size_bytes": random.randint(100, 10000),
            "response_size_bytes": response_size,
            "total_latency_ms": round(response_time * 1000, 2),
            "gateway_latency_ms": round(random.uniform(1, 50), 2),
            "upstream_latency_ms": round(response_time * 1000 - random.uniform(1, 50), 2)
        },
        "tracing": {
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "span_id": span_id
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_gateway_request_flow(base_time=None, correlation_id=None, is_anomalous=False):
    """Generate a sequence of related API gateway calls that form a workflow"""
    if base_time is None:
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Define a typical workflow:
    # For example: authenticate -> get user -> get products -> create order -> process payment
    flow_pattern = [
        {"service": "user-service", "route": "/api/v1/auth", "method": "POST"},
        {"service": "user-service", "route": "/api/v1/users/{id}", "method": "GET"},
        {"service": "product-service", "route": "/api/v1/products/search", "method": "GET"},
        {"service": "order-service", "route": "/api/v1/cart", "method": "POST"},
        {"service": "order-service", "route": "/api/v1/checkout", "method": "POST"},
        {"service": "payment-service", "route": "/api/v1/payments", "method": "POST"}
    ]
    
    # Determine if we introduce an error and at which step
    error_step = random.randint(0, len(flow_pattern) - 1) if is_anomalous else None
    
    # Generate logs for each step in the flow
    flow_logs = []
    current_time = base_time
    
    for i, step in enumerate(flow_pattern):
        # Add some time between requests
        current_time += datetime.timedelta(milliseconds=random.randint(100, 2000))
        
        # Determine if this step should be anomalous
        step_is_anomalous = is_anomalous and (i == error_step or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_gateway_log_entry(
            timestamp=current_time,
            correlation_id=correlation_id,
            is_anomalous=step_is_anomalous
        )
        
        # Override with flow-specific values
        if "{id}" in step["route"]:
            # Replace placeholder with ID
            route = step["route"].replace("{id}", str(uuid.uuid4())[:8])
        else:
            route = step["route"]
        
        log_entry["request"]["method"] = step["method"]
        log_entry["request"]["uri"] = route
        log_entry["upstream"]["service"] = step["service"]
        log_entry["upstream"]["host"] = UPSTREAM_SERVICES[step["service"]]["host"]
        log_entry["upstream"]["port"] = UPSTREAM_SERVICES[step["service"]]["port"]
        
        flow_logs.append(log_entry)
        
        # If we get an error, we might stop the flow
        if step_is_anomalous and log_entry["response"]["status"] >= 400 and random.random() < 0.7:
            break
    
    return flow_logs

def generate_gateway_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of API gateway logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.5)):  # 50% of normal logs are individual
        logs.append(generate_gateway_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_gateway_log_entry(is_anomalous=True))
    
    # Generate normal gateway flows
    num_normal_flows = int(num_normal * 0.5 / 6)  # Approximately 6 logs per flow
    for _ in range(num_normal_flows):
        flow_logs.extend(generate_gateway_request_flow(is_anomalous=False))
    
    # Generate anomalous gateway flows
    num_anomalous_flows = int(num_anomalous * 0.7 / 6)  # Approximately 6 logs per flow
    for _ in range(num_anomalous_flows):
        flow_logs.extend(generate_gateway_request_flow(is_anomalous=True))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_gateway_logs(logs, format='json', filename='gateway_api_logs'):
    """Save gateway logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} gateway logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "gateway_type": log["gateway"]["type"],
                "gateway_environment": log["gateway"]["environment"],
                "gateway_region": log["gateway"]["region"],
                "gateway_instance_id": log["gateway"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_uri": log["request"]["uri"],
                "request_query_string": log["request"]["query_string"],
                "client_ip": log["request"]["client_ip"],
                "request_size": log["request"]["size"],
                "response_status": log["response"]["status"],
                "response_status_message": log["response"]["status_message"],
                "response_size": log["response"]["size"],
                "response_error": log["response"]["error"],
                "upstream_service": log["upstream"]["service"],
                "upstream_host": log["upstream"]["host"],
                "upstream_port": log["upstream"]["port"],
                "upstream_latency_ms": log["upstream"]["latency_ms"],
                "route_id": log["route"]["id"],
                "route_path": log["route"]["path"],
                "total_latency_ms": log["metrics"]["total_latency_ms"],
                "gateway_latency_ms": log["metrics"]["gateway_latency_ms"],
                "correlation_id": log["tracing"]["correlation_id"],
                "trace_id": log["tracing"]["trace_id"],
                "span_id": log["tracing"]["span_id"],
                "is_anomalous": log["is_anomalous"]
            }
            
            # Add some key request headers
            if "user-agent" in log["request"]["headers"]:
                flat_log["user_agent"] = log["request"]["headers"]["user-agent"]
            
            if "x-api-version" in log["request"]["headers"]:
                flat_log["api_version"] = log["request"]["headers"]["x-api-version"]
            
            if "x-client-type" in log["request"]["headers"]:
                flat_log["client_type"] = log["request"]["headers"]["x-client-type"]
            
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} gateway logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_gateway_logs(logs):
    """Print analysis of the generated gateway logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by gateway type
    gateway_types = {}
    for log in logs:
        gw_type = log['gateway']['type']
        gateway_types[gw_type] = gateway_types.get(gw_type, 0) + 1
    
    # Count by environment
    environments = {}
    for log in logs:
        env = log['gateway']['environment']
        environments[env] = environments.get(env, 0) + 1
    
    # Count by upstream service
    services = {}
    for log in logs:
        service = log['upstream']['service']
        services[service] = services.get(service, 0) + 1
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['response']['status']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Count unique trace IDs and correlation IDs
    trace_ids = set(log['tracing']['trace_id'] for log in logs)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    # Calculate average response times
    normal_times = [log['metrics']['total_latency_ms'] for log in logs if not log.get('is_anomalous', False)]
    anomalous_times = [log['metrics']['total_latency_ms'] for log in logs if log.get('is_anomalous', False)]
    
    avg_normal_time = sum(normal_times) / len(normal_times) if normal_times else 0
    avg_anomalous_time = sum(anomalous_times) / len(anomalous_times) if anomalous_times else 0
    
    print("\n=== Gateway Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Unique traces: {len(trace_ids)}")
    print(f"Unique request flows (correlation IDs): {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Gateway Type Distribution ===")
    for gw_type, count in sorted(gateway_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{gw_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Environment Distribution ===")
    for env, count in sorted(environments.items(), key=lambda x: x[1], reverse=True):
        print(f"{env}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Service Distribution ===")
    for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True):
        print(f"{service}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_logs(num_logs=1000, anomaly_percentage=20):
    """Generate logs from multiple sources that share correlation IDs for interconnected request flows"""
    
    # Total logs to generate
    total_logs = num_logs
    
    # Generate correlation IDs for interconnected flows
    num_flows = int(total_logs / 15)  # Approximately 15 logs per flow
    correlation_ids = [str(uuid.uuid4()) for _ in range(num_flows)]
    
    # Determine which flows will be anomalous
    num_anomalous_flows = int(num_flows * (anomaly_percentage / 100))
    anomalous_correlations = set(random.sample(correlation_ids, num_anomalous_flows))
    
    # Initialize log collections
    gateway_logs = []
    service_logs = []
    db_logs = []
    
    # Generate logs for each flow
    for correlation_id in correlation_ids:
        is_anomalous = correlation_id in anomalous_correlations
        
        # Generate a base timestamp for this flow
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # Gateway logs (entry point)
        flow_gateway_logs = generate_gateway_request_flow(
            base_time=base_time,
            correlation_id=correlation_id,
            is_anomalous=is_anomalous
        )
        gateway_logs.extend(flow_gateway_logs)
        
        # For each gateway log, potentially generate service and database logs
        for gateway_log in flow_gateway_logs:
            # Service could generate 0-2 service logs per gateway request
            num_service_logs = random.randint(0, 2)
            for i in range(num_service_logs):
                service_time = datetime.datetime.fromisoformat(gateway_log["timestamp"]) + datetime.timedelta(milliseconds=random.randint(5, 50))
                
                # Create a service log connected to this gateway request
                service_log = {
                    "timestamp": service_time.isoformat(),
                    "service": {
                        "type": gateway_log["upstream"]["service"],
                        "environment": gateway_log["gateway"]["environment"],
                        "metadata": {
                            "instance_id": f"svc-{uuid.uuid4().hex[:8]}"
                        }
                    },
                    "request": {
                        "method": gateway_log["request"]["method"],
                        "path": gateway_log["request"]["uri"],
                        "headers": {
                            "x-request-id": gateway_log["request"]["id"],
                            "x-gateway-instance": gateway_log["gateway"]["instance_id"]
                        }
                    },
                    "correlation": {
                        "id": correlation_id,
                        "trace_id": gateway_log["tracing"]["trace_id"],
                        "span_id": f"span-{uuid.uuid4().hex[:8]}",
                        "parent_id": gateway_log["tracing"]["span_id"]
                    },
                    "is_anomalous": is_anomalous
                }
                service_logs.append(service_log)
                
                # Database logs - Each service may generate 0-2 database operations
                num_db_logs = random.randint(0, 2)
                for j in range(num_db_logs):
                    db_time = service_time + datetime.timedelta(milliseconds=random.randint(5, 100))
                    
                    # Create a database log connected to this service
                    db_log = {
                        "timestamp": db_time.isoformat(),
                        "db_type": random.choice(["mysql", "postgres", "mongodb"]),
                        "operation": random.choice(["query", "insert", "update", "delete"]),
                        "collection": gateway_log["upstream"]["service"].replace("-service", "s"),
                        "execution_time_sec": random.uniform(0.001, 0.1),
                        "correlation_id": correlation_id,
                        "trace_id": gateway_log["tracing"]["trace_id"],
                        "span_id": f"span-{uuid.uuid4().hex[:8]}",
                        "parent_span_id": service_log["correlation"]["span_id"],
                        "is_anomalous": is_anomalous
                    }
                    db_logs.append(db_log)
    
    print(f"Generated {len(gateway_logs)} gateway logs")
    print(f"Generated {len(service_logs)} service logs")
    print(f"Generated {len(db_logs)} database logs")
    
    return {
        "gateway_logs": gateway_logs,
        "service_logs": service_logs,
        "db_logs": db_logs
    }

def save_interconnected_logs(logs, output_dir="."):
    """Save interconnected logs to separate files"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save gateway logs
    gateway_path = os.path.join(output_dir, "gateway_logs.json")
    with open(gateway_path, 'w') as f:
        json.dump(logs["gateway_logs"], f, indent=2)
    
    # Save service logs
    service_path = os.path.join(output_dir, "service_logs.json")
    with open(service_path, 'w') as f:
        json.dump(logs["service_logs"], f, indent=2)
    
    # Save database logs
    db_path = os.path.join(output_dir, "db_logs.json")
    with open(db_path, 'w') as f:
        json.dump(logs["db_logs"], f, indent=2)
    
    print(f"Saved interconnected logs to {output_dir}")
    return gateway_path, service_path, db_path

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate logs
    print("Generating API gateway logs...")
    logs = generate_gateway_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_gateway_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_gateway_logs(logs, format='json')
    csv_path = save_gateway_logs(logs, format='csv')
    
    print(f"\nGateway logs have been saved to {json_path} and {csv_path}")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Gateway Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2)[:1000] + "... (truncated)")
    
    print("\n=== Sample Anomalous Gateway Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2)[:1000] + "... (truncated)")
    
    # Generate interconnected logs (optional)
    print("\nGenerating interconnected logs across multiple services...")
    interconnected_logs = generate_interconnected_logs(num_logs=1500, anomaly_percentage=20)
    
    # Save interconnected logs
    output_dir = "interconnected_logs"
    gateway_path, service_path, db_path = save_interconnected_logs(interconnected_logs, output_dir)
    
    print(f"\nInterconnected logs have been saved to {output_dir}/")
    print("You can use these files to analyze request flows across different services.")