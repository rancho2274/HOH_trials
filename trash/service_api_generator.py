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

# Define constants specific to service API
SERVICE_ENVIRONMENTS = ["on-premise", "aws-cloud", "azure-cloud", "gcp-cloud"]
SERVICE_TYPES = ["auth-service", "user-service", "product-service", "order-service", "payment-service"]
SERVICE_VERSIONS = ["v1", "v2", "v3"]

# Service-specific endpoints
SERVICE_ENDPOINTS = {
    "auth-service": ["/authenticate", "/validate", "/refresh-token", "/revoke-token"],
    "user-service": ["/users", "/users/{id}", "/users/search", "/users/validate"],
    "product-service": ["/products", "/products/{id}", "/products/search", "/products/categories"],
    "order-service": ["/orders", "/orders/{id}", "/orders/status", "/orders/history"],
    "payment-service": ["/payments", "/payments/{id}", "/payments/process", "/payments/refund"]
}

# Service status codes including internal service-specific codes
SERVICE_STATUS_CODES = {
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    # Service-specific status codes
    550: "Service Configuration Error",
    551: "Service Dependency Failure",
    552: "Service Rate Limited"
}

# Service-specific errors
SERVICE_ERRORS = {
    400: ["Invalid service parameters", "Missing required field", "Validation failed"],
    401: ["Invalid service token", "Expired service credentials", "Service authentication required"],
    403: ["Service access denied", "Service quota exceeded", "Service subscription inactive"],
    404: ["Service endpoint not found", "Requested resource unavailable", "Service version not supported"],
    500: ["Internal service error", "Unexpected service condition", "Service execution failed"],
    550: ["Service misconfigured", "Environment configuration error", "Service initialization failed"],
    551: ["Downstream service unavailable", "Required dependency offline", "Service chain broken"],
    552: ["Service rate limit reached", "Too many service requests", "Service throttled"]
}

def generate_service_correlation_id():
    """Generate a consistent format for service correlation IDs"""
    return f"svc-{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}"

def generate_service_trace_id():
    """Generate a trace ID for distributed tracing"""
    return f"trace-{uuid.uuid4().hex[:16]}"

def generate_service_headers(service_type):
    """Generate service-specific headers"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Service-Name": service_type,
        "X-Service-Version": random.choice(SERVICE_VERSIONS),
        "X-Request-ID": str(uuid.uuid4())
    }
    
    # Add authentication headers for some services
    if service_type in ["auth-service", "user-service", "payment-service"]:
        headers["X-API-Key"] = f"svc-key-{uuid.uuid4().hex[:16]}"
    
    return headers

def generate_service_request_body(service_type, endpoint, is_anomalous=False):
    """Generate service-specific request bodies"""
    if service_type == "auth-service":
        if "/authenticate" in endpoint:
            body = {
                "client_id": f"client-{uuid.uuid4().hex[:8]}",
                "client_secret": f"secret-{uuid.uuid4().hex[:16]}",
                "grant_type": "client_credentials"
            }
            if is_anomalous and random.random() < 0.5:
                # Invalid client credentials
                del body["client_secret"]
        elif "/validate" in endpoint:
            body = {
                "token": f"token-{uuid.uuid4().hex[:32]}",
                "scope": random.choice(["read", "write", "admin"])
            }
            if is_anomalous and random.random() < 0.5:
                # Invalid token
                body["token"] = "invalid-token-format"
        else:
            body = {
                "token": f"token-{uuid.uuid4().hex[:32]}"
            }
    
    elif service_type == "user-service":
        if "/users/search" in endpoint:
            body = {
                "query": fake.word(),
                "filters": {
                    "status": random.choice(["active", "inactive", "pending"]),
                    "role": random.choice(["user", "admin", "guest"])
                },
                "page": random.randint(1, 10),
                "limit": random.randint(10, 50)
            }
            if is_anomalous and random.random() < 0.7:
                # Invalid pagination
                body["page"] = -1
                body["limit"] = 1000
        else:
            body = {
                "id": str(uuid.uuid4()),
                "name": fake.name(),
                "email": fake.email()
            }
            if is_anomalous and random.random() < 0.6:
                # Invalid email format
                body["email"] = "not-an-email"
    
    elif service_type == "product-service":
        body = {
            "id": str(uuid.uuid4()),
            "name": fake.word(),
            "category": random.choice(["electronics", "clothing", "food"]),
            "price": round(random.uniform(10, 1000), 2),
            "inventory": random.randint(0, 100)
        }
        if is_anomalous and random.random() < 0.6:
            # Negative price or inventory
            if random.random() < 0.5:
                body["price"] = -50.0
            else:
                body["inventory"] = -10
    
    elif service_type == "order-service":
        body = {
            "order_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "products": [
                {"id": str(uuid.uuid4()), "quantity": random.randint(1, 5)}
                for _ in range(random.randint(1, 3))
            ],
            "status": random.choice(["pending", "processing", "shipped", "delivered"]),
            "total": round(random.uniform(20, 2000), 2)
        }
        if is_anomalous and random.random() < 0.7:
            # Empty products list or invalid status
            if random.random() < 0.5:
                body["products"] = []
            else:
                body["status"] = "invalid_status"
    
    elif service_type == "payment-service":
        body = {
            "payment_id": str(uuid.uuid4()),
            "order_id": str(uuid.uuid4()),
            "amount": round(random.uniform(10, 2000), 2),
            "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
            "status": random.choice(["pending", "completed", "failed", "refunded"])
        }
        if is_anomalous and random.random() < 0.7:
            # Amount mismatch or invalid payment method
            if random.random() < 0.5:
                body["amount"] = -100.0
            else:
                body["payment_method"] = "unsupported_method"
    
    else:
        # Generic body for unknown service types
        body = {"data": {"request": fake.sentence()}}
    
    return body

def generate_service_metadata(service_type, environment):
    """Generate service-specific metadata"""
    metadata = {
        "service_name": service_type,
        "service_version": random.choice(SERVICE_VERSIONS),
        "environment": environment,
        "instance_id": f"inst-{uuid.uuid4().hex[:8]}",
        "region": random.choice(["us-east", "us-west", "eu-central", "ap-south"]),
        "deployed_at": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 90))).isoformat()
    }
    
    # Add environment-specific metadata
    if "cloud" in environment:
        cloud_provider = environment.split("-")[0]  # aws, azure, gcp
        metadata["cloud_provider"] = cloud_provider
        metadata["instance_type"] = f"{cloud_provider}-{random.choice(['small', 'medium', 'large'])}"
        metadata["auto_scaling_group"] = f"asg-{service_type}-{uuid.uuid4().hex[:6]}"
    else:  # on-premise
        metadata["datacenter"] = random.choice(["dc-east", "dc-west", "dc-central"])
        metadata["rack"] = f"rack-{random.randint(1, 20)}"
        metadata["server"] = f"srv-{random.randint(100, 999)}"
    
    return metadata

def generate_service_log_entry(timestamp=None, service_type=None, environment=None, 
                             correlation_id=None, trace_id=None, is_anomalous=False):
    """Generate a service API log entry with service-specific format"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Select random service type and environment if not provided
    if service_type is None:
        service_type = random.choice(SERVICE_TYPES)
    
    if environment is None:
        environment = random.choice(SERVICE_ENVIRONMENTS)
    
    # Get available endpoints for this service type
    available_endpoints = SERVICE_ENDPOINTS.get(service_type, ["/default"])
    endpoint = random.choice(available_endpoints)
    
    # Replace {id} placeholders with actual IDs
    if "{id}" in endpoint:
        endpoint = endpoint.replace("{id}", str(uuid.uuid4()))
    
    # Determine method based on endpoint and service
    if endpoint.endswith("search"):
        method = "GET"
    elif "process" in endpoint or "authenticate" in endpoint:
        method = "POST"
    elif any(word in endpoint for word in ["revoke", "delete"]):
        method = "DELETE"
    elif any(word in endpoint for word in ["update", "modify"]):
        method = "PUT"
    else:
        method = random.choice(["GET", "POST", "PUT", "DELETE"])
    
    # Generate correlation and trace IDs if not provided
    if correlation_id is None:
        correlation_id = generate_service_correlation_id()
    
    if trace_id is None:
        trace_id = generate_service_trace_id()
    
    # Determine status code based on anomaly flag
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous requests
        if random.random() < 0.6:
            # Service-specific errors are more common
            status_code = random.choice([400, 401, 403, 404, 422, 429, 500, 550, 551, 552])
        else:
            # Standard HTTP errors
            status_code = random.choice([400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
    else:
        # Normal requests mostly succeed
        weights = [0.95, 0.05]  # 95% success, 5% error
        status_category = random.choices(["success", "error"], weights=weights)[0]
        
        if status_category == "success":
            status_code = random.choice([200, 201, 202, 204])
        else:
            status_code = random.choice([400, 401, 403, 404, 422, 429, 500])
    
    # Generate response time based on environment and status
    if "cloud" in environment:
        base_response_time = random.uniform(0.02, 0.2)  # Cloud is generally faster
    else:
        base_response_time = random.uniform(0.05, 0.5)  # On-premise is a bit slower
    
    # Errors and anomalies take longer
    if status_code >= 500 or is_anomalous:
        response_time = base_response_time * random.uniform(5, 20)  # 5-20x slower
    elif status_code >= 400:
        response_time = base_response_time * random.uniform(2, 5)  # 2-5x slower
    else:
        response_time = base_response_time
    
    # Generate request headers and body
    request_headers = generate_service_headers(service_type)
    request_body = generate_service_request_body(service_type, endpoint, is_anomalous)
    
    # Generate response size
    if status_code == 204:
        response_size = 0
    elif 200 <= status_code < 300:
        response_size = random.randint(512, 10240)
    else:
        response_size = random.randint(128, 1024)
    
    # Generate error message if applicable
    if status_code >= 400:
        if status_code in SERVICE_ERRORS:
            error_message = random.choice(SERVICE_ERRORS[status_code])
        else:
            error_message = "Unexpected service error"
        error_code = f"E{status_code}_{random.randint(1000, 9999)}"
    else:
        error_message = None
        error_code = None
    
    # Generate metadata
    metadata = generate_service_metadata(service_type, environment)
    
    # Create the service log entry with service-specific format
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "service": {
            "type": service_type,
            "environment": environment,
            "metadata": metadata
        },
        "request": {
            "method": method,
            "path": endpoint,
            "headers": request_headers,
            "body": request_body,
            "source_ip": str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
        },
        "response": {
            "status_code": status_code,
            "size": response_size,
            "time_ms": round(response_time * 1000, 2),  # Convert to milliseconds
            "error": {
                "message": error_message,
                "code": error_code
            } if error_message else None
        },
        "correlation": {
            "id": correlation_id,
            "trace_id": trace_id,
            "span_id": f"span-{uuid.uuid4().hex[:8]}",
            "parent_id": f"parent-{uuid.uuid4().hex[:8]}" if random.random() < 0.7 else None
        },
        "metrics": {
            "cpu_usage": round(random.uniform(10, 95), 2),
            "memory_usage": round(random.uniform(20, 90), 2),
            "active_connections": random.randint(5, 1000),
            "queue_size": random.randint(0, 50)
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_service_flow(base_time=None, is_anomalous=False):
    """Generate a sequence of related service calls that form a workflow"""
    if base_time is None:
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    # Generate correlation_id and trace_id for the entire flow
    correlation_id = generate_service_correlation_id()
    trace_id = generate_service_trace_id()
    
    # Define a service flow pattern:
    # For example: auth-service -> user-service -> product-service -> order-service -> payment-service
    flow_pattern = [
        {"service": "auth-service", "environment": random.choice(SERVICE_ENVIRONMENTS)},
        {"service": "user-service", "environment": random.choice(SERVICE_ENVIRONMENTS)},
        {"service": random.choice(["product-service", "order-service"]), 
         "environment": random.choice(SERVICE_ENVIRONMENTS)},
        {"service": "payment-service", "environment": random.choice(SERVICE_ENVIRONMENTS)}
    ]
    
    # Determine if we introduce an error and at which step
    error_step = random.randint(0, len(flow_pattern) - 1) if is_anomalous else None
    
    # Generate logs for each step in the flow
    flow_logs = []
    current_time = base_time
    
    for i, step in enumerate(flow_pattern):
        # Add some time between service calls
        current_time += datetime.timedelta(milliseconds=random.randint(5, 200))
        
        # Determine if this step should be anomalous
        step_is_anomalous = is_anomalous and (i == error_step or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_service_log_entry(
            timestamp=current_time,
            service_type=step["service"],
            environment=step["environment"],
            correlation_id=correlation_id,
            trace_id=trace_id,
            is_anomalous=step_is_anomalous
        )
        
        flow_logs.append(log_entry)
        
        # If we get an error, we might stop the flow
        if step_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return flow_logs

def generate_service_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of service API logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.5)):  # 50% of normal logs are individual
        logs.append(generate_service_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_service_log_entry(is_anomalous=True))
    
    # Generate normal service flows
    num_normal_flows = int(num_normal * 0.5 / 4)  # Approximately 4 logs per flow
    for _ in range(num_normal_flows):
        flow_logs.extend(generate_service_flow(is_anomalous=False))
    
    # Generate anomalous service flows
    num_anomalous_flows = int(num_anomalous * 0.7 / 4)  # Approximately 4 logs per flow
    for _ in range(num_anomalous_flows):
        flow_logs.extend(generate_service_flow(is_anomalous=True))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_service_logs(logs, format='json', filename='service_api_logs'):
    """Save service logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} service logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "service_type": log["service"]["type"],
                "service_environment": log["service"]["environment"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "request_source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_size": log["response"]["size"],
                "response_time_ms": log["response"]["time_ms"],
                "error_message": log["response"].get("error", {}).get("message") if log["response"].get("error") else None,
                "error_code": log["response"].get("error", {}).get("code") if log["response"].get("error") else None,
                "correlation_id": log["correlation"]["id"],
                "trace_id": log["correlation"]["trace_id"],
                "span_id": log["correlation"]["span_id"],
                "parent_id": log["correlation"]["parent_id"],
                "cpu_usage": log["metrics"]["cpu_usage"],
                "memory_usage": log["metrics"]["memory_usage"],
                "active_connections": log["metrics"]["active_connections"],
                "is_anomalous": log["is_anomalous"]
            }
            
            # Add service metadata as flattened fields
            for key, value in log["service"]["metadata"].items():
                flat_log[f"metadata_{key}"] = value
            
            # Add request headers as flattened fields
            for key, value in log["request"]["headers"].items():
                flat_log[f"header_{key}"] = value
            
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} service logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_service_logs(logs):
    """Print analysis of the generated service logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by service type
    service_types = {}
    for log in logs:
        service_type = log['service']['type']
        service_types[service_type] = service_types.get(service_type, 0) + 1
    
    # Count by environment
    environments = {}
    for log in logs:
        env = log['service']['environment']
        environments[env] = environments.get(env, 0) + 1
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['response']['status_code']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Count unique correlation IDs (service flows)
    correlation_ids = set(log['correlation']['id'] for log in logs)
    trace_ids = set(log['correlation']['trace_id'] for log in logs)
    
    print("\n=== Service Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Unique service flows (correlation IDs): {len(correlation_ids)}")
    print(f"Unique traces: {len(trace_ids)}")
    
    print("\n=== Service Type Distribution ===")
    for svc_type, count in sorted(service_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{svc_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Environment Distribution ===")
    for env, count in sorted(environments.items(), key=lambda x: x[1], reverse=True):
        print(f"{env}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate 1000 logs with 20% anomalies
    logs = generate_service_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_service_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_service_logs(logs, format='json')
    csv_path = save_service_logs(logs, format='csv')
    
    print(f"\nService logs have been saved to {json_path} and {csv_path}")
    print("You can use these files for training your anomaly detection model.")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Service Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2))
    
    print("\n=== Sample Anomalous Service Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2))