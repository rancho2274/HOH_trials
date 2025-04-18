from flask import Flask, request, jsonify
import random
import time
import logging
import uuid
import datetime
import psutil
import json
import threading
import statistics
from collections import defaultdict, deque

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mock_api")

app = Flask(__name__)

# Data structures for metrics tracking
response_times = defaultdict(lambda: deque(maxlen=1000))  # Store last 1000 response times per endpoint
error_counts = defaultdict(int)
request_counts = defaultdict(int)
db_query_times = defaultdict(lambda: deque(maxlen=1000))

# Simulate database with query timing
class MockDatabase:
    def query(self, query_type):
        start_time = time.time()
        # Simulate different query types and performance characteristics
        if query_type == "read":
            time.sleep(random.uniform(0.01, 0.05))
        elif query_type == "write":
            time.sleep(random.uniform(0.02, 0.08))
        elif query_type == "complex":
            time.sleep(random.uniform(0.05, 0.15))
        
        query_time = time.time() - start_time
        return query_time, {"result": "data"}

db = MockDatabase()

# Generate a unique request ID
def generate_request_id():
    return str(uuid.uuid4())

# Simulate different environments
ENVIRONMENTS = ["on-premises", "aws-cloud", "azure-cloud", "gcp-cloud"]

# Dictionary to store simulated API health status
api_health = {
    "/users": "healthy",
    "/products": "healthy",
    "/orders": "healthy",
    "/payments": "healthy"
}

# Track third party dependencies
third_party_services = {
    "auth-service": {"url": "http://auth.example.com/verify", "status": "healthy"},
    "payment-gateway": {"url": "http://payments.example.com/process", "status": "healthy"},
    "analytics-service": {"url": "http://analytics.example.com/track", "status": "healthy"}
}

# Calculate percentiles for a given list of values
def calculate_percentiles(values, percentiles=[95, 99]):
    if not values:
        return {f"p{p}": 0 for p in percentiles}
    
    result = {}
    for p in percentiles:
        try:
            result[f"p{p}"] = statistics.quantiles(values, n=100)[p-1]
        except (ValueError, IndexError):
            # Handle cases with too few values
            result[f"p{p}"] = max(values) if values else 0
    
    return result

# Background thread to periodically calculate and log metrics
def metrics_reporter():
    while True:
        try:
            for endpoint, times in response_times.items():
                if times:
                    times_list = list(times)
                    percentiles = calculate_percentiles(times_list)
                    error_rate = error_counts[endpoint] / max(request_counts[endpoint], 1) * 100
                    
                    # Log the metrics
                    logger.info(
                        f"METRICS - Endpoint: {endpoint}, "
                        f"Total Requests: {request_counts[endpoint]}, "
                        f"Error Rate: {error_rate:.2f}%, "
                        f"Avg Response Time: {sum(times_list)/len(times_list):.4f}s, "
                        f"p95: {percentiles['p95']:.4f}s, "
                        f"p99: {percentiles['p99']:.4f}s, "
                        f"CPU: {psutil.cpu_percent()}%, "
                        f"Memory: {psutil.virtual_memory().percent}%"
                    )
            
            # Wait 60 seconds before the next report
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in metrics reporter: {str(e)}")
            time.sleep(60)

# Start the metrics reporter in a background thread
metrics_thread = threading.Thread(target=metrics_reporter, daemon=True)
metrics_thread.start()

# Simulate a database query
def perform_db_query(endpoint, query_type):
    query_time, result = db.query(query_type)
    # Store the query time for metrics
    db_query_times[endpoint].append(query_time)
    return query_time, result

# Simulate a call to a third-party service
def call_third_party_service(service_name):
    service = third_party_services.get(service_name)
    if not service:
        return 0, False
    
    start_time = time.time()
    # Simulate the API call based on service health
    if random.random() < 0.05:  # 5% chance of failure
        success = False
        time.sleep(random.uniform(0.5, 2.0))  # Failing calls often take longer
    else:
        success = True
        time.sleep(random.uniform(0.05, 0.3))
    
    duration = time.time() - start_time
    return duration, success

@app.route('/users', methods=['GET'])
def get_users():
    # Capture start time
    start_time = time.time()
    
    # Generate a request ID
    request_id = generate_request_id()
    
    # Capture system resources at start
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().percent
    
    # Increment request counter
    endpoint = "/users"
    request_counts[endpoint] += 1
    
    # Simulate processing time (between 50ms and 200ms normally)
    if api_health[endpoint] == "degraded":
        processing_time = random.uniform(0.3, 0.8)  # Slower when degraded
    elif api_health[endpoint] == "failing":
        processing_time = random.uniform(0.8, 2.0)  # Very slow when failing
    else:
        processing_time = random.uniform(0.05, 0.2)  # Normal performance
    
    time.sleep(processing_time)
    
    # Perform a database query
    db_time, db_result = perform_db_query(endpoint, "read")
    
    # Call authentication service
    auth_time, auth_success = call_third_party_service("auth-service")
    
    # Capture system resources at end
    end_cpu = psutil.cpu_percent()
    end_memory = psutil.virtual_memory().percent
    
    # Randomly decide if we'll have an error (1% chance normally)
    error_threshold = 0.01 if api_health[endpoint] == "healthy" else 0.1
    has_error = random.random() < error_threshold
    
    # Calculate response time
    response_time = time.time() - start_time
    response_times[endpoint].append(response_time)
    
    # Select a random environment
    environment = random.choice(ENVIRONMENTS)
    
    if has_error or not auth_success:
        # Increment error counter
        error_counts[endpoint] += 1
        
        # Log the error with enhanced data
        error_info = {
            "request_id": request_id,
            "endpoint": endpoint,
            "status": 500,
            "response_time": response_time,
            "environment": environment,
            "component_errors": {
                "auth_service": not auth_success,
                "database": db_time > 0.1,  # Flag slow DB queries
                "application": has_error
            },
            "resource_usage": {
                "cpu_start": start_cpu,
                "cpu_end": end_cpu,
                "memory_start": start_memory,
                "memory_end": end_memory
            },
            "dependencies": {
                "auth_service": {
                    "duration": auth_time,
                    "success": auth_success
                },
                "database": {
                    "duration": db_time,
                    "success": True
                }
            }
        }
        logger.error(f"ERROR - {json.dumps(error_info)}")
        return jsonify({"error": "Internal Server Error"}), 500
    else:
        # Log the successful request with enhanced data
        success_info = {
            "request_id": request_id,
            "endpoint": endpoint,
            "status": 200,
            "response_time": response_time,
            "environment": environment,
            "db_query_time": db_time,
            "resource_usage": {
                "cpu_delta": end_cpu - start_cpu,
                "memory_delta": end_memory - start_memory
            },
            "dependencies": {
                "auth_service": {
                    "duration": auth_time,
                    "success": True
                },
                "database": {
                    "duration": db_time,
                    "success": True
                }
            },
            "business_transaction": "user_profile_view",
            "user_type": random.choice(["free", "premium", "enterprise"])
        }
        logger.info(f"INFO - {json.dumps(success_info)}")
        return jsonify({"users": ["user1", "user2", "user3"]}), 200

@app.route('/products', methods=['GET'])
def get_products():
    # Similar to get_users but with product-specific logic
    start_time = time.time()
    request_id = generate_request_id()
    endpoint = "/products"
    request_counts[endpoint] += 1
    
    # Capture system resources
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().percent
    
    # Simulate processing and database queries
    # This endpoint intentionally has more complex logic and higher error rates
    
    # Perform multiple database queries for product information
    db_read_time, _ = perform_db_query(endpoint, "read")
    db_complex_time, _ = perform_db_query(endpoint, "complex")
    total_db_time = db_read_time + db_complex_time
    
    # Call analytics service
    analytics_time, analytics_success = call_third_party_service("analytics-service")
    
    # Determine if we'll have an error - products API is less stable
    has_error = random.random() < 0.3  # 30% error rate to match your logs
    
    # Calculate response time
    time.sleep(random.uniform(0.1, 0.3))  # Additional processing time
    response_time = time.time() - start_time
    response_times[endpoint].append(response_time)
    
    # Capture end resources
    end_cpu = psutil.cpu_percent()
    end_memory = psutil.virtual_memory().percent
    
    # Select environment
    environment = random.choice(ENVIRONMENTS)
    
    if has_error:
        error_counts[endpoint] += 1
        
        error_category = random.choice([
            "database_connection_error", 
            "timeout_error", 
            "validation_error",
            "resource_not_found"
        ])
        
        error_info = {
            "request_id": request_id,
            "endpoint": endpoint,
            "status": 500,
            "response_time": response_time,
            "environment": environment,
            "error_category": error_category,
            "component_errors": {
                "analytics_service": not analytics_success,
                "database": total_db_time > 0.2,
                "application": True
            },
            "resource_usage": {
                "cpu_delta": end_cpu - start_cpu,
                "memory_delta": end_memory - start_memory
            },
            "dependencies": {
                "analytics_service": {
                    "duration": analytics_time,
                    "success": analytics_success
                },
                "database": {
                    "duration": total_db_time,
                    "success": total_db_time < 0.2
                }
            }
        }
        logger.error(f"ERROR - {json.dumps(error_info)}")
        return jsonify({"error": "Failed to retrieve products"}), 500
    else:
        success_info = {
            "request_id": request_id,
            "endpoint": endpoint,
            "status": 200,
            "response_time": response_time,
            "environment": environment,
            "db_query_time": total_db_time,
            "resource_usage": {
                "cpu_delta": end_cpu - start_cpu,
                "memory_delta": end_memory - start_memory
            },
            "dependencies": {
                "analytics_service": {
                    "duration": analytics_time,
                    "success": analytics_success
                },
                "database": {
                    "read_duration": db_read_time,
                    "complex_duration": db_complex_time,
                    "success": True
                }
            },
            "business_transaction": "product_catalog_view",
            "product_count": random.randint(10, 50)
        }
        logger.info(f"INFO - {json.dumps(success_info)}")
        return jsonify({"products": ["product1", "product2", "product3"]}), 200

@app.route('/health/<endpoint>', methods=['POST'])
def change_health(endpoint):
    """Endpoint to change the health status of APIs for testing"""
    endpoint_path = f"/{endpoint}"
    if endpoint_path not in api_health:
        return jsonify({"error": "Unknown endpoint"}), 404
    
    data = request.json
    new_status = data.get('status')
    
    if new_status not in ["healthy", "degraded", "failing"]:
        return jsonify({"error": "Invalid status"}), 400
    
    api_health[endpoint_path] = new_status
    
    # Log the health status change
    logger.warning(
        f"HEALTH_CHANGE - Endpoint: {endpoint_path}, "
        f"New Status: {new_status}, "
        f"Timestamp: {datetime.datetime.now().isoformat()}"
    )
    
    return jsonify({"message": f"Health status of {endpoint_path} changed to {new_status}"}), 200

# New endpoint to get metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = {}
    
    for endpoint in api_health.keys():
        if endpoint in response_times and response_times[endpoint]:
            times_list = list(response_times[endpoint])
            percentiles = calculate_percentiles(times_list)
            error_rate = error_counts[endpoint] / max(request_counts[endpoint], 1) * 100
            
            metrics[endpoint] = {
                "total_requests": request_counts[endpoint],
                "error_rate": f"{error_rate:.2f}%",
                "avg_response_time": f"{sum(times_list)/len(times_list):.4f}s",
                "p95": f"{percentiles['p95']:.4f}s",
                "p99": f"{percentiles['p99']:.4f}s",
                "health_status": api_health[endpoint]
            }
            
            if endpoint in db_query_times and db_query_times[endpoint]:
                db_times = list(db_query_times[endpoint])
                db_percentiles = calculate_percentiles(db_times)
                metrics[endpoint]["db_query_metrics"] = {
                    "avg_time": f"{sum(db_times)/len(db_times):.4f}s",
                    "p95": f"{db_percentiles['p95']:.4f}s"
                }
    
    # Add system resource metrics
    metrics["system"] = {
        "cpu": f"{psutil.cpu_percent()}%",
        "memory": f"{psutil.virtual_memory().percent}%",
        "disk": f"{psutil.disk_usage('/').percent}%"
    }
    
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)