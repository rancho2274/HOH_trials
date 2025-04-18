from flask import Flask, request, jsonify
import requests
import random
import time
import logging
import uuid
import json
import psutil
import threading
import statistics
from collections import defaultdict, deque

# Set up logging similar to the first API
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("order_service_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("order_service")

app = Flask(__name__)

# Data structures for metrics tracking
response_times = defaultdict(lambda: deque(maxlen=1000))
error_counts = defaultdict(int)
request_counts = defaultdict(int)
journey_times = defaultdict(lambda: deque(maxlen=1000))  # Track end-to-end journey times

# Generate a unique request ID
def generate_request_id():
    return str(uuid.uuid4())

# Simulate different environments
ENVIRONMENTS = ["on-premises", "aws-cloud", "azure-cloud", "gcp-cloud"]

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
            
            # Process journey metrics
            if journey_times['order_creation']:
                journey_list = list(journey_times['order_creation'])
                journey_percentiles = calculate_percentiles(journey_list)
                logger.info(
                    f"JOURNEY_METRICS - Transaction: order_creation, "
                    f"Total: {len(journey_list)}, "
                    f"Avg Time: {sum(journey_list)/len(journey_list):.4f}s, "
                    f"p95: {journey_percentiles['p95']:.4f}s, "
                    f"p99: {journey_percentiles['p99']:.4f}s"
                )
            
            # Wait 60 seconds before the next report
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in metrics reporter: {str(e)}")
            time.sleep(60)

# Start the metrics reporter in a background thread
metrics_thread = threading.Thread(target=metrics_reporter, daemon=True)
metrics_thread.start()

@app.route('/create-order', methods=['POST'])
def create_order():
    # Capture start time
    start_time = time.time()
    
    # Generate a request ID
    request_id = generate_request_id()
    
    # Capture system resources at start
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().percent
    
    # Increment request counter
    endpoint = "/create-order"
    request_counts[endpoint] += 1
    
    # Select a random environment
    environment = random.choice(ENVIRONMENTS)
    
    # Create context dictionary to track the entire journey
    context = {
        "request_id": request_id,
        "start_time": start_time,
        "environment": environment,
        "dependencies": {},
        "journey_steps": [],
        "success": True
    }
    
    # Log the start of this request
    request_start_data = {
        'request_id': request_id,
        'endpoint': endpoint,
        'environment': environment,
        'timestamp': time.time()
    }
    logger.info(f"REQUEST_START - {json.dumps(request_start_data)}")
    
    # First, call the user service to validate the user
    try:
        user_service_start = time.time()
        user_response = requests.get('http://localhost:5000/users')
        user_service_time = time.time() - user_service_start
        
        # Add to journey steps
        context["journey_steps"].append({
            "step": "user_validation",
            "start_time": user_service_start,
            "duration": user_service_time,
            "status_code": user_response.status_code,
            "success": user_response.status_code == 200
        })
        
        # Add to dependencies
        context["dependencies"]["user_service"] = {
            "duration": user_service_time,
            "status_code": user_response.status_code,
            "success": user_response.status_code == 200
        }
        
        # Log the call to the user service
        dependency_call_data = {
            'request_id': request_id,
            'source': endpoint,
            'target': 'user-service',
            'status': user_response.status_code,
            'response_time': user_service_time,
            'environment': environment
        }
        logger.info(f"DEPENDENCY_CALL - {json.dumps(dependency_call_data)}")
        
        if user_response.status_code != 200:
            context["success"] = False
            error_counts[endpoint] += 1
            
            # Log the error
            dependency_error_data = {
                'request_id': request_id,
                'source': endpoint,
                'target': 'user-service',
                'status': user_response.status_code,
                'environment': environment,
                'error_type': 'dependency_failure'
            }
            logger.error(f"DEPENDENCY_ERROR - {json.dumps(dependency_error_data)}")
            
            # Capture end metrics
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent
            response_time = time.time() - start_time
            response_times[endpoint].append(response_time)
            
            # Log the journey failure
            journey_failure_data = {
                'request_id': request_id,
                'journey': 'order_creation',
                'failed_step': 'user_validation',
                'total_time': response_time,
                'environment': environment,
                'resource_usage': {
                    'cpu_delta': end_cpu - start_cpu,
                    'memory_delta': end_memory - start_memory
                },
                'journey_context': context
            }
            logger.error(f"JOURNEY_FAILURE - {json.dumps(journey_failure_data)}")
            
            return jsonify({"error": "Failed to validate user"}), 500
            
    except Exception as e:
        context["success"] = False
        error_counts[endpoint] += 1
        
        # Capture error metrics
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        response_time = time.time() - start_time
        response_times[endpoint].append(response_time)
        
        # Log the exception
        exception_data = {
            'request_id': request_id,
            'source': endpoint,
            'target': 'user-service',
            'error': str(e),
            'environment': environment,
            'response_time': response_time,
            'resource_usage': {
                'cpu_delta': end_cpu - start_cpu,
                'memory_delta': end_memory - start_memory
            }
        }
        logger.error(f"EXCEPTION - {json.dumps(exception_data)}")
        
        return jsonify({"error": "User service unavailable"}), 503
    
    # Then, call the product service to check inventory
    try:
        product_service_start = time.time()
        product_response = requests.get('http://localhost:5000/products')
        product_service_time = time.time() - product_service_start
        
        # Add to journey steps
        context["journey_steps"].append({
            "step": "inventory_check",
            "start_time": product_service_start,
            "duration": product_service_time,
            "status_code": product_response.status_code,
            "success": product_response.status_code == 200
        })
        
        # Add to dependencies
        context["dependencies"]["product_service"] = {
            "duration": product_service_time,
            "status_code": product_response.status_code,
            "success": product_response.status_code == 200
        }
        
        # Log the call to the product service
        dependency_call_data = {
            'request_id': request_id,
            'source': endpoint,
            'target': 'product-service',
            'status': product_response.status_code,
            'response_time': product_service_time,
            'environment': environment
        }
        logger.info(f"DEPENDENCY_CALL - {json.dumps(dependency_call_data)}")
        
        if product_response.status_code != 200:
            context["success"] = False
            error_counts[endpoint] += 1
            
            # Log the error
            dependency_error_data = {
                'request_id': request_id,
                'source': endpoint,
                'target': 'product-service',
                'status': product_response.status_code,
                'environment': environment,
                'error_type': 'dependency_failure'
            }
            logger.error(f"DEPENDENCY_ERROR - {json.dumps(dependency_error_data)}")
            
            # Capture end metrics
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent
            response_time = time.time() - start_time
            response_times[endpoint].append(response_time)
            
            # Log the journey failure
            journey_failure_data = {
                'request_id': request_id,
                'journey': 'order_creation',
                'failed_step': 'inventory_check',
                'total_time': response_time,
                'environment': environment,
                'resource_usage': {
                    'cpu_delta': end_cpu - start_cpu,
                    'memory_delta': end_memory - start_memory
                },
                'journey_context': context
            }
            logger.error(f"JOURNEY_FAILURE - {json.dumps(journey_failure_data)}")
            
            return jsonify({"error": "Failed to check inventory"}), 500
            
    except Exception as e:
        context["success"] = False
        error_counts[endpoint] += 1
        
        # Capture error metrics
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        response_time = time.time() - start_time
        response_times[endpoint].append(response_time)
        
        # Log the exception
        exception_data = {
            'request_id': request_id,
            'source': endpoint,
            'target': 'product-service',
            'error': str(e),
            'environment': environment,
            'response_time': response_time,
            'resource_usage': {
                'cpu_delta': end_cpu - start_cpu,
                'memory_delta': end_memory - start_memory
            }
        }
        logger.error(f"EXCEPTION - {json.dumps(exception_data)}")
        
        return jsonify({"error": "Product service unavailable"}), 503
    
    # Simulate payment processing
    payment_service_start = time.time()
    payment_success = random.random() > 0.05  # 5% chance of payment failure
    time.sleep(random.uniform(0.1, 0.3))  # Payment processing time
    payment_service_time = time.time() - payment_service_start
    
    # Add to journey steps
    context["journey_steps"].append({
        "step": "payment_processing",
        "start_time": payment_service_start,
        "duration": payment_service_time,
        "success": payment_success
    })
    
    # Add to dependencies
    context["dependencies"]["payment_service"] = {
        "duration": payment_service_time,
        "success": payment_success
    }
    
    # Log the payment dependency
    payment_call_data = {
        'request_id': request_id,
        'source': endpoint,
        'target': 'payment-service',
        'status': 200 if payment_success else 500,
        'response_time': payment_service_time,
        'environment': environment
    }
    logger.info(f"DEPENDENCY_CALL - {json.dumps(payment_call_data)}")
    
    if not payment_success:
        context["success"] = False
        error_counts[endpoint] += 1
        
        # Capture end metrics
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        response_time = time.time() - start_time
        response_times[endpoint].append(response_time)
        
        # Log the journey failure
        journey_failure_data = {
            'request_id': request_id,
            'journey': 'order_creation',
            'failed_step': 'payment_processing',
            'total_time': response_time,
            'environment': environment,
            'resource_usage': {
                'cpu_delta': end_cpu - start_cpu,
                'memory_delta': end_memory - start_memory
            },
            'journey_context': context
        }
        logger.error(f"JOURNEY_FAILURE - {json.dumps(journey_failure_data)}")
        
        return jsonify({"error": "Payment processing failed"}), 500
        
    # Calculate total response time
    response_time = time.time() - start_time
    response_times[endpoint].append(response_time)
    journey_times['order_creation'].append(response_time)
    
    # Capture end system metrics
    end_cpu = psutil.cpu_percent()
    end_memory = psutil.virtual_memory().percent
    
    # Generate order ID
    order_id = f"ORD-{random.randint(10000, 99999)}"
    
    # Log the completed request with detailed metrics
    success_info = {
        "request_id": request_id,
        "endpoint": endpoint,
        "journey": "order_creation",
        "order_id": order_id,
        "status": 200,
        "response_time": response_time,
        "environment": environment,
        "resource_usage": {
            "cpu_delta": end_cpu - start_cpu,
            "memory_delta": end_memory - start_memory
        },
        "dependencies": context["dependencies"],
        "journey_steps": context["journey_steps"],
        "business_metrics": {
            "order_value": round(random.uniform(50, 500), 2),
            "items_count": random.randint(1, 10),
            "customer_type": random.choice(["new", "returning", "premium"])
        }
    }
    
    logger.info(f"JOURNEY_SUCCESS - {json.dumps(success_info)}")
    
    # Return a successful response
    return jsonify({
        "order_id": order_id,
        "status": "created"
    }), 200

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = {}
    
    # API metrics
    endpoint = "/create-order"
    if endpoint in response_times and response_times[endpoint]:
        times_list = list(response_times[endpoint])
        percentiles = calculate_percentiles(times_list)
        error_rate = error_counts[endpoint] / max(request_counts[endpoint], 1) * 100
        
        metrics["api"] = {
            "total_requests": request_counts[endpoint],
            "error_rate": f"{error_rate:.2f}%",
            "avg_response_time": f"{sum(times_list)/len(times_list):.4f}s",
            "p95": f"{percentiles['p95']:.4f}s",
            "p99": f"{percentiles['p99']:.4f}s"
        }
    
    # Journey metrics
    if journey_times['order_creation']:
        journey_list = list(journey_times['order_creation'])
        journey_percentiles = calculate_percentiles(journey_list)
        
        metrics["journey"] = {
            "name": "order_creation",
            "total_executions": len(journey_list),
            "avg_duration": f"{sum(journey_list)/len(journey_list):.4f}s",
            "p95": f"{journey_percentiles['p95']:.4f}s",
            "p99": f"{journey_percentiles['p99']:.4f}s"
        }
    
    # Add system resource metrics
    metrics["system"] = {
        "cpu": f"{psutil.cpu_percent()}%",
        "memory": f"{psutil.virtual_memory().percent}%",
        "disk": f"{psutil.disk_usage('/').percent}%"
    }
    
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)