import requests
import time
import random
import threading
import datetime
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("load_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("load_generator")

# Base URLs for our APIs
USER_API_URL = "http://localhost:5000/users"
PRODUCT_API_URL = "http://localhost:5000/products"
ORDER_API_URL = "http://localhost:5001/create-order"
METRICS_URL = "http://localhost:5000/metrics"
ORDER_METRICS_URL = "http://localhost:5001/metrics"

# URLs for changing API health for testing anomalies
HEALTH_URL_TEMPLATE = "http://localhost:5000/health/{}"

# Track request statistics
requests_stats = {
    "users": {"success": 0, "error": 0, "total_time": 0, "requests": 0},
    "products": {"success": 0, "error": 0, "total_time": 0, "requests": 0},
    "orders": {"success": 0, "error": 0, "total_time": 0, "requests": 0}
}

def make_api_request(url, name, with_payload=False):
    """Make a request to the specified API and print result"""
    try:
        start_time = time.time()
        
        payload = {}
        if with_payload and name.lower() == "order":
            # Generate more realistic order data
            num_items = random.randint(1, 5)
            items = []
            for _ in range(num_items):
                items.append({
                    "product_id": f"PROD-{random.randint(1000, 9999)}",
                    "quantity": random.randint(1, 10),
                    "price": round(random.uniform(9.99, 99.99), 2)
                })
            
            payload = {
                "customer_id": f"CUST-{random.randint(10000, 99999)}",
                "order_items": items,
                "shipping_address": {
                    "street": "123 Main St",
                    "city": "Anytown",
                    "state": "ST",
                    "zip": "12345"
                },
                "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"])
            }
        
        if url == ORDER_API_URL:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)
        
        duration = time.time() - start_time
        
        # Record statistics - fixed by mapping the API name to the key
        api_key = name.lower()
        if api_key == "user":
            api_key = "users"  # Fix for the key error
        elif api_key == "product":
            api_key = "products"
        elif api_key == "order":
            api_key = "orders"
            
        requests_stats[api_key]["requests"] += 1
        requests_stats[api_key]["total_time"] += duration
        
        if response.status_code >= 200 and response.status_code < 300:
            requests_stats[api_key]["success"] += 1
            logger.info(f"{name} API: {response.status_code}, Time: {duration:.4f}s, Success")
        else:
            requests_stats[api_key]["error"] += 1
            logger.warning(f"{name} API: {response.status_code}, Time: {duration:.4f}s, Error")
            
    except Exception as e:
        logger.error(f"Error calling {name} API: {str(e)}")
        api_key = name.lower()
        if api_key == "user":
            api_key = "users"  # Fix for the key error
        elif api_key == "product":
            api_key = "products"
        elif api_key == "order":
            api_key = "orders"
            
        requests_stats[api_key]["requests"] += 1
        requests_stats[api_key]["error"] += 1

def change_api_health(endpoint, new_status):
    """Change the health status of an API"""
    try:
        url = HEALTH_URL_TEMPLATE.format(endpoint)
        response = requests.post(url, json={"status": new_status})
        logger.info(f"Changed {endpoint} health to {new_status}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error changing API health: {str(e)}")

def normal_traffic_pattern():
    """Generate normal traffic to the APIs"""
    while True:
        # Randomly choose which API to call with weighted probability
        api_choice = random.choices(
            ["users", "products", "orders"],
            weights=[0.4, 0.4, 0.2],  # 40% users, 40% products, 20% orders
            k=1
        )[0]
        
        if api_choice == "users":
            make_api_request(USER_API_URL, "User")
        elif api_choice == "products":
            make_api_request(PRODUCT_API_URL, "Product")
        else:
            make_api_request(ORDER_API_URL, "Order", with_payload=True)
        
        # Sleep for a random time between requests (50-200ms)
        time.sleep(random.uniform(0.05, 0.2))

def create_anomaly():
    """Periodically create anomalies in the system"""
    while True:
        # Sleep for 2-5 minutes before creating an anomaly (shortened for testing)
        sleep_time = random.uniform(120, 300)
        logger.info(f"Next anomaly in {sleep_time:.1f} seconds")
        time.sleep(sleep_time)
        
        # Choose which API to degrade
        api_to_degrade = random.choice(["users", "products"])
        
        # Choose type of anomaly
        anomaly_type = random.choice(["degraded", "failing"])
        
        logger.info(f"Creating {anomaly_type} anomaly in {api_to_degrade} API")
        change_api_health(api_to_degrade, anomaly_type)
        
        # Keep the anomaly for 1-3 minutes (shortened for testing)
        anomaly_duration = random.uniform(60, 180)
        time.sleep(anomaly_duration)
        
        # Restore normal operation
        logger.info(f"Restoring {api_to_degrade} API to healthy")
        change_api_health(api_to_degrade, "healthy")

def burst_traffic():
    """Periodically create traffic bursts"""
    while True:
        # Sleep for 3-10 minutes before creating a traffic burst (shortened for testing)
        sleep_time = random.uniform(180, 600)
        logger.info(f"Next traffic burst in {sleep_time:.1f} seconds")
        time.sleep(sleep_time)
        
        logger.info("Starting traffic burst")
        
        # Create a burst of traffic for 30-90 seconds (shortened for testing)
        burst_end_time = time.time() + random.uniform(30, 90)
        
        while time.time() < burst_end_time:
            # Make multiple concurrent requests
            threads = []
            for _ in range(random.randint(5, 15)):
                api_choice = random.choice(["users", "products", "orders"])
                
                if api_choice == "users":
                    t = threading.Thread(target=make_api_request, args=(USER_API_URL, "User"))
                elif api_choice == "products":
                    t = threading.Thread(target=make_api_request, args=(PRODUCT_API_URL, "Product"))
                else:
                    t = threading.Thread(target=make_api_request, args=(ORDER_API_URL, "Order", True))
                
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
            
            # Small pause between bursts
            time.sleep(random.uniform(0.01, 0.05))
        
        logger.info("Traffic burst completed")

def collect_metrics():
    """Periodically collect and log metrics"""
    while True:
        try:
            # Get metrics from the user/product API
            response = requests.get(METRICS_URL)
            if response.status_code == 200:
                api_metrics = response.json()
                logger.info(f"API Metrics: {json.dumps(api_metrics)}")
            else:
                logger.warning(f"Failed to get API metrics: {response.status_code}")
            
            # Get metrics from the order API
            response = requests.get(ORDER_METRICS_URL)
            if response.status_code == 200:
                order_metrics = response.json()
                logger.info(f"Order API Metrics: {json.dumps(order_metrics)}")
            else:
                logger.warning(f"Failed to get Order API metrics: {response.status_code}")
            
            # Calculate and log current load generator statistics
            for api, stats in requests_stats.items():
                if stats["requests"] > 0:
                    success_rate = (stats["success"] / stats["requests"]) * 100
                    avg_time = stats["total_time"] / stats["requests"] if stats["requests"] > 0 else 0
                    logger.info(
                        f"Load Stats - API: {api}, "
                        f"Requests: {stats['requests']}, "
                        f"Success Rate: {success_rate:.2f}%, "
                        f"Avg Time: {avg_time:.4f}s"
                    )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
        
        # Sleep for 60 seconds before next collection
        time.sleep(60)

def latency_variation():
    """Simulate network latency variation over time"""
    base_latency = 0.05  # base 50ms latency
    while True:
        # Normal variation (small fluctuations)
        for _ in range(20):
            # Vary latency by small amounts
            current_latency = base_latency * random.uniform(0.8, 1.2)
            time.sleep(5)  # Update every 5 seconds
        
        # Occasional latency spike
        if random.random() < 0.3:  # 30% chance
            spike_duration = random.uniform(10, 30)  # 10-30 seconds
            spike_multiplier = random.uniform(2, 5)  # 2-5x normal latency
            logger.warning(f"Network latency spike: {base_latency * spike_multiplier:.4f}s for {spike_duration:.1f} seconds")
            time.sleep(spike_duration)

def dependency_simulation():
    """Simulate periodic third-party dependency issues"""
    while True:
        # Sleep for 5-15 minutes before dependency issue
        sleep_time = random.uniform(300, 900)
        time.sleep(sleep_time)
        
        # Choose which dependency to affect
        dependency = random.choice(["database", "auth-service", "payment-gateway"])
        issue_type = random.choice(["slowdown", "error", "timeout"])
        
        logger.warning(f"Simulating {issue_type} in {dependency} dependency")
        
        # Duration of the dependency issue
        issue_duration = random.uniform(60, 240)  # 1-4 minutes
        time.sleep(issue_duration)
        
        logger.info(f"Dependency {dependency} restored to normal operation")

if __name__ == "__main__":
    logger.info("Starting load generator...")
    
    # Start normal traffic pattern in a thread
    normal_traffic_thread = threading.Thread(target=normal_traffic_pattern)
    normal_traffic_thread.daemon = True
    normal_traffic_thread.start()
    
    # Start anomaly creator in a thread
    anomaly_thread = threading.Thread(target=create_anomaly)
    anomaly_thread.daemon = True
    anomaly_thread.start()
    
    # Start burst traffic generator in a thread
    burst_thread = threading.Thread(target=burst_traffic)
    burst_thread.daemon = True
    burst_thread.start()
    
    # Start metrics collector in a thread
    metrics_thread = threading.Thread(target=collect_metrics)
    metrics_thread.daemon = True
    metrics_thread.start()
    
    # Start latency variation simulator in a thread
    latency_thread = threading.Thread(target=latency_variation)
    latency_thread.daemon = True
    latency_thread.start()
    
    # Start dependency simulation in a thread
    dependency_thread = threading.Thread(target=dependency_simulation)
    dependency_thread.daemon = True
    dependency_thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down load generator...")