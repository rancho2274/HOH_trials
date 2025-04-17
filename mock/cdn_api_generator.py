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

# Define constants for CDN log generation
CDN_PROVIDERS = ["cloudfront", "akamai", "fastly", "cloudflare", "custom-cdn"]
CDN_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]
CDN_EDGE_LOCATIONS = [
    "iad", "sfo", "dfw", "lhr", "fra", "sin", "nrt", "syd", "gru", "bom", 
    "cdg", "ams", "ewr", "sea", "yyz", "icn", "mad", "mxp", "dub"
]

# Content types
CONTENT_TYPES = {
    "html": ["text/html", "application/xhtml+xml"],
    "image": ["image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"],
    "css": ["text/css"],
    "javascript": ["application/javascript", "text/javascript"],
    "json": ["application/json"],
    "font": ["font/woff", "font/woff2", "font/ttf", "font/otf"],
    "video": ["video/mp4", "video/webm"],
    "audio": ["audio/mpeg", "audio/ogg"]
}

# Cache status values
CACHE_STATUS = {
    "HIT": 0.70,    # 70% of requests are cache hits (good performance)
    "MISS": 0.20,   # 20% are cache misses (first time or expired content)
    "EXPIRED": 0.05, # 5% expired content (requires revalidation)
    "UPDATING": 0.02, # 2% content being updated
    "BYPASS": 0.03  # 3% bypass cache (special requests)
}

# HTTP status codes specific to CDN operations
CDN_STATUS_CODES = {
    200: "OK",
    206: "Partial Content",
    301: "Moved Permanently",
    302: "Found",
    304: "Not Modified",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    416: "Range Not Satisfiable",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout"
}

# Asset paths for CDN
ASSET_PATHS = {
    "image": [
        "/assets/images/products/{id}.jpg",
        "/assets/images/banners/{id}.png",
        "/assets/images/avatars/{id}.webp",
        "/assets/images/logos/{id}.svg",
        "/assets/images/icons/{id}.png"
    ],
    "css": [
        "/assets/css/main.css",
        "/assets/css/style.min.css",
        "/assets/css/theme/{id}.css",
        "/assets/css/components.css"
    ],
    "javascript": [
        "/assets/js/main.js",
        "/assets/js/vendor.min.js",
        "/assets/js/components/{id}.js",
        "/assets/js/app.js"
    ],
    "font": [
        "/assets/fonts/{id}.woff2",
        "/assets/fonts/{id}.woff",
        "/assets/fonts/{id}.ttf"
    ],
    "html": [
        "/index.html",
        "/products/{id}.html",
        "/category/{id}.html",
        "/about.html",
        "/contact.html"
    ],
    "json": [
        "/api/content/{id}.json",
        "/api/settings.json",
        "/api/meta/{id}.json"
    ],
    "video": [
        "/assets/videos/{id}.mp4",
        "/assets/videos/promo.mp4"
    ],
    "audio": [
        "/assets/audio/{id}.mp3",
        "/assets/audio/notification.mp3"
    ]
}

# Referrers
REFERRERS = [
    "https://www.google.com/",
    "https://www.facebook.com/",
    "https://twitter.com/",
    "https://www.instagram.com/",
    "https://www.linkedin.com/",
    "https://www.youtube.com/",
    "https://www.reddit.com/",
    "https://www.pinterest.com/",
    None  # Direct visits
]

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 10; Mobile; rv:88.0) Gecko/88.0 Firefox/88.0"
]

def generate_asset_path(content_type):
    """Generate a realistic asset path based on content type"""
    paths = ASSET_PATHS.get(content_type, ASSET_PATHS["image"])
    path = random.choice(paths)
    
    # Replace any {id} placeholders with realistic IDs
    if "{id}" in path:
        if "product" in path:
            path = path.replace("{id}", f"product-{random.randint(1000, 9999)}")
        elif "banner" in path:
            path = path.replace("{id}", f"banner-{random.randint(1, 20)}")
        elif "avatar" in path:
            path = path.replace("{id}", f"user-{random.randint(1, 10000)}")
        elif "theme" in path or "component" in path:
            path = path.replace("{id}", random.choice(["light", "dark", "blue", "red", "green", "corporate", "modern"]))
        elif "font" in path:
            path = path.replace("{id}", random.choice(["opensans", "roboto", "lato", "montserrat", "poppins"]))
        elif "category" in path:
            path = path.replace("{id}", random.choice(["electronics", "clothing", "home", "beauty", "sports", "books"]))
        else:
            path = path.replace("{id}", str(uuid.uuid4())[:8])
    
    return path

def select_content_type():
    """Select a content type based on realistic distribution"""
    weights = {
        "image": 0.45,  # Images are most common
        "javascript": 0.20,
        "css": 0.15,
        "html": 0.10,
        "font": 0.05,
        "json": 0.03,
        "video": 0.01,
        "audio": 0.01
    }
    
    content_type = random.choices(
        list(weights.keys()),
        weights=list(weights.values())
    )[0]
    
    return content_type, random.choice(CONTENT_TYPES[content_type])

def generate_cdn_log_entry(timestamp=None, correlation_id=None, is_anomalous=False):
    """Generate a single CDN log entry"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Generate CDN details
    cdn_provider = random.choice(CDN_PROVIDERS)
    cdn_region = random.choice(CDN_REGIONS)
    edge_location = random.choice(CDN_EDGE_LOCATIONS)
    pop_id = f"{edge_location}-{random.randint(1, 5)}"
    
    # Generate correlation ID if not provided (for tracing requests across services)
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Generate request ID
    request_id = f"cdn-req-{uuid.uuid4().hex[:16]}"
    
    # Select content type and MIME type
    content_type, mime_type = select_content_type()
    
    # Generate asset path
    path = generate_asset_path(content_type)
    
    # Generate client details
    client_ip = str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
    user_agent = random.choice(USER_AGENTS)
    referrer = random.choice(REFERRERS)
    
    # Determine cache status based on weights
    cache_statuses = list(CACHE_STATUS.keys())
    cache_weights = list(CACHE_STATUS.values())
    cache_status = random.choices(cache_statuses, weights=cache_weights)[0]
    
    # Determine HTTP status code
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous requests
        if random.random() < 0.7:
            # Server errors more common for anomalies
            status_code = random.choice([500, 502, 503, 504])
        else:
            # Client errors
            status_code = random.choice([400, 401, 403, 404, 408, 416, 429])
    else:
        # Normal requests mostly succeed
        weights = [0.95, 0.04, 0.01]  # 95% success, 4% client error, 1% server error
        status_category = random.choices(["success", "client_error", "server_error"], weights=weights)[0]
        
        if status_category == "success":
            if content_type == "html" and random.random() < 0.1:
                status_code = 304  # Not Modified (for HTML with If-Modified-Since)
            elif content_type in ["image", "video"] and random.random() < 0.2:
                status_code = 206  # Partial Content (for range requests)
            else:
                status_code = 200
        elif status_category == "client_error":
            status_code = random.choice([400, 403, 404, 429])
        else:
            status_code = random.choice([500, 502, 503, 504])
    
    # Calculate response size
    if status_code == 304:
        # Not Modified - no response body
        response_size = 0
    elif status_code >= 400:
        # Error responses are small
        response_size = random.randint(200, 1000)
    else:
        # Successful responses vary by content type
        if content_type == "image":
            response_size = random.randint(5000, 2000000)  # 5KB - 2MB
        elif content_type == "video":
            response_size = random.randint(1000000, 50000000)  # 1MB - 50MB
        elif content_type == "audio":
            response_size = random.randint(500000, 10000000)  # 500KB - 10MB
        elif content_type == "javascript" or content_type == "css":
            response_size = random.randint(2000, 500000)  # 2KB - 500KB
        elif content_type == "font":
            response_size = random.randint(10000, 100000)  # 10KB - 100KB
        elif content_type == "html":
            response_size = random.randint(1000, 100000)  # 1KB - 100KB
        else:
            response_size = random.randint(500, 50000)  # 500B - 50KB
    
    # Calculate response time
    if is_anomalous and random.random() < 0.8:
        # Anomalous response times are much higher
        base_time = random.uniform(0.5, 5.0)  # 500ms - 5s
    else:
        if cache_status == "HIT":
            # Cache hits are fast
            base_time = random.uniform(0.005, 0.050)  # 5-50ms
        elif cache_status == "MISS":
            # Cache misses are slower (need to fetch from origin)
            base_time = random.uniform(0.050, 0.500)  # 50-500ms
        elif cache_status in ["EXPIRED", "UPDATING"]:
            # Need to validate or update
            base_time = random.uniform(0.020, 0.300)  # 20-300ms
        else:  # BYPASS
            # Direct to origin
            base_time = random.uniform(0.100, 0.800)  # 100-800ms
    
    # Adjust time based on content type and size
    content_multiplier = 1.0
    if content_type == "video" or content_type == "audio":
        content_multiplier = random.uniform(1.2, 2.0)
    elif content_type == "image" and response_size > 1000000:  # Large images
        content_multiplier = random.uniform(1.1, 1.5)
    
    # Adjust for status code
    if status_code >= 500:
        status_multiplier = random.uniform(1.5, 3.0)
    elif status_code >= 400:
        status_multiplier = random.uniform(0.8, 1.2)
    else:
        status_multiplier = 1.0
    
    response_time = base_time * content_multiplier * status_multiplier
    
    # Create the CDN log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "cdn": {
            "provider": cdn_provider,
            "region": cdn_region,
            "pop": pop_id,
            "edge_location": edge_location
        },
        "request": {
            "id": request_id,
            "method": "GET",  # Almost always GET for CDN assets
            "path": path,
            "host": f"cdn-{random.randint(1, 5)}.example.com",
            "query_string": "" if random.random() < 0.9 else f"v={random.randint(1, 100)}",
            "protocol": "https",
            "headers": {
                "user-agent": user_agent,
                "referer": referrer,
                "accept": mime_type
            },
            "client_ip": client_ip,
            "country": fake.country_code(),
            "city": fake.city()
        },
        "response": {
            "status": status_code,
            "content_type": mime_type,
            "content_length": response_size,
            "cache_status": cache_status,
            "ttl": random.randint(60, 86400) if cache_status != "BYPASS" else 0,
            "age": random.randint(0, 3600) if cache_status in ["HIT", "EXPIRED"] else 0
        },
        "performance": {
            "response_time_sec": round(response_time, 6),
            "origin_fetch_time_sec": round(response_time * 0.8, 6) if cache_status != "HIT" else 0,
            "edge_processing_time_sec": round(response_time * 0.2, 6)
        },
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_cdn_request_flow(base_time=None, correlation_id=None, page_type=None, is_anomalous=False):
    """Generate a sequence of related CDN requests that would typically load a web page"""
    if base_time is None:
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Define different page types with typical resource patterns
    if page_type is None:
        page_type = random.choice(["product", "category", "homepage", "checkout"])
    
    # Define common resources needed for different page types
    page_patterns = {
        "product": {
            "html": 1,
            "css": 2,
            "javascript": 4,
            "image": 8,  # Product + related product images
            "font": 2,
            "json": 1
        },
        "category": {
            "html": 1,
            "css": 2,
            "javascript": 3,
            "image": 15,  # Many product thumbnails
            "font": 2,
            "json": 2  # Additional data like filters
        },
        "homepage": {
            "html": 1,
            "css": 3,
            "javascript": 5,
            "image": 12,  # Banner, featured products, etc.
            "font": 2,
            "json": 1,
            "video": 1  # Homepage might have a promo video
        },
        "checkout": {
            "html": 1,
            "css": 2,
            "javascript": 6,  # More JS for forms, validation, payment processing
            "image": 4,  # Fewer images
            "font": 2,
            "json": 3  # More API calls for pricing, shipping, etc.
        }
    }
    
    # Get the pattern for the selected page type
    pattern = page_patterns.get(page_type, page_patterns["product"])
    
    # Generate logs for each resource type
    flow_logs = []
    current_time = base_time
    
    # First, the HTML document
    html_log = generate_cdn_log_entry(
        timestamp=current_time,
        correlation_id=correlation_id,
        is_anomalous=is_anomalous and random.random() < 0.3
    )
    # Override with HTML-specific values
    html_log["request"]["path"] = f"/{page_type}.html" if page_type != "product" else f"/products/product-{random.randint(1000, 9999)}.html"
    html_log["response"]["content_type"] = "text/html"
    flow_logs.append(html_log)
    
    # For each resource type in the pattern
    for content_type, count in pattern.items():
        if content_type == "html":
            continue  # Already added the HTML document
        
        # Add a small delay for each subsequent resource
        for i in range(count):
            current_time += datetime.timedelta(milliseconds=random.randint(5, 50))
            
            # Determine if this specific resource should be anomalous
            resource_is_anomalous = is_anomalous and random.random() < 0.2
            
            resource_log = generate_cdn_log_entry(
                timestamp=current_time,
                correlation_id=correlation_id,
                is_anomalous=resource_is_anomalous
            )
            
            # Override to ensure content type matches
            mime_type = random.choice(CONTENT_TYPES[content_type])
            resource_log["response"]["content_type"] = mime_type
            resource_log["request"]["path"] = generate_asset_path(content_type)
            
            flow_logs.append(resource_log)
    
    return flow_logs

def generate_cdn_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of CDN logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.4)):  # 40% of normal logs are individual
        logs.append(generate_cdn_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_cdn_log_entry(is_anomalous=True))
    
    # Generate normal page flows
    num_normal_flows = int(num_normal * 0.6 / 10)  # Approximately 10 resources per page
    for _ in range(num_normal_flows):
        flow_logs.extend(generate_cdn_request_flow(is_anomalous=False))
    
    # Generate anomalous page flows
    num_anomalous_flows = int(num_anomalous * 0.7 / 10)  # Approximately 10 resources per page
    for _ in range(num_anomalous_flows):
        flow_logs.extend(generate_cdn_request_flow(is_anomalous=True))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_cdn_logs(logs, format='json', filename='cdn_logs'):
    """Save CDN logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} CDN logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "cdn_provider": log["cdn"]["provider"],
                "cdn_region": log["cdn"]["region"],
                "cdn_pop": log["cdn"]["pop"],
                "cdn_edge_location": log["cdn"]["edge_location"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "request_host": log["request"]["host"],
                "request_query_string": log["request"]["query_string"],
                "request_protocol": log["request"]["protocol"],
                "client_ip": log["request"]["client_ip"],
                "client_country": log["request"]["country"],
                "client_city": log["request"]["city"],
                "response_status": log["response"]["status"],
                "response_content_type": log["response"]["content_type"],
                "response_content_length": log["response"]["content_length"],
                "cache_status": log["response"]["cache_status"],
                "response_ttl": log["response"]["ttl"],
                "response_age": log["response"]["age"],
                "response_time_sec": log["performance"]["response_time_sec"],
                "origin_fetch_time_sec": log["performance"]["origin_fetch_time_sec"],
                "edge_processing_time_sec": log["performance"]["edge_processing_time_sec"],
                "correlation_id": log["tracing"]["correlation_id"],
                "is_anomalous": log["is_anomalous"]
            }
            
            # Add user agent
            if "user-agent" in log["request"]["headers"]:
                flat_log["user_agent"] = log["request"]["headers"]["user-agent"]
            
            # Add referrer if available
            if "referer" in log["request"]["headers"]:
                flat_log["referrer"] = log["request"]["headers"]["referer"]
            
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} CDN logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_cdn_logs(logs):
    """Print analysis of the generated CDN logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by content type
    content_types = {}
    for log in logs:
        content_type = log['response']['content_type']
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    # Group content types into categories
    content_categories = {}
    for content_type, count in content_types.items():
        category = next((cat for cat, mime_types in CONTENT_TYPES.items() 
                         if content_type in mime_types), "other")
        content_categories[category] = content_categories.get(category, 0) + count
    
    # Count by cache status
    cache_statuses = {}
    for log in logs:
        status = log['response']['cache_status']
        cache_statuses[status] = cache_statuses.get(status, 0) + 1
    
    # Count by HTTP status code
    status_codes = {}
    for log in logs:
        status = log['response']['status']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Count unique correlation IDs
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    # Calculate average response times
    normal_times = [log['performance']['response_time_sec'] for log in logs if not log.get('is_anomalous', False)]
    anomalous_times = [log['performance']['response_time_sec'] for log in logs if log.get('is_anomalous', False)]
    
    avg_normal_time = sum(normal_times) / len(normal_times) if normal_times else 0
    avg_anomalous_time = sum(anomalous_times) / len(anomalous_times) if anomalous_times else 0
    
    print("\n=== CDN Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Unique page loads (correlation IDs): {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time*1000:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time*1000:.2f} ms")
    
    print("\n=== Content Type Distribution ===")
    for content_type, count in sorted(content_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"{content_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Cache Status Distribution ===")
    for status, count in sorted(cache_statuses.items(), key=lambda x: x[1], reverse=True):
        print(f"{status}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== HTTP Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_cdn_logs(gateway_logs, num_logs=500):
    """Generate CDN logs that share correlation IDs with existing gateway logs"""
    cdn_logs = []
    
    # Extract correlation IDs from gateway logs
    correlation_ids = []
    for log in gateway_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            correlation_ids.append(log["tracing"]["correlation_id"])
    
    # Ensure we have correlation IDs to work with
    if not correlation_ids:
        print("No correlation IDs found in gateway logs. Generating standalone CDN logs.")
        return generate_cdn_logs(num_logs)
    
    # Generate CDN logs with existing correlation IDs
    for _ in range(num_logs):
        correlation_id = random.choice(correlation_ids)
        cdn_log = generate_cdn_log_entry(correlation_id=correlation_id)
        cdn_logs.append(cdn_log)
    
    return cdn_logs

if __name__ == "__main__":
    import os
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate CDN logs
    print("Generating CDN logs...")
    logs = generate_cdn_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_cdn_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_cdn_logs(logs, format='json')
    csv_path = save_cdn_logs(logs, format='csv')
    
    print(f"\nCDN logs have been saved to {json_path} and {csv_path}")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal CDN Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2)[:1000] + "... (truncated)")
    
    print("\n=== Sample Anomalous CDN Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2)[:1000] + "... (truncated)")
    
    # Generate page load examples
    print("\nGenerating example page load flows...")
    page_types = ["product", "category", "homepage", "checkout"]
    for page_type in page_types:
        page_flow = generate_cdn_request_flow(page_type=page_type)
        print(f"Generated {len(page_flow)} resources for {page_type} page")