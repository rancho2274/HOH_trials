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

# Define constants for Search Service log generation - limited to location search only
SEARCH_TYPES = ["location_search", "destination_info"]

SEARCH_ENVIRONMENTS = ["dev", "staging", "production", "test"]
SEARCH_SERVERS = ["search-primary", "search-replica", "search-cache"]
SEARCH_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Limit destinations to only New York, Paris, and London
POPULAR_DESTINATIONS = ["New York", "Paris", "London"]

# Hotel chains for location search results
HOTEL_CHAINS = [
    "Marriott", "Hilton", "Hyatt", "InterContinental", "Accor",
    "Wyndham", "Best Western", "Radisson", "Four Seasons",
    "Shangri-La", "Mandarin Oriental", "Ritz-Carlton",
    "Holiday Inn", "Sheraton", "Westin", "Novotel"
]

ROOM_TYPES = ["Single", "Double", "Twin", "Suite", "Deluxe", "Presidential"]

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "TravelApp/2.3.0 iOS/15.4.1",
    "TravelApp/2.4.1 Android/12.0"
]

# HTTP Methods
HTTP_METHODS = ["GET", "POST"]

# Status codes with their general meanings
SUCCESS_STATUS_CODES = [200, 201, 204]
ERROR_STATUS_CODES = {
    "client_error": [400, 401, 403, 404, 422, 429],
    "server_error": [500, 502, 503, 504]
}

# Currencies
CURRENCIES = ["USD", "EUR", "GBP"]

# Error messages by status code
ERROR_MESSAGES = {
    400: [
        "Invalid search parameters",
        "Missing required destination",
        "Invalid date format",
        "Search criteria too broad"
    ],
    401: [
        "Authentication required",
        "Session expired",
        "Invalid API key",
        "Missing authentication token"
    ],
    403: [
        "Search access denied",
        "Geo-restriction applies",
        "Rate limit exceeded for this API key",
        "Premium feature access denied"
    ],
    404: [
        "No results found",
        "Destination not supported",
        "No available inventory"
    ],
    422: [
        "Unprocessable search parameters",
        "Invalid filter combination",
        "Date range too wide",
        "Location not recognized"
    ],
    429: [
        "Search rate limit exceeded",
        "Too many search requests",
        "Temporary search restriction",
        "API quota exceeded"
    ],
    500: [
        "Search engine error",
        "Unable to process search",
        "Internal server error",
        "Unexpected condition"
    ],
    502: [
        "Bad gateway",
        "Upstream provider error",
        "External service failure",
        "API gateway error"
    ],
    503: [
        "Search service temporarily unavailable",
        "Maintenance in progress",
        "Provider service unavailable",
        "System overloaded"
    ],
    504: [
        "Gateway timeout",
        "Search timed out",
        "Provider response timeout",
        "Request processing timeout"
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
        "User-Agent": random.choice(USER_AGENTS),
        "X-Request-ID": str(uuid.uuid4())
    }
    
    if auth_required:
        auth_type = random.choice(["api-key", "jwt", "oauth2"])
        if auth_type == "api-key":
            headers["X-API-Key"] = f"apk_{uuid.uuid4().hex[:24]}"
        elif auth_type in ["jwt", "oauth2"]:
            headers["Authorization"] = f"Bearer {uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}"
    
    return headers

def generate_travel_dates():
    """Generate realistic travel dates"""
    today = datetime.datetime.now()
    departure_offset = random.randint(7, 180)  # 1 week to 6 months in the future
    departure_date = today + datetime.timedelta(days=departure_offset)
    
    trip_length = random.randint(1, 21)  # 1 day to 3 weeks
    return_date = departure_date + datetime.timedelta(days=trip_length)
    
    return departure_date.strftime("%Y-%m-%d"), return_date.strftime("%Y-%m-%d")
def write_logs_as_json_array(log_entries, file_path):
    """
    Writes log entries as a valid JSON array to a file.
    Creates a new file or overwrites an existing one.
    """
    import json
    import os
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Write logs as a valid JSON array
        with open(file_path, 'w') as f:
            # Start with an opening bracket
            f.write('[\n')
            
            # Write each log entry followed by a comma (except the last one)
            for i, log in enumerate(log_entries):
                f.write(json.dumps(log))
                if i < len(log_entries) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            
            # End with a closing bracket
            f.write(']\n')
            
        print(f"Successfully wrote {len(log_entries)} logs to {file_path}")
        return True
    except Exception as e:
        import traceback
        print(f"ERROR writing logs to {file_path}: {str(e)}")
        print(traceback.format_exc())
        return False

def calculate_response_time(search_type, status_code, is_anomalous=False):
    """Calculate realistic response times for search operations"""
    # Base times in seconds
    base_times = {
        "location_search": 0.3,
        "destination_info": 0.1
    }
    
    # Get base time for this search type
    base_time = base_times.get(search_type, 0.3)
    
    # Add variability
    response_time = base_time * random.uniform(0.8, 1.2)
    
    # Adjust for status code
    if status_code >= 500:
        # Server errors are often slower
        response_time *= random.uniform(2.0, 5.0)
    elif status_code == 429:
        # Rate limiting checks are quick
        response_time *= 0.5
    elif status_code >= 400:
        # Client errors are generally fast
        response_time *= random.uniform(0.7, 1.0)
    
    # Anomalous times
    if is_anomalous:
        if random.random() < 0.7:
            # Most anomalies are slower
            response_time *= random.uniform(3.0, 10.0)
        else:
            # Some anomalies are suspiciously fast
            response_time *= random.uniform(0.1, 0.3)
    
    return round(response_time, 3)

def get_error_message(status_code):
    """Get an appropriate error message based on status code"""
    if status_code < 400:
        return None
    
    if status_code in ERROR_MESSAGES:
        return random.choice(ERROR_MESSAGES[status_code])
    else:
        return "Unknown error occurred"

def generate_search_log_entry(timestamp=None, search_type=None, 
                            correlation_id=None, user_id=None, 
                            is_anomalous=False, parent_request_id=None):
    """Generate a single search service log entry"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    # Select search type if not provided
    if search_type is None:
        search_type = random.choice(SEARCH_TYPES)
    
    # Environment and region
    environment = random.choice(SEARCH_ENVIRONMENTS)
    region = random.choice(SEARCH_REGIONS)
    server = random.choice(SEARCH_SERVERS)
    instance_id = f"{server}-{region}-{random.randint(1, 5)}"
    
    # Request details
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    http_method = "GET" if random.random() < 0.8 else "POST"  # Most searches are GET, some are POST with complex params
    
    # Get path based on search type
    if search_type == "location_search":
        path = "/api/search/locations"
    elif search_type == "destination_info":
        path = "/api/destinations"
    else:
        path = f"/api/search/{search_type}"
    
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    client_type = random.choice(["web-app", "mobile-ios", "mobile-android", "partner-api", "internal"])
    source_ip = generate_ip()
    
    # Determine if authentication is needed (80% of searches require auth)
    auth_required = user_id is not None or random.random() < 0.8
    
    # Generate request headers
    request_headers = generate_request_headers(auth_required=auth_required)
    
    # Generate travel dates for search
    check_in_date, check_out_date = generate_travel_dates()
    
    # Generate request body based on search type
    request_body = {}
    
    if search_type == "location_search":
        destination = random.choice(POPULAR_DESTINATIONS)
            
        request_body = {
            "destination": destination,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "rooms": random.randint(1, 3),
            "guests": {
                "adults": random.randint(1, 4),
                "children": random.randint(0, 2)
            },
            "star_rating_min": random.randint(1, 5)
        }
        
        if is_anomalous and random.random() < 0.5:
            # Create anomalous request
            request_body["check_in_date"] = check_out_date  # Check-in after check-out
            request_body["check_out_date"] = check_in_date
            
    elif search_type == "destination_info":
        request_body = {
            "destination": random.choice(POPULAR_DESTINATIONS),
            "include_attractions": True,
            "include_weather": True,
            "include_travel_advisories": random.choice([True, False])
        }
    
    # Determine status code
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous requests
        error_type = random.choice(["client_error", "server_error"])
        status_code = random.choice(ERROR_STATUS_CODES[error_type])
    else:
        # Normal requests mostly succeed
        weights = [0.95, 0.05]  # 95% success, 5% error
        status_category = random.choices(["success", "error"], weights=weights)[0]
        
        if status_category == "success":
            status_code = random.choice(SUCCESS_STATUS_CODES)
        else:
            error_type = random.choice(["client_error", "server_error"])
            status_code = random.choice(ERROR_STATUS_CODES[error_type])
    
    # Generate response body
    response_body = {}
    
    if status_code < 400:  # Success responses
        if search_type == "location_search":
            num_results = random.randint(1, 15) if random.random() < 0.9 else 0
            destination = request_body.get("destination", "Unknown")
            
            hotels = []
            for _ in range(num_results):
                hotel_id = f"hotel-{uuid.uuid4().hex[:8]}"
                
                # Generate hotel name based on destination
                hotel_chain = random.choice(HOTEL_CHAINS)
                hotel_name = f"{hotel_chain} {destination}" if random.random() < 0.3 else \
                            f"{destination} {hotel_chain}" if random.random() < 0.5 else \
                            f"{hotel_chain} {random.choice(['Resort', 'Hotel', 'Suites', 'Inn'])}"
                
                hotel = {
                    "hotel_id": hotel_id,
                    "name": hotel_name,
                    "destination": destination,
                    "star_rating": random.randint(request_body.get("star_rating_min", 1), 5),
                    "user_rating": round(random.uniform(3.0, 5.0), 1),
                    "price_per_night": {
                        "amount": round(random.uniform(50, 1000), 2),
                        "currency": random.choice(CURRENCIES)
                    },
                    "available_rooms": random.randint(1, 20),
                    "amenities": random.sample(["pool", "wifi", "breakfast", "parking", "gym", "spa", "restaurant"], 
                                             k=random.randint(2, 7))
                }
                hotels.append(hotel)
            
            response_body = {
                "search_id": f"search-{uuid.uuid4().hex[:8]}",
                "destination": destination,
                "num_results": len(hotels),
                "hotels": hotels
            }
            
        elif search_type == "destination_info":
            destination = request_body.get("destination", "Unknown")
            
            response_body = {
                "destination": destination,
                "country": "France" if destination == "Paris" else 
                          "United Kingdom" if destination == "London" else
                          "United States" if destination == "New York" else
                          fake.country(),
                "description": fake.paragraph(nb_sentences=5),
                "popular_attractions": [fake.text(max_nb_chars=20) for _ in range(random.randint(3, 8))],
                "weather": {
                    "temp_celsius_avg": random.randint(10, 35),
                    "precipitation_chance": random.randint(0, 100),
                    "forecast": random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy"])
                } if request_body.get("include_weather", False) else None,
                "travel_advisory": random.choice(["Low Risk", "Medium Risk", "Exercise Caution", "Reconsider Travel"]) 
                if request_body.get("include_travel_advisories", False) else None
            }
    else:
        # Error response
        error_message = get_error_message(status_code)
        response_body = {
            "error": "error_code",
            "error_description": error_message,
            "search_id": f"search-{uuid.uuid4().hex[:8]}",
            "status": status_code,
            "timestamp": timestamp.isoformat()
        }
    
    # Calculate response time
    response_time = calculate_response_time(search_type, status_code, is_anomalous)
    
    # Create the search log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "search_service": {
            "type": search_type,
            "environment": environment,
            "region": region,
            "instance_id": instance_id
        },
        "request": {
            "id": request_id,
            "method": http_method,
            "path": path,
            "headers": request_headers,
            "body": request_body,
            "client_id": client_id,
            "client_type": client_type,
            "source_ip": source_ip
        },
        "response": {
            "status_code": status_code,
            "body": response_body,
            "time_ms": round(response_time * 1000, 2)
        },
        "user": {
            "user_id": user_id,
            "authenticated": auth_required
        },
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "parent_request_id": parent_request_id
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_related_searches(correlation_id, user_id=None, base_timestamp=None, is_anomalous=False):
    """Generate a sequence of related search requests with the same correlation ID"""
    related_logs = []
    
    if base_timestamp is None:
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    current_timestamp = base_timestamp
    
    # Choose a search sequence pattern - simplified to match our limited search types
    search_patterns = [
        # Location-focused pattern
        ["destination_info", "location_search"],
        # Simple location search
        ["location_search"],
        # Information first
        ["destination_info"]
    ]
    
    pattern = random.choice(search_patterns)
    
    # Generate logs for each search in the pattern
    for i, search_type in enumerate(pattern):
        # Add some time between searches
        current_timestamp += datetime.timedelta(seconds=random.uniform(2, 15))
        
        # Determine if this search should be anomalous
        search_is_anomalous = is_anomalous and (i == len(pattern) - 1 or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_search_log_entry(
            timestamp=current_timestamp,
            search_type=search_type,
            correlation_id=correlation_id,
            user_id=user_id,
            is_anomalous=search_is_anomalous,
            parent_request_id=None if i == 0 else related_logs[-1]["request"]["id"]
        )
        
        related_logs.append(log_entry)
        
        # If we get an error, we might stop the sequence
        if search_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return related_logs

def generate_search_logs(num_logs=1000, anomaly_percentage=15):
    """Generate a dataset of search service logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    print(f"Generating {int(num_normal * 0.6)} individual normal search logs...")
    for _ in range(int(num_normal * 0.6)):  # 60% of normal logs are individual
        logs.append(generate_search_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    print(f"Generating {int(num_anomalous * 0.4)} individual anomalous search logs...")
    for _ in range(int(num_anomalous * 0.4)):  # 40% of anomalous logs are individual
        logs.append(generate_search_log_entry(is_anomalous=True))
    
    # Generate normal search sequences
    num_normal_flows = int(num_normal * 0.4 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_normal_flows} normal search flows...")
    for _ in range(num_normal_flows):
        # Generate user ID (80% of flows have a user ID)
        user_id = str(uuid.uuid4()) if random.random() < 0.8 else None
        
        correlation_id = generate_correlation_id()
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_searches(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_timestamp,
            is_anomalous=False
        ))
    
    # Generate anomalous search sequences
    num_anomalous_flows = int(num_anomalous * 0.6 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_anomalous_flows} anomalous search flows...")
    for _ in range(num_anomalous_flows):
        # Generate user ID (80% of flows have a user ID)
        user_id = str(uuid.uuid4()) if random.random() < 0.8 else None
        
        correlation_id = generate_correlation_id()
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_searches(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_timestamp,
            is_anomalous=True
        ))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_logs_to_file(logs, format='json', filename='search_logs'):
    """Save logs to a file in the specified format"""
    import json
    import pandas as pd
    
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        # Use the new function to write as JSON array
        write_logs_as_json_array(logs, file_path)
        return file_path
    
    elif format.lower() == 'csv':
        # CSV output remains unchanged
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "search_type": log["search_service"]["type"],
                "environment": log["search_service"]["environment"],
                "region": log["search_service"]["region"],
                "instance_id": log["search_service"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "client_id": log["request"]["client_id"],
                "client_type": log["request"]["client_type"],
                "source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_time_ms": log["response"]["time_ms"],
                "user_id": log["user"]["user_id"],
                "authenticated": log["user"]["authenticated"],
                "correlation_id": log["tracing"]["correlation_id"],
                "parent_request_id": log["tracing"].get("parent_request_id"),
                "is_anomalous": log["is_anomalous"]
            }
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_search_logs(logs):
    """Print analysis of the generated search logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by search type
    search_types = {}
    for log in logs:
        search_type = log['search_service']['type']
        search_types[search_type] = search_types.get(search_type, 0) + 1
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['response']['status_code']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Success vs. failure rate
    success_count = sum(1 for log in logs if log['response']['status_code'] < 400)
    failure_count = total_logs - success_count
    
    # Calculate average response times
    normal_times = [log['response']['time_ms'] for log in logs if not log.get('is_anomalous', False)]
    anomalous_times = [log['response']['time_ms'] for log in logs if log.get('is_anomalous', False)]
    
    avg_normal_time = sum(normal_times) / len(normal_times) if normal_times else 0
    avg_anomalous_time = sum(anomalous_times) / len(anomalous_times) if anomalous_times else 0
    
    # Count authenticated searches
    auth_count = sum(1 for log in logs if log['user']['authenticated'])
    
    # Count unique correlation IDs (search flows)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    # Count destinations
    destinations = {}
    for log in logs:
        if 'body' in log['request'] and 'destination' in log['request']['body']:
            destination = log['request']['body']['destination']
            destinations[destination] = destinations.get(destination, 0) + 1
    
    print("\n=== Search Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Success rate: {success_count/total_logs*100:.2f}%")
    print(f"Failure rate: {failure_count/total_logs*100:.2f}%")
    print(f"Authenticated searches: {auth_count/total_logs*100:.2f}%")
    print(f"Unique search flows: {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Search Type Distribution ===")
    for search_type, count in sorted(search_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{search_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Destination Distribution ===")
    for destination, count in sorted(destinations.items(), key=lambda x: x[1], reverse=True):
        print(f"{destination}: {count} searches")

def generate_interconnected_search_logs(auth_logs=None, num_logs=500):
    """Generate search logs that share correlation IDs with existing auth logs"""
    search_logs = []
    
    # Extract correlation IDs from auth logs
    correlation_ids = []
    user_id_map = {}
    for log in auth_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            correlation_ids.append(log["tracing"]["correlation_id"])
            
            # Map user IDs to correlation IDs
            if "operation" in log and "user_id" in log["operation"] and log["operation"]["user_id"]:
                user_id_map[log["tracing"]["correlation_id"]] = log["operation"]["user_id"]
    
    # If no correlation IDs found, generate standalone logs
    if not correlation_ids:
        print("No correlation IDs found in auth logs. Generating standalone search logs.")
        return generate_search_logs(num_logs)
    
    # For each correlation ID, generate a small search flow
    print(f"Generating search logs connected to {len(correlation_ids)} auth flows...")
    for correlation_id in correlation_ids:
        # Get user ID if available
        user_id = user_id_map.get(correlation_id)
        
        # Generate a base timestamp
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # Generate search flow with this correlation ID
        flow_logs = generate_related_searches(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_time,
            is_anomalous=random.random() < 0.15  # 15% chance of anomalous flow
        )
        
        search_logs.extend(flow_logs)
        
        # Stop if we've generated enough logs
        if len(search_logs) >= num_logs:
            break
    
    # If we need more logs, add some individual ones
    while len(search_logs) < num_logs:
        # 50% chance to reuse an existing correlation ID or create a new one
        if random.random() < 0.5 and correlation_ids:
            correlation_id = random.choice(correlation_ids)
            user_id = user_id_map.get(correlation_id)
        else:
            correlation_id = generate_correlation_id()
            user_id = None
            
        search_logs.append(generate_search_log_entry(
            correlation_id=correlation_id,
            user_id=user_id
        ))