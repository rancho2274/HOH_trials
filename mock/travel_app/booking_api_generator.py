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

# Define constants for Booking Service log generation
BOOKING_OPERATIONS = [
    "create_booking", "update_booking", "cancel_booking", 
    "get_booking_details", "add_passenger", "remove_passenger", 
    "add_special_request", "booking_confirmation"
]

BOOKING_ENVIRONMENTS = ["dev", "staging", "production", "test"]
BOOKING_SERVERS = ["booking-primary", "booking-replica", "booking-backup"]
BOOKING_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Booking types
BOOKING_TYPES = ["flight", "hotel", "car", "package"]

# Booking statuses
BOOKING_STATUSES = [
    "pending", "confirmed", "cancelled", "completed", "on_hold"
]

# Passenger types
PASSENGER_TYPES = ["adult", "child", "infant"]

# Special requests
SPECIAL_REQUESTS = [
    "wheelchair_assistance", "vegetarian_meal", "vegan_meal", 
    "kosher_meal", "halal_meal", "extra_legroom",
    "bassinet", "assistance_animal", "airport_pickup"
]

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
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

# Status codes with their general meanings
SUCCESS_STATUS_CODES = [200, 201, 204]
ERROR_STATUS_CODES = {
    "client_error": [400, 401, 403, 404, 422, 429],
    "server_error": [500, 502, 503, 504]
}
def save_booking_logs(logs, format='json', filename='booking_logs', single_line=True):
    """Save logs to a file in the specified format"""
    import json
    import pandas as pd
    
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        
        if single_line:
            # Write each log as a separate JSON line (JSON Lines format)
            with open(file_path, 'w') as f:
                for log in logs:
                    f.write(json.dumps(log) + '\n')
        else:
            # Original pretty-printed format
            with open(file_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        print(f"Saved {len(logs)} logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV (implementation unchanged)
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "booking_type": log["booking_service"]["type"],
                "operation": log["booking_service"]["operation"],
                "environment": log["booking_service"]["environment"],
                "region": log["booking_service"]["region"],
                "instance_id": log["booking_service"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "client_id": log["request"]["client_id"],
                "client_type": log["request"]["client_type"],
                "source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_time_ms": log["response"]["time_ms"],
                "user_id": log["user"]["user_id"],
                "booking_id": log["booking_id"],
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

# Currencies
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD", "AED"]

# Error messages by status code
ERROR_MESSAGES = {
    400: [
        "Invalid booking details",
        "Missing passenger information",
        "Invalid date format",
        "Booking parameters incomplete"
    ],
    401: [
        "Authentication required for booking",
        "Session expired",
        "Invalid API key",
        "Missing authentication token"
    ],
    403: [
        "Booking permission denied",
        "Account restricted",
        "Rate limit exceeded for this API key",
        "Premium feature access denied"
    ],
    404: [
        "Booking not found",
        "Requested inventory not found",
        "Passenger record not found",
        "Resource not available"
    ],
    422: [
        "Booking validation failed",
        "Inconsistent traveler information",
        "Invalid booking parameters",
        "Business rule validation failed"
    ],
    429: [
        "Booking rate limit exceeded",
        "Too many booking attempts",
        "API quota exceeded",
        "Please retry later"
    ],
    500: [
        "Booking system error",
        "Unable to confirm booking",
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
        "Booking service temporarily unavailable",
        "Maintenance in progress",
        "Provider service unavailable",
        "System overloaded"
    ],
    504: [
        "Gateway timeout",
        "Booking process timed out",
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

def generate_request_headers(auth_required=True):
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

def generate_passenger_details(num_passengers=None):
    """Generate passenger details for bookings"""
    if num_passengers is None:
        num_passengers = random.randint(1, 5)
    
    passengers = []
    
    for _ in range(num_passengers):
        passenger_type = random.choices(
            PASSENGER_TYPES, weights=[0.8, 0.15, 0.05])[0]
        
        passenger = {
            "id": str(uuid.uuid4()),
            "type": passenger_type,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "nationality": fake.country_code(),
            "date_of_birth": fake.date_of_birth(minimum_age=18 if passenger_type == "adult" else 0, 
                                               maximum_age=80 if passenger_type == "adult" else 
                                               (12 if passenger_type == "child" else 2)).strftime("%Y-%m-%d"),
        }
        
        if passenger_type == "adult":
            passenger["email"] = fake.email()
            passenger["phone"] = fake.phone_number()
        
        passengers.append(passenger)
    
    return passengers
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
def calculate_response_time(operation, status_code, is_anomalous=False):
    """Calculate realistic response times for booking operations"""
    # Base times in seconds
    base_times = {
        "create_booking": 0.3,
        "update_booking": 0.2,
        "cancel_booking": 0.2,
        "get_booking_details": 0.1,
        "add_passenger": 0.15,
        "remove_passenger": 0.15,
        "add_special_request": 0.1,
        "booking_confirmation": 0.2
    }
    
    # Get base time for this operation
    base_time = base_times.get(operation, 0.2)
    
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

def generate_booking_log_entry(timestamp=None, operation=None, 
                            correlation_id=None, user_id=None, booking_id=None,
                            search_data=None, is_anomalous=False, parent_request_id=None, session_id=None):
    """Generate a single booking service log entry"""
    
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
    
    # User ID is required for bookings
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    # Select operation if not provided
    if operation is None:
        operation = random.choice(BOOKING_OPERATIONS)
    
    # Environment and region
    environment = random.choice(BOOKING_ENVIRONMENTS)
    region = random.choice(BOOKING_REGIONS)
    server = random.choice(BOOKING_SERVERS)
    instance_id = f"{server}-{region}-{random.randint(1, 5)}"
    
    # Request details
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    
    # Determine HTTP method based on operation
    if operation in ["get_booking_details"]:
        http_method = "GET"
    elif operation in ["create_booking", "booking_confirmation", "add_passenger", "add_special_request"]:
        http_method = "POST"
    elif operation in ["update_booking"]:
        http_method = "PUT"
    elif operation in ["cancel_booking", "remove_passenger"]:
        http_method = "DELETE"
    else:
        http_method = random.choice(["POST", "PUT"])
    
    # Generate booking ID if not provided
    if booking_id is None:
        booking_id = f"booking-{uuid.uuid4().hex[:8]}"
    
    # Get path based on operation
    if operation == "create_booking":
        path = "/api/bookings"
    elif operation == "update_booking":
        path = f"/api/bookings/{booking_id}"
    elif operation == "cancel_booking":
        path = f"/api/bookings/{booking_id}/cancel"
    elif operation == "get_booking_details":
        path = f"/api/bookings/{booking_id}"
    elif operation == "add_passenger":
        path = f"/api/bookings/{booking_id}/passengers"
    elif operation == "remove_passenger":
        path = f"/api/bookings/{booking_id}/passengers/{uuid.uuid4().hex[:8]}"
    elif operation == "add_special_request":
        path = f"/api/bookings/{booking_id}/requests"
    elif operation == "booking_confirmation":
        path = f"/api/bookings/{booking_id}/confirm"
    else:
        path = f"/api/bookings/{operation.replace('_', '/')}"
    
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    client_type = random.choice(["web-app", "mobile-ios", "mobile-android", "partner-api", "internal"])
    source_ip = generate_ip()
    
    # Generate request headers
    request_headers = generate_request_headers(auth_required=True)
    
    # Generate travel dates if not provided via search data
    if search_data and "departure_date" in search_data and "return_date" in search_data:
        departure_date = search_data["departure_date"]
        return_date = search_data["return_date"]
    else:
        departure_date, return_date = generate_travel_dates()
    
    # Generate booking data based on operation
    booking_status = random.choice(BOOKING_STATUSES)
    booking_type = random.choice(BOOKING_TYPES)
    item_id = f"item-{uuid.uuid4().hex[:8]}"
    
    # If search data is provided, use it to create coherent booking details
    if search_data:
        if "flight_id" in search_data:
            booking_type = "flight"
            item_id = search_data["flight_id"]
        elif "hotel_id" in search_data:
            booking_type = "hotel"
            item_id = search_data["hotel_id"]
        elif "car_id" in search_data:
            booking_type = "car"
            item_id = search_data["car_id"]
        elif "package_id" in search_data:
            booking_type = "package"
            item_id = search_data["package_id"]
    
    # Generate request body based on operation
    request_body = {}
    
    if operation == "create_booking":
        passengers = generate_passenger_details()
        
        request_body = {
            "booking_type": booking_type,
            "item_id": item_id,
            "user_id": user_id,
            "departure_date": departure_date,
            "return_date": return_date if booking_type in ["flight", "package"] else None,
            "passengers": passengers,
            "contact_info": {
                "email": fake.email(),
                "phone": fake.phone_number()
            },
            "special_requests": random.sample(SPECIAL_REQUESTS, k=random.randint(0, 2)) if random.random() < 0.3 else []
        }
        
        if is_anomalous and random.random() < 0.5:
            # Create anomalous request
            if random.random() < 0.5:
                # Missing required field
                del request_body["item_id"]
            else:
                # Invalid dates
                request_body["departure_date"] = return_date
                request_body["return_date"] = departure_date
    
    elif operation == "update_booking":
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "updates": {
                "departure_date": (datetime.datetime.strptime(departure_date, "%Y-%m-%d") + 
                                 datetime.timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                "return_date": (datetime.datetime.strptime(return_date, "%Y-%m-%d") + 
                               datetime.timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d") if booking_type in ["flight", "package"] else None,
                "special_requests": random.sample(SPECIAL_REQUESTS, k=random.randint(0, 2))
            }
        }
    
    elif operation == "cancel_booking":
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "reason": random.choice(["change_of_plans", "found_better_option", "emergency", "other"]),
            "refund_requested": random.choice([True, False])
        }
    
    elif operation == "get_booking_details":
        # GET request typically doesn't have a body
        request_body = {}
    
    elif operation == "add_passenger":
        new_passenger = generate_passenger_details(1)[0]
        request_body = {
            "booking_id": booking_id,
            "passenger": new_passenger
        }
    
    elif operation == "remove_passenger":
        passenger_id = uuid.uuid4().hex[:8]
        request_body = {
            "booking_id": booking_id,
            "passenger_id": passenger_id
        }
    
    elif operation == "add_special_request":
        request_body = {
            "booking_id": booking_id,
            "special_request": random.choice(SPECIAL_REQUESTS),
            "details": fake.sentence()
        }
    
    elif operation == "booking_confirmation":
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "payment_confirmed": True,
            "confirmation_email": fake.email()
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
        if operation == "create_booking":
            response_body = {
                "booking_id": booking_id,
                "status": "pending",
                "booking_type": request_body.get("booking_type", "unknown"),
                "item_id": request_body.get("item_id", "unknown"),
                "user_id": user_id,
                "departure_date": request_body.get("departure_date"),
                "return_date": request_body.get("return_date"),
                "passengers": request_body.get("passengers", []),
                "total_price": {
                    "amount": round(random.uniform(100, 5000), 2),
                    "currency": random.choice(CURRENCIES)
                },
                "created_at": timestamp.isoformat(),
                "expiration_time": (timestamp + datetime.timedelta(minutes=30)).isoformat()
            }
            
        elif operation == "update_booking":
            response_body = {
                "booking_id": booking_id,
                "status": "updated",
                "updates_applied": list(request_body.get("updates", {}).keys()),
                "updated_at": timestamp.isoformat()
            }
            
        elif operation == "cancel_booking":
            response_body = {
                "booking_id": booking_id,
                "status": "cancelled",
                "cancellation_fee": {
                    "amount": round(random.uniform(0, 100), 2),
                    "currency": random.choice(CURRENCIES)
                } if random.random() < 0.7 else None,
                "refund_status": "processing" if request_body.get("refund_requested", False) else "not_requested",
                "cancelled_at": timestamp.isoformat()
            }
            
        elif operation == "get_booking_details":
            passengers = generate_passenger_details(random.randint(1, 4))
            
            response_body = {
                "booking_id": booking_id,
                "user_id": user_id,
                "status": booking_status,
                "booking_type": booking_type,
                "item_id": item_id,
                "item_details": {
                    "name": fake.text(max_nb_chars=20),
                    "description": fake.sentence()
                },
                "departure_date": departure_date,
                "return_date": return_date if booking_type in ["flight", "package"] else None,
                "passengers": passengers,
                "total_price": {
                    "amount": round(random.uniform(100, 5000), 2),
                    "currency": random.choice(CURRENCIES)
                },
                "payment_status": random.choice(["paid", "pending", "failed"]),
                "special_requests": random.sample(SPECIAL_REQUESTS, k=random.randint(0, 2)) if random.random() < 0.3 else [],
                "created_at": (timestamp - datetime.timedelta(days=random.randint(1, 30))).isoformat(),
                "updated_at": timestamp.isoformat() if random.random() < 0.5 else None
            }
            
        elif operation == "add_passenger":
            response_body = {
                "booking_id": booking_id,
                "passenger_added": request_body.get("passenger", {}),
                "updated_at": timestamp.isoformat()
            }
            
        elif operation == "remove_passenger":
            response_body = {
                "booking_id": booking_id,
                "passenger_removed": request_body.get("passenger_id", "unknown"),
                "updated_at": timestamp.isoformat()
            }
            
        elif operation == "add_special_request":
            response_body = {
                "booking_id": booking_id,
                "special_request_added": request_body.get("special_request", "unknown"),
                "status": "confirmed" if random.random() < 0.8 else "pending_review",
                "updated_at": timestamp.isoformat()
            }
            
        elif operation == "booking_confirmation":
            response_body = {
                "booking_id": booking_id,
                "status": "confirmed",
                "confirmation_code": f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10000, 99999)}",
                "payment_status": "completed",
                "confirmation_email_sent": True,
                "confirmed_at": timestamp.isoformat()
            }
    else:
        # Error response
        error_message = get_error_message(status_code)
        response_body = {
            "error": "error_code",
            "error_description": error_message,
            "booking_id": booking_id if operation != "create_booking" else None,
            "status": status_code,
            "timestamp": timestamp.isoformat()
        }
    
    # Calculate response time
    response_time = calculate_response_time(operation, status_code, is_anomalous)
    if session_id is None:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
    # Create the booking log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "booking_service": {
            "type": booking_type,
            "operation": operation,
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
            "user_id": user_id
        },
        "booking_id": booking_id,
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "parent_request_id": parent_request_id,
            "session_id": session_id  # Add session_id to tracing information
        },
        "is_anomalous": is_anomalous
    }

    return log_entry

def generate_related_bookings(correlation_id, user_id=None, search_data=None, base_timestamp=None, is_anomalous=False):
    """Generate a sequence of related booking requests with the same correlation ID"""
    related_logs = []
    
    if base_timestamp is None:
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    current_timestamp = base_timestamp
    
    # Generate user ID if not provided
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    # Generate a booking ID
    booking_id = f"booking-{uuid.uuid4().hex[:8]}"
    
    # Choose a booking sequence pattern
    booking_patterns = [
        # Basic flow: create and confirm
        ["create_booking", "booking_confirmation"],
        # Flow with passenger modification
        ["create_booking", "add_passenger", "booking_confirmation"],
        # Flow with special request
        ["create_booking", "add_special_request", "booking_confirmation"],
        # Flow with update
        ["create_booking", "update_booking", "booking_confirmation"],
        # Cancellation flow
        ["create_booking", "cancel_booking"],
        # Detailed flow
        ["create_booking", "get_booking_details", "add_passenger", "booking_confirmation"]
    ]
    
    pattern = random.choice(booking_patterns)
    
    # Keep track of parent request ID for linking
    parent_request_id = None
    
    # Generate logs for each booking operation in the pattern
    for i, operation in enumerate(pattern):
        # Add some time between operations
        current_timestamp += datetime.timedelta(seconds=random.uniform(2, 10))
        
        # Determine if this operation should be anomalous
        op_is_anomalous = is_anomalous and (i == len(pattern) - 1 or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_booking_log_entry(
            timestamp=current_timestamp,
            operation=operation,
            correlation_id=correlation_id,
            user_id=user_id,
            booking_id=booking_id,
            search_data=search_data,
            is_anomalous=op_is_anomalous,
            parent_request_id=parent_request_id
        )
        
        related_logs.append(log_entry)
        
        # Update parent request ID for the next operation
        parent_request_id = log_entry["request"]["id"]
        
        # If we get an error, we might stop the sequence
        if op_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return related_logs

def generate_booking_logs(num_logs=1000, anomaly_percentage=15):
    """Generate a dataset of booking service logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    print(f"Generating {int(num_normal * 0.4)} individual normal booking logs...")
    for _ in range(int(num_normal * 0.4)):  # 40% of normal logs are individual
        logs.append(generate_booking_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    print(f"Generating {int(num_anomalous * 0.3)} individual anomalous booking logs...")
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_booking_log_entry(is_anomalous=True))
    
    # Generate normal booking sequences
    num_normal_flows = int(num_normal * 0.6 / 3)  # Approximately 3 logs per flow
    print(f"Generating {num_normal_flows} normal booking flows...")
    for _ in range(num_normal_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_bookings(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_timestamp,
            is_anomalous=False
        ))
    
    # Generate anomalous booking sequences
    num_anomalous_flows = int(num_anomalous * 0.7 / 3)  # Approximately 3 logs per flow
    print(f"Generating {num_anomalous_flows} anomalous booking flows...")
    for _ in range(num_anomalous_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_bookings(
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

def save_booking_logs(logs, format='json', filename='booking_logs'):
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
                "booking_type": log["booking_service"]["type"],
                "operation": log["booking_service"]["operation"],
                "environment": log["booking_service"]["environment"],
                "region": log["booking_service"]["region"],
                "instance_id": log["booking_service"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "client_id": log["request"]["client_id"],
                "client_type": log["request"]["client_type"],
                "source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_time_ms": log["response"]["time_ms"],
                "user_id": log["user"]["user_id"],
                "booking_id": log["booking_id"],
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



def analyze_booking_logs(logs):
    """Print analysis of the generated booking logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by booking type
    booking_types = {}
    for log in logs:
        booking_type = log['booking_service']['type']
        booking_types[booking_type] = booking_types.get(booking_type, 0) + 1
    
    # Count by operation
    operations = {}
    for log in logs:
        operation = log['booking_service']['operation']
        operations[operation] = operations.get(operation, 0) + 1
    
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
    
    # Count unique correlation IDs (booking flows)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    print("\n=== Booking Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Success rate: {success_count/total_logs*100:.2f}%")
    print(f"Failure rate: {failure_count/total_logs*100:.2f}%")
    print(f"Unique booking flows: {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Booking Type Distribution ===")
    for booking_type, count in sorted(booking_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{booking_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Operation Distribution ===")
    for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        print(f"{operation}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_booking_logs(auth_logs=None, search_logs=None, num_logs=500):
    """Generate booking logs that share correlation IDs with existing auth/search logs"""
    booking_logs = []
    
    # Extract correlation IDs from existing logs
    correlation_ids = set()
    user_id_map = {}
    search_data_map = {}
    
    # Process auth logs if available
    if auth_logs:
        for log in auth_logs:
            if "tracing" in log and "correlation_id" in log["tracing"]:
                corr_id = log["tracing"]["correlation_id"]
                correlation_ids.add(corr_id)
                
                # Map user IDs to correlation IDs
                if "operation" in log and "user_id" in log["operation"]:
                    user_id_map[corr_id] = log["operation"]["user_id"]
    
    # Process search logs if available
    if search_logs:
        for log in search_logs:
            if "tracing" in log and "correlation_id" in log["tracing"]:
                corr_id = log["tracing"]["correlation_id"]
                correlation_ids.add(corr_id)
                
                # Map user IDs to correlation IDs if not already mapped from auth logs
                if "user" in log and "user_id" in log["user"] and log["user"]["user_id"] and corr_id not in user_id_map:
                    user_id_map[corr_id] = log["user"]["user_id"]
                
                # Extract search data for booking if applicable
                if log["response"]["status_code"] < 400 and "body" in log["response"]:
                    response_body = log["response"]["body"]
                    search_type = log["search_service"]["type"] if "search_service" in log and "type" in log["search_service"] else None
                    
                    search_data = {}
                    
                    # Extract travel dates from request
                    if "request" in log and "body" in log["request"]:
                        request_body = log["request"]["body"]
                        if "departure_date" in request_body:
                            search_data["departure_date"] = request_body["departure_date"]
                        if "return_date" in request_body:
                            search_data["return_date"] = request_body["return_date"]
                        if "check_in_date" in request_body:
                            search_data["departure_date"] = request_body["check_in_date"]
                        if "check_out_date" in request_body:
                            search_data["return_date"] = request_body["check_out_date"]
                    
                    # Extract item IDs based on search type
                    if search_type == "flight_search" and "flights" in response_body and response_body["flights"]:
                        search_data["flight_id"] = response_body["flights"][0]["flight_id"]
                    elif search_type == "hotel_search" and "hotels" in response_body and response_body["hotels"]:
                        search_data["hotel_id"] = response_body["hotels"][0]["hotel_id"]
                    elif search_type == "car_rental_search" and "cars" in response_body and response_body["cars"]:
                        search_data["car_id"] = response_body["cars"][0]["car_id"]
                    elif search_type == "package_search" and "packages" in response_body and response_body["packages"]:
                        search_data["package_id"] = response_body["packages"][0]["package_id"]
                    
                    if search_data:
                        search_data_map[corr_id] = search_data
    
    # If no correlation IDs found, generate standalone logs
    if not correlation_ids:
        print("No correlation IDs found in input logs. Generating standalone booking logs.")
        return generate_booking_logs(num_logs)
    
    # Convert to list for random.choice
    correlation_ids = list(correlation_ids)
    
    # For each correlation ID, generate a booking flow
    print(f"Generating booking logs connected to {len(correlation_ids)} existing flows...")
    corr_ids_processed = 0
    
    while len(booking_logs) < num_logs and corr_ids_processed < len(correlation_ids):
        # Pick a correlation ID
        correlation_id = correlation_ids[corr_ids_processed]
        corr_ids_processed += 1
        
        # Get user ID if available
        user_id = user_id_map.get(correlation_id)
        
        # Get search data if available
        search_data = search_data_map.get(correlation_id)
        
        # Generate a base timestamp
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # Generate booking flow with this correlation ID
        flow_logs = generate_related_bookings(
            correlation_id=correlation_id,
            user_id=user_id,
            search_data=search_data,
            base_timestamp=base_time,
            is_anomalous=random.random() < 0.15  # 15% chance of anomalous flow
        )
        
        booking_logs.extend(flow_logs)
    
    # If we need more logs, add some individual ones
    while len(booking_logs) < num_logs:
        booking_logs.append(generate_booking_log_entry())
    
    return booking_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Booking API logs')
    parser.add_argument('--num-logs', type=int, default=1000, help='Number of logs to generate')
    parser.add_argument('--anomaly-percentage', type=int, default=15, help='Percentage of anomalous logs')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json', help='Output file format')
    parser.add_argument('--output-filename', default='booking_logs', help='Output filename (without extension)')
    parser.add_argument('--analyze', action='store_true', help='Analyze generated logs')
    parser.add_argument('--auth-logs', help='Path to auth logs file to connect with')
    parser.add_argument('--search-logs', help='Path to search logs file to connect with')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load existing logs if provided
    auth_logs = None
    search_logs = None
    
    if args.auth_logs:
        try:
            print(f"Loading auth logs from {args.auth_logs}...")
            with open(args.auth_logs, 'r') as f:
                auth_logs = json.load(f)
            print(f"Loaded {len(auth_logs)} auth logs")
        except Exception as e:
            print(f"Error loading auth logs: {e}")
    
    if args.search_logs:
        try:
            print(f"Loading search logs from {args.search_logs}...")
            with open(args.search_logs, 'r') as f:
                search_logs = json.load(f)
            print(f"Loaded {len(search_logs)} search logs")
        except Exception as e:
            print(f"Error loading search logs: {e}")
    
    # Generate logs
    if args.auth_logs or args.search_logs:
        logs = generate_interconnected_booking_logs(auth_logs, search_logs, args.num_logs)
    else:
        logs = generate_booking_logs(args.num_logs, args.anomaly_percentage)
    
    # Save logs to file
    save_logs_to_file(logs, args.output_format, args.output_filename)
    
    # Analyze logs if requested
    if args.analyze:
        analyze_booking_logs(logs)