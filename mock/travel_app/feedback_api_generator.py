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

# Define constants for Feedback Service log generation
FEEDBACK_OPERATIONS = [
    "submit_review", "update_review", "get_review", 
    "submit_rating", "submit_complaint", "respond_to_feedback"
]

FEEDBACK_ENVIRONMENTS = ["dev", "staging", "production", "test"]
FEEDBACK_SERVERS = ["feedback-primary", "feedback-backup"]
FEEDBACK_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Rating categories
RATING_CATEGORIES = [
    "overall_experience", "booking_process", "customer_service", 
    "price_value", "airline_quality", "hotel_quality", 
    "car_rental_experience", "website_usability", "mobile_app"
]

# Complaint categories
COMPLAINT_CATEGORIES = [
    "booking_issues", "payment_problems", "refund_delays", 
    "cancellation_issues", "customer_service", "technical_problems", 
    "incorrect_information", "hidden_fees", "service_quality"
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
HTTP_METHODS = ["GET", "POST", "PUT"]

# Status codes with their general meanings
SUCCESS_STATUS_CODES = [200, 201, 204]
ERROR_STATUS_CODES = {
    "client_error": [400, 401, 403, 404, 422, 429],
    "server_error": [500, 502, 503, 504]
}

# Error messages by status code
ERROR_MESSAGES = {
    400: [
        "Invalid feedback format",
        "Rating out of range",
        "Missing required fields",
        "Invalid review content"
    ],
    401: [
        "Authentication required for feedback",
        "Session expired",
        "Invalid API key",
        "Missing authentication token"
    ],
    403: [
        "Not authorized to submit feedback",
        "Review period expired",
        "Account restricted",
        "Only customers can leave reviews"
    ],
    404: [
        "Review not found",
        "Booking reference not found",
        "User not found",
        "Response ID not found"
    ],
    422: [
        "Feedback validation failed",
        "Rating must be between 1 and 5",
        "Review text too short",
        "Review contains prohibited content"
    ],
    429: [
        "Feedback submission rate limit exceeded",
        "Too many review attempts",
        "API quota exceeded",
        "Please try again later"
    ],
    500: [
        "Feedback system error",
        "Unable to save review",
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
        "Feedback service temporarily unavailable",
        "Maintenance in progress",
        "Database unavailable",
        "System overloaded"
    ],
    504: [
        "Gateway timeout",
        "Review processing timed out",
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
    """Calculate realistic response times for feedback operations"""
    # Base times in seconds
    base_times = {
        "submit_review": 0.2,
        "update_review": 0.15,
        "get_review": 0.05,
        "submit_rating": 0.1,
        "submit_complaint": 0.2,
        "respond_to_feedback": 0.15
    }
    
    # Get base time for this operation
    base_time = base_times.get(operation, 0.15)
    
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

def generate_feedback_log_entry(timestamp=None, operation=None, 
                             correlation_id=None, user_id=None, 
                             booking_id=None, review_id=None, 
                             booking_data=None, is_anomalous=False):
    """Generate a single feedback service log entry"""
    
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
    
    # User ID is required for most feedback operations (except for anonymous feedback)
    if user_id is None and random.random() < 0.9:  # 90% of feedback is associated with a user
        user_id = str(uuid.uuid4())
    
    # Select operation if not provided
    if operation is None:
        operation = random.choice(FEEDBACK_OPERATIONS)
    
    # Environment and region
    environment = random.choice(FEEDBACK_ENVIRONMENTS)
    region = random.choice(FEEDBACK_REGIONS)
    server = random.choice(FEEDBACK_SERVERS)
    instance_id = f"{server}-{region}-{random.randint(1, 5)}"
    
    # Request details
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    
    # Determine HTTP method based on operation
    if operation in ["get_review"]:
        http_method = "GET"
    elif operation in ["submit_review", "submit_rating", "submit_complaint"]:
        http_method = "POST"
    elif operation in ["update_review", "respond_to_feedback"]:
        http_method = "PUT"
    else:
        http_method = "POST"
    
    # Generate booking ID if not provided
    if booking_id is None and booking_data and "booking_id" in booking_data:
        booking_id = booking_data["booking_id"]
    elif booking_id is None:
        booking_id = f"booking-{uuid.uuid4().hex[:8]}"
    
    # Generate review ID if needed and not provided
    if review_id is None and operation != "submit_review":
        review_id = f"review-{uuid.uuid4().hex[:8]}"
    
    # Get path based on operation
    if operation == "submit_review":
        path = "/api/feedback/reviews"
    elif operation == "update_review":
        path = f"/api/feedback/reviews/{review_id}"
    elif operation == "get_review":
        path = f"/api/feedback/reviews/{review_id}"
    elif operation == "submit_rating":
        path = "/api/feedback/ratings"
    elif operation == "submit_complaint":
        path = "/api/feedback/complaints"
    elif operation == "respond_to_feedback":
        path = f"/api/feedback/reviews/{review_id}/responses"
    else:
        path = f"/api/feedback/{operation.replace('_', '/')}"
    
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    client_type = random.choice(["web-app", "mobile-ios", "mobile-android", "partner-api", "internal"])
    source_ip = generate_ip()
    
    # Generate request headers
    auth_required = user_id is not None
    request_headers = generate_request_headers(auth_required=auth_required)
    
    # Determine booking type if available
    booking_type = None
    if booking_data and "booking_type" in booking_data:
        booking_type = booking_data["booking_type"]
    else:
        booking_type = random.choice(["flight", "hotel", "car", "package"])
    
    # Generate request body based on operation
    request_body = {}
    
    if operation == "submit_review":
        # Generate a more detailed review
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "booking_type": booking_type,
            "rating": random.randint(1, 5),
            "title": fake.sentence(nb_words=random.randint(3, 8)),
            "content": fake.paragraph(nb_sentences=random.randint(3, 8)),
            "categories": {
                category: random.randint(1, 5) for category in random.sample(RATING_CATEGORIES, k=random.randint(2, 5))
            },
            "recommend": random.choice([True, False, None]),
            "photos": [f"photo-{uuid.uuid4().hex[:8]}" for _ in range(random.randint(0, 2))] if random.random() < 0.2 else []
        }
        
        if is_anomalous and random.random() < 0.5:
            # Create anomalous request
            if random.random() < 0.5:
                # Invalid rating
                request_body["rating"] = random.choice([0, 6, -1, None])
            else:
                # Empty content
                request_body["content"] = ""
    
    elif operation == "update_review":
        request_body = {
            "review_id": review_id,
            "user_id": user_id,
            "updates": {
                "rating": random.randint(1, 5),
                "content": fake.paragraph(nb_sentences=random.randint(2, 6))
            }
        }
    
    elif operation == "get_review":
        # GET request typically doesn't have a body
        request_body = {}
    
    elif operation == "submit_rating":
        # Simpler rating without detailed review
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "booking_type": booking_type,
            "rating": random.randint(1, 5),
            "categories": {
                category: random.randint(1, 5) for category in random.sample(RATING_CATEGORIES, k=random.randint(1, 3))
            }
        }
    
    elif operation == "submit_complaint":
        request_body = {
            "booking_id": booking_id,
            "user_id": user_id,
            "booking_type": booking_type,
            "category": random.choice(COMPLAINT_CATEGORIES),
            "description": fake.paragraph(nb_sentences=random.randint(2, 5)),
            "severity": random.choice(["low", "medium", "high"]),
            "contact_requested": random.choice([True, False])
        }
    
    elif operation == "respond_to_feedback":
        request_body = {
            "review_id": review_id,
            "responder_id": f"staff-{uuid.uuid4().hex[:8]}",
            "response": fake.paragraph(nb_sentences=random.randint(2, 4)),
            "public": random.choice([True, False])
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
        if operation == "submit_review":
            # Generate review ID if not provided
            if review_id is None:
                review_id = f"review-{uuid.uuid4().hex[:8]}"
                
            response_body = {
                "review_id": review_id,
                "booking_id": booking_id,
                "user_id": user_id,
                "status": "published",
                "submitted_at": timestamp.isoformat(),
                "moderation_status": random.choice(["approved", "pending"]),
                "thank_you_points": random.randint(10, 100) if random.random() < 0.7 else None
            }
            
        elif operation == "update_review":
            response_body = {
                "review_id": review_id,
                "status": "updated",
                "updates_applied": list(request_body.get("updates", {}).keys()),
                "updated_at": timestamp.isoformat(),
                "moderation_status": random.choice(["approved", "pending"])
            }
            
        elif operation == "get_review":
            review_date = timestamp - datetime.timedelta(days=random.randint(1, 90))
            
            response_body = {
                "review_id": review_id,
                "booking_id": booking_id,
                "user_id": user_id,
                "booking_type": booking_type,
                "rating": random.randint(1, 5),
                "title": fake.sentence(nb_words=random.randint(3, 8)),
                "content": fake.paragraph(nb_sentences=random.randint(3, 8)),
                "categories": {
                    category: random.randint(1, 5) for category in random.sample(RATING_CATEGORIES, k=random.randint(2, 5))
                },
                "recommend": random.choice([True, False, None]),
                "helpful_votes": random.randint(0, 100),
                "submitted_at": review_date.isoformat(),
                "updated_at": (review_date + datetime.timedelta(days=random.randint(1, 7))).isoformat() if random.random() < 0.3 else None,
                "responses": [] if random.random() < 0.7 else [{
                    "responder_id": f"staff-{uuid.uuid4().hex[:8]}",
                    "responder_name": f"{fake.first_name()}, Customer Service",
                    "response": fake.paragraph(nb_sentences=random.randint(2, 4)),
                    "responded_at": (review_date + datetime.timedelta(days=random.randint(1, 5))).isoformat()
                }]
            }
            
        elif operation == "submit_rating":
            response_body = {
                "rating_id": f"rating-{uuid.uuid4().hex[:8]}",
                "booking_id": booking_id,
                "user_id": user_id,
                "status": "recorded",
                "thank_you_message": "Thank you for your feedback!",
                "submitted_at": timestamp.isoformat()
            }
            
        elif operation == "submit_complaint":
            response_body = {
                "complaint_id": f"complaint-{uuid.uuid4().hex[:8]}",
                "booking_id": booking_id,
                "status": "received",
                "reference_number": f"COMP-{random.randint(10000, 99999)}",
                "expected_response_time": "24 hours" if request_body.get("severity") == "high" else 
                                       "48 hours" if request_body.get("severity") == "medium" else
                                       "72 hours",
                "submitted_at": timestamp.isoformat()
            }
            
        elif operation == "respond_to_feedback":
            response_body = {
                "response_id": f"response-{uuid.uuid4().hex[:8]}",
                "review_id": review_id,
                "status": "published" if request_body.get("public", True) else "private",
                "responder_id": request_body.get("responder_id"),
                "responded_at": timestamp.isoformat()
            }
    else:
        # Error response
        error_message = get_error_message(status_code)
        response_body = {
            "error": "error_code",
            "error_description": error_message,
            "review_id": review_id if operation != "submit_review" else None,
            "status": status_code,
            "timestamp": timestamp.isoformat()
        }
    
    # Calculate response time
    response_time = calculate_response_time(operation, status_code, is_anomalous)
    
    # Create the feedback log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "feedback_service": {
            "type": "feedback",
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
            "user_id": user_id,
            "anonymous": user_id is None
        },
        "booking_id": booking_id,
        "review_id": review_id,
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_related_feedback(correlation_id, user_id=None, booking_id=None, 
                             booking_data=None, base_timestamp=None, is_anomalous=False):
    """Generate a sequence of related feedback requests with the same correlation ID"""
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
    
    # Generate booking ID if not provided
    if booking_id is None and booking_data and "booking_id" in booking_data:
        booking_id = booking_data["booking_id"]
    elif booking_id is None:
        booking_id = f"booking-{uuid.uuid4().hex[:8]}"
    
    # Choose a feedback sequence pattern
    feedback_patterns = [
        # Basic review
        ["submit_review"],
        # Review and update
        ["submit_review", "update_review"],
        # Submit and check
        ["submit_review", "get_review"],
        # Review with response
        ["submit_review", "respond_to_feedback"],
        # Rating and complaint
        ["submit_rating", "submit_complaint"],
        # Complete cycle
        ["submit_review", "get_review", "respond_to_feedback"]
    ]
    
    pattern = random.choice(feedback_patterns)
    
    # Keep track of review ID for the sequence
    review_id = None
    
    # Generate logs for each feedback operation in the pattern
    for i, operation in enumerate(pattern):
        # Add some time between operations
        current_timestamp += datetime.timedelta(seconds=random.uniform(5, 30))
        
        if operation == "respond_to_feedback":
            # Add more time for staff response
            current_timestamp += datetime.timedelta(hours=random.uniform(2, 48))
        
        # Determine if this operation should be anomalous
        op_is_anomalous = is_anomalous and (i == len(pattern) - 1 or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_feedback_log_entry(
            timestamp=current_timestamp,
            operation=operation,
            correlation_id=correlation_id,
            user_id=user_id if operation != "respond_to_feedback" else None,  # Staff response
            booking_id=booking_id,
            review_id=review_id,
            booking_data=booking_data,
            is_anomalous=op_is_anomalous
        )
        
        related_logs.append(log_entry)
        
        # Extract review ID from the response for subsequent operations
        if operation == "submit_review":
            if log_entry["response"]["status_code"] < 400 and "body" in log_entry["response"]:
                review_id = log_entry["response"]["body"].get("review_id")
        
        # If we get an error, we might stop the sequence
        if op_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return related_logs

def generate_feedback_logs(num_logs=1000, anomaly_percentage=15):
    """Generate a dataset of feedback service logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    print(f"Generating {int(num_normal * 0.5)} individual normal feedback logs...")
    for _ in range(int(num_normal * 0.5)):  # 50% of normal logs are individual
        logs.append(generate_feedback_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    print(f"Generating {int(num_anomalous * 0.3)} individual anomalous feedback logs...")
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_feedback_log_entry(is_anomalous=True))
    
    # Generate normal feedback sequences
    num_normal_flows = int(num_normal * 0.5 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_normal_flows} normal feedback flows...")
    for _ in range(num_normal_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_feedback(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_timestamp,
            is_anomalous=False
        ))
    
    # Generate anomalous feedback sequences
    num_anomalous_flows = int(num_anomalous * 0.7 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_anomalous_flows} anomalous feedback flows...")
    for _ in range(num_anomalous_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_feedback(
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

def save_feedback_logs(logs, format='json', filename='feedback_logs'):
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
                "feedback_operation": log["feedback_service"]["operation"],
                "environment": log["feedback_service"]["environment"],
                "region": log["feedback_service"]["region"],
                "instance_id": log["feedback_service"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "client_id": log["request"]["client_id"],
                "client_type": log["request"]["client_type"],
                "source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_time_ms": log["response"]["time_ms"],
                "user_id": log["user"]["user_id"],
                "anonymous": log["user"]["anonymous"],
                "booking_id": log["booking_id"],
                "review_id": log["review_id"],
                "correlation_id": log["tracing"]["correlation_id"],
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

def analyze_feedback_logs(logs):
    """Print analysis of the generated feedback logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by operation
    operations = {}
    for log in logs:
        operation = log['feedback_service']['operation']
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
    
    # Count anonymous feedback
    anonymous_count = sum(1 for log in logs if log['user']['anonymous'])
    
    # Count unique correlation IDs (feedback flows)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    print("\n=== Feedback Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Success rate: {success_count/total_logs*100:.2f}%")
    print(f"Failure rate: {failure_count/total_logs*100:.2f}%")
    print(f"Anonymous feedback: {anonymous_count} ({anonymous_count/total_logs*100:.2f}%)")
    print(f"Unique feedback flows: {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Operation Distribution ===")
    for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        print(f"{operation}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_feedback_logs(auth_logs=None, booking_logs=None, num_logs=500):
    """Generate feedback logs that share correlation IDs with existing auth/booking logs"""
    feedback_logs = []
    
    # Extract correlation IDs from existing logs
    correlation_ids = set()
    user_id_map = {}
    booking_data_map = {}
    
    # Process auth logs if available
    if auth_logs:
        for log in auth_logs:
            if "tracing" in log and "correlation_id" in log["tracing"]:
                corr_id = log["tracing"]["correlation_id"]
                correlation_ids.add(corr_id)
                
                # Map user IDs to correlation IDs
                if "operation" in log and "user_id" in log["operation"]:
                    user_id_map[corr_id] = log["operation"]["user_id"]
    
    # Process booking logs if available
    if booking_logs:
        for log in booking_logs:
            if "tracing" in log and "correlation_id" in log["tracing"]:
                corr_id = log["tracing"]["correlation_id"]
                correlation_ids.add(corr_id)
                
                # Map user IDs to correlation IDs if not already mapped from auth logs
                if corr_id not in user_id_map and "user" in log and "user_id" in log["user"] and log["user"]["user_id"]:
                    user_id_map[corr_id] = log["user"]["user_id"]
                
                # Extract booking data for feedback
                if "booking_id" in log and log["booking_id"]:
                    booking_data = {
                        "booking_id": log["booking_id"]
                    }
                    
                    # Try to extract booking type
                    if "booking_service" in log and "type" in log["booking_service"]:
                        booking_data["booking_type"] = log["booking_service"]["type"]
                    
                    booking_data_map[corr_id] = booking_data
    
    # If no correlation IDs found, generate standalone logs
    if not correlation_ids:
        print("No correlation IDs found in input logs. Generating standalone feedback logs.")
        return generate_feedback_logs(num_logs)
    
    # Convert to list for random.choice
    correlation_ids = list(correlation_ids)
    
    # For each correlation ID, generate a feedback flow
    print(f"Generating feedback logs connected to {len(correlation_ids)} existing flows...")
    corr_ids_processed = 0
    
    while len(feedback_logs) < num_logs and corr_ids_processed < len(correlation_ids):
        # Pick a correlation ID
        correlation_id = correlation_ids[corr_ids_processed]
        corr_ids_processed += 1
        
        # Get user ID if available
        user_id = user_id_map.get(correlation_id)
        
        # Get booking data if available
        booking_data = booking_data_map.get(correlation_id)
        
        # Generate a base timestamp
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # Generate feedback flow with this correlation ID
        flow_logs = generate_related_feedback(
            correlation_id=correlation_id,
            user_id=user_id,
            booking_data=booking_data,
            base_timestamp=base_time,
            is_anomalous=random.random() < 0.15  # 15% chance of anomalous flow
        )
        
        feedback_logs.extend(flow_logs)
    
    # If we need more logs, add some individual ones
    while len(feedback_logs) < num_logs:
        feedback_logs.append(generate_feedback_log_entry())
    
    return feedback_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Feedback API logs')
    parser.add_argument('--num-logs', type=int, default=1000, help='Number of logs to generate')
    parser.add_argument('--anomaly-percentage', type=int, default=15, help='Percentage of anomalous logs')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json', help='Output file format')
    parser.add_argument('--output-filename', default='feedback_logs', help='Output filename (without extension)')
    parser.add_argument('--analyze', action='store_true', help='Analyze generated logs')
    parser.add_argument('--auth-logs', help='Path to auth logs file to connect with')
    parser.add_argument('--booking-logs', help='Path to booking logs file to connect with')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load existing logs if provided
    auth_logs = None
    booking_logs = None
    
    if args.auth_logs:
        try:
            print(f"Loading auth logs from {args.auth_logs}...")
            with open(args.auth_logs, 'r') as f:
                auth_logs = json.load(f)
            print(f"Loaded {len(auth_logs)} auth logs")
        except Exception as e:
            print(f"Error loading auth logs: {e}")
    
    if args.booking_logs:
        try:
            print(f"Loading booking logs from {args.booking_logs}...")
            with open(args.booking_logs, 'r') as f:
                booking_logs = json.load(f)
            print(f"Loaded {len(booking_logs)} booking logs")
        except Exception as e:
            print(f"Error loading booking logs: {e}")
    
    print("Generating feedback service logs...")
    
    # Generate logs
    if args.auth_logs or args.booking_logs:
        logs = generate_interconnected_feedback_logs(auth_logs, booking_logs, args.num_logs)
    else:
        logs = generate_feedback_logs(args.num_logs, args.anomaly_percentage)
    
    # Analyze logs
    analyze_feedback_logs(logs)
    
    # Save logs to file
    json_path = save_logs_to_file(logs, format='json', filename=args.output_filename)
    csv_path = save_logs_to_file(logs, format='csv', filename=args.output_filename)
    
    print(f"\nFeedback logs have been saved to {json_path} and {csv_path}")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Feedback Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2)[:1000] + "... (truncated)")
    
    print("\n=== Sample Anomalous Feedback Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2)[:1000] + "... (truncated)")
    
    # Generate feedback flow examples
    print("\nGenerating example feedback flows...")
    feedback_flows = [
        ["submit_review"],
        ["submit_review", "update_review"],
        ["submit_review", "respond_to_feedback"]
    ]
    
    for flow in feedback_flows:
        flow_name = " -> ".join(flow)
        print(f"\nExample flow: {flow_name}")
        user_id = str(uuid.uuid4())
        correlation_id = generate_correlation_id()
        flow_logs = generate_related_feedback(
            correlation_id=correlation_id,
            user_id=user_id,
            is_anomalous=False
        )
        
        print(f"Generated {len(flow_logs)} logs for this flow")