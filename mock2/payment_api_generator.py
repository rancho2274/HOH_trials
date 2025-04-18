   # Calculate response time
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

# Define constants for Payment Service log generation
PAYMENT_OPERATIONS = [
    "process_payment", "refund_payment", "payment_status_check", 
    "authorize_payment", "capture_payment", "void_payment"
]

PAYMENT_ENVIRONMENTS = ["dev", "staging", "production", "test"]
PAYMENT_SERVERS = ["payment-primary", "payment-replica", "payment-secure"]
PAYMENT_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Payment methods
PAYMENT_METHODS = [
    "credit_card", "debit_card", "paypal", "bank_transfer", 
    "apple_pay", "google_pay", "travel_wallet", "gift_card"
]

# Payment statuses
PAYMENT_STATUSES = [
    "authorized", "captured", "settled", "failed", "refunded", 
    "partially_refunded", "voided", "pending"
]

# Credit card types
CREDIT_CARD_TYPES = ["visa", "mastercard", "amex", "discover"]

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

# Currencies
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD", "AED"]

# Error messages by status code
ERROR_MESSAGES = {
    400: [
        "Invalid payment information",
        "Missing required payment fields",
        "Invalid card details",
        "Currency not supported"
    ],
    401: [
        "Payment authentication failed",
        "Unauthorized payment attempt",
        "Invalid API key",
        "Missing authentication token"
    ],
    403: [
        "Payment method not allowed",
        "Suspicious activity detected",
        "Geographic restriction",
        "Payment operation not allowed"
    ],
    404: [
        "Payment record not found",
        "Transaction not found",
        "Payment method not found",
        "User account not found"
    ],
    422: [
        "Payment processing failed",
        "Card verification failed",
        "Insufficient funds",
        "Card expired"
    ],
    429: [
        "Too many payment attempts",
        "Payment rate limit exceeded",
        "Try again later",
        "API quota exceeded"
    ],
    500: [
        "Payment system error",
        "Payment gateway unavailable",
        "Internal server error",
        "Unexpected condition"
    ],
    502: [
        "Bad gateway",
        "Payment provider error",
        "External service failure",
        "API gateway error"
    ],
    503: [
        "Payment service temporarily unavailable",
        "Maintenance in progress",
        "Provider service unavailable",
        "System overloaded"
    ],
    504: [
        "Gateway timeout",
        "Payment process timed out",
        "Provider response timeout",
        "Request processing timeout"
    ]
    response_time = calculate_response_time(operation, status_code, is_anomalous)
    
    # Create the payment log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "payment_service": {
            "type": "payment",
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
        "payment_id": payment_id,
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "parent_request_id": parent_request_id
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_related_payments(correlation_id, user_id=None, booking_id=None, 
                             booking_data=None, base_timestamp=None, is_anomalous=False):
    """Generate a sequence of related payment requests with the same correlation ID"""
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
    
    # Choose a payment sequence pattern
    payment_patterns = [
        # Basic flow: process payment
        ["process_payment"],
        # Auth and capture flow
        ["authorize_payment", "capture_payment"],
        # Refund flow
        ["process_payment", "refund_payment"],
        # Auth and void flow
        ["authorize_payment", "void_payment"],
        # Complex flow with status check
        ["process_payment", "payment_status_check", "refund_payment"]
    ]
    
    pattern = random.choice(payment_patterns)
    
    # Keep track of payment ID for the sequence
    payment_id = None
    
    # Keep track of parent request ID for linking
    parent_request_id = None
    
    # Generate logs for each payment operation in the pattern
    for i, operation in enumerate(pattern):
        # Add some time between operations
        current_timestamp += datetime.timedelta(seconds=random.uniform(2, 10))
        
        # Determine if this operation should be anomalous
        op_is_anomalous = is_anomalous and (i == len(pattern) - 1 or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_payment_log_entry(
            timestamp=current_timestamp,
            operation=operation,
            correlation_id=correlation_id,
            user_id=user_id,
            booking_id=booking_id,
            payment_id=payment_id,
            booking_data=booking_data,
            is_anomalous=op_is_anomalous,
            parent_request_id=parent_request_id
        )
        
        related_logs.append(log_entry)
        
        # Extract payment ID from the response for subsequent operations
        if operation == "process_payment" or operation == "authorize_payment":
            if log_entry["response"]["status_code"] < 400 and "body" in log_entry["response"]:
                payment_id = log_entry["response"]["body"].get("payment_id")
        
        # Update parent request ID for the next operation
        parent_request_id = log_entry["request"]["id"]
        
        # If we get an error, we might stop the sequence
        if op_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return related_logs

def generate_payment_logs(num_logs=1000, anomaly_percentage=15):
    """Generate a dataset of payment service logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    print(f"Generating {int(num_normal * 0.4)} individual normal payment logs...")
    for _ in range(int(num_normal * 0.4)):  # 40% of normal logs are individual
        logs.append(generate_payment_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    print(f"Generating {int(num_anomalous * 0.3)} individual anomalous payment logs...")
    for _ in range(int(num_anomalous * 0.3)):  # 30% of anomalous logs are individual
        logs.append(generate_payment_log_entry(is_anomalous=True))
    
    # Generate normal payment sequences
    num_normal_flows = int(num_normal * 0.6 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_normal_flows} normal payment flows...")
    for _ in range(num_normal_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_payments(
            correlation_id=correlation_id,
            user_id=user_id,
            base_timestamp=base_timestamp,
            is_anomalous=False
        ))
    
    # Generate anomalous payment sequences
    num_anomalous_flows = int(num_anomalous * 0.7 / 2)  # Approximately 2 logs per flow
    print(f"Generating {num_anomalous_flows} anomalous payment flows...")
    for _ in range(num_anomalous_flows):
        correlation_id = generate_correlation_id()
        user_id = str(uuid.uuid4())
        base_timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        flow_logs.extend(generate_related_payments(
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

def save_logs_to_file(logs, format='json', filename='payment_logs'):
    """Save logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "payment_operation": log["payment_service"]["operation"],
                "environment": log["payment_service"]["environment"],
                "region": log["payment_service"]["region"],
                "instance_id": log["payment_service"]["instance_id"],
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
                "payment_id": log["payment_id"],
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

def analyze_payment_logs(logs):
    """Print analysis of the generated payment logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by operation
    operations = {}
    for log in logs:
        operation = log['payment_service']['operation']
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
    
    # Count unique correlation IDs (payment flows)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    print("\n=== Payment Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Success rate: {success_count/total_logs*100:.2f}%")
    print(f"Failure rate: {failure_count/total_logs*100:.2f}%")
    print(f"Unique payment flows: {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Operation Distribution ===")
    for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        print(f"{operation}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_payment_logs(auth_logs=None, booking_logs=None, num_logs=500):
    """Generate payment logs that share correlation IDs with existing auth/booking logs"""
    payment_logs = []
    
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
                if "user" in log and "user_id" in log["user"] and log["user"]["user_id"] and corr_id not in user_id_map:
                    user_id_map[corr_id] = log["user"]["user_id"]
                
                # Extract booking data for payment
                if corr_id not in booking_data_map and "booking_id" in log and log["booking_id"]:
                    booking_data = {
                        "booking_id": log["booking_id"]
                    }
                    
                    # Try to extract price information
                    if "response" in log and "body" in log["response"] and log["response"]["status_code"] < 400:
                        response_body = log["response"]["body"]
                        if "total_price" in response_body:
                            booking_data["total_price"] = response_body["total_price"]
                    
                    booking_data_map[corr_id] = booking_data
    
    # If no correlation IDs found, generate standalone logs
    if not correlation_ids:
        print("No correlation IDs found in input logs. Generating standalone payment logs.")
        return generate_payment_logs(num_logs)
    
    # Convert to list for random.choice
    correlation_ids = list(correlation_ids)
    
    # For each correlation ID, generate a payment flow
    print(f"Generating payment logs connected to {len(correlation_ids)} existing flows...")
    corr_ids_processed = 0
    
    while len(payment_logs) < num_logs and corr_ids_processed < len(correlation_ids):
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
        
        # Generate payment flow with this correlation ID
        flow_logs = generate_related_payments(
            correlation_id=correlation_id,
            user_id=user_id,
            booking_data=booking_data,
            base_timestamp=base_time,
            is_anomalous=random.random() < 0.15  # 15% chance of anomalous flow
        )
        
        payment_logs.extend(flow_logs)
    
    # If we need more logs, add some individual ones
    while len(payment_logs) < num_logs:
        payment_logs.append(generate_payment_log_entry())
    
    return payment_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Payment API logs')
    parser.add_argument('--num-logs', type=int, default=1000, help='Number of logs to generate')
    parser.add_argument('--anomaly-percentage', type=int, default=15, help='Percentage of anomalous logs')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json', help='Output file format')
    parser.add_argument('--output-filename', default='payment_logs', help='Output filename (without extension)')
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
    
    # Generate logs
    if args.auth_logs or args.booking_logs:
        logs = generate_interconnected_payment_logs(auth_logs, booking_logs, args.num_logs)
    else:
        logs = generate_payment_logs(args.num_logs, args.anomaly_percentage)
    
    # Save logs to file
    save_logs_to_file(logs, args.output_format, args.output_filename)
    
    # Analyze logs if requested
    if args.analyze:
        analyze_payment_logs(logs)