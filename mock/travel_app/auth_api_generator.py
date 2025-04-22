import os
import json
import random
import datetime
import uuid
import ipaddress
import time
import hmac
import hashlib
import base64
from faker import Faker
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize Faker for generating realistic data
fake = Faker()

# Define constants for Auth Service log generation
AUTH_TYPES = ["oauth2", "jwt", "api-key", "basic", "saml", "oidc"]
AUTH_ENVIRONMENTS = ["dev", "staging", "production", "test"]
AUTH_SERVERS = ["auth-primary", "auth-replica", "auth-failover", "auth-backup"]
AUTH_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Auth operations
AUTH_OPERATIONS = {
    "login": 0.30,           # 30% of auth operations are logins
    "token_validation": 0.25, # 25% are token validations
    "token_refresh": 0.15,    # 15% are token refreshes
    "logout": 0.10,           # 10% are logouts
    "register": 0.05,         # 5% are registrations
    "password_reset": 0.03,   # 3% are password resets
    "mfa_validation": 0.05,   # 5% are MFA validations
    "permission_check": 0.07  # 7% are permission checks
}

# Auth clients
AUTH_CLIENTS = [
    "web-app", "mobile-ios", "mobile-android", "desktop-app", 
    "third-party-service", "internal-service", "admin-portal"
]

# HTTP Methods for different operations
AUTH_METHODS = {
    "login": "POST",
    "token_validation": "POST",
    "token_refresh": "POST",
    "logout": "POST",
    "register": "POST",
    "password_reset": "POST",
    "mfa_validation": "POST",
    "permission_check": "GET"
}

# Auth status codes and messages
AUTH_SUCCESS_CODES = {
    "login": 200,
    "token_validation": 200,
    "token_refresh": 200,
    "logout": 204,
    "register": 201,
    "password_reset": 200,
    "mfa_validation": 200,
    "permission_check": 200
}

AUTH_ERROR_CODES = {
    "invalid_credentials": 401,
    "invalid_token": 401,
    "expired_token": 401,
    "insufficient_permissions": 403,
    "account_locked": 403,
    "invalid_request": 400,
    "rate_limited": 429,
    "server_error": 500,
    "service_unavailable": 503
}

AUTH_ERROR_MESSAGES = {
    "invalid_credentials": ["Invalid username or password", "Authentication failed"],
    "invalid_token": ["Invalid authentication token", "Token verification failed"],
    "expired_token": ["Token has expired", "Authentication token expired"],
    "insufficient_permissions": ["Insufficient permissions for this operation", "Access denied"],
    "account_locked": ["Account has been locked due to too many failed attempts", "Account temporarily suspended"],
    "invalid_request": ["Invalid request parameters", "Malformed request"],
    "rate_limited": ["Too many requests", "Rate limit exceeded"],
    "server_error": ["Internal server error", "Unexpected error occurred"],
    "service_unavailable": ["Authentication service temporarily unavailable", "Service down for maintenance"]
}

# MFA Types
MFA_TYPES = ["sms", "email", "app", "hardware_token"]

# User roles
USER_ROLES = ["admin", "user", "guest", "support", "api", "system"]

# Token types
TOKEN_TYPES = ["access", "refresh", "id"]

def generate_auth_headers(auth_type, include_security_headers=True):
    """Generate authentication related headers"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "PostmanRuntime/7.28.0",
            "python-requests/2.25.1",
            "Internal Service Client/1.0"
        ]),
        "X-Request-ID": str(uuid.uuid4())
    }
    
    # Add auth specific headers
    if auth_type == "api-key":
        headers["X-API-Key"] = f"apk_{uuid.uuid4().hex[:24]}"
    elif auth_type == "jwt" or auth_type == "oauth2":
        headers["Authorization"] = f"Bearer {uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}"
    elif auth_type == "basic":
        fake_token = base64.b64encode(f"{fake.user_name()}:{uuid.uuid4().hex[:12]}".encode()).decode()
        headers["Authorization"] = f"Basic {fake_token}"
    
    # Add security headers
    if include_security_headers:
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["X-Frame-Options"] = "DENY"
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return headers

def generate_user_context():
    """Generate a user context for auth operations"""
    user_id = str(uuid.uuid4())
    
    return {
        "user_id": user_id,
        "username": fake.user_name(),
        "email": fake.email(),
        "role": random.choice(USER_ROLES),
        "tenant_id": f"tenant-{random.randint(1, 100)}",
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365))).isoformat(),
        "last_login": (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30))).isoformat() if random.random() < 0.8 else None,
        "mfa_enabled": random.random() < 0.6,  # 60% of users have MFA enabled
        "account_status": random.choices(
            ["active", "suspended", "pending", "locked"], 
            weights=[0.9, 0.05, 0.03, 0.02]
        )[0]
    }

def generate_token_data(token_type="access", user_id=None):
    """Generate token data"""
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    now = datetime.datetime.now()
    issued_at = int(time.mktime(now.timetuple()))
    
    # Set expiry time based on token type
    if token_type == "access":
        expires_in = 3600  # 1 hour
    elif token_type == "refresh":
        expires_in = 2592000  # 30 days
    else:  # id token
        expires_in = 3600  # 1 hour
    
    expiry = issued_at + expires_in
    
    return {
        "jti": str(uuid.uuid4()),
        "iss": "auth.example.com",
        "sub": user_id,
        "iat": issued_at,
        "exp": expiry,
        "token_type": token_type,
        "scope": " ".join(random.sample(["read", "write", "delete", "admin"], random.randint(1, 3)))
    }

def generate_request_body(operation, auth_type, is_anomalous=False):
    """Generate a request body based on the operation type"""
    body = {}
    
    if operation == "login":
        body = {
            "username": fake.user_name(),
            "password": "********" if not is_anomalous else (None if random.random() < 0.5 else ""),
            "grant_type": "password"
        }
        
        # Add client info for OAuth flows
        if auth_type in ["oauth2", "oidc"]:
            body.update({
                "client_id": f"client-{uuid.uuid4().hex[:8]}",
                "client_secret": f"secret-{uuid.uuid4().hex[:16]}"
            })
    
    elif operation == "token_validation":
        body = {
            "token": f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}",
            "token_type_hint": random.choice(TOKEN_TYPES)
        }
        
        if is_anomalous and random.random() < 0.5:
            body["token"] = "invalid.token.format" if random.random() < 0.5 else ""
    
    elif operation == "token_refresh":
        body = {
            "refresh_token": f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}",
            "client_id": f"client-{uuid.uuid4().hex[:8]}",
            "grant_type": "refresh_token"
        }
        
        if is_anomalous and random.random() < 0.5:
            body["refresh_token"] = "expired.refresh.token" if random.random() < 0.5 else ""
    
    elif operation == "register":
        body = {
            "username": fake.user_name(),
            "email": fake.email(),
            "password": "********",
            "confirm_password": "********"
        }
        
        if is_anomalous and random.random() < 0.7:
            if random.random() < 0.5:
                # Username/email already exists
                pass
            else:
                # Password mismatch
                body["confirm_password"] = "different_password"
    
    elif operation == "password_reset":
        if random.random() < 0.5:
            # Request password reset
            body = {
                "email": fake.email()
            }
        else:
            # Confirm password reset
            body = {
                "token": f"reset-{uuid.uuid4().hex[:16]}",
                "password": "newpassword",
                "confirm_password": "newpassword"
            }
            
            if is_anomalous and random.random() < 0.6:
                body["token"] = "invalid-reset-token"
    
    elif operation == "mfa_validation":
        mfa_type = random.choice(MFA_TYPES)
        
        if mfa_type == "sms" or mfa_type == "email":
            body = {
                "code": str(random.randint(100000, 999999)),
                "mfa_type": mfa_type
            }
        elif mfa_type == "app":
            body = {
                "totp_code": str(random.randint(100000, 999999)),
                "mfa_type": mfa_type
            }
        else:  # hardware token
            body = {
                "token_response": base64.b64encode(os.urandom(20)).decode('utf-8'),
                "mfa_type": mfa_type
            }
            
        if is_anomalous and random.random() < 0.6:
            if mfa_type in ["sms", "email", "app"]:
                body["code" if mfa_type in ["sms", "email"] else "totp_code"] = "invalid"
    
    elif operation == "permission_check":
        body = {
            "resource": f"resource-{random.randint(1, 100)}",
            "action": random.choice(["read", "write", "delete", "admin"])
        }
    
    return body

def generate_response_body(operation, status_code, auth_type, user_id=None):
    """Generate a response body based on the operation and status code"""
    # Success responses
    if status_code < 400:
        if operation == "login":
            access_token_data = generate_token_data("access", user_id)
            refresh_token_data = generate_token_data("refresh", user_id)
            
            response = {
                "access_token": f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}" if auth_type in ["oauth2", "oidc"] else None
            }
            
            # Add id_token for OIDC
            if auth_type == "oidc":
                response["id_token"] = f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}"
            
            return response
        
        elif operation == "token_validation":
            return {
                "active": True,
                "scope": "read write",
                "client_id": f"client-{uuid.uuid4().hex[:8]}",
                "username": fake.user_name(),
                "exp": int(time.time()) + 3600
            }
        
        elif operation == "token_refresh":
            return {
                "access_token": f"{uuid.uuid4().hex}.{uuid.uuid4().hex}.{uuid.uuid4().hex}",
                "token_type": "Bearer",
                "expires_in": 3600
            }
        
        elif operation == "register":
            return {
                "id": user_id or str(uuid.uuid4()),
                "username": fake.user_name(),
                "email": fake.email(),
                "created_at": datetime.datetime.now().isoformat()
            }
        
        elif operation == "password_reset":
            if "token" in generate_request_body(operation, auth_type):
                return {
                    "message": "Password has been reset successfully"
                }
            else:
                return {
                    "message": "Password reset instructions sent to email"
                }
        
        elif operation == "mfa_validation":
            return {
                "success": True,
                "message": "MFA validation successful"
            }
        
        elif operation == "permission_check":
            return {
                "allowed": True,
                "resource": f"resource-{random.randint(1, 100)}",
                "action": random.choice(["read", "write", "delete", "admin"])
            }
        
        elif operation == "logout":
            return {}  # Usually empty for successful logout (status 204)
    
    # Error responses
    else:
        error_code = next((k for k, v in AUTH_ERROR_CODES.items() if v == status_code), "server_error")
        error_message = random.choice(AUTH_ERROR_MESSAGES.get(error_code, ["Unknown error"]))
        
        return {
            "error": error_code,
            "error_description": error_message,
            "status": status_code,
            "timestamp": datetime.datetime.now().isoformat()
        }

def calculate_auth_response_time(operation, status_code, is_anomalous=False):
    """Calculate realistic response times for auth operations"""
    # Base times in seconds
    base_times = {
        "login": 0.2,
        "token_validation": 0.05,
        "token_refresh": 0.1,
        "logout": 0.05,
        "register": 0.3,
        "password_reset": 0.2,
        "mfa_validation": 0.2,
        "permission_check": 0.05
    }
    
    # Get base time for this operation
    base_time = base_times.get(operation, 0.1)
    
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
    
    return response_time

def get_auth_path(operation):
    """Get the API path for different auth operations"""
    if operation == "login":
        return "/auth/token"
    elif operation == "token_validation":
        return "/auth/token/validate"
    elif operation == "token_refresh":
        return "/auth/token/refresh"
    elif operation == "logout":
        return "/auth/logout"
    elif operation == "register":
        return "/auth/register"
    elif operation == "password_reset":
        return "/auth/password/reset"
    elif operation == "mfa_validation":
        return "/auth/mfa/validate"
    elif operation == "permission_check":
        return "/auth/permissions/check"
    else:
        return f"/auth/{operation}"

def generate_auth_log_entry(timestamp=None, operation=None, auth_type=None, 
                          correlation_id=None, is_anomalous=False, session_id=None):
    """Generate a single auth service log entry"""
    
    if session_id is None:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Generate auth details
    if auth_type is None:
        auth_type = random.choice(AUTH_TYPES)
    
    # Select random operation if not provided
    if operation is None:
        operations = list(AUTH_OPERATIONS.keys())
        weights = list(AUTH_OPERATIONS.values())
        operation = random.choices(operations, weights=weights)[0]
    
    # Environment and region
    environment = random.choice(AUTH_ENVIRONMENTS)
    region = random.choice(AUTH_REGIONS)
    server = random.choice(AUTH_SERVERS)
    instance_id = f"{server}-{region}-{random.randint(1, 5)}"
    
    # Generate correlation ID if not provided (for tracing requests across services)
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Request details
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    http_method = AUTH_METHODS.get(operation, "POST")
    path = get_auth_path(operation)
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    client_type = random.choice(AUTH_CLIENTS)
    source_ip = str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
    
    # Generate user context
    user_id = str(uuid.uuid4())
    user_context = generate_user_context()
    user_id = user_context["user_id"]
    
    # Set headers
    request_headers = generate_auth_headers(auth_type)
    
    # Generate request body
    request_body = generate_request_body(operation, auth_type, is_anomalous)
    
    # Determine status code based on anomaly flag
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous requests
        error_types = list(AUTH_ERROR_CODES.keys())
        error_type = random.choice(error_types)
        status_code = AUTH_ERROR_CODES[error_type]
    else:
        # Normal requests mostly succeed
        weights = [0.95, 0.05]  # 95% success, 5% error
        status_category = random.choices(["success", "error"], weights=weights)[0]
        
        if status_category == "success":
            status_code = AUTH_SUCCESS_CODES.get(operation, 200)
        else:
            error_types = list(AUTH_ERROR_CODES.keys())
            error_type = random.choice(error_types)
            status_code = AUTH_ERROR_CODES[error_type]
    
    # Generate response body
    response_body = generate_response_body(operation, status_code, auth_type, user_id)
    
    # Calculate response time
    response_time = calculate_auth_response_time(operation, status_code, is_anomalous)
    
    # Create the auth log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "auth_service": {
            "type": auth_type,
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
        "operation": {
            "type": operation,
            "success": status_code < 400,
            "user_id": user_id if status_code < 400 or operation != "login" else None,
            "tenant_id": user_context["tenant_id"]
        },
        "security": {
            "mfa_used": operation == "mfa_validation" or (operation == "login" and request_body.get("mfa_type") is not None),
            "ip_reputation": random.choices(["safe", "suspicious", "malicious"], weights=[0.95, 0.04, 0.01])[0],
            "rate_limit": {
                "limit": 100,
                "remaining": random.randint(0, 100),
                "reset": int(time.time()) + random.randint(1, 3600)
            }
        },
        "tracing": {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "session_id": session_id  # Add session_id to tracing information
        },
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_auth_flow(base_time=None, correlation_id=None, is_anomalous=False):
    """Generate a sequence of related auth operations that form a workflow"""
    if base_time is None:
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Randomly select auth type for this flow
    auth_type = random.choice(AUTH_TYPES)
    
    # Define possible auth flows
    flow_options = [
        # Login flow with MFA
        ["login", "mfa_validation", "token_validation", "permission_check", "logout"],
        # Register and login flow
        ["register", "login", "token_validation", "permission_check"],
        # Password reset flow
        ["password_reset", "login", "token_validation"],
        # Token refresh flow
        ["token_validation", "token_refresh", "token_validation"],
        # Simple login-logout flow
        ["login", "permission_check", "logout"]
    ]
    
    # Select a flow
    flow = random.choice(flow_options)
    
    # Determine if we introduce an error and at which step
    error_step = random.randint(0, len(flow) - 1) if is_anomalous else None
    
    # Generate logs for each step in the flow
    flow_logs = []
    current_time = base_time
    user_id = str(uuid.uuid4())  # Same user for the whole flow
    
    for i, operation in enumerate(flow):
        # Add some time between operations
        current_time += datetime.timedelta(milliseconds=random.randint(100, 2000))
        
        # Determine if this step should be anomalous
        step_is_anomalous = is_anomalous and (i == error_step or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_auth_log_entry(
            timestamp=current_time,
            operation=operation,
            auth_type=auth_type,
            correlation_id=correlation_id,
            is_anomalous=step_is_anomalous
        )
        
        # Ensure user_id is consistent across the flow
        log_entry["operation"]["user_id"] = user_id if log_entry["response"]["status_code"] < 400 or operation != "login" else None
        
        flow_logs.append(log_entry)
        
        # If we get an error, we might stop the flow
        if step_is_anomalous and log_entry["response"]["status_code"] >= 400 and random.random() < 0.7:
            break
    
    return flow_logs

def generate_auth_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of auth service logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.6)):  # 60% of normal logs are individual
        logs.append(generate_auth_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.4)):  # 40% of anomalous logs are individual
        logs.append(generate_auth_log_entry(is_anomalous=True))
    
    # Generate normal auth flows
    num_normal_flows = int(num_normal * 0.4 / 4)  # Approximately 4 logs per flow
    for _ in range(num_normal_flows):
        flow_logs.extend(generate_auth_flow(is_anomalous=False))
    
    # Generate anomalous auth flows
    num_anomalous_flows = int(num_anomalous * 0.6 / 4)  # Approximately 4 logs per flow
    for _ in range(num_anomalous_flows):
        flow_logs.extend(generate_auth_flow(is_anomalous=True))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

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
    
def write_logs_as_text(log_entries, file_path):
    """
    Writes log entries to a simple text file.
    Each log entry is written as a single line of JSON, 
    no commas between entries, no enclosing square brackets, 
    and entries are separated by newlines.
    
    Args:
        log_entries: List of log entries (dictionaries)
        file_path: Path to the output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    import os
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Write each log as a separate JSON line
        with open(file_path, 'w') as f:
            for log in log_entries:
                f.write(json.dumps(log) + '\n')
            
        print(f"Successfully wrote {len(log_entries)} logs to {file_path}")
        return True
    except Exception as e:
        import traceback
        print(f"ERROR writing logs to {file_path}: {str(e)}")
        print(traceback.format_exc())
        return False

# Add this to your existing save_auth_logs function in auth_api_generator.py

def save_auth_logs(logs, format='json', filename='auth_logs'):
    """Save auth logs to a file in the specified format"""
    import json
    import pandas as pd
    
    # First, save logs to a text file (simple format with no commas or brackets)
    text_file_path = f"{filename}.txt"
    try:
        with open(text_file_path, 'w') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')
        print(f"Saved {len(logs)} auth logs to {text_file_path}")
    except Exception as e:
        print(f"Error saving logs to text file: {e}")
    
    # Then proceed with the original JSON or CSV saving
    if format.lower() == 'json':
        file_path = f"{filename}.json"

        write_logs_as_json_array(logs, file_path)
        return file_path
        
    elif format.lower() == 'csv':
        # CSV output remains the same
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "auth_type": log["auth_service"]["type"],
                "environment": log["auth_service"]["environment"],
                "region": log["auth_service"]["region"],
                "instance_id": log["auth_service"]["instance_id"],
                "request_id": log["request"]["id"],
                "request_method": log["request"]["method"],
                "request_path": log["request"]["path"],
                "client_id": log["request"]["client_id"],
                "client_type": log["request"]["client_type"],
                "source_ip": log["request"]["source_ip"],
                "response_status_code": log["response"]["status_code"],
                "response_time_ms": log["response"]["time_ms"],
                "operation_type": log["operation"]["type"],
                "operation_success": log["operation"]["success"],
                "user_id": log["operation"]["user_id"],
                "tenant_id": log["operation"]["tenant_id"],
                "mfa_used": log["security"]["mfa_used"],
                "ip_reputation": log["security"]["ip_reputation"],
                "correlation_id": log["tracing"]["correlation_id"],
                "is_anomalous": log["is_anomalous"]
            }
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} auth logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_auth_logs(logs):
    """Print analysis of the generated auth logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by auth type
    auth_types = {}
    for log in logs:
        auth_type = log['auth_service']['type']
        auth_types[auth_type] = auth_types.get(auth_type, 0) + 1
    
    # Count by operation
    operations = {}
    for log in logs:
        operation = log['operation']['type']
        operations[operation] = operations.get(operation, 0) + 1
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['response']['status_code']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Success vs. failure rate
    success_count = sum(1 for log in logs if log['operation']['success'])
    failure_count = total_logs - success_count
    
    # Calculate average response times
    normal_times = [log['response']['time_ms'] for log in logs if not log.get('is_anomalous', False)]
    anomalous_times = [log['response']['time_ms'] for log in logs if log.get('is_anomalous', False)]
    
    avg_normal_time = sum(normal_times) / len(normal_times) if normal_times else 0
    avg_anomalous_time = sum(anomalous_times) / len(anomalous_times) if anomalous_times else 0
    
    # Count MFA usage
    mfa_count = sum(1 for log in logs if log['security']['mfa_used'])
    
    # Count unique correlation IDs (auth flows)
    correlation_ids = set(log['tracing']['correlation_id'] for log in logs)
    
    print("\n=== Auth Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Success rate: {success_count/total_logs*100:.2f}%")
    print(f"Failure rate: {failure_count/total_logs*100:.2f}%")
    print(f"MFA usage: {mfa_count/total_logs*100:.2f}%")
    print(f"Unique auth flows: {len(correlation_ids)}")
    print(f"Average response time (normal): {avg_normal_time:.2f} ms")
    print(f"Average response time (anomalous): {avg_anomalous_time:.2f} ms")
    
    print("\n=== Auth Type Distribution ===")
    for auth_type, count in sorted(auth_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{auth_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Operation Distribution ===")
    for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        print(f"{operation}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"HTTP {code}: {count} logs ({count/total_logs*100:.2f}%)")

def generate_interconnected_auth_logs(gateway_logs, num_logs=500):
    """Generate auth logs that share correlation IDs with existing gateway logs"""
    auth_logs = []
    
    # Extract correlation IDs from gateway logs
    correlation_ids = []
    for log in gateway_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            correlation_ids.append(log["tracing"]["correlation_id"])
    
    # If no correlation IDs found, generate standalone logs
    if not correlation_ids:
        print("No correlation IDs found in gateway logs. Generating standalone auth logs.")
        return generate_auth_logs(num_logs)
    
    # For each correlation ID, generate a small auth flow
    for correlation_id in correlation_ids:
        # Generate a base timestamp
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # Generate auth flow with this correlation ID
        flow_logs = generate_auth_flow(
            base_time=base_time,
            correlation_id=correlation_id,
            is_anomalous=random.random() < 0.2  # 20% chance of anomalous flow
        )
        
        auth_logs.extend(flow_logs)
        
        # Stop if we've generated enough logs
        if len(auth_logs) >= num_logs:
            break
    
    # If we need more logs, add some individual ones
    while len(auth_logs) < num_logs:
        correlation_id = random.choice(correlation_ids)
        auth_logs.append(generate_auth_log_entry(correlation_id=correlation_id))
    
    return auth_logs

if __name__ == "__main__":
    import os
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate auth logs
    print("Generating auth service logs...")
    logs = generate_auth_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_auth_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_auth_logs(logs, format='json')
    csv_path = save_auth_logs(logs, format='csv')
    
    print(f"\nAuth logs have been saved to {json_path} and {csv_path}")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Auth Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2)[:1000] + "... (truncated)")
    
    print("\n=== Sample Anomalous Auth Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2)[:1000] + "... (truncated)")
    
    # Generate auth flow examples
    print("\nGenerating example auth flows...")
    auth_flows = [
        ["login", "mfa_validation", "token_validation", "logout"],
        ["register", "login", "token_validation"],
        ["password_reset", "login"]
    ]
    
    for flow in auth_flows:
        flow_name = " -> ".join(flow)
        print(f"\nExample flow: {flow_name}")
        auth_type = random.choice(AUTH_TYPES)
        flow_logs = []
        
        for operation in flow:
            log = generate_auth_log_entry(operation=operation, auth_type=auth_type)
            flow_logs.append(log)
        
        print(f"Generated {len(flow_logs)} logs for this flow")