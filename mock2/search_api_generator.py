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

# Define constants for Search Service log generation
SEARCH_TYPES = [
    "flight_search", "hotel_search", "car_rental_search", 
    "package_search", "destination_info", "availability_check"
]

SEARCH_ENVIRONMENTS = ["dev", "staging", "production", "test"]
SEARCH_SERVERS = ["search-primary", "search-replica", "search-cache"]
SEARCH_REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "global"]

# Popular destinations
POPULAR_DESTINATIONS = [
    "New York", "Paris", "Tokyo", "London", "Sydney", "Dubai",
    "Rome", "Bangkok", "Singapore", "Hong Kong", "Barcelona",
    "Istanbul", "Prague", "Amsterdam", "Rio de Janeiro",
    "Vienna", "Madrid", "Toronto", "San Francisco", "Berlin"
]

AIRLINES = [
    "Air France", "British Airways", "Lufthansa", "Emirates",
    "Delta Airlines", "United Airlines", "Qatar Airways",
    "Singapore Airlines", "Japan Airlines", "KLM",
    "American Airlines", "Turkish Airlines", "Cathay Pacific",
    "Etihad Airways", "Qantas", "Eva Air", "ANA", "Virgin Atlantic"
]

HOTEL_CHAINS = [
    "Marriott", "Hilton", "Hyatt", "InterContinental", "Accor",
    "Wyndham", "Best Western", "Radisson", "Four Seasons",
    "Shangri-La", "Mandarin Oriental", "Ritz-Carlton",
    "Holiday Inn", "Sheraton", "Westin", "Novotel"
]

CAR_RENTAL_COMPANIES = [
    "Hertz", "Avis", "Enterprise", "Budget", "Sixt",
    "Europcar", "National", "Alamo", "Dollar", "Thrifty"
]

CABIN_CLASSES = ["Economy", "Premium Economy", "Business", "First"]
ROOM_TYPES = ["Single", "Double", "Twin", "Suite", "Deluxe", "Presidential"]
CAR_CATEGORIES = ["Economy", "Compact", "Midsize", "Full-size", "SUV", "Luxury"]

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
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD", "AED"]

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
        "Route not available",
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

def calculate_response_time(search_type, status_code, is_anomalous=False):
    """Calculate realistic response times for search operations"""
    # Base times in seconds
    base_times = {
        "flight_search": 0.4,
        "hotel_search": 0.3,
        "car_rental_search": 0.2,
        "package_search": 0.5,
        "destination_info": 0.1,
        "availability_check": 0.2
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
    if search_type == "flight_search":
        path = "/api/search/flights"
    elif search_type == "hotel_search":
        path = "/api/search/hotels"
    elif search_type == "car_rental_search":
        path = "/api/search/cars"
    elif search_type == "package_search":
        path = "/api/search/packages"
    elif search_type == "destination_info":
        path = "/api/destinations"
    elif search_type == "availability_check":
        path = "/api/availability"
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
    departure_date, return_date = generate_travel_dates()
    
    # Generate request body based on search type
    request_body = {}
    
    if search_type == "flight_search":
        origin = random.choice(POPULAR_DESTINATIONS)
        destination = random.choice([d for d in POPULAR_DESTINATIONS if d != origin])
        
        request_body = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date if random.random() < 0.8 else None,  # 80% round trips
            "passengers": {
                "adults": random.randint(1, 4),
                "children": random.randint(0, 2),
                "infants": random.randint(0, 1)
            },
            "cabin_class": random.choice(CABIN_CLASSES),
            "flexible_dates": random.choice([True, False]),
            "direct_flights_only": random.choice([True, False])
        }
        
        if is_anomalous and random.random() < 0.5:
            # Create anomalous request
            if random.random() < 0.5:
                # Missing required field
                del request_body["destination"]
            else:
                # Invalid date format
                request_body["departure_date"] = "invalid-date"
                
    elif search_type == "hotel_search":
        request_body = {
            "destination": random.choice(POPULAR_DESTINATIONS),
            "check_in_date": departure_date,
            "check_out_date": return_date,
            "rooms": random.randint(1, 3),
            "guests": {
                "adults": random.randint(1, 4),
                "children": random.randint(0, 2)
            },
            "star_rating_min": random.randint(1, 5),
            "amenities": random.sample(["pool", "wifi", "breakfast", "parking", "gym", "spa", "restaurant"], 
                                      k=random.randint(0, 4))
        }
        
        if is_anomalous and random.random() < 0.5:
            # Create anomalous request
            request_body["check_in_date"] = return_date  # Check-in after check-out
            request_body["check_out_date"] = departure_date
            
    elif search_type == "car_rental_search":
        # Create time strings for pickup and dropoff
        pickup_hour = random.randint(0, 23)
        pickup_minute = random.choice(["00", "30"])
        pickup_time = f"{pickup_hour:02d}:{pickup_minute}"
        
        dropoff_hour = random.randint(0, 23)
        dropoff_minute = random.choice(["00", "30"])
        dropoff_time = f"{dropoff_hour:02d}:{dropoff_minute}"
        
        request_body = {
            "pick_up_location": random.choice(POPULAR_DESTINATIONS),
            "drop_off_location": random.choice(POPULAR_DESTINATIONS),
            "pick_up_date": departure_date,
            "pick_up_time": pickup_time,
            "drop_off_date": return_date,
            "drop_off_time": dropoff_time,
            "car_category": random.choice(CAR_CATEGORIES),
            "driver_age": random.randint(21, 75)
        }
        
    elif search_type == "package_search":
        origin = random.choice(POPULAR_DESTINATIONS)
        destination = random.choice([d for d in POPULAR_DESTINATIONS if d != origin])
        
        request_body = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": {
                "adults": random.randint(1, 4),
                "children": random.randint(0, 2)
            },
            "include_flight": True,
            "include_hotel": True,
            "include_car": random.choice([True, False]),
            "star_rating_min": random.randint(3, 5)
        }
        
    elif search_type == "destination_info":
        request_body = {
            "destination": random.choice(POPULAR_DESTINATIONS),
            "include_attractions": True,
            "include_weather": True,
            "include_travel_advisories": random.choice([True, False])
        }
        
    elif search_type == "availability_check":
        request_body = {
            "item_type": random.choice(["flight", "hotel", "car", "package"]),
            "item_id": f"item-{uuid.uuid4().hex[:8]}",
            "date": departure_date
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
        if search_type == "flight_search":
            num_results = random.randint(1, 10) if random.random() < 0.9 else 0  # Sometimes return empty results
            
            flights = []
            for _ in range(num_results):
                # Create formatted departure and arrival times
                departure_hour = random.randint(0, 23)
                departure_minute = random.choice(["00", "15", "30", "45"])
                departure_time = f"{departure_date}T{departure_hour:02d}:{departure_minute}:00"
                
                arrival_hour = random.randint(0, 23)
                arrival_minute = random.choice(["00", "15", "30", "45"])
                arrival_time = f"{departure_date}T{arrival_hour:02d}:{arrival_minute}:00"
                
                flight = {
                    "flight_id": f"flight-{uuid.uuid4().hex[:8]}",
                    "airline": random.choice(AIRLINES),
                    "origin": request_body.get("origin", "Unknown"),
                    "destination": request_body.get("destination", "Unknown"),
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "duration_minutes": random.randint(60, 1200),
                    "stops": random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0],
                    "cabin_class": request_body.get("cabin_class", "Economy"),
                    "price": {
                        "amount": round(random.uniform(100, 5000), 2),
                        "currency": random.choice(CURRENCIES)
                    },
                    "seats_available": random.randint(1, 50)
                }
                flights.append(flight)
            
            response_body = {
                "search_id": f"search-{uuid.uuid4().hex[:8]}",
                "num_results": len(flights),
                "flights": flights
            }
            
        elif search_type == "hotel_search":
            num_results = random.randint(1, 15) if random.random() < 0.9 else 0
            
            hotels = []
            for _ in range(num_results):
                hotel = {
                    "hotel_id": f"hotel-{uuid.uuid4().hex[:8]}",
                    "name": f"{random.choice(HOTEL_CHAINS)} {random.choice(['Resort', 'Hotel', 'Suites', 'Inn'])}",
                    "destination": request_body.get("destination", "Unknown"),
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
                "num_results": len(hotels),
                "hotels": hotels
            }
            
        elif search_type == "car_rental_search":
            num_results = random.randint(1, 8) if random.random() < 0.9 else 0
            
            cars = []
            for _ in range(num_results):
                car = {
                    "car_id": f"car-{uuid.uuid4().hex[:8]}",
                    "company": random.choice(CAR_RENTAL_COMPANIES),
                    "category": request_body.get("car_category", "Economy"),
                    "model": f"{fake.company()} {fake.last_name()}",
                    "pick_up_location": request_body.get("pick_up_location", "Unknown"),
                    "drop_off_location": request_body.get("drop_off_location", "Unknown"),
                    "price": {
                        "amount": round(random.uniform(20, 200), 2),
                        "currency": random.choice(CURRENCIES),
                        "rate": "daily"
                    },
                    "features": random.sample(["GPS", "A/C", "Automatic", "Unlimited mileage", "Bluetooth"], 
                                             k=random.randint(1, 5))
                }
                cars.append(car)
            
            response_body = {
                "search_id": f"search-{uuid.uuid4().hex[:8]}",
                "num_results": len(cars),
                "cars": cars
            }
            
        elif search_type == "package_search":
            num_results = random.randint(1, 5) if random.random() < 0.9 else 0
            
            packages = []
            for _ in range(num_results):
                package = {
                    "package_id": f"package-{uuid.uuid4().hex[:8]}",
                    "name": f"{request_body.get('destination', 'Destination')} {random.choice(['Getaway', 'Adventure', 'Escape', 'Package'])}",
                    "origin": request_body.get("origin", "Unknown"),
                    "destination": request_body.get("destination", "Unknown"),
                    "includes_flight": request_body.get("include_flight", True),
                    "includes_hotel": request_body.get("include_hotel", True),
                    "includes_car": request_body.get("include_car", False),
                    "duration_days": (datetime.datetime.strptime(return_date, "%Y-%m-%d") - 
                                    datetime.datetime.strptime(departure_date, "%Y-%m-%d")).days,
                    "total_price": {
                        "amount": round(random.uniform(500, 10000), 2),
                        "currency": random.choice(CURRENCIES)
                    }
                }
                packages.append(package)
            
            response_body = {
                "search_id": f"search-{uuid.uuid4().hex[:8]}",
                "num_results": len(packages),
                "packages": packages
            }
            
        elif search_type == "destination_info":
            response_body = {
                "destination": request_body.get("destination", "Unknown"),
                "country": fake.country(),
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
            
        elif search_type == "availability_check":
            is_available = random.random() < 0.8  # 80% of checks return available
            
            response_body = {
                "item_type": request_body.get("item_type", "Unknown"),
                "item_id": request_body.get("item_id", "Unknown"),
                "date": request_body.get("date", "Unknown"),
                "available": is_available,
                "quantity_available": random.randint(1, 20) if is_available else 0,
                "price": {
                    "amount": round(random.uniform(50, 1000), 2),
                    "currency": random.choice(CURRENCIES)
                } if is_available else None
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
    
    # Choose a search sequence pattern
    search_patterns = [
        # Flight-focused pattern
        ["flight_search", "flight_search", "availability_check"],
        # Hotel-focused pattern
        ["destination_info", "hotel_search", "hotel_search"],
        # Car rental pattern
        ["destination_info", "car_rental_search", "availability_check"],
        # Package search pattern
        ["destination_info", "package_search", "availability_check"],
        # Comprehensive search pattern
        ["flight_search", "hotel_search", "car_rental_search"]
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
    num_normal_flows = int(num_normal * 0.4 / 3)  # Approximately 3 logs per flow
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
    num_anomalous_flows = int(num_anomalous * 0.6 / 3)  # Approximately 3 logs per flow
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

def generate_interconnected_search_logs(auth_logs, num_logs=500):
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
    
    return search_logs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Search API logs')
    parser.add_argument('--num-logs', type=int, default=1000, help='Number of logs to generate')
    parser.add_argument('--anomaly-percentage', type=int, default=15, help='Percentage of anomalous logs')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json', help='Output file format')
    parser.add_argument('--output-filename', default='search_logs', help='Output filename (without extension)')
    parser.add_argument('--analyze', action='store_true', help='Analyze generated logs')
    parser.add_argument('--connect-with', help='Path to auth logs file to connect with')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Generating search service logs...")
    
    # Generate logs
    if args.connect_with:
        try:
            print(f"Loading auth logs from: {args.connect_with}")
            with open(args.connect_with, 'r') as f:
                auth_logs = json.load(f)
            print(f"Loaded {len(auth_logs)} auth logs")
            logs = generate_interconnected_search_logs(auth_logs, args.num_logs)
        except Exception as e:
            print(f"Error loading auth logs: {e}")
            print("Falling back to standalone log generation")
            logs = generate_search_logs(args.num_logs, args.anomaly_percentage)
    else:
        logs = generate_search_logs(args.num_logs, args.anomaly_percentage)
    
    # Analyze logs
    analyze_search_logs(logs)
    
    # Save logs to file
    json_path = save_logs_to_file(logs, format='json', filename=args.output_filename)
    csv_path = save_logs_to_file(logs, format='csv', filename=args.output_filename)
    
    print(f"\nSearch logs have been saved to {json_path} and {csv_path}")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Search Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2)[:1000] + "... (truncated)")
    
    print("\n=== Sample Anomalous Search Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2)[:1000] + "... (truncated)")
    
    # Generate search flow examples
    print("\nGenerating example search flows...")
    search_flow_examples = [
        ["flight_search", "availability_check"],
        ["destination_info", "hotel_search"],
        ["flight_search", "hotel_search", "car_rental_search"]
    ]
    
    for flow in search_flow_examples:
        flow_name = " -> ".join(flow)
        print(f"\nExample flow: {flow_name}")
        user_id = str(uuid.uuid4())
        correlation_id = generate_correlation_id()
        
        # Create a custom flow with the specific pattern
        flow_logs = []
        current_timestamp = datetime.datetime.now()
        
        for i, search_type in enumerate(flow):
            # Add some time between operations
            current_timestamp += datetime.timedelta(seconds=random.randint(2, 10))
            
            # Generate the log entry
            parent_request_id = None if i == 0 else flow_logs[-1]["request"]["id"]
            
            log_entry = generate_search_log_entry(
                timestamp=current_timestamp,
                search_type=search_type,
                correlation_id=correlation_id,
                user_id=user_id,
                is_anomalous=False,
                parent_request_id=parent_request_id
            )
            
            flow_logs.append(log_entry)
        
        print(f"Generated {len(flow_logs)} logs for this flow")