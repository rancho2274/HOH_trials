from datetime import datetime
from auth_api_generator import generate_auth_log_entry
from search_api_generator import generate_search_log_entry
from booking_api_generator import generate_booking_log_entry
from payment_api_generator import generate_payment_log_entry
from feedback_api_generator import generate_feedback_log_entry

# === AUTH ===
def generate_normal_auth_log():
    return generate_auth_log_entry(timestamp=datetime.now(), is_anomalous=False)

def generate_anomalous_auth_log():
    return generate_auth_log_entry(timestamp=datetime.now(), is_anomalous=True)

# === SEARCH ===
def generate_normal_search_log():
    return generate_search_log_entry(timestamp=datetime.now(), search_type="hotel_search", is_anomalous=False)

def generate_anomalous_search_log():
    return generate_search_log_entry(timestamp=datetime.now(), search_type="hotel_search", is_anomalous=True)

# === BOOKING ===
def generate_normal_booking_log():
    return generate_booking_log_entry(timestamp=datetime.now(), operation="create_booking", is_anomalous=False)

def generate_anomalous_booking_log():
    return generate_booking_log_entry(timestamp=datetime.now(), operation="create_booking", is_anomalous=True)

# === PAYMENT ===
def generate_normal_payment_log():
    return generate_payment_log_entry(timestamp=datetime.now(), operation="process_payment", is_anomalous=False)

def generate_anomalous_payment_log():
    return generate_payment_log_entry(timestamp=datetime.now(), operation="process_payment", is_anomalous=True)

# === FEEDBACK ===
def generate_normal_feedback_log():
    return generate_feedback_log_entry(timestamp=datetime.now(), operation="submit_review", is_anomalous=False)

def generate_anomalous_feedback_log():
    return generate_feedback_log_entry(timestamp=datetime.now(), operation="submit_review", is_anomalous=True)
