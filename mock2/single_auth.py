from hackohire.HOH.mock.travel_app.auth_api_generator import generate_auth_log_entry
import json
import datetime

# Optional: Set a specific timestamp (e.g., current time)
timestamp = datetime.datetime.now()

# Generate one log entry for a successful login
auth_log = generate_auth_log_entry(
    timestamp=timestamp,
    operation="login",
    is_anomalous=False  # Set to True if you want to simulate an anomaly
)

# Save it to a file
filename = "auth_interactions.json"
with open(filename, "a") as f:
    f.write(json.dumps(auth_log, indent=2) + "\n")

print(f"Auth log saved to {filename}")
