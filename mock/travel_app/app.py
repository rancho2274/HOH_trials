# Modified app.py functions to write single-line JSON logs
from flask import Flask, render_template, request, redirect, url_for
from search_api_generator import generate_search_log_entry
from booking_api_generator import generate_booking_log_entry
from payment_api_generator import generate_payment_log_entry
from feedback_api_generator import generate_feedback_log_entry
from auth_api_generator import generate_auth_log_entry
import json
import os
import traceback
from datetime import datetime

app = Flask(__name__)

# Single-line JSON writer function
# Define a consistent function for writing logs in JSON array format
def append_log_to_text_file(log_entry, file_path):
    """
    Appends a single log entry to a text file.
    If the file doesn't exist, creates a new file.
    Each log is a JSON string on a single line.
    """
    import json
    import os
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Append the log entry to the text file
        with open(file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n\n')
        
        print(f"Successfully appended log to text file: {file_path}")
        return True
    except Exception as e:
        import traceback
        print(f"ERROR appending log to {file_path}: {str(e)}")
        print(traceback.format_exc())
        return False
def append_log_to_json_array(log_entry, file_path):
    """
    Appends a single log entry to a JSON array file.
    If the file doesn't exist, creates a new file with a JSON array containing the log entry.
    """
    import json
     
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # If file doesn't exist, create it with a new JSON array
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('[\n')
                f.write(json.dumps(log_entry))
                f.write('\n]\n')
            print(f"Created new JSON array file with 1 log: {file_path}")
            return True
        
        # File exists, need to append to the JSON array
        with open(file_path, 'r+') as f:
            # Move cursor to position before the closing bracket
            f.seek(0, os.SEEK_END)  # Go to end of file
            pos = f.tell()  # Get position
            
            # Search backwards to find the last bracket
            while pos > 0:
                pos -= 1
                f.seek(pos)
                char = f.read(1)
                if char == ']':
                    break
            
            if pos <= 0:
                # File is not a valid JSON array, overwrite it
                f.seek(0)
                f.truncate()
                f.write('[\n')
                f.write(json.dumps(log_entry))
                f.write('\n]\n')
            else:
                # Position cursor before the closing bracket
                f.seek(pos)
                f.truncate()  # Remove closing bracket
                
                # Check if there are existing entries (need a comma)
                f.seek(0)
                content = f.read(pos)
                need_comma = '}' in content or ']' in content
                
                # Move cursor back to truncation point
                f.seek(pos)
                
                # Add comma if needed
                if need_comma:
                    f.write(',\n')
                
                # Write new log entry and closing bracket
                f.write(json.dumps(log_entry))
                f.write('\n]\n')
        
        print(f"Successfully appended log to JSON array: {file_path}")
        return True
    except Exception as e:
        import traceback
        print(f"ERROR appending log to {file_path}: {str(e)}")
        print(traceback.format_exc())
        return False


# Direct file writing approach
def write_auth_log(log_entry):
    """Write auth log to a file as part of a JSON array and to a text file"""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "auth_interactions.json")
    text_file_path = os.path.join(current_dir, "combine_logs\euth_interactions1.json")
    
    # Write to JSON array
    json_success = append_log_to_json_array(log_entry, json_file_path)
    
    # Write to text file
    text_success = append_log_to_text_file(log_entry, text_file_path)
    
    return json_success and text_success

def write_booking_log(log_entry):
    """Write booking log to a file as part of a JSON array and to a text file"""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "booking_interactions.json")
    text_file_path = os.path.join(current_dir, "combine_logs\mooking_interactions1.json")
    
    # Write to JSON array
    json_success = append_log_to_json_array(log_entry, json_file_path)
    
    # Write to text file
    text_success = append_log_to_text_file(log_entry, text_file_path)
    
    return json_success and text_success

def write_payment_log(log_entry):
    """Write payment log to a file as part of a JSON array and to a text file"""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "payment_interactions.json")
    text_file_path = os.path.join(current_dir, "combine_logs\payment_interactions1.json")
    
    # Write to JSON array
    json_success = append_log_to_json_array(log_entry, json_file_path)
    
    # Write to text file
    text_success = append_log_to_text_file(log_entry, text_file_path)
    
    return json_success and text_success

def write_feedback_log(log_entry):
    """Write feedback log to a file as part of a JSON array and to a text file"""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "feedback_interactions.json")
    text_file_path = os.path.join(current_dir, "combine_logs\ceedback_interactions1.json")
    
    # Write to JSON array
    json_success = append_log_to_json_array(log_entry, json_file_path)
    
    # Write to text file
    text_success = append_log_to_text_file(log_entry, text_file_path)
    
    return json_success and text_success

def write_search_log(log_entry):
    """Write search log to a file as part of a JSON array and to a text file"""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "search_interactions.json")
    text_file_path = os.path.join(current_dir, "combine_logs\search_interactions1.json")
    
    # Write to JSON array
    json_success = append_log_to_json_array(log_entry, json_file_path)
    
    # Write to text file
    text_success = append_log_to_text_file(log_entry, text_file_path)
    
    return json_success and text_success



@app.route('/', methods=['GET', 'POST'])
def login():
    # Handle login form submission
    if request.method == 'POST':
        try:
            action = request.form.get('action', 'login')  # Default to login if not specified
            print(f"Login form submitted with action: {action}")
            
            # Create auth log entry
            # Login is anomalous, signup is non-anomalous
            is_anomalous = (action == "login")
            
            auth_log = generate_auth_log_entry(
                timestamp=datetime.now(),
                operation="login" if action == "login" else "signin",
                auth_type="form_login",
                is_anomalous=is_anomalous
            )
            
            # Try to write directly with our specialized function
            success = write_auth_log(auth_log)
            
            if success:
                print(f"Auth log written successfully for {action}!")
            else:
                print(f"Failed to write auth log for {action}")
                
            # Redirect to search page after successful login
            return redirect(url_for('search'))
        except Exception as e:
            print(f"Exception in login route: {str(e)}")
            print(traceback.format_exc())
            # Still redirect to search even if logging fails
            return redirect(url_for('search'))
    
    # Display login form for GET requests
    return render_template('login.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    # This route now only handles search functionality, not authentication
    if request.method == 'POST':
        # Check if it's the icon search (which should be anomalous)
        is_icon_search = request.form.get('search_type') == 'icon'
        
        # Generate search log when search form is submitted
        search_log = generate_search_log_entry(
            timestamp=datetime.now(),
            search_type="initial_search",
            is_anomalous=is_icon_search  # Icon search is anomalous
        )
        
        # Save search log
        write_search_log(search_log)
    
    return render_template('search.html')

@app.route('/hotels', methods=['POST'])
def hotels():
    destination = request.form.get('destination')
    checkin = request.form.get('checkin')
    checkout = request.form.get('checkout')
    guests = request.form.get('guests')
    hotel_data = {
        "paris": [
            {"name": "Hotel Eiffel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "paris1.jpg", "description": "Beachside hotel with pool and cozy rooms."},
            {"name": "Paris Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "paris2.jpg", "description": "Modern hotel near Baga Beach with free parking."},
            {"name": "Louvre Palace", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "paris3.jpg", "description": "Modern comfort near the Louvre Museum."}
        ],
        "new york": [
             {"name": "Times Square Hotel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "ny1.jpg", "description": "Chic city views just steps from Times Square."},
            {"name": "NY Comfort Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "ny2.jpg", "description": "Cozy stay close to Broadway theaters."},
            {"name": "Empire Lodge", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "ny3.jpg", "description": "Modern comfort in the heart of Manhattan."}
        ],
         "london": [
            {"name": "London Bridge Hotel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "london1.jpg", "description": "Historic stay with Thames River views."},
            {"name": "The Big Ben Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "london2.jpg", "description": "Urban chic hotel in vibrant Soho."},
            {"name": "Buckingham Suites", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "london3.jpg", "description": "Classic elegance near Buckingham Palace."}
        ],
    }

    hotels = hotel_data.get(destination, [])
    search_log = generate_search_log_entry(
        timestamp=datetime.now(),
        search_type="location_search",
        is_anomalous=False
    )
    
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_interactions.json")
    append_log_to_json_array(log_file, search_log)
    
    return render_template(
        'search.html',
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=guests
    )

@app.route('/hotels/<location>')
def hotels_by_location(location):
    checkin = request.args.get('checkin')
    checkout = request.args.get('checkout')
    guests = request.args.get('guests')
    hotels_data = {
        "paris": [
            {"name": "Hotel Eiffel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "paris1.jpg", "description": "Beachside hotel with pool and cozy rooms."},
            {"name": "Paris Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "paris2.jpg", "description": "Modern hotel near Baga Beach with free parking."},
            {"name": "Louvre Palace", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "paris3.jpg", "description": "Modern comfort near the Louvre Museum."}
            ],
        "newyork": [
            {"name": "Times Square Hotel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "ny1.jpg", "description": "Chic city views just steps from Times Square."},
            {"name": "NY Comfort Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "ny2.jpg", "description": "Cozy stay close to Broadway theaters."},
            {"name": "Empire Lodge", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "ny3.jpg", "description": "Modern comfort in the heart of Manhattan."}
            ],
        "london": [
            {"name": "London Bridge Hotel", "stars": "★★★", "rating_label": "Exceptional", "reviews": "128", "image": "london1.jpg", "description": "Historic stay with Thames River views."},
            {"name": "The Big Ben Inn", "stars": "★★", "rating_label": "Good", "reviews": "63", "image": "london2.jpg", "description": "Urban chic hotel in vibrant Soho."},
            {"name": "Buckingham Suites", "stars": "★★★★", "rating_label": "Excellent", "reviews": "265", "image": "london3.jpg", "description": "Classic elegance near Buckingham Palace."}
            ],
    }
    search_log = generate_search_log_entry(
        timestamp=datetime.now(),
        search_type="hotel_details_search",
        is_anomalous=False
    )

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_interactions.json")
    append_log_to_json_array(log_file, search_log)

    hotels = hotels_data.get(location, [])
    return render_template("hotels.html", location = location.title(), hotels=hotels, checkin=checkin, checkout=checkout, guests=guests)

@app.route('/book', methods=['POST'])
def book():
    hotel = request.form.get('hotel')
    destination = request.form.get('destination')
    checkin = request.form.get('checkin')
    checkout = request.form.get('checkout')
    guests = request.form.get('guests')
    
    # Check if this is an anomalous booking (3rd hotel)
    is_anomalous = request.form.get('is_anomalous') == 'true'
    print(f"Booking hotel: {hotel}, Anomalous: {is_anomalous}")

    # Parse date strings into datetime objects
    date_format = "%Y-%m-%d"
    checkin_date = datetime.strptime(checkin, date_format)
    checkout_date = datetime.strptime(checkout, date_format)

    # Calculate number of nights
    nights = (checkout_date - checkin_date).days

    # Price setup
    price_per_night = 4999
    taxes = 1000
    total_price = (price_per_night * nights) + taxes
    
    # Generate booking log with appropriate anomalous flag
    booking_log = generate_booking_log_entry(
        timestamp=datetime.now(),
        operation="create_booking",
        is_anomalous=is_anomalous
    )
    
    # Write booking log
    write_booking_log(booking_log)
    
    return render_template("book.html",
        hotel=hotel,
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=guests,
        nights=nights,
        price_per_night=price_per_night,
        total_price=total_price,
        taxes=taxes
    )

@app.route('/payment', methods=['POST'])
def payment():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        arrival = request.form.get('arrival_date')
        departure = request.form.get('departure_date')
        method = request.form.get('method')
        rooms = request.form.get('rooms')
        hotel = request.form.get('hotel')
        nights = int(request.form.get('nights', 1))
        price_per_night = int(request.form.get('price_per_night', 4999))
        taxes = int(request.form.get('taxes', 1000))
        total_price = int(request.form.get('total_price', price_per_night * nights + taxes))

        print(f"Payment form submitted with method: {method}")
        
        try:
            rooms_int = int(rooms)
        except:
            rooms_int = 1

        # UPI payment is considered anomalous (will fail)
        is_anomalous = method == 'upi'

        # Generate payment log
        payment_log = generate_payment_log_entry(
            timestamp=datetime.now(),
            operation="process_payment",
            is_anomalous=is_anomalous
        )
        
        # Try to write directly with our specialized function
        success = write_payment_log(payment_log)
        
        if success:
            print("Payment log written successfully!")
        else:
            print("Failed to write payment log")

        if is_anomalous:
            # Payment failed (UPI method)
            return render_template('book.html',
                name=name,
                email=email,
                arrival=arrival,
                departure=departure,
                rooms=rooms_int,
                hotel=hotel,
                nights=nights,
                price_per_night=price_per_night,
                taxes=taxes,
                total_price=total_price,
                payment_failed=True
            )
        else:
            # Payment successful (Netbanking method)
            return redirect(url_for('feedback'))
    except Exception as e:
        print(f"Exception in payment route: {str(e)}")
        print(traceback.format_exc())
        return redirect(url_for('search'))

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    # Only generate logs for GET requests to avoid duplicate logging
    if request.method == 'GET':
        feedback_log = generate_feedback_log_entry(
            timestamp=datetime.now(),
            operation="view_feedback_form",
            is_anomalous=False
        )
        
        write_feedback_log(feedback_log)
    
    return render_template('feedback.html')

@app.route('/thankyou', methods=['POST'])
def thankyou():
    try:
        # Get feedback text and anomalous flag from form
        feedback_text = request.form.get('feedback', '')
        is_anomalous = request.form.get('is_anomalous') == 'true'
        
        print(f"Feedback submitted. Is 5-star (anomalous): {is_anomalous}")
        print(f"Feedback text: {feedback_text[:50]}...")  # Print first 50 chars
        
        # Generate feedback log with appropriate anomalous flag
        feedback_log = generate_feedback_log_entry(
            timestamp=datetime.now(),
            operation="submit_feedback",
            is_anomalous=is_anomalous
        )
        
        # Write feedback log
        write_feedback_log(feedback_log)
        
        return render_template('thankyou.html')
    except Exception as e:
        print(f"Exception in thankyou route: {str(e)}")
        print(traceback.format_exc())
        return render_template('thankyou.html')

if __name__ == '__main__':
    # Print working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=True, port=5050)