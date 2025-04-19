from flask import Flask, render_template, request, redirect, url_for
from search_api_generator import generate_search_log_entry
from booking_api_generator import generate_booking_log_entry
from payment_api_generator import generate_payment_log_entry
from feedback_api_generator import generate_feedback_log_entry
from auth_api_generator import generate_auth_log_entry
import json
import os
from datetime import datetime



app = Flask(__name__)

# Mock hotel data
hotels_data = {
    "paris": ["Hotel Eiffel", "Paris Inn", "Louvre Palace"],
    "new york": ["Times Square Hotel", "NY Comfort Inn", "Empire Lodge"],
    "london": ["London Bridge Hotel", "The Big Ben Inn", "Buckingham Suites"]
}
def append_log_to_json_array(file_path, log_entry):
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)

                        if not isinstance(data, list):
                            data = [data]
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            data.append(log_entry)

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
@app.route('/', methods=['GET', 'POST'])
def login():
    # Handle login form submission
    if request.method == 'POST':
        action = request.form.get('action')  # Detect which button was clicked
        
        # Create auth log entry when login button is clicked
        auth_log = generate_auth_log_entry(
            timestamp=datetime.now(),
            operation="user_login",  # Or you can use the 'action' value from the form
            auth_type="form_login",
            is_anomalous=False
        )
        
        # Save auth log
        append_log_to_json_array("auth_interactions.json", auth_log)
        
        # Redirect to search page after successful login
        return redirect(url_for('search'))
    
    # Display login form for GET requests
    return render_template('login.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    # This route now only handles search functionality, not authentication
    if request.method == 'POST':
        # Generate search log when search form is submitted
        search_log = generate_search_log_entry(
            timestamp=datetime.now(),
            search_type="initial_search",
            is_anomalous=False
        )
        
        # Save search log
        append_log_to_json_array("search_interactions.json", search_log)
    
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
        
        # Add more destinations
    }


    hotels = hotel_data.get(destination, [])
    search_log = generate_search_log_entry(
        timestamp=datetime.now(),
        search_type="location_search",
        is_anomalous=False
    )
    append_log_to_json_array("search_interactions.json", search_log)
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

    append_log_to_json_array("search_interactions.json", search_log)


    hotels = hotels_data.get(location, [])
    return render_template("hotels.html", location = location.title(), hotels=hotels, checkin=checkin, checkout=checkout, guests=guests)
       



from datetime import datetime

@app.route('/book', methods=['POST'])
def book():
    hotel = request.form.get('hotel')
    destination = request.form.get('destination')
    checkin = request.form.get('checkin')
    checkout = request.form.get('checkout')
    guests = request.form.get('guests')

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
    booking_log = generate_booking_log_entry(
        timestamp=datetime.now(),
        operation="create_booking",
        is_anomalous=False
    )
    append_log_to_json_array("booking_interactions.json", booking_log)

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

    try:
        rooms_int = int(rooms)
    except:
        rooms_int = 1

    is_anomalous = method == 'upi'

    # Generate payment log
    payment_log = generate_payment_log_entry(
        timestamp=datetime.now(),
        operation="process_payment",
        is_anomalous=is_anomalous
    )
    append_log_to_json_array("payment_interactions.json", payment_log)

    if is_anomalous:
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
        return redirect(url_for('feedback'))


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_log = generate_feedback_log_entry(
        timestamp=datetime.now(),
        operation="submit_feedback",
        is_anomalous=False
    )
    append_log_to_json_array("feedback_interactions.json", feedback_log)
    return render_template('feedback.html')


@app.route('/thankyou', methods=['POST'])
def thankyou():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
