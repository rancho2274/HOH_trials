from flask import Flask, render_template, request, redirect, url_for
from search_api_generator import generate_search_log_entry
from booking_api_generator import generate_booking_log_entry
from payment_api_generator import generate_payment_log_entry
from feedback_api_generator import generate_feedback_log_entry
from auth_api_generator import generate_auth_log_entry
import json
from datetime import datetime



app = Flask(__name__)

# Mock hotel data
hotels_data = {
    "paris": ["Hotel Eiffel", "Paris Inn", "Louvre Palace"],
    "new york": ["Times Square Hotel", "NY Comfort Inn", "Empire Lodge"],
    "london": ["London Bridge Hotel", "The Big Ben Inn", "Buckingham Suites"]
}

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    # Generate auth log on login (POST)
    if request.method == 'POST':
        timestamp = datetime.now()
        auth_log = generate_auth_log_entry(
            timestamp=timestamp,
            operation="login",
            is_anomalous=False
        )

        # Save to file
        with open("auth_interactions.json", "a") as f:
            f.write(json.dumps(auth_log, indent=2) + "\n")

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
        search_type="hotel_search",
        is_anomalous=False
    )
    with open("search_interactions.json", "a") as f:
        f.write(json.dumps(search_log, indent=2) + "\n")
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
    with open("booking_interactions.json", "a") as f:
        f.write(json.dumps(booking_log, indent=2) + "\n")

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
    
    # Convert rooms to integer BEFORE passing to template
    rooms = request.form.get('rooms')
    try:
        rooms_int = int(rooms)
    except:
        rooms_int = 1  # fallback in case of error

    total = rooms_int * 4999
    # Generate payment log
    payment_log = generate_payment_log_entry(
        timestamp=datetime.now(),
        operation="process_payment",
        is_anomalous=False
    )
    with open("payment_interactions.json", "a") as f:
        f.write(json.dumps(payment_log, indent=2) + "\n")

    return render_template('payment.html', name=name, email=email, arrival=arrival, departure=departure, rooms=rooms_int, total=total)



@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_log = generate_feedback_log_entry(
        timestamp=datetime.now(),
        operation="submit_feedback",
        is_anomalous=False
    )
    with open("feedback_interactions.json", "a") as f:
        f.write(json.dumps(feedback_log, indent=2) + "\n")
    return render_template('feedback.html')


@app.route('/thankyou', methods=['POST'])
def thankyou():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
