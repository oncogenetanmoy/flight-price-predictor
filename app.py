from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Placeholder for the scraping function (replace with your actual implementation)
def scrape_skyscanner_flights_selenium(origin, destination, date):
    """
    Placeholder function for scraping Skyscanner.
    Replace with your actual Selenium scraping logic adapted for Render.
    This function should return a list of dictionaries, where each dictionary
    represents a flight and includes keys like 'Airline', 'Departure Time',
    'Arrival Time', 'Duration', 'Total Stops', and 'Price'.
    """
    print(f"Placeholder scraping called for {origin} to {destination} on {date}")
    # For demonstration, returning dummy data. In your real app,
    # implement the Selenium scraping logic here that extracts actual data.
    # Since we are not focusing on Skyscanner, this data isn't strictly needed
    # for the "normal" price summary, but the function is called.
    dummy_flights_data = [
        {'Airline': 'Indigo', 'Departure Time': '08:00', 'Arrival Time': '10:00', 'Duration': '2h 0m', 'Total Stops': '0', 'Price': '₹ 4800'},
        {'Airline': 'SpiceJet', 'Departure Time': '09:30', 'Arrival Time': '12:00', 'Duration': '2h 30m', 'Total Stops': '1', 'Price': '₹ 6200'},
        {'Airline': 'AirAsia', 'Departure Time': '10:00', 'Arrival Time': '13:00', 'Duration': '3h 0m', 'Total Stops': '0', 'Price': '₹ 5500'}
    ]
    return dummy_flights_data


app = Flask(__name__)

# Load the trained model and preprocessor when the app starts
try:
    # Load the original model and preprocessor (before fuel costs)
    model_xgb = joblib.load('model_xgb.pkl') # Load the original model
    preprocessor = joblib.load('preprocessor.pkl') # Load the original preprocessor
    print("Original Model and preprocessor loaded successfully!")
except FileNotFoundError:
    print("Error loading original model or preprocessor. Make sure 'model_xgb.pkl' and 'preprocessor.pkl' are in the correct location.")
    model_xgb = None # Set to None if loading fails
    preprocessor = None
except Exception as e:
    print(f"An unexpected error occurred during model or preprocessor loading: {e}")
    model_xgb = None
    preprocessor = None


# Load the dataset to get unique cities and for normal price summary calculation
# Note: If your original model used a dataset without the merged fuel costs,
# you should ensure this df loading matches what was used for that training.
# For calculating normal price summary from original data, load your base dataset.
df = None # Initialize df to None
fuel_df = None # Initialize fuel_df to None # Fuel_df is likely not needed if reverting

try:
    # Load your main flight data (the one used for training the original model)
    df = pd.read_csv("your_output.csv") # Assuming this was your original dataset
    # Ensure the 'Price' column is numeric before calculating stats
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Price'], inplace=True) # Drop rows where price couldn't be converted

    available_sources = sorted(df['Source'].unique().tolist()) # Get unique source cities and sort them
    available_destinations = sorted(df['Destination'].unique().tolist()) # Get unique destination cities and sort them
    print("Dataset loaded and unique cities extracted.")

    # Fuel cost data is likely not needed if reverting the model

except FileNotFoundError:
    print("Error loading dataset. Make sure 'your_output.csv' is in the correct path.")
    available_sources = []
    available_destinations = []
except Exception as e:
    print(f"An unexpected error occurred during dataset loading: {e}")
    available_sources = []
    available_destinations = []


# Define the holidays dictionary
holidays = {
    (1, 1): "New Year's Day", (1, 6): "Guru Govind Singh Jayanti", (1, 14): "Pongal/Makar Sankranti/Hazarat Ali's Birthday",
    (1, 26): "Republic Day", (2, 2): "Vasant Panchami", (2, 12): "Guru Ravidas Jayanti", (2, 19): "Shivaji Jayanti",
    (2, 23): "Maharishi Dayanand Saraswati Jayanti", (2, 26): "Maha Shivaratri/Shivaratri", (3, 2): "Ramadan Start",
    (3, 13): "Holika Dahana", (3, 14): "Holi/Dolyatra", (3, 28): "Jamat Ul-Vida", (3, 30): "Chaitra Sukhladi/Ugadi/Gudi Padwa",
    (3, 31): "Ramzan Id", (4, 6): "Rama Navami", (4, 10): "Mahavir Jayanti", (4, 13): "Vaisakhi",
    (4, 14): "Mesadi/Ambedkar Jayanti", (4, 15): "Bahag Bihu/Vaisakhadi", (4, 18): "Good Friday", (4, 20): "Easter Day",
    (5, 9): "Birthday of Rabindranath", (5, 12): "Buddha Purnima/Vesak", (6, 7): "Bakrid", (6, 27): "Rath Yatra",
    (7, 6): "Muharram/Ashura", (8, 9): "Raksha Bandhan (Rakhi)", (8, 15): "Independence Day/Janmashtami (Smarta)/Parsi New Year",
    (8, 16): "Janmashtami", (8, 27): "Ganesh Chaturthi/Vinayaka Chaturthi", (9, 5): "Milad un-Nabi/Id-e-Milad/Onam",
    (9, 22): "First Day of Sharad Navratri", (9, 28): "First Day of Durga Puja Festivities", (9, 29): "Maha Saptami",
    (9, 30): "Maha Ashtami", (10, 1): "Maha Navami", (10, 2): "Mahatma Gandhi Jayanti/Dussehra",
    (10, 7): "Maharishi Valmiki Jayanti", (10, 10): "Karaka Chaturthi (Karva Chauth)", (10, 20): "Naraka Chaturdasi/Diwali/Deepavali",
    (10, 22): "Govardhan Puja", (10, 23): "Bhai Duj", (10, 28): "Chhat Puja (Pratihar Sashthi/Surya Sashthi)",
    (11, 5): "Guru Nanak Jayanti", (11, 24): "Guru Tegh Bahadur's Martyrdom Day", (12, 24): "Christmas Eve", (12, 25): "Christmas"
}

# Dictionary to map cities (source or destination) to image filenames
# Ensure you have images for ALL possible source and destination cities in your static/images folder
destination_images = {
    'New Delhi': 'new_delhi.jpg',
    'Banglore': 'banglore.jpg',
    'Cochin': 'cochin.jpeg',
    'Kolkata': 'kolkata.jpg',
    'Delhi': 'delhi.jpg',
    'Hyderabad': 'hyderabad.jpg'
    # Add other cities if needed, with a default image placeholder
}


@app.route('/')
def home():
    # Pass the available cities to the template
    return render_template(
        'index.html',
        available_sources=available_sources,
        available_destinations=available_destinations
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model, preprocessor, and dataset are loaded
    if model_xgb is None or preprocessor is None or df is None:
        return "Error: Required data, model or preprocessor not loaded. Ensure files are in the correct location.", 500

    # Get data from the form
    source = request.form['source']
    destination = request.form['destination']
    departure_datetime_str = request.form['departure_datetime']
    total_stops = int(request.form['total_stops'])

    # --- Get new inputs for price breakdown ---
    # Use .get() with a default value to avoid errors if the checkbox/radio buttons are not selected
    is_round_trip = request.form.get('is_round_trip') == 'yes' # Check if round trip is selected (assuming value='yes' for checked)
    luggage_option = request.form.get('luggage_option', 'none') # Get luggage option, default to 'none'
    seat_preference = request.form.get('seat_preference', 'economy') # Get seat preference, default to 'economy'

    # Get return date/time only if round trip is selected
    return_datetime_str = None
    return_datetime = None # Initialize return_datetime
    if is_round_trip:
        return_datetime_str = request.form.get('return_datetime') # Use .get() as it might not be present if checkbox was unchecked via inspection
        # Validate return date is after departure date if round trip is selected
        if return_datetime_str:
            try:
                dep_datetime_check = datetime.strptime(departure_datetime_str, "%Y-%m-%dT%H:%M")
                return_datetime = datetime.strptime(return_datetime_str, "%Y-%m-%dT%H:%M")
                if return_datetime <= dep_datetime_check:
                    return "Error: Return date and time must be after departure date and time.", 400
            except ValueError:
                 return "Error: Invalid return date or time format.", 400
            except Exception as e:
                 return f"An error occurred validating return date: {e}", 400
        else:
             # If round trip is checked but return_datetime is missing (shouldn't happen with 'required' in HTML but good practice)
             return "Error: Return date and time is required for round trip.", 400


    # Convert datetime strings to datetime objects
    try:
        dep_datetime = datetime.strptime(departure_datetime_str, "%Y-%m-%dT%H:%M")
        current_date = datetime.now() # Get current date

        # Arrival datetime is no longer a form input.
        # We need a dummy arrival datetime to calculate duration for the model,
        # or adjust the model features if duration was critical and cannot be inferred.
        # A simple approach is to assume a fixed duration based on average from data,
        # or calculate based on a dummy arrival time (e.g., 2 hours after departure).
        # Let's use a dummy arrival time for feature calculation based on departure
        dummy_duration_hours = 3 # Example average duration - ADJUST THIS BASED ON YOUR DATA'S AVERAGE
        dummy_arrival_datetime = dep_datetime + timedelta(hours=dummy_duration_hours) # Example dummy arrival time

        # Process return datetime if available (primarily for display or future use)
        # return_datetime is already parsed above if is_round_trip is True

    except ValueError:
        return "Error: Invalid date or time format. Please use the provided date and time picker.", 400
    except Exception as e:
         return f"An error occurred processing dates: {e}", 400


    # Extract features for the model (using dummy_arrival_datetime for duration if needed)
    dep_hours = dep_datetime.hour
    dep_min = dep_datetime.minute
    month = dep_datetime.month
    year = dep_datetime.year

    # Calculate duration using the dummy arrival datetime for feature extraction
    duration = dummy_arrival_datetime - dep_datetime
    duration_hours = int(duration.total_seconds() // 3600)
    duration_min = int((duration.total_seconds() % 3600) // 60)

    # Day of week for weekend check
    day_of_week = dep_datetime.weekday() # Monday is 0, Sunday is 6
    is_weekend = day_of_week >= 5 # Saturday (5) or Sunday (6)


    # Create a DataFrame for prediction based on features used by the ORIGINAL model
    # Adjust this list based on the features used by your original model_xgb.pkl
    # Ensure features derived from arrival time/duration match what the preprocessor expects
    input_data = pd.DataFrame({
        'Airline': ['IndiGo'], # Keep as placeholder or allow user selection
        'Source': [source],
        'Destination': [destination],
        'Dep_hours': [dep_hours],
        'Dep_min': [dep_min],
        # Use features derived from the dummy arrival time if your model needs them
        'Arrival_hours': [dummy_arrival_datetime.hour],
        'Arrival_min': [dummy_arrival_datetime.minute],
        'Duration_hours': [duration_hours],
        'Duration_min': [duration_min],
        'Total_Stops': [total_stops],
        'Month': [month],
        'Year': [year],
        # Include features used by your original model
        'Day_of_Week': [day_of_week] # Keep or remove based on original model features
    })

    # Define the feature columns that were used during training for the ORIGINAL model
    # THIS LIST IS CRUCIAL and must match the features/order used by model_xgb.pkl
    # Assuming original features include Arrival time components, Duration components, and Day_of_Week:
    feature_cols = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Month', 'Year',
                    'Dep_hours', 'Dep_min', 'Arrival_hours', 'Arrival_min',
                    'Duration_hours', 'Duration_min', 'Day_of_Week']

    # Ensure the input data has the same columns as the training data in the correct order
    try:
        input_data = input_data[feature_cols]
    except KeyError as e:
        return f"Error: Missing feature in input data for the original model. Please check the form inputs and defined feature columns. Missing: {e}", 400
    except Exception as e:
        return f"An error occurred preparing input features: {e}", 400


    # Preprocess the input data using the ORIGINAL preprocessor
    try:
        input_data_processed = preprocessor.transform(input_data)
    except ValueError as e:
        # If the error is due to unknown categories, provide a more specific message
        if "unknown categories" in str(e):
             return f"Error: Invalid city or airline entered (based on original model features). Please select from the available options.", 400
        else:
            return f"Error processing input data: {e}.", 400
    except Exception as e:
        return f"An error occurred during preprocessing: {e}", 400


    # Make prediction - This will be the base predicted price from the ORIGINAL model
    base_predicted_price_transformed = model_xgb.predict(input_data_processed)[0]
    # Use the correct inverse transformation based on how 'Price' was transformed for the ORIGINAL model
    # Assuming np.log was used originally:
    base_predicted_price = np.exp(base_predicted_price_transformed)


    # --- Implement NEW price adjustments based on provided logics ---
    # Start with the base predicted price
    final_predicted_price = base_predicted_price
    adjustment_details = [] # List to store details of each adjustment


    # e> if current date and booking date differs by 1 to 6 months then educe actual price by 20%
    # Calculate difference in months (approximate) - Do this first to apply discount to base price
    # Ensure dep_datetime is in the future compared to current_date for this logic to make sense
    if dep_datetime > current_date:
         diff_months = (dep_datetime.year - current_date.year) * 12 + dep_datetime.month - current_date.month
         # Adjust diff_months calculation for precision if needed, e.g., using total days / 30.44

         if 1 <= diff_months <= 6:
             advance_booking_discount = final_predicted_price * 0.20 # 20% reduction
             final_predicted_price -= advance_booking_discount
             adjustment_details.append({'Description': 'Advance Booking Discount (1-6 Months)', 'Amount': f"- ₹ {advance_booking_discount:.2f}"})
    # else: # Handle cases where departure is in the past or very soon - no discount or maybe a hike?
         # Example: if diff_months is 0 (less than 1 month), maybe a last-minute hike?
         # last_minute_hike = final_predicted_price * 0.10 # 10% hike
         # final_predicted_price += last_minute_hike
         # adjustment_details.append({'Description': 'Last Minute Booking Hike (Estimate)', 'Amount': f"+ ₹ {last_minute_hike:.2f}"})


    # a> journey time... if time is night slightly reduce the price by 10%
    # Assuming "night" is between 8 PM (20:00) and 6 AM (06:00) departure time
    if dep_hours >= 20 or dep_hours < 6:
        night_discount = final_predicted_price * 0.10 # 10% reduction
        final_predicted_price -= night_discount
        adjustment_details.append({'Description': 'Night Journey Discount', 'Amount': f"- ₹ {night_discount:.2f}"})


    # c>if falls on weekend...hike the price by 20%
    if is_weekend:
        weekend_hike = final_predicted_price * 0.20 # 20% hike
        final_predicted_price += weekend_hike
        adjustment_details.append({'Description': 'Weekend Hike (Estimate)', 'Amount': f"+ ₹ {weekend_hike:.2f}"})


    # d> if seating choice is economy show actual price
    # if business class.then 20% hike
    # 40%hike for first class or premium
    # Note: 'economy' seating means no price change from base here
    seat_hike_amount = 0
    if seat_preference == 'business':
        seat_hike_amount = final_predicted_price * 0.20 # 20% hike for business (applied to current price)
        final_predicted_price += seat_hike_amount
        adjustment_details.append({'Description': 'Business Class Hike (Estimate)', 'Amount': f"+ ₹ {seat_hike_amount:.2f}"})
    elif seat_preference == 'first': # Assuming 'premium' is also handled by 'first' logic
        seat_hike_amount = final_predicted_price * 0.40 # 40% hike for first/premium (applied to current price)
        final_predicted_price += seat_hike_amount
        adjustment_details.append({'Description': 'First/Premium Class Hike (Estimate)', 'Amount': f"+ ₹ {seat_hike_amount:.2f}"})
    # If seat_preference is 'economy', seat_hike_amount remains 0, no adjustment needed here.


    # b> luggage if no luggage or checked luggage...keep the actual price
    # if eXtra luggage add 1500 more to the price
    # Note: 'none' or 'checked' luggage options mean no price change from base here
    # 'extra' luggage option adds a fixed cost
    luggage_add_cost = 0
    if luggage_option == 'extra':
        luggage_add_cost = 1500 # Add 1500 for extra luggage
        final_predicted_price += luggage_add_cost
        adjustment_details.append({'Description': 'Extra Luggage Cost', 'Amount': f"+ ₹ {luggage_add_cost:.2f}"})
    # If luggage_option is 'none' or 'checked', luggage_add_cost remains 0, no adjustment needed here.


    # Round Trip Adjustment
    # New logic: if the the trip is round trip..show the price = one way price * 1.8
    # This adjustment should likely apply to the *final* one-way price after all other adjustments
    if is_round_trip:
        # Calculate the round trip price based on the final calculated one-way price
        round_trip_price = final_predicted_price * 1.8
        round_trip_add_amount = round_trip_price - final_predicted_price # The amount added for the return leg
        # Update the final predicted price to the calculated round trip price
        final_predicted_price = round_trip_price
        adjustment_details.append({'Description': 'Round Trip (1.8x One-Way)', 'Amount': f"Total: ₹ {final_predicted_price:.2f}"}) # Show total round trip price here


    # Holiday Hike (Keep or remove this based on your preference, it might interact with weekend hike)
    # If you want both, consider how they should combine.
    # Currently, the weekend hike is applied earlier. Applying holiday after round trip might make sense.
    departure_month_day = (dep_datetime.month, dep_datetime.day)
    # holiday_message = None # Separate message if needed, otherwise adjustment_details is enough
    # holiday_hike_amount = 0 # Already initialized

    # Re-check if departure_month_day is a holiday AFTER all other adjustments
    if departure_month_day in holidays:
         holiday_name = holidays[departure_month_day]
         # Apply hike to the *final* price including all adjustments so far
         hike_percentage = 0.25 # Example: 25% hike. Adjust as needed.
         holiday_hike_amount = final_predicted_price * hike_percentage
         final_predicted_price += holiday_hike_amount
         # Decide if you want a separate holiday message or include it in adjustment_details
         # holiday_message = f"Attention! Your departure date ({dep_datetime.strftime('%Y-%m-%d')}) falls on {holiday_name}. Price may be higher than usual."
         if holiday_hike_amount > 0:
              adjustment_details.append({'Description': f'{holiday_name} Holiday Hike (Estimate)', 'Amount': f"+ ₹ {holiday_hike_amount:.2f}"})


    # --- Optional: Add a minimum threshold for the final price after all adjustments ---
    # This can prevent the final price from being unrealistically low in some cases
    # minimum_final_price = 4000 # Example minimum final price. Adjust as needed.
    # if final_predicted_price < minimum_final_price:
    #     final_predicted_price = minimum_final_price
    #     # Optionally add a note to adjustment_details


    # --- Call the Skyscanner scraping function (still placeholder as per user request) ---
    # Note: skyscanner_origin_code and skyscanner_destination_code are defined here,
    # but used *inside* the placeholder function. This is why the NameError occurred.
    # The fix was to remove the use of the outer variable name inside the function print statement.
    skyscanner_origin_code = source[:3].upper()
    skyscanner_destination_code = destination[:3].upper()
    skyscanner_date = dep_datetime.strftime('%Y-%m-%d')
    print(f"Attempting to call placeholder Skyscanner scraping for {skyscanner_origin_code} to {skyscanner_destination_code} on {skyscanner_date}")
    # This will still return dummy data unless you replace the function body
    scraped_flights_data = scrape_skyscanner_flights_selenium(skyscanner_origin_code, skyscanner_destination_code, skyscanner_date)


    # --- Calculate Best, Average, and Highest Prices from YOUR DATASET for the route ---
    # This uses the loaded 'df' which should match the data used for the original model training
    normal_best_price = 'N/A'
    normal_average_price = 'N/A'
    normal_highest_price = 'N/A'

    if df is not None: # Ensure the dataset was loaded
        # Filter the dataset for the specific source and destination
        route_data = df[(df['Source'] == source) & (df['Destination'] == destination)]

        if not route_data.empty:
            # Calculate statistics from the 'Price' column
            # Ensure the 'Price' column in route_data is numeric before calculating stats
            prices_for_route = pd.to_numeric(route_data['Price'], errors='coerce').dropna()

            if not prices_for_route.empty:
                 normal_best_price = f"₹ {prices_for_route.min():.2f}"
                 normal_average_price = f"₹ {prices_for_route.mean():.2f}"
                 normal_highest_price = f"₹ {prices_for_route.max():.2f}"
            else:
                 print(f"Warning: No numeric price data found for route {source} to {destination} in the dataset.")
        else:
             print(f"Warning: No data found for route {source} to {destination} in the dataset.")


    # --- Determine suggested airline based on the final predicted price ---
    # Adjust these thresholds based on the price range expected from your ORIGINAL model
    if final_predicted_price < 8000: # Adjust thresholds
        suggested_airline = "SpiceJet or IndiGo (Budget)"
    elif 8000 <= final_predicted_price < 20000: # Adjust thresholds
        suggested_airline = "IndiGo or Vistara (Mid-Range)"
    else: # Adjust thresholds
        suggested_airline = "Vistara or Air India (Premium)"


    # Determine the image filenames based on the source and destination cities
    # Use .get() with a default value for robustness
    departure_image_filename = destination_images.get(source, 'default_image.jpg') # Get image for source city
    destination_image_filename = destination_images.get(destination, 'default_image.jpg') # Get image for destination city


    # Render the results template
    return render_template(
        'results.html',
        base_predicted_price=f"{base_predicted_price:.2f}", # Pass base price
        final_predicted_price=f"{final_predicted_price:.2f}", # Pass final price
        adjustment_details=adjustment_details, # Pass the list of adjustment details
        # holiday_message is now part of adjustment_details if applicable
        suggested_airline=suggested_airline,
        departure_city_name=source, # Pass city names for text
        destination_city_name=destination, # Pass city names for text
        departure_image=departure_image_filename, # Pass departure image filename
        destination_image=destination_image_filename, # Pass destination image filename
        # Pass Best, Average, Highest prices from YOUR DATASET
        normal_best_price=normal_best_price,
        normal_average_price=normal_average_price,
        normal_highest_price=normal_highest_price,
        # Pass the scraped data to the template (will show dummy data unless scraping is implemented)
        scraped_flights=scraped_flights_data,
        is_round_trip=is_round_trip, # Pass this to results.html to show return date if applicable
        return_datetime=return_datetime # Pass return datetime if available (will be None for one-way)
    )


if __name__ == '__main__':
    app.run(debug=True)
