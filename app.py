from flask import Flask, render_template, request, url_for # Import url_for
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os # Import the os module

app = Flask(__name__)

# Load the trained model and preprocessor when the app starts
try:
    model_xgb = joblib.load('model_xgb.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    print("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    print("Error loading model or preprocessor. Make sure 'model_xgb.pkl' and 'preprocessor.pkl' are in the same directory.")
    model_xgb = None # Set to None if loading fails
    preprocessor = None

# Load the dataset to get unique cities (and potentially other data if needed later)
try:
    # --- Debugging Print Statements (Optional for deployment) ---
    # print(f"Attempting to load dataset from path: your_output.csv") # Print the path being used
    # print(f"Current working directory: {os.getcwd()}") # Print the current working directory

    # # List files in the current directory
    # print("Files in current directory:")
    # try:
    #     for item in os.listdir('.')[:-1]: # Avoid printing potentially large model files
    #         print(item)
    # except Exception as e:
    #     print(f"Error listing files: {e}")
    # # ---------------------------------

    df = pd.read_csv("your_output.csv") # Use the relative path for deployment
    available_sources = sorted(df['Source'].unique().tolist()) # Get unique source cities and sort them
    available_destinations = sorted(df['Destination'].unique().tolist()) # Get unique destination cities and sort them
    print("Dataset loaded and unique cities extracted.")
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
# Ensure you have images for ALL possible source and destination cities
destination_images = {
    'New Delhi': 'new_delhi.jpg', # Update with your actual filenames and ensure images exist
    'Banglore': 'banglore.jpg',
    'Cochin': 'cochin.jpg',
    'Kolkata': 'kolkata.jpg',
    'Delhi': 'delhi.jpg',
    'Hyderabad': 'hyderabad.jpg',
    # Add entries for any source cities that are not also destinations if needed
    # 'Mumbai': 'mumbai.jpg',
    # 'Chennai': 'chennai.jpg',
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
    if model_xgb is None or preprocessor is None:
        return "Error: Model or preprocessor not loaded.", 500 # Return an error if model not loaded

    # Get data from the form
    source = request.form['source']
    destination = request.form['destination']
    departure_datetime_str = request.form['departure_datetime']
    arrival_datetime_str = request.form['arrival_datetime']
    total_stops = int(request.form['total_stops'])

    # Convert datetime strings to datetime objects
    dep_datetime = datetime.strptime(departure_datetime_str, "%Y-%m-%dT%H:%M")
    arrival_datetime = datetime.strptime(arrival_datetime_str, "%Y-%m-%dT%H:%M")

    # Extract features
    dep_hours = dep_datetime.hour
    dep_min = dep_datetime.minute
    month = dep_datetime.month
    year = dep_datetime.year

    duration = arrival_datetime - dep_datetime
    duration_hours = int(duration.total_seconds() // 3600)
    duration_min = int((duration.total_seconds() % 3600) // 60)

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Airline': ['IndiGo'], # You might want to make this dynamic later if you predict for different airlines
        'Source': [source],
        'Destination': [destination],
        'Dep_hours': [dep_hours],
        'Dep_min': [dep_min],
        'Arrival_hours': [arrival_datetime.hour],
        'Arrival_min': [arrival_datetime.minute],
        'Duration_hours': [duration_hours],
        'Duration_min': [duration_min],
        'Total_Stops': [total_stops],
        'Month': [month],
        'Year': [year]
        # 'Date' column is not used by the model, so we don't include it here for prediction
    })

    # Preprocess the input data
    feature_cols = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Month', 'Year',
                    'Dep_hours', 'Dep_min', 'Arrival_hours', 'Arrival_min',
                    'Duration_hours', 'Duration_min']
    input_data = input_data[feature_cols]

    try:
        input_data_processed = preprocessor.transform(input_data)
    except ValueError as e:
        # If the error is due to unknown categories, provide a more specific message
        if "unknown categories" in str(e):
             return f"Error: Invalid city entered. Please select from the available cities listed on the homepage.", 400
        else:
            return f"Error processing input data: {e}.", 400


    # Make prediction
    predicted_price_transformed = model_xgb.predict(input_data_processed)[0]
    actual_predicted_price = np.exp(predicted_price_transformed) # Inverse transform


    # Apply holiday hike logic
    departure_month_day = (dep_datetime.month, dep_datetime.day)
    holiday_message = None
    final_predicted_price = actual_predicted_price

    if departure_month_day in holidays:
        holiday_name = holidays[departure_month_day]
        hike_percentage = random.uniform(0.20, 0.30)
        final_predicted_price = actual_predicted_price * (1 + hike_percentage)
        holiday_message = f"Attention! Your departure date ({dep_datetime.strftime('%Y-%m-%d')}) falls on {holiday_name}. Price may be higher than usual."


    # Determine suggested airline
    if final_predicted_price < 5000:
        suggested_airline = "SpiceJet"
    elif 5000 <= final_predicted_price < 10000:
        suggested_airline = "IndiGo"
    else:
        suggested_airline = "Vistara"

    # Determine the image filenames based on the source and destination cities
    # Use .get() with a default value for robustness
    departure_image_filename = destination_images.get(source, 'default_image.jpg') # Get image for source city
    destination_image_filename = destination_images.get(destination, 'default_image.jpg') # Get image for destination city


    # Render the results template
    return render_template(
        'results.html',
        predicted_price=f"{final_predicted_price:.2f}", # Format to 2 decimal places
        holiday_message=holiday_message,
        suggested_airline=suggested_airline, # Pass suggested_airline to the template
        departure_city_name=source, # Pass city names for text
        destination_city_name=destination, # Pass city names for text
        departure_image=departure_image_filename, # Pass departure image filename
        destination_image=destination_image_filename # Pass destination image filename
    )


if __name__ == '__main__':
    app.run(debug=True)
