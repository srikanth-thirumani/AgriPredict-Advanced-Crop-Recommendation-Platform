from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
import joblib
import os
import re
import requests
import folium
import google.generativeai as genai
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def clean_string(value):
    """
    Clean and standardize string values
    """
    if isinstance(value, str):
        # Remove extra whitespace and convert to lowercase
        return re.sub(r'\s+', ' ', value).strip().lower()
    return value

def get_gemini_fertilizer_recommendation(crop, soil_type, location):
    """
    Get fertilizer recommendations using Gemini API
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        
        # Use the correct model name
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Prompt for fertilizer recommendation
        prompt = f"""
        Provide a detailed fertilizer recommendation for {crop} grown in {soil_type} soil type in {location}.
        Include:
        1. Recommended NPK fertilizer ratio
        2. Quantity of fertilizer per hectare
        3. Application method
        4. Timing of fertilizer application
        5. Additional soil health recommendations

        Ensure the response is practical, scientifically accurate, and formatted in clear, readable markdown.
        """

        # Generate the content
        response = model.generate_content(prompt)
        
        # Return the text of the response
        return response.text
    except Exception as e:
        # More detailed error handling
        return f"""
        ### Fertilizer Recommendation Unavailable

        Unable to fetch fertilizer recommendation due to an API error. 
        Possible reasons:
        - API key may be invalid
        - Service might be temporarily unavailable
        - Network connectivity issues

        General Fertilizer Recommendation Tips:
        1. Consult local agricultural experts
        2. Get a soil test from a local laboratory
        3. Consider crop-specific nutrient requirements
        4. Use balanced NPK fertilizers
        5. Follow recommended application rates
        """

def get_weather_data(location):
    """
    Fetch weather data using OpenWeatherMap API
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': os.environ.get('OPENWEATHER_API_KEY'),
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'latitude': data['coord']['lat'],
                'longitude': data['coord']['lon']
            }
        else:
            return None
    except Exception as e:
        return None

def create_location_map(latitude, longitude):
    """
    Create a Folium map for location visualization
    """
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.Marker(
        [latitude, longitude], 
        popup='Your Location'
    ).add_to(m)
    
    # Save map to HTML string
    map_html = BytesIO()
    m.save(map_html, close_file=False)
    return map_html.getvalue().decode()

def load_and_preprocess_recommendation_data():
    """
    Load and preprocess crop recommendation dataset
    """
    try:
        crop = pd.read_csv("dataset/Crop_recommendation.csv")
        crop.drop(crop[crop.label == 'muskmelon'].index, inplace=True)
        data = crop.copy().drop_duplicates()
        encod = LabelEncoder()
        data['Encoded_label'] = encod.fit_transform(data.label)
        
        classes = pd.DataFrame({'label': pd.unique(data.label), 'encoded': pd.unique(data.Encoded_label)})
        classes = classes.sort_values('encoded').set_index('label')
        return data, classes, encod
    except FileNotFoundError:
        return None, None, None

def train_crop_recommendation_model(data):
    """
    Train Random Forest Classifier for crop recommendation
    """
    x = data.iloc[:,:-2]
    y = data.Encoded_label
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    param = {
        'n_estimators': [10, 50, 100], 
        'criterion': ['gini', 'entropy'], 
        'max_depth': [5, 10, 15, None]
    }
    rand = RandomForestClassifier(random_state=42)
    grid_rand = GridSearchCV(rand, param, cv=5, n_jobs=-1, verbose=1)
    grid_rand.fit(x_train_scaled, y_train)
    
    pred_rand = grid_rand.predict(x_test_scaled)
    print('Classification Report:\n', classification_report(y_test, pred_rand))
    return grid_rand, x.columns, scaler

def load_and_preprocess_production_data():
    """
    Load and preprocess crop production dataset with improved error handling
    """
    try:
        # Read the CSV file
        crop_data = pd.read_csv("dataset/crop_production.csv")
        
        # Apply string cleaning to all object columns
        for col in crop_data.select_dtypes(include=['object']).columns:
            crop_data[col] = crop_data[col].apply(clean_string)
        
        # Clean and standardize crop names
        crop_data['Crop'] = crop_data['Crop'].replace({
            'moth': 'mothbeans', 
            'peas  (vegetable)': 'pigeonpeas', 
            'bean': 'kidneybeans',
            'moong(green gram)': 'mungbean', 
            'pome granet': 'pomegranate',
            'water melon': 'watermelon', 
            'cotton(lint)': 'cotton', 
            'gram': 'chickpea'
        })
        
        # Define valid crops
        valid_crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
            'banana', 'mango', 'grapes', 'watermelon', 'apple', 'orange', 
            'papaya', 'coconut', 'cotton', 'jute', 'coffee'
        ]
        
        # Filter and clean data
        crop_data = crop_data[crop_data['Crop'].isin(valid_crops)]
        
        # Remove columns that might cause issues
        columns_to_drop = [col for col in ['State_Name', 'District_Name'] if col in crop_data.columns]
        if columns_to_drop:
            crop_data = crop_data.drop(columns_to_drop, axis=1)
        
        # Convert numeric columns, handling potential string values
        numeric_columns = ['Crop_Year', 'Area', 'Production']
        for col in numeric_columns:
            # First clean any non-numeric characters
            crop_data[col] = crop_data[col].apply(lambda x: re.sub(r'[^\d.]', '', str(x)) if isinstance(x, str) else x)
            
            # Attempt to convert to numeric, replacing errors with NaN
            crop_data[col] = pd.to_numeric(crop_data[col], errors='coerce')
        
        # Additional check to remove rows with non-numeric values in critical columns
        crop_data = crop_data.dropna(subset=numeric_columns)
        
        # Ensure crop year is integer
        crop_data['Crop_Year'] = crop_data['Crop_Year'].astype(int)
        
        return crop_data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def train_crop_production_model(data):
    """
    Train Random Forest Regressor for crop production prediction
    """
    # Ensure data is not None
    if data is None:
        return None, None, None, None, None
    
    # Use median crop year as default
    default_crop_year = int(data['Crop_Year'].median())
    
    # Perform one-hot encoding
    dummy = pd.get_dummies(data, columns=['Crop'])
    
    # Verify data types before training
    numeric_columns = dummy.select_dtypes(include=[np.number]).columns
    dummy = dummy[numeric_columns]
    
    x = dummy.drop(["Production"], axis=1)
    y = dummy["Production"]
    
    # Scale the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Check if we have enough data
    if len(x) == 0 or len(y) == 0:
        return None, None, None, None, None
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=5)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    rf_predict = model.predict(x_test)
    
    # Calculate R-squared and other metrics
    r1 = r2_score(y_test, rf_predict)
    mse = mean_squared_error(y_test, rf_predict)
    rmse = np.sqrt(mse)
    
    print(f"R2 score: {r1}")
    print(f"Root Mean Squared Error: {rmse}")
    
    return model, x, x.columns.tolist(), scaler, default_crop_year

def preprocess_input_data(input_data, feature_columns, recommendation_scaler):
    """
    Preprocess input data for crop recommendation
    """
    # Convert season to numeric encoding
    season_mapping = {
        'kharif': 0, 
        'rabi': 1, 
        'zaid': 2, 
        'whole year': 2, 
        'summer': 2
    }
    # Clean and map the season
    season = clean_string(input_data[0])
    input_data[0] = season_mapping.get(season, input_data[0])
    
    # Create a dictionary with feature values
    features_dict = {
        'N': input_data[1],
        'P': input_data[2],
        'K': input_data[3],
        'temperature': 25,  # Default average temperature
        'humidity': 70,     # Default average humidity
        'ph': 6.5,          # Default average pH
        'rainfall': 1500    # Default average rainfall
    }
    
    # Get feature values in the original order
    features = [features_dict[col] for col in feature_columns]
    
    # Scale the features
    features_scaled = recommendation_scaler.transform([features])
    
    return features_scaled[0]

def predict_crop_and_production(recommendation_model, production_model, x_train, production_columns, classes, feature_columns, recommendation_scaler, production_scaler, default_crop_year, input_data):
    """
    Predict recommended crop and estimated production
    """
    # Preprocess input data for recommendation
    preprocessed_input = preprocess_input_data(input_data, feature_columns, recommendation_scaler)
    
    crop_probabilities = recommendation_model.predict_proba([preprocessed_input])
    crop_prob_df = pd.DataFrame(data=np.round(crop_probabilities.T*100, 2), index=classes.index, columns=['predicted_values'])
    recommended_crop = crop_prob_df.predicted_values.idxmax()
    
    # Prepare test row with one-hot encoded crop
    test_row = x_train.head(1).copy()
    test_row.iloc[0] = 0  # Reset all values to 0
    test_row['Crop_Year'] = default_crop_year
    test_row['Area'] = input_data[4]
    
    # Set the crop column for the recommended crop
    crop_column = f'Crop_{recommended_crop}'
    if crop_column in test_row.columns:
        test_row[crop_column] = 1
    
    # Scale the test row features
    test_row_scaled = production_scaler.transform(test_row)
    
    # Predict production
    production = production_model.predict(test_row_scaled)[0]
    yield_per_area = production / test_row['Area'].values[0]
    
    return recommended_crop, production, yield_per_area, crop_prob_df.sort_values('predicted_values', ascending=False)

def train_and_save_models():
    """
    Train and save models
    """
    # Training recommendation model
    recommendation_data, classes, encoder = load_and_preprocess_recommendation_data()
    if recommendation_data is not None:
        recommendation_model, feature_columns, recommendation_scaler = train_crop_recommendation_model(recommendation_data)
        
        # Save components as a dictionary
        recommendation_model_dict = {
            'model': recommendation_model,
            'classes': classes,
            'encoder': encoder,
            'scaler': recommendation_scaler,
            'feature_columns': feature_columns
        }
        
        joblib.dump(recommendation_model_dict, 'crop_recommendation_model.pkl')
        result = "Crop Recommendation Model Trained and Saved!"
    else:
        result = "Error: Crop recommendation dataset not found!"
        return jsonify({"status": "error", "message": result})
    
    # Training production model
    production_data = load_and_preprocess_production_data()
    if production_data is not None:
        production_model, x_train, production_columns, production_scaler, default_crop_year = train_crop_production_model(production_data)
        if production_model is not None:
            # Save all components together in a dictionary
            production_model_dict = {
                'model': production_model, 
                'x_train': x_train,
                'production_columns': production_columns,
                'scaler': production_scaler, 
                'default_crop_year': default_crop_year
            }
            joblib.dump(production_model_dict, 'crop_production_model.pkl')
            result += " Crop Production Model Trained and Saved!"
            return jsonify({"status": "success", "message": result})
        else:
            return jsonify({"status": "error", "message": "Error training production model!"})
    else:
        return jsonify({"status": "error", "message": "Error: Crop production dataset not found!"})

def load_saved_models():
    """
    Load pre-trained models and scalers with robust error handling
    """
    try:
        # Check if model files exist
        if not (os.path.exists('crop_recommendation_model.pkl') and 
                os.path.exists('crop_production_model.pkl')):
            return None

        # Load recommendation model components
        recommendation_model_dict = joblib.load('crop_recommendation_model.pkl')
        
        # Verify all required keys are present
        recommendation_keys = ['model', 'classes', 'encoder', 'scaler', 'feature_columns']
        if not all(key in recommendation_model_dict for key in recommendation_keys):
            return None

        # Load production model components
        production_model_dict = joblib.load('crop_production_model.pkl')
        
        # Verify all required keys are present
        production_keys = ['model', 'x_train', 'production_columns', 'scaler', 'default_crop_year']
        if not all(key in production_model_dict for key in production_keys):
            return None

        # Unpack components
        recommendation_model = recommendation_model_dict['model']
        classes = recommendation_model_dict['classes']
        encoder = recommendation_model_dict['encoder']
        recommendation_scaler = recommendation_model_dict['scaler']
        feature_columns = recommendation_model_dict['feature_columns']

        production_model = production_model_dict['model']
        x_train = production_model_dict['x_train']
        production_columns = production_model_dict['production_columns']
        production_scaler = production_model_dict['scaler']
        default_crop_year = production_model_dict['default_crop_year']
        
        return (recommendation_model, classes, encoder, recommendation_scaler, 
                feature_columns, production_model, x_train, production_columns, 
                production_scaler, default_crop_year)
    
    except Exception as e:
        return None

@app.route('/')
def index():
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Silt', 'Chalky', 'Peaty']
    return render_template('index.html', soil_types=soil_types)

@app.route('/train', methods=['GET'])
def train_models():
    return render_template('train.html')

@app.route('/train_models', methods=['POST'])
def train_models_api():
    return train_and_save_models()

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    data = request.form
    season = data.get('season')
    nitrogen = float(data.get('nitrogen'))
    phosphorus = float(data.get('phosphorus'))
    potassium = float(data.get('potassium'))
    area = float(data.get('area'))
    location = data.get('location')
    soil_type = data.get('soil_type')
    
    # Load models
    model_components = load_saved_models()
    
    if model_components is None:
        return jsonify({
            "status": "error", 
            "message": "Models not found or incomplete. Please train the models first."
        })
    
    # Unpack model components
    (recommendation_model, classes, encoder, recommendation_scaler, 
     feature_columns, production_model, x_train, production_columns, 
     production_scaler, default_crop_year) = model_components
    
    # Create input data array
    input_data = [season, nitrogen, phosphorus, potassium, area]
    
    try:
        # Get weather data
        weather_data = get_weather_data(location)
        
        if not weather_data:
            return jsonify({
                "status": "error", 
                "message": "Weather data could not be retrieved for the given location."
            })
        
        # Predict crop and production
        recommended_crop, production, yield_per_area, crop_prob_df = predict_crop_and_production(
            recommendation_model, production_model, x_train, production_columns, 
            classes, feature_columns, recommendation_scaler, 
            production_scaler, default_crop_year, input_data
        )
        
        # Get fertilizer recommendation
        fertilizer_recommendation = get_gemini_fertilizer_recommendation(
            recommended_crop, soil_type, location
        )
        
        # Create map
        map_html = create_location_map(
            weather_data['latitude'], 
            weather_data['longitude']
        )
        
        # Prepare crop probabilities for JSON response
        crop_probs = crop_prob_df.head(5).reset_index()
        crop_probs_list = []
        for _, row in crop_probs.iterrows():
            crop_probs_list.append({
                "crop": row['label'].capitalize(),
                "probability": float(row['predicted_values'])
            })
        
        # Build response
        response = {
            "status": "success",
            "recommended_crop": recommended_crop.capitalize(),
            "production": round(float(production), 2),
            "yield_per_hectare": round(float(yield_per_area), 2),
            "weather": {
                "temperature": weather_data['temperature'],
                "humidity": weather_data['humidity'],
                "description": weather_data['description'].capitalize(),
                "wind_speed": weather_data['wind_speed']
            },
            "crop_probabilities": crop_probs_list,
            "fertilizer_recommendation": fertilizer_recommendation,
            "map_html": map_html
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred during prediction: {str(e)}"
        })

if __name__ == "__main__":
    app.run(debug=True)