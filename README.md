# 🌱 Advanced Crop Recommendation System

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## Overview

This Advanced Crop Recommendation System is an AI-powered web application that helps farmers make informed decisions about crop selection based on soil composition, location, and weather conditions. The system combines machine learning models with real-time data to provide personalized recommendations for optimal crop selection, production estimates, and fertilizer application guidance.

## 🌟 Key Features

- **Intelligent Crop Recommendation**: Suggests the most suitable crops based on soil nutrients (N-P-K), season, and environmental factors
- **Production Forecasting**: Estimates potential crop yield and production based on area and recommended crop
- **Real-Time Weather Integration**: Fetches current weather data for the specified location
- **Fertilizer Recommendations**: Provides detailed fertilizer guidance using Google's Gemini AI
- **Interactive Location Mapping**: Visualizes farm location with integrated Folium maps
- **Model Training Interface**: Allows retraining of models with new data

## 📋 Tech Stack

- **Backend**: Flask
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (RandomForest algorithms)
- **Visualization**: Folium
- **AI Integration**: Google Gemini API
- **External Services**: OpenWeatherMap API

## 🛠️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/srikanth-thirumani/AgriPredict-Advanced-Crop-Recommendation-Platform.git
   cd crop-recommendation-system
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   ```bash
   # Linux/MacOS
   export GEMINI_API_KEY="your_gemini_api_key"
   export OPENWEATHER_API_KEY="your_openweather_api_key"
   
   # Windows
   set GEMINI_API_KEY=your_gemini_api_key
   set OPENWEATHER_API_KEY=your_openweather_api_key
   ```

5. Ensure your dataset folder structure is correct
   ```
   dataset/
   ├── Crop_recommendation.csv
   └── crop_production.csv
   ```

## 🚀 Usage

1. Start the Flask application
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`

3. If you're running the application for the first time, navigate to the training page at `http://127.0.0.1:5000/train` and click "Train Models"

4. On the main page, fill in the required information:
   - Season (Kharif, Rabi, Zaid, etc.)
   - Soil NPK values
   - Cultivable area
   - Location
   - Soil type

5. Submit the form to receive customized crop recommendations, production estimates, weather data, and fertilizer advice

## 📊 Models and Data

### Crop Recommendation Model
- **Algorithm**: Random Forest Classifier with GridSearchCV optimization
- **Features**: Soil nutrients (N, P, K), temperature, humidity, pH, rainfall
- **Target**: Crop type
- **Metrics**: Classification report including precision, recall, and F1-score

### Crop Production Model
- **Algorithm**: Random Forest Regressor
- **Features**: Crop type (one-hot encoded), area, crop year
- **Target**: Production quantity
- **Metrics**: R² score, RMSE

## 📝 API Endpoints

- **/** - Home page with input form
- **/train** - Page to train or retrain models
- **/train_models** (POST) - API endpoint for model training
- **/predict** (POST) - API endpoint for crop prediction and recommendations

## 🔄 Data Flow

1. User inputs soil and location data
2. System retrieves real-time weather data
3. ML models predict optimal crop and expected production
4. Gemini AI generates fertilizer recommendations
5. System creates an interactive location map
6. All information is presented in a unified interface

## ⚙️ Configuration

The system requires two API keys:
- **Google Gemini API Key**: For generating fertilizer recommendations
- **OpenWeatherMap API Key**: For fetching real-time weather data

These should be set as environment variables before running the application.

## 📁 Project Structure

```
crop-recommendation-system/
├── app.py                    # Main Flask application
├── dataset/                  # Data for model training
│   ├── Crop_recommendation.csv
│   └── crop_production.csv
├── static/                   # CSS, JavaScript, and images
├── templates/                # HTML templates
│   ├── index.html            # Main interface
│   └── train.html            # Model training page
├── crop_recommendation_model.pkl    # Saved recommendation model
├── crop_production_model.pkl        # Saved production model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 🧪 Features in Development

- Integration with soil sensor APIs
- Mobile application for field use
- Crop disease prediction using image recognition
- Multi-language support
- Historical data analysis and trends

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Project Link: [https://github.com/srikanth-thirumani/crop-recommendation-system](https://github.com/srikanth-thirumani/crop-recommendation-system)
