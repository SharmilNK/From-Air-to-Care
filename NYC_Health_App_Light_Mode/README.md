# NYC Health Prediction System

A machine learning web application that predicts respiratory and asthma hospitalizations in New York City based on weather patterns, air quality data, and historical health records.

## Overview

This project combines multiple data sources (weather, air quality, and hospitalization records from 2017-2024) to forecast hospital admission rates across NYC's five boroughs. The system uses gradient boosting models to provide both risk classifications and admission count predictions.

## Features

- **Dual Prediction Models**: Classification for risk level (High/Moderate/Low) and regression for expected admission counts
- **Interactive Web Interface**: Simple two-input form (date + borough) with clear visual results
- **Historical Context**: Compares predictions against historical averages for the same date
- **Real-Time Predictions**: Fast inference with confidence scoring
- **Responsive Design**: Clean, professional UI that works across all devices

## Tech Stack

- **Backend**: Flask (Python 3.12)
- **ML Models**: Scikit-learn (Gradient Boosting Classifier & Regressor)
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Data Sources**: NYC OpenData, NOAA Weather Data, EPA Air Quality Data

## Project Structure
```
NYC_Health_App_Light_Mode/
├── app.py                              # Flask application
├── rebuild_models.py                   # Script to retrain models
├── merged_data_before_features.csv     # Processed dataset
├── prediction_artifacts.pkl            # Trained ML models
├── templates/
│   └── index.html                      # Web interface
├── static/
│   ├── css/
│   │   └── style.css                   # Styling
│   └── js/
│       └── script.js                   # Frontend logic
└── requirements.txt                    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/SharmilNK/From-Air-to-Care.git
   cd From-Air-to-Care/NYC_HEALTH_APP_LIGHT_MODE
```

2. **Create a virtual environment** (recommended)
```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Verify data files**
   
   Ensure these files are in your project directory:
   - `merged_data_before_features.csv`
   - `prediction_artifacts.pkl`
   
   If `prediction_artifacts.pkl` is missing or incompatible, rebuild it:
```bash
   python rebuild_models.py
```

## Running the Application

1. **Start the Flask server**
```bash
   python app.py
```

2. **Open your browser**
   
   Navigate to `http://localhost:5000`

3. **Make predictions**
   
   - Select a date from the date picker
   - Choose a borough from the dropdown
   - Click "Predict" to see results

## Usage

### Web Interface

The application provides four key metrics:

1. **Risk Level**: Classification of hospitalization risk (High/Moderate/Low)
2. **Expected Admissions**: Predicted count of respiratory and asthma cases
3. **Historical Average**: Baseline from similar dates in past years
4. **Confidence**: Reliability indicator of the prediction

### API Endpoint

You can also make predictions programmatically:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-01-15", "borough": "brooklyn"}'
```

Response:
```json
{
  "success": true,
  "borough": "Brooklyn",
  "date": "January 15, 2024",
  "risk_level": "Moderate",
  "risk_probability": 0.65,
  "expected_admissions": 75.3,
  "historical_average": 71.2,
  "confidence": "High (ML Model)"
}
```

## Model Information

### Training Data

- **Time Period**: 2017-2024 (excluding 2020-2022 due to COVID-19 anomalies)
- **Records**: ~12,000 daily borough-level observations
- **Features**: Weather metrics (temperature, humidity, precipitation), air quality indicators (PM2.5, ozone, NO2), temporal features (day, month, season)

### Model Performance

- **Classification AUROC**: ~0.85
- **Regression R²**: ~0.75
- **Mean Absolute Error**: ±15-20 admissions

### Feature Engineering

The models use engineered features including:
- Temporal patterns (day of week, month, season)
- Weather interactions (temperature-humidity combinations)
- Air quality indices
- Historical trend indicators