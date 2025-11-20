from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

class PredictionSystem:
    def __init__(self):
        self.class_models = None
        self.reg_models = None
        self.scaler = None
        self.features = None
        self.df_final = None
        self.models_loaded = False
        
    def load_models(self):
        """Load trained models and data"""
        print("\n" + "="*70)
        print("LOADING MODELS AND DATA")
        print("="*70)
        
        try:
            if os.path.exists('prediction_artifacts.pkl'):
                print("Loading prediction_artifacts.pkl...")
                with open('prediction_artifacts.pkl', 'rb') as f:
                    artifacts = pickle.load(f)
                    
                self.class_models = artifacts['classification_models']
                self.reg_models = artifacts['regression_models']
                self.scaler = artifacts['scaler']
                self.features = artifacts['feature_cols']  # Note: feature_cols not features
                self.df_final = artifacts.get('df_final', None)
                
                # Get best models (Gradient Boosting usually best)
                self.best_class_model = self.class_models['Gradient Boosting']
                self.best_reg_model = self.reg_models['Gradient Boosting Regressor']
                
                print(f"  ✓ Classification model: Gradient Boosting")
                print(f"  ✓ Regression model: Gradient Boosting Regressor")
                print(f"  ✓ Features: {len(self.features)}")
                print(f"  ✓ Scaler loaded")
                
                self.models_loaded = True
                print("  ✓ Models loaded successfully!")
            else:
                print("  ⚠ prediction_artifacts.pkl not found")
        
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
        
        print("="*70)
        if self.models_loaded:
            print("✓ READY: Using ML models for predictions")
        else:
            print("⚠ FALLBACK MODE: Using historical averages")
        print("="*70 + "\n")
    
    def get_historical_average(self, borough, date):
        """Get historical average from df_final"""
        if self.df_final is None:
            return None
        
        try:
            borough_data = self.df_final[self.df_final['borough'].str.lower() == borough.lower()]
            
            if len(borough_data) == 0:
                return None
            
            target_month = date.month
            target_day = date.day
            
            similar_dates = borough_data[
                (borough_data['Date'].dt.month == target_month) &
                (borough_data['Date'].dt.day == target_day)
            ]
            
            if len(similar_dates) > 0 and 'Total_Hospitalization' in similar_dates.columns:
                return round(similar_dates['Total_Hospitalization'].mean(), 2)
            
            monthly_data = borough_data[borough_data['Date'].dt.month == target_month]
            if len(monthly_data) > 0 and 'Total_Hospitalization' in monthly_data.columns:
                return round(monthly_data['Total_Hospitalization'].mean(), 2)
                
        except Exception as e:
            print(f"Error getting historical average: {e}")
        
        return None
    
    def predict(self, borough, date):
        """Make predictions"""
        result = {
            'success': False,
            'borough': borough.title(),
            'date': date.strftime('%B %d, %Y'),
            'risk_level': 'Unknown',
            'risk_probability': 0.0,
            'expected_admissions': 0.0,
            'confidence': 'Low',
            'historical_average': None
        }
        
        result['historical_average'] = self.get_historical_average(borough, date)
        
        if not self.models_loaded:
            print("⚠ Using fallback predictions")
            if result['historical_average']:
                result['expected_admissions'] = result['historical_average']
                result['risk_level'] = 'Moderate' if result['historical_average'] > 50 else 'Low'
                result['risk_probability'] = min(result['historical_average'] / 100, 0.95)
                result['confidence'] = 'Medium (Historical Data)'
                result['success'] = True
            else:
                borough_baselines = {
                    'manhattan': 65, 'brooklyn': 75, 'queens': 70,
                    'bronx': 85, 'staten island': 45
                }
                result['expected_admissions'] = borough_baselines.get(borough.lower(), 60)
                result['risk_level'] = 'Moderate'
                result['risk_probability'] = 0.45
                result['confidence'] = 'Low (Baseline)'
                result['success'] = True
            return result
        
        try:
            print(f"🔮 Making ML prediction for {borough} on {date.strftime('%Y-%m-%d')}")
            
            # Create feature vector matching your model's expectations
            month = date.month
            day = date.day
            weekday = date.weekday()
            year = date.year
            
            # Seasonal defaults based on month
            if month in [12, 1, 2]:  # Winter
                temp_max, temp_min, humidity = 5, -2, 65
            elif month in [3, 4, 5]:  # Spring
                temp_max, temp_min, humidity = 18, 8, 60
            elif month in [6, 7, 8]:  # Summer
                temp_max, temp_min, humidity = 28, 20, 70
            else:  # Fall
                temp_max, temp_min, humidity = 15, 7, 65
            
            # Create feature array (must match your training features exactly)
            n_features = len(self.features)
            X = np.zeros((1, n_features))
            
            # Map common features (this is a simplified version)
            # You may need to adjust based on your actual feature engineering
            feature_dict = {
                'month': month,
                'day': day,
                'day_of_week': weekday,
                'is_weekend': 1 if weekday >= 5 else 0,
                'year': year,
                'quarter': (month - 1) // 3,
                'Temp_Max_C': temp_max,
                'Temp_Min_C': temp_min,
                'Humidity_Avg': humidity,
                'Temp_Range': temp_max - temp_min,
                'is_winter': 1 if month in [12, 1, 2] else 0,
                'is_summer': 1 if month in [6, 7, 8] else 0,
            }
            
            # Fill in features we have
            for i, feat in enumerate(self.features):
                if feat in feature_dict:
                    X[0, i] = feature_dict[feat]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            risk_pred = self.best_class_model.predict(X_scaled)[0]
            risk_proba = self.best_class_model.predict_proba(X_scaled)[0]
            admissions_pred = self.best_reg_model.predict(X_scaled)[0]
            
            # Format results
            result['risk_level'] = 'High' if risk_pred == 1 else 'Low'
            result['risk_probability'] = float(risk_proba[1])
            result['expected_admissions'] = float(max(0, admissions_pred))
            result['confidence'] = 'High (ML Model)'
            result['success'] = True
            
            print(f"Prediction: {result['risk_level']} risk, {result['expected_admissions']:.0f} admissions")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to historical
            if result['historical_average']:
                result['expected_admissions'] = result['historical_average']
                result['risk_probability'] = min(result['historical_average'] / 100, 0.95)
                result['risk_level'] = 'High' if result['expected_admissions'] > 75 else 'Moderate'
                result['confidence'] = 'Medium (Historical)'
                result['success'] = True
        
        return result

prediction_system = PredictionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        borough = data.get('borough', '').strip()
        date_str = data.get('date', '')
        
        if not borough or not date_str:
            return jsonify({'success': False, 'error': 'Please provide both borough and date'})
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date format'})
        
        result = prediction_system.predict(borough, date)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def stats():
    try:
        stats_data = {
            'total_boroughs': 5,
            'boroughs': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
            'date_range': 'January 2017 - December 2024',
            'models_loaded': prediction_system.models_loaded
        }
        
        if prediction_system.df_final is not None:
            stats_data['total_records'] = len(prediction_system.df_final)
            if 'Total_Hospitalization' in prediction_system.df_final.columns:
                stats_data['avg_daily_admissions'] = float(
                    prediction_system.df_final['Total_Hospitalization'].mean()
                )
        
        return jsonify(stats_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    prediction_system.load_models()
    
    print("\n" + "="*70)
    print("NYC HEALTH PREDICTION WEB APP")
    print("="*70)
    print("\nOpen: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)