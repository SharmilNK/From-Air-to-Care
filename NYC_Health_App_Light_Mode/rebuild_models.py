"""
Rebuild pickle file with current scikit-learn version
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print("Loading your data...")
df = pd.read_csv('merged_data_before_features.csv')
df['Date'] = pd.to_datetime(df['Date'])  # FIXED: was read_datetime

# Quick and dirty: train simple models
print("Training quick models...")

# Prepare simple features
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['weekday'] = df['Date'].dt.dayofweek

# Select numeric features
feature_cols = ['month', 'day', 'weekday']
if 'Temp_Max_C' in df.columns:
    feature_cols.append('Temp_Max_C')
if 'Humidity_Avg' in df.columns:
    feature_cols.append('Humidity_Avg')

X = df[feature_cols].fillna(0)
y_class = (df['Total_Hospitalization'] > df['Total_Hospitalization'].quantile(0.75)).astype(int)
y_reg = df['Total_Hospitalization']

# Remove NaN rows
mask = ~(y_reg.isna())
X = X[mask]
y_class = y_class[mask]
y_reg = y_reg[mask]

# Train models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

print("Training classification model...")
class_model.fit(X_scaled, y_class)

print("Training regression model...")
reg_model.fit(X_scaled, y_reg)

# Save new pickle
artifacts = {
    'classification_models': {'Gradient Boosting': class_model},
    'regression_models': {'Gradient Boosting Regressor': reg_model},
    'scaler': scaler,
    'feature_cols': feature_cols,
    'df_final': df
}

with open('prediction_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("New pickle saved! Now run app.py")