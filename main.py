"""
NYC Health Prediction Pipeline with Interactive CLI
-------------------------------------------------------
End-to-end modelling pipeline:
  • Merges weather, air quality, respiratory & asthma data
  • Predicts high hospitalization risk using multiple models
  • Evaluates with Accuracy, AUROC, Recall metrics
  • Interactive CLI for predictions
-------------------------------------------------------
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, 
                             precision_score, f1_score, confusion_matrix, 
                             roc_curve, classification_report)

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ============================================================================

def load_all_data():
    """Load all CSV files."""
    print("="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    print("\nLoading weather data...")
    df_weather = pd.read_csv("nyc_weather_by_borough_2017-2024.csv", index_col=1, parse_dates=True)
    
    print("Loading respiratory data...")
    df_resp = pd.read_csv("Respiratory.csv", index_col=6, parse_dates=True)
    
    print("Loading asthma data...")
    df_asthma = pd.read_csv("Asthama.csv", index_col=6, parse_dates=True)
    
    print("Loading air quality data...")
    df_airq = pd.read_csv("Air_Quality.csv", index_col=6, parse_dates=True)
    
    print("\n✓ All data loaded successfully!")
    return df_weather, df_resp, df_asthma, df_airq


# ============================================================================
# STEP 2: CLEAN AND PREPARE EACH DATASET
# ============================================================================

def reset_date_index(df, date_col_name='Date'):
    """Reset index and ensure Date column exists."""
    if isinstance(df.index, pd.DatetimeIndex) and date_col_name not in df.columns:
        df = df.reset_index()
        df.columns = [date_col_name if 'DATE' in col.upper() or i == 0 
                      else col for i, col in enumerate(df.columns)]
    
    if date_col_name in df.columns:
        df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    
    return df


def prepare_weather_data(df_weather):
    """Clean and prepare weather data."""
    print("\n" + "="*70)
    print("STEP 2A: PREPARING WEATHER DATA")
    print("="*70)
    
    df = reset_date_index(df_weather, 'Date')
    
    # Rename columns for clarity
    weather_cols = {
        'TMAX': 'Temp_Max_C',
        'TMIN': 'Temp_Min_C',
        'PRCP': 'Precip_mm',
        'AWND': 'WindSpeed_mps',
        'RHAV': 'Humidity_Avg',
        'RHMX': 'Humidity_Max',
        'RHMN': 'Humidity_Min',
        'SNWD': 'Snow_Depth',
        'SNOW': 'Snowfall',
        'WDF2': 'FastWind2m_deg',
        'WDF5': 'FastWind5s_deg',
        'ADPT': 'DewPoint_Temp',
        'ASLP': 'AirPres',
        'ASTP': 'AirPres_Station',
        'AWBT': 'AirTemp_WetBulb',
        'AWDR': 'Avg_Wind_Dir'
    }
    
    df = df.rename(columns=weather_cols)
    
    # Keep relevant columns
    keep_cols = ['Date', 'borough', 'year'] + list(weather_cols.values())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # Standardize borough names
    df['borough'] = df['borough'].astype(str).str.strip().str.lower()
    
    # Aggregate by Date x Borough to avoid duplicates
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_dict = {col: 'mean' for col in numeric_cols}
    df = df.groupby(['Date', 'borough'], as_index=False).agg(agg_dict)
    
    print(f"Weather data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Boroughs: {df['borough'].unique()}")
    
    return df


def prepare_respiratory_asthma_data(df_resp, df_asthma):
    """Aggregate respiratory and asthma ER visits."""
    print("\n" + "="*70)
    print("STEP 2B: PREPARING RESPIRATORY & ASTHMA DATA")
    print("="*70)
    
    # Reset indices
    df_resp = reset_date_index(df_resp, 'Date')
    df_asthma = reset_date_index(df_asthma, 'Date')
    
    # Rename borough column
    df_resp = df_resp.rename(columns={'Dim1Value': 'borough'})
    df_asthma = df_asthma.rename(columns={'Dim1Value': 'borough'})
    
    # Standardize borough names
    df_resp['borough'] = df_resp['borough'].astype(str).str.strip().str.lower()
    df_asthma['borough'] = df_asthma['borough'].astype(str).str.strip().str.lower()
    
    # Convert Count to numeric (handles mixed types)
    df_resp['Count'] = pd.to_numeric(df_resp['Count'], errors='coerce')
    df_asthma['Count'] = pd.to_numeric(df_asthma['Count'], errors='coerce')
    
    # Aggregate by Date x Borough
    resp_agg = (df_resp.groupby(['Date', 'borough'], as_index=False)['Count']
                .sum().rename(columns={'Count': 'Respiratory_Count'}))
    
    asth_agg = (df_asthma.groupby(['Date', 'borough'], as_index=False)['Count']
                .sum().rename(columns={'Count': 'Asthma_Count'}))
    
    # Merge both
    df_health = pd.merge(resp_agg, asth_agg, on=['Date', 'borough'], how='outer')
    df_health = df_health.fillna(0)
    
    # Ensure numeric types
    df_health['Respiratory_Count'] = pd.to_numeric(df_health['Respiratory_Count'], errors='coerce').fillna(0)
    df_health['Asthma_Count'] = pd.to_numeric(df_health['Asthma_Count'], errors='coerce').fillna(0)
    
    # Create total hospitalization column
    df_health['Total_Hospitalization'] = (df_health['Respiratory_Count'] + 
                                          df_health['Asthma_Count'])
    
    # Filter to relevant years
    df_health['year'] = df_health['Date'].dt.year
    df_health = df_health[df_health['year'].isin([2017, 2018, 2019, 2023, 2024])].copy()
    
    print(f"Health data shape: {df_health.shape}")
    print(f"Date range: {df_health['Date'].min()} to {df_health['Date'].max()}")
    print(f"Total hospitalizations: {df_health['Total_Hospitalization'].sum():,.0f}")
    
    return df_health


def prepare_air_quality_data(df_airq):
    """Clean and prepare air quality data."""
    print("\n" + "="*70)
    print("STEP 2C: PREPARING AIR QUALITY DATA")
    print("="*70)
    
    df = reset_date_index(df_airq, 'Start_Date')
    df = df.rename(columns={'Start_Date': 'Date'})
    
    # Rename borough column
    if 'Geo Place Name' in df.columns:
        df = df.rename(columns={'Geo Place Name': 'borough'})
    
    # Standardize borough names
    df['borough'] = df['borough'].astype(str).str.strip().str.lower()
    
    # Pivot to get different pollutants as columns
    # Extract pollutant name and measure
    if 'Name' in df.columns and 'Data Value' in df.columns:
        df_pivot = df.pivot_table(
            index=['Date', 'borough'],
            columns='Name',
            values='Data Value',
            aggfunc='mean'
        ).reset_index()
        
        # Clean column names
        df_pivot.columns.name = None
        pollutant_cols = [col for col in df_pivot.columns if col not in ['Date', 'borough']]
        
        # Rename pollutant columns for clarity
        rename_map = {}
        for col in pollutant_cols:
            clean_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
            rename_map[col] = f"AQ_{clean_name}"
        
        df_pivot = df_pivot.rename(columns=rename_map)
        
        print(f"Air quality data shape: {df_pivot.shape}")
        print(f"Pollutants tracked: {list(rename_map.values())}")
        
        return df_pivot
    
    return df


# ============================================================================
# STEP 3: MERGE ALL DATASETS
# ============================================================================

def merge_all_datasets(df_weather, df_health, df_airq):
    """Merge all datasets on Date and Borough."""
    print("\n" + "="*70)
    print("STEP 3: MERGING ALL DATASETS")
    print("="*70)
    
    # Start with health data (our target)
    df_merged = df_health.copy()
    
    print(f"\nStarting with health data: {df_merged.shape}")
    print(f"  Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
    print(f"  Boroughs: {sorted(df_merged['borough'].unique())}")
    
    # Merge with weather
    df_merged = pd.merge(df_merged, df_weather, on=['Date', 'borough'], how='left')
    print(f"\nAfter merging weather: {df_merged.shape}")
    
    # Check air quality data before merge
    print(f"\nAir quality data: {df_airq.shape}")
    if len(df_airq) > 0:
        print(f"  Date range: {df_airq['Date'].min()} to {df_airq['Date'].max()}")
        print(f"  Boroughs: {sorted(df_airq['borough'].unique())}")
        
        # Merge with air quality
        df_merged = pd.merge(df_merged, df_airq, on=['Date', 'borough'], how='left')
        print(f"\nAfter merging air quality: {df_merged.shape}")
    else:
        print("  ⚠️  Air quality data is empty, skipping merge")
    
    # Filter valid boroughs
    valid_boroughs = ['brooklyn', 'bronx', 'manhattan', 'staten island', 'queens']
    df_merged = df_merged[df_merged['borough'].isin(valid_boroughs)]
    
    print(f"\nAfter filtering to valid boroughs: {df_merged.shape}")
    
    return df_merged


# ============================================================================
# STEP 4: IMPUTE MISSING DATA
# ============================================================================

def impute_missing_data(df):
    """Impute missing values using forward/backward fill by borough."""
    print("\n" + "="*70)
    print("STEP 4: IMPUTING MISSING DATA")
    print("="*70)
    
    print("\nMissing values before imputation:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])
    
    df = df.sort_values(['borough', 'Date']).reset_index(drop=True)
    
    # Forward fill then backward fill by borough
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df.groupby('borough')[col].transform(lambda x: x.ffill().bfill())
    
    # If still missing, use overall mean (or 0 if all are NaN)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(mean_val, inplace=True)
    
    print("\nMissing values after imputation:")
    missing_after = df.isnull().sum()
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values!")
    
    # Drop columns that are still all NaN or all zeros (not useful)
    cols_to_drop = []
    for col in df.columns:
        if df[col].isnull().all() or (df[col] == 0).all():
            if col not in ['Date', 'borough', 'Total_Hospitalization', 'Respiratory_Count', 'Asthma_Count']:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"\nDropping {len(cols_to_drop)} columns with all NaN or all zeros:")
        print(cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    
    return df


# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """Create additional features for modeling."""
    print("\n" + "="*70)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*70)
    
    print(f"Input shape: {df.shape}")
    
    df = df.copy()
    
    # Date features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['season'] = df['month'].map({12:1, 1:1, 2:1,  # Winter
                                     3:2, 4:2, 5:2,   # Spring
                                     6:3, 7:3, 8:3,   # Summer
                                     9:4, 10:4, 11:4}) # Fall
    
    # One-hot encode borough BEFORE sorting
    if 'borough' in df.columns:
        df = pd.get_dummies(df, columns=['borough'], prefix='borough')
        print(f"After one-hot encoding borough: {df.shape}")
    
    # Sort by date for lag features
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Lag features
    lag_cols = ['Total_Hospitalization', 'Respiratory_Count', 'Asthma_Count',
                'Temp_Max_C', 'Humidity_Avg', 'WindSpeed_mps']
    
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    '''
    # Rolling averages (lagged by 3 days to avoid data leakage for 3-day ahead predictions)
    rolling_cols = ['Total_Hospitalization', 'Temp_Max_C', 'Humidity_Avg']
    for col in rolling_cols:
        if col in df.columns:
            # Calculate 7-day rolling average and shift by 3 days
            df[f'{col}_roll7'] = df[col].rolling(window=7, min_periods=1).mean().shift(3)
    '''
    # Temperature range
    if 'Temp_Max_C' in df.columns and 'Temp_Min_C' in df.columns:
        df['Temp_Range'] = df['Temp_Max_C'] - df['Temp_Min_C']
    
    # Humidity range
    if 'Humidity_Max' in df.columns and 'Humidity_Min' in df.columns:
        df['Humidity_Range'] = df['Humidity_Max'] - df['Humidity_Min']
    
    # Drop NaN in target only
    df = df.dropna(subset=['Total_Hospitalization'])
    
    print(f"\nAfter feature engineering: {df.shape}")
    
    return df


# ============================================================================
# STEP 6: CREATE TARGET VARIABLE
# ============================================================================

def create_target_variable(df, threshold_percentile=75):
    """Create binary target: high hospitalization risk."""
    print("\n" + "="*70)
    print("STEP 6: CREATING TARGET VARIABLE")
    print("="*70)
    
    threshold = df['Total_Hospitalization'].quantile(threshold_percentile / 100)
    df['High_Risk'] = (df['Total_Hospitalization'] >= threshold).astype(int)
    
    print(f"\nThreshold ({threshold_percentile}th percentile): {threshold:.2f}")
    print(f"\nClass distribution:")
    print(df['High_Risk'].value_counts())
    
    return df


# ============================================================================
# STEP 7: PREPARE DATA FOR MODELING
# ============================================================================

def prepare_for_modeling(df):
    """Split into train/test and scale features."""
    print("\n" + "="*70)
    print("STEP 7: PREPARING FOR MODELING")
    print("="*70)
    
    # Exclude non-feature columns
    exclude_cols = ['Date', 'Total_Hospitalization', 'Respiratory_Count', 
                    'Asthma_Count', 'High_Risk', 'month_period',
                    'AQ_Asthma_emergency_department_visits_due_to_PM25',
                    'AQ_Asthma_emergency_departments_visits_due_to_Ozone',
                    'AQ_Asthma_hospitalizations_due_to_Ozone',
                    'AQ_Cardiac_and_respiratory_deaths_due_to_Ozone',
                    'AQ_Cardiovascular_hospitalizations_due_to_PM25_age_40+',
                    'AQ_Deaths_due_to_PM25',
                    'AQ_Respiratory_hospitalizations_due_to_PM25_age_20+',
                    'Total_Hospitalization_lag1', 'Asthma_Count_lag1','Respiratory_Count_lag1',
                    'Temp_Max_C_lag1','Humidity_Avg_lag1','WindSpeed_mps_lag1']
    
    # Also exclude any 'year' columns
    exclude_cols.extend([col for col in df.columns if 'year' in col.lower()])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols 
                    and not col.startswith('Unnamed')]
    
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    
    X = df[feature_cols].copy()
    y_class = df['High_Risk'].copy()
    y_reg = df['Total_Hospitalization'].copy()
    
    # Check for columns with all NaN
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\nDropping {len(all_nan_cols)} columns with all NaN values")
        X = X.drop(columns=all_nan_cols)
        feature_cols = [col for col in feature_cols if col not in all_nan_cols]
    
    # Fill any remaining NaN with 0
    if X.isnull().any().any():
        print(f"\nFilling remaining NaN values with 0")
        X = X.fillna(0)
    
    dates = df.loc[X.index, 'Date'].copy()
    
    print(f"\nAfter cleaning:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    
    # Split by year
    years = dates.dt.year
    
    train_mask = years.isin([2017, 2018, 2019])
    val_mask = years == 2023
    test_mask = years == 2024
    
    print(f"\nSplit sizes:")
    print(f"  Train (2017-2019): {train_mask.sum()} samples")
    print(f"  Val (2023): {val_mask.sum()} samples")
    print(f"  Test (2024): {test_mask.sum()} samples")
    
    # Check if we have enough data
    if train_mask.sum() == 0:
        print("\n⚠️  WARNING: No training data found for 2017-2019!")
        print("Adjusting split strategy to use 70-15-15 split instead...")
        
        X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.15, random_state=42, stratify=y_class
        )
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
            X_temp, y_class_temp, y_reg_temp, test_size=0.176, random_state=42, stratify=y_class_temp
        )
    else:
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        y_class_train = y_class[train_mask]
        y_class_val = y_class[val_mask]
        y_class_test = y_class[test_mask]
        y_reg_train = y_reg[train_mask]
        y_reg_val = y_reg[val_mask]
        y_reg_test = y_reg[test_mask]
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_class_train, y_class_val, y_class_test,
            y_reg_train, y_reg_val, y_reg_test,
            feature_cols, scaler)


# ============================================================================
# STEP 8: TRAIN AND COMPARE MODELS
# ============================================================================

def train_classification_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train multiple classification models for high-risk prediction."""
    print("\n" + "="*70)
    print("STEP 8A: TRAINING CLASSIFICATION MODELS")
    print("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, 
                                                 min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        for set_name, X_set, y_set in [('Train', X_train, y_train),
                                        ('Val', X_val, y_val),
                                        ('Test', X_test, y_test)]:
            y_pred = model.predict(X_set)
            y_pred_proba = model.predict_proba(X_set)[:, 1]
            
            accuracy = accuracy_score(y_set, y_pred)
            auroc = roc_auc_score(y_set, y_pred_proba)
            recall = recall_score(y_set, y_pred)
            precision = precision_score(y_set, y_pred)
            f1 = f1_score(y_set, y_pred)
            
            results.append({
                'Model': name,
                'Dataset': set_name,
                'Accuracy': accuracy,
                'AUROC': auroc,
                'Recall': recall,
                'Precision': precision,
                'F1-Score': f1
            })
            
            print(f"  {set_name}: Acc={accuracy:.3f} | AUROC={auroc:.3f} | Recall={recall:.3f}")
    
    results_df = pd.DataFrame(results)
    
    return trained_models, results_df


def train_regression_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train regression models to predict actual hospitalization counts."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    
    print("\n" + "="*70)
    print("STEP 8B: TRAINING REGRESSION MODELS")
    print("="*70)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=10,
                                                         min_samples_split=5, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        for set_name, X_set, y_set in [('Train', X_train, y_train),
                                        ('Val', X_val, y_val),
                                        ('Test', X_test, y_test)]:
            y_pred = model.predict(X_set)
            
            if set_name == 'Test':
                predictions[name] = y_pred
            
            mse = mean_squared_error(y_set, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_set, y_pred)
            r2 = r2_score(y_set, y_pred)
            
            mask = y_set != 0
            if mask.sum() > 0:
                mape = mean_absolute_percentage_error(y_set[mask], y_pred[mask]) * 100
            else:
                mape = np.nan
            
            results.append({
                'Model': name,
                'Dataset': set_name,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            })
            
            print(f"  {set_name}: RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.3f}")
    
    results_df = pd.DataFrame(results)
    
    return trained_models, results_df, predictions


# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================

def visualize_results(class_results_df, reg_results_df, class_models, reg_predictions, 
                     X_test, y_class_test, y_reg_test):
    """Create comprehensive visualizations."""
    print("\n" + "="*70)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(24, 16))
    
    test_class = class_results_df[class_results_df['Dataset'] == 'Test'].copy()
    test_reg = reg_results_df[reg_results_df['Dataset'] == 'Test'].copy()
    
    # 1. Classification metrics
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['Accuracy', 'AUROC', 'Recall']
    x = np.arange(len(test_class))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, test_class[metric], width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Performance')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(test_class['Model'], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    ax2 = plt.subplot(3, 3, 2)
    for name, model in class_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_class_test, y_pred_proba)
        auroc = roc_auc_score(y_class_test, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'{name} ({auroc:.3f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(3, 3, 3)
    best_class_name = test_class.loc[test_class['AUROC'].idxmax(), 'Model']
    best_class_model = class_models[best_class_name]
    y_pred = best_class_model.predict(X_test)
    cm = confusion_matrix(y_class_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Normal', 'High Risk'],
                yticklabels=['Normal', 'High Risk'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix - {best_class_name}')
    
    # 4. Regression Performance
    ax4 = plt.subplot(3, 3, 4)
    x_reg = np.arange(len(test_reg))
    width_reg = 0.35
    
    ax4_twin = ax4.twinx()
    ax4.bar(x_reg, test_reg['RMSE'], width_reg, label='RMSE', alpha=0.7, color='steelblue')
    ax4.bar(x_reg + width_reg, test_reg['MAE'], width_reg, label='MAE', alpha=0.7, color='coral')
    ax4_twin.plot(x_reg + width_reg/2, test_reg['R²'], 'go-', linewidth=2, markersize=8, label='R²')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Error (RMSE / MAE)')
    ax4_twin.set_ylabel('R² Score')
    ax4.set_title('Regression Performance')
    ax4.set_xticks(x_reg + width_reg/2)
    ax4.set_xticklabels(test_reg['Model'], rotation=45, ha='right', fontsize=8)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Predicted vs Actual
    ax5 = plt.subplot(3, 3, 5)
    best_reg_name = test_reg.loc[test_reg['R²'].idxmax(), 'Model']
    best_reg_pred = reg_predictions[best_reg_name]
    
    ax5.scatter(y_reg_test, best_reg_pred, alpha=0.5, s=20)
    ax5.plot([y_reg_test.min(), y_reg_test.max()], 
             [y_reg_test.min(), y_reg_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax5.set_xlabel('Actual Hospitalizations')
    ax5.set_ylabel('Predicted Hospitalizations')
    ax5.set_title(f'Predicted vs Actual - {best_reg_name}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Residual Plot
    ax6 = plt.subplot(3, 3, 6)
    residuals = y_reg_test - best_reg_pred
    ax6.scatter(best_reg_pred, residuals, alpha=0.5, s=20)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Predicted Hospitalizations')
    ax6.set_ylabel('Residuals')
    ax6.set_title(f'Residual Plot - {best_reg_name}')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hospitalization_prediction_full_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved")
    
    return fig


# ============================================================================
# STEP 10: SAVE MODELS AND ARTIFACTS
# ============================================================================

def save_models_and_artifacts(class_models, reg_models, scaler, feature_cols, df_final):
    """Save trained models and necessary artifacts for CLI prediction."""
    print("\n" + "="*70)
    print("STEP 10: SAVING MODELS AND ARTIFACTS")
    print("="*70)
    
    # Get best models
    artifacts = {
        'classification_models': class_models,
        'regression_models': reg_models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'df_final': df_final
    }
    
    with open('prediction_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("✓ Models and artifacts saved to 'prediction_artifacts.pkl'")
    
    # Save feature columns to CSV
    features_df = pd.DataFrame({
        'Feature_Name': feature_cols,
        'Feature_Index': range(len(feature_cols))
    })
    features_df.to_csv('selected_features.csv', index=False)
    print("✓ Feature columns saved to 'selected_features.csv'")


# ============================================================================
# STEP 11: REGIONAL ANALYSIS VISUALIZATIONS
# ============================================================================

def create_regional_analysis_plots(df_merged_original, df_final):
    """Create comprehensive regional analysis plots."""
    print("\n" + "="*70)
    print("STEP 11: CREATING REGIONAL ANALYSIS PLOTS")
    print("="*70)
    
    # Prepare data - need to get borough info back
    df_analysis = df_merged_original.copy()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # ========================================================================
    # 1. TOTAL CASES BY BOROUGH
    # ========================================================================
    ax1 = plt.subplot(3, 3, 1)
    borough_totals = df_analysis.groupby('borough')['Total_Hospitalization'].sum().sort_values(ascending=False)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(borough_totals)))
    
    borough_totals.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Total Hospitalizations by Borough', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Borough', fontsize=12)
    ax1.set_ylabel('Total Cases', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(borough_totals.values):
        ax1.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # 2. ASTHMA CASES BY BOROUGH
    # ========================================================================
    ax2 = plt.subplot(3, 3, 2)
    asthma_by_borough = df_analysis.groupby('borough')['Asthma_Count'].sum().sort_values(ascending=False)
    colors_asthma = plt.cm.Oranges(np.linspace(0.4, 0.9, len(asthma_by_borough)))
    
    asthma_by_borough.plot(kind='bar', ax=ax2, color=colors_asthma)
    ax2.set_title('Asthma Cases by Borough', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Borough', fontsize=12)
    ax2.set_ylabel('Total Asthma Cases', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(asthma_by_borough.values):
        ax2.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # 3. RESPIRATORY CASES BY BOROUGH
    # ========================================================================
    ax3 = plt.subplot(3, 3, 3)
    resp_by_borough = df_analysis.groupby('borough')['Respiratory_Count'].sum().sort_values(ascending=False)
    colors_resp = plt.cm.Blues(np.linspace(0.4, 0.9, len(resp_by_borough)))
    
    resp_by_borough.plot(kind='bar', ax=ax3, color=colors_resp)
    ax3.set_title('Respiratory Cases by Borough', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Borough', fontsize=12)
    ax3.set_ylabel('Total Respiratory Cases', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(resp_by_borough.values):
        ax3.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # 4. PM2.5 IMPACT BY BOROUGH
    # ========================================================================
    ax4 = plt.subplot(3, 3, 4)
    
    # Check if PM2.5 related columns exist
    pm25_cols = [col for col in df_analysis.columns if 'PM25' in col or 'PM2.5' in col or 'pm25' in col.lower()]
    
    if pm25_cols and 'Total_Hospitalization' in df_analysis.columns:
        # Create PM2.5 severity bins
        if len(pm25_cols) > 0:
            df_analysis['PM25_avg'] = df_analysis[pm25_cols].mean(axis=1)
            
            # Filter out zeros and create high PM2.5 days (top 25%)
            df_pm25 = df_analysis[df_analysis['PM25_avg'] > 0].copy()
            if len(df_pm25) > 0:
                pm25_threshold = df_pm25['PM25_avg'].quantile(0.75)
                df_pm25_high = df_pm25[df_pm25['PM25_avg'] >= pm25_threshold]
                
                pm25_impact = df_pm25_high.groupby('borough')['Total_Hospitalization'].sum().sort_values(ascending=False)
                colors_pm25 = plt.cm.Purples(np.linspace(0.4, 0.9, len(pm25_impact)))
                
                pm25_impact.plot(kind='bar', ax=ax4, color=colors_pm25)
                ax4.set_title('Cases During High PM2.5 Days by Borough', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Borough', fontsize=12)
                ax4.set_ylabel('Cases (High PM2.5 Days)', fontsize=12)
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3, axis='y')
                
                for i, v in enumerate(pm25_impact.values):
                    ax4.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No PM2.5 data available', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No PM2.5 columns found', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'PM2.5 data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Cases During High PM2.5 Days', fontsize=14, fontweight='bold')
    
    # ========================================================================
    # 5. OVERALL AIR QUALITY IMPACT BY BOROUGH
    # ========================================================================
    ax5 = plt.subplot(3, 3, 5)
    
    aq_cols = [col for col in df_analysis.columns if col.startswith('AQ_') and 'due_to' not in col]
    
    if aq_cols and len(aq_cols) > 0:
        # Create composite air quality score
        df_analysis['AQ_composite'] = df_analysis[aq_cols].mean(axis=1)
        
        # High pollution days (top 25%)
        df_aq = df_analysis[df_analysis['AQ_composite'] > 0].copy()
        if len(df_aq) > 0:
            aq_threshold = df_aq['AQ_composite'].quantile(0.75)
            df_aq_high = df_aq[df_aq['AQ_composite'] >= aq_threshold]
            
            aq_impact = df_aq_high.groupby('borough')['Total_Hospitalization'].sum().sort_values(ascending=False)
            colors_aq = plt.cm.Greens(np.linspace(0.4, 0.9, len(aq_impact)))
            
            aq_impact.plot(kind='bar', ax=ax5, color=colors_aq)
            ax5.set_title('Cases During High Pollution Days by Borough', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Borough', fontsize=12)
            ax5.set_ylabel('Cases (High Pollution Days)', fontsize=12)
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(aq_impact.values):
                ax5.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No air quality data available', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Air quality data not available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Cases During High Pollution Days', fontsize=14, fontweight='bold')
    
    # ========================================================================
    # 6. EXTREME WEATHER IMPACT BY BOROUGH
    # ========================================================================
    ax6 = plt.subplot(3, 3, 6)
    
    weather_cols = ['Temp_Max_C', 'Temp_Min_C', 'Humidity_Avg', 'WindSpeed_mps']
    available_weather = [col for col in weather_cols if col in df_analysis.columns]
    
    if available_weather:
        # Define extreme weather: high temp (>30°C) or high humidity (>80%) or low temp (<0°C)
        df_weather = df_analysis.copy()
        
        extreme_conditions = pd.Series(False, index=df_weather.index)
        
        if 'Temp_Max_C' in df_weather.columns:
            extreme_conditions |= (df_weather['Temp_Max_C'] > 30)
        if 'Temp_Min_C' in df_weather.columns:
            extreme_conditions |= (df_weather['Temp_Min_C'] < 0)
        if 'Humidity_Avg' in df_weather.columns:
            extreme_conditions |= (df_weather['Humidity_Avg'] > 80)
        
        df_extreme = df_weather[extreme_conditions]
        
        if len(df_extreme) > 0:
            weather_impact = df_extreme.groupby('borough')['Total_Hospitalization'].sum().sort_values(ascending=False)
            colors_weather = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(weather_impact)))
            
            weather_impact.plot(kind='bar', ax=ax6, color=colors_weather)
            ax6.set_title('Cases During Extreme Weather by Borough', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Borough', fontsize=12)
            ax6.set_ylabel('Cases (Extreme Weather Days)', fontsize=12)
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(weather_impact.values):
                ax6.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No extreme weather events found', ha='center', va='center', transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'Weather data not available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Cases During Extreme Weather', fontsize=14, fontweight='bold')
    
    # ========================================================================
    # 7. AVERAGE CASES PER DAY BY BOROUGH
    # ========================================================================
    ax7 = plt.subplot(3, 3, 7)
    borough_avg = df_analysis.groupby('borough')['Total_Hospitalization'].mean().sort_values(ascending=False)
    colors_avg = plt.cm.Spectral(np.linspace(0.2, 0.8, len(borough_avg)))
    
    borough_avg.plot(kind='bar', ax=ax7, color=colors_avg)
    ax7.set_title('Average Daily Hospitalizations by Borough', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Borough', fontsize=12)
    ax7.set_ylabel('Avg Cases per Day', fontsize=12)
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(borough_avg.values):
        ax7.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # 8. ASTHMA VS RESPIRATORY BREAKDOWN
    # ========================================================================
    ax8 = plt.subplot(3, 3, 8)
    
    borough_breakdown = df_analysis.groupby('borough')[['Asthma_Count', 'Respiratory_Count']].sum()
    # Calculate total for sorting
    borough_breakdown['Total'] = borough_breakdown['Asthma_Count'] + borough_breakdown['Respiratory_Count']
    borough_breakdown = borough_breakdown.sort_values('Total', ascending=False)
    
    x = np.arange(len(borough_breakdown))
    width = 0.35
    
    ax8.bar(x - width/2, borough_breakdown['Asthma_Count'], width, label='Asthma', color='coral', alpha=0.8)
    ax8.bar(x + width/2, borough_breakdown['Respiratory_Count'], width, label='Respiratory', color='skyblue', alpha=0.8)
    
    ax8.set_title('Asthma vs Respiratory Cases by Borough', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Borough', fontsize=12)
    ax8.set_ylabel('Total Cases', fontsize=12)
    ax8.set_xticks(x)
    ax8.set_xticklabels(borough_breakdown.index, rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 9. SUMMARY STATISTICS TABLE
    # ========================================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary statistics
    summary_stats = []
    
    most_affected_total = borough_totals.index[0]
    most_affected_asthma = asthma_by_borough.index[0]
    most_affected_resp = resp_by_borough.index[0]
    
    summary_text = f"""
    REGIONAL ANALYSIS SUMMARY
    {'='*45}
    
    MOST AFFECTED REGIONS:
    
    📊 Total Hospitalizations:
       🥇 {most_affected_total.title()}
       Cases: {int(borough_totals.iloc[0]):,}
    
    🫁 Asthma Cases:
       🥇 {most_affected_asthma.title()}
       Cases: {int(asthma_by_borough.iloc[0]):,}
    
    🌬️  Respiratory Cases:
       🥇 {most_affected_resp.title()}
       Cases: {int(resp_by_borough.iloc[0]):,}
    
    📈 OVERALL STATISTICS:
    Total Cases (All Boroughs): {int(borough_totals.sum()):,}
    Average per Borough: {int(borough_totals.mean()):,}
    Highest Daily Average: {borough_avg.iloc[0]:.1f}
       ({borough_avg.index[0].title()})
    
    {'='*45}
    Data covers: {df_analysis['Date'].min().strftime('%Y-%m-%d')} 
    to {df_analysis['Date'].max().strftime('%Y-%m-%d')}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('regional_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ Regional analysis visualization saved as 'regional_analysis_comprehensive.png'")
    
    # ========================================================================
    # CREATE SUMMARY CSV
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING REGIONAL SUMMARY CSV")
    print("="*70)
    
    summary_data = {
        'Borough': borough_totals.index,
        'Total_Cases': borough_totals.values,
        'Asthma_Cases': [asthma_by_borough.get(b, 0) for b in borough_totals.index],
        'Respiratory_Cases': [resp_by_borough.get(b, 0) for b in borough_totals.index],
        'Avg_Daily_Cases': [borough_avg.get(b, 0) for b in borough_totals.index]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('regional_summary.csv', index=False)
    print("✓ Regional summary saved as 'regional_summary.csv'")
    
    return fig


# ============================================================================
# INTERACTIVE CLI FOR PREDICTIONS
# ============================================================================

def predict_for_date_borough(date_str, borough, current_capacity=80):
    """
    Predict hospitalizations for a specific date and borough.
    
    Parameters:
    -----------
    date_str : str
        Date in format 'YYYY-MM-DD'
    borough : str
        Borough name (brooklyn, bronx, manhattan, staten island, queens)
    current_capacity : int
        Current bed capacity (default: 80)
    """
    
    print("\n" + "="*70)
    print("PREDICTION FOR SPECIFIC DATE AND BOROUGH")
    print("="*70)
    
    try:
        # Load artifacts
        with open('prediction_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        class_models = artifacts['classification_models']
        reg_models = artifacts['regression_models']
        scaler = artifacts['scaler']
        feature_cols = artifacts['feature_cols']
        df_final = artifacts['df_final']
        
        # Parse input date
        pred_date = pd.to_datetime(date_str)
        borough = borough.lower().strip()
        
        # Validate borough
        valid_boroughs = ['brooklyn', 'bronx', 'manhattan', 'staten island', 'queens']
        if borough not in valid_boroughs:
            print(f"\n❌ Error: Invalid borough '{borough}'")
            print(f"Valid boroughs: {', '.join(valid_boroughs)}")
            return
        
        print(f"\nInput Date: {pred_date.strftime('%Y-%m-%d (%A)')}")
        print(f"Borough: {borough.title()}")
        print(f"Current Capacity: {current_capacity} beds")
        
        # Find closest date in historical data
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        
        # Filter by borough (check one-hot encoded columns)
        borough_col = f'borough_{borough}'
        if borough_col not in df_final.columns:
            print(f"\n❌ Error: No data available for borough '{borough}'")
            return
        
        borough_data = df_final[df_final[borough_col] == 1].copy()
        
        if len(borough_data) == 0:
            print(f"\n❌ Error: No historical data found for {borough.title()}")
            return
        
        # Find the closest date
        borough_data['date_diff'] = abs((borough_data['Date'] - pred_date).dt.days)
        closest_idx = borough_data['date_diff'].idxmin()
        closest_row = df_final.loc[closest_idx]
        closest_date = closest_row['Date']
        
        print(f"\nUsing data from: {closest_date.strftime('%Y-%m-%d')} (closest available)")
        print(f"Date difference: {abs((closest_date - pred_date).days)} days")
        
        # Prepare features
        X_pred = closest_row[feature_cols].values.reshape(1, -1)
        X_pred = np.nan_to_num(X_pred, nan=0.0)  # Handle any NaN
        X_pred_scaled = scaler.transform(X_pred)
        
        # Get best models
        best_class_model = class_models['Random Forest']
        best_reg_model = reg_models['Random Forest Regressor']
        
        # Make predictions
        high_risk_prob = best_class_model.predict_proba(X_pred_scaled)[0, 1]
        predicted_count = best_reg_model.predict(X_pred_scaled)[0]
        predicted_count = max(0, predicted_count)  # Ensure non-negative
        
        # Calculate additional beds needed
        additional_beds = max(0, int(np.ceil(predicted_count - current_capacity)))
        utilization = (predicted_count / current_capacity) * 100
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        print(f"\n📊 HOSPITALIZATION FORECAST:")
        print(f"   Expected Admissions: {predicted_count:.1f} patients")
        print(f"   High-Risk Probability: {high_risk_prob*100:.1f}%")
        
        print(f"\n🏥 CAPACITY ANALYSIS:")
        print(f"   Current Capacity: {current_capacity} beds")
        print(f"   Expected Utilization: {utilization:.1f}%")
        
        if additional_beds > 0:
            print(f"   ⚠️  ADDITIONAL BEDS NEEDED: {additional_beds} beds")
            print(f"   Status: CAPACITY EXCEEDED")
        else:
            print(f"   ✓ Additional Beds Needed: 0 beds")
            print(f"   Status: WITHIN CAPACITY")
        
        print(f"\n📈 RISK LEVEL:")
        if high_risk_prob >= 0.7:
            print(f"   🔴 HIGH RISK (≥70%)")
        elif high_risk_prob >= 0.4:
            print(f"   🟡 MODERATE RISK (40-70%)")
        else:
            print(f"   🟢 LOW RISK (<40%)")
        
        # Historical context
        historical_avg = borough_data['Total_Hospitalization'].mean()
        print(f"\n📉 HISTORICAL CONTEXT:")
        print(f"   Average Admissions ({borough.title()}): {historical_avg:.1f}")
        print(f"   Predicted vs Average: {((predicted_count/historical_avg - 1) * 100):+.1f}%")
        
        print("\n" + "="*70)
        
        return {
            'date': pred_date,
            'borough': borough,
            'predicted_count': predicted_count,
            'high_risk_probability': high_risk_prob,
            'current_capacity': current_capacity,
            'additional_beds_needed': additional_beds,
            'utilization_percent': utilization
        }
        
    except FileNotFoundError:
        print("\n❌ Error: 'prediction_artifacts.pkl' not found!")
        print("Please run the main pipeline first to train models.")
        return None
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STEP 12: CALCULATE PRESENTATION STATISTICS
# ============================================================================

def calculate_presentation_stats(df_merged_for_analysis, class_results, reg_results):
    """Calculate all statistics needed for VC presentation."""
    print("\n" + "="*80)
    print("PRESENTATION STATISTICS - COPY THESE TO YOUR SLIDES")
    print("="*80)
    
    df = df_merged_for_analysis.copy()
    
    # ========================================================================
    # 1. HIGH PM2.5 IMPACT BY BOROUGH
    # ========================================================================
    print("\n" + "="*80)
    print("1. HIGH PM2.5 IMPACT BY BOROUGH")
    print("="*80)
    
    pm25_cols = [col for col in df.columns if 'PM25' in col or 'PM2.5' in col or 'pm25' in col.lower()]
    
    if pm25_cols and len(pm25_cols) > 0:
        # Create average PM2.5 column
        df['PM25_avg'] = df[pm25_cols].mean(axis=1)
        
        # Filter out zeros
        df_pm25 = df[df['PM25_avg'] > 0].copy()
        
        if len(df_pm25) > 0:
            # Define high PM2.5 threshold (top 25%)
            pm25_threshold = df_pm25['PM25_avg'].quantile(0.75)
            print(f"\nPM2.5 High Threshold (75th percentile): {pm25_threshold:.2f} μg/m³")
            
            # Calculate impact by borough
            print("\n📊 PM2.5 IMPACT BY BOROUGH:")
            print("-" * 80)
            
            for borough in ['bronx', 'brooklyn', 'manhattan', 'queens', 'staten island']:
                borough_data = df_pm25[df_pm25['borough'] == borough]
                
                if len(borough_data) > 0:
                    # High PM2.5 days
                    high_pm25_data = borough_data[borough_data['PM25_avg'] >= pm25_threshold]
                    normal_pm25_data = borough_data[borough_data['PM25_avg'] < pm25_threshold]
                    
                    if len(high_pm25_data) > 0 and len(normal_pm25_data) > 0:
                        high_avg = high_pm25_data['Total_Hospitalization'].mean()
                        normal_avg = normal_pm25_data['Total_Hospitalization'].mean()
                        
                        percent_increase = ((high_avg - normal_avg) / normal_avg) * 100
                        
                        print(f"{borough.title():15} | High PM2.5 days: +{percent_increase:5.1f}% admissions")
                        print(f"                | Avg on high PM2.5 days: {high_avg:.1f} patients")
                        print(f"                | Avg on normal days: {normal_avg:.1f} patients")
                        print(f"                | Absolute increase: +{high_avg - normal_avg:.1f} patients")
                        print("-" * 80)
        else:
            print("⚠️  No PM2.5 data available after filtering zeros")
    else:
        print("⚠️  No PM2.5 columns found in data")
    
    # ========================================================================
    # 2. EXTREME TEMPERATURE IMPACT
    # ========================================================================
    print("\n" + "="*80)
    print("2. EXTREME TEMPERATURE IMPACT")
    print("="*80)
    
    if 'Temp_Max_C' in df.columns and 'Temp_Min_C' in df.columns:
        # Define normal temperature range
        normal_temp_data = df[(df['Temp_Max_C'] >= 10) & (df['Temp_Max_C'] <= 25)]
        normal_temp_avg = normal_temp_data['Total_Hospitalization'].mean()
        
        # Extreme heat (>30°C)
        extreme_heat_data = df[df['Temp_Max_C'] > 30]
        if len(extreme_heat_data) > 0:
            extreme_heat_avg = extreme_heat_data['Total_Hospitalization'].mean()
            heat_increase = ((extreme_heat_avg - normal_temp_avg) / normal_temp_avg) * 100
            
            print(f"\n🌡️  EXTREME HEAT (>30°C):")
            print(f"   Days with extreme heat: {len(extreme_heat_data)}")
            print(f"   Avg admissions on extreme heat days: {extreme_heat_avg:.1f}")
            print(f"   Avg admissions on normal days: {normal_temp_avg:.1f}")
            print(f"   ➤ INCREASE: +{heat_increase:.1f}% admissions")
            
            # By condition type
            if 'Respiratory_Count' in df.columns:
                heat_resp_avg = extreme_heat_data['Respiratory_Count'].mean()
                normal_resp_avg = normal_temp_data['Respiratory_Count'].mean()
                resp_increase = ((heat_resp_avg - normal_resp_avg) / normal_resp_avg) * 100
                print(f"   ➤ Respiratory cases: +{resp_increase:.1f}%")
            
            if 'Asthma_Count' in df.columns:
                heat_asthma_avg = extreme_heat_data['Asthma_Count'].mean()
                normal_asthma_avg = normal_temp_data['Asthma_Count'].mean()
                asthma_increase = ((heat_asthma_avg - normal_asthma_avg) / normal_asthma_avg) * 100
                print(f"   ➤ Asthma cases: +{asthma_increase:.1f}%")
        else:
            print("\n⚠️  No extreme heat days (>30°C) found in data")
        
        # Extreme cold (<0°C)
        extreme_cold_data = df[df['Temp_Min_C'] < 0]
        if len(extreme_cold_data) > 0:
            extreme_cold_avg = extreme_cold_data['Total_Hospitalization'].mean()
            cold_increase = ((extreme_cold_avg - normal_temp_avg) / normal_temp_avg) * 100
            
            print(f"\n❄️  EXTREME COLD (<0°C):")
            print(f"   Days with extreme cold: {len(extreme_cold_data)}")
            print(f"   Avg admissions on extreme cold days: {extreme_cold_avg:.1f}")
            print(f"   Avg admissions on normal days: {normal_temp_avg:.1f}")
            print(f"   ➤ INCREASE: +{cold_increase:.1f}% admissions")
            
            # By condition type
            if 'Respiratory_Count' in df.columns:
                cold_resp_avg = extreme_cold_data['Respiratory_Count'].mean()
                resp_increase = ((cold_resp_avg - normal_resp_avg) / normal_resp_avg) * 100
                print(f"   ➤ Respiratory cases: +{resp_increase:.1f}%")
        else:
            print("\n⚠️  No extreme cold days (<0°C) found in data")
    else:
        print("⚠️  Temperature data not available")
    
    # ========================================================================
    # 3. HIGH HUMIDITY IMPACT
    # ========================================================================
    print("\n" + "="*80)
    print("3. HIGH HUMIDITY IMPACT")
    print("="*80)
    
    if 'Humidity_Avg' in df.columns:
        # Define normal humidity range
        normal_humidity_data = df[(df['Humidity_Avg'] >= 40) & (df['Humidity_Avg'] <= 70)]
        normal_humidity_avg = normal_humidity_data['Total_Hospitalization'].mean()
        
        if 'Asthma_Count' in df.columns:
            normal_asthma_avg = normal_humidity_data['Asthma_Count'].mean()
        
        # High humidity (>80%)
        high_humidity_data = df[df['Humidity_Avg'] > 80]
        if len(high_humidity_data) > 0:
            high_humidity_avg = high_humidity_data['Total_Hospitalization'].mean()
            humidity_increase = ((high_humidity_avg - normal_humidity_avg) / normal_humidity_avg) * 100
            
            print(f"\n💧 HIGH HUMIDITY (>80%):")
            print(f"   Days with high humidity: {len(high_humidity_data)}")
            print(f"   Avg admissions on high humidity days: {high_humidity_avg:.1f}")
            print(f"   Avg admissions on normal days: {normal_humidity_avg:.1f}")
            print(f"   ➤ TOTAL INCREASE: +{humidity_increase:.1f}% admissions")
            
            # Asthma-specific
            if 'Asthma_Count' in df.columns:
                high_asthma_avg = high_humidity_data['Asthma_Count'].mean()
                asthma_increase = ((high_asthma_avg - normal_asthma_avg) / normal_asthma_avg) * 100
                print(f"   ➤ ASTHMA CASES: +{asthma_increase:.1f}%")
                print(f"      High humidity days: {high_asthma_avg:.1f} asthma cases")
                print(f"      Normal days: {normal_asthma_avg:.1f} asthma cases")
        else:
            print("\n⚠️  No high humidity days (>80%) found in data")
    else:
        print("⚠️  Humidity data not available")
    
    # ========================================================================
    # 4. BOROUGH TOTAL CASES (ACTUAL NUMBERS)
    # ========================================================================
    print("\n" + "="*80)
    print("4. BOROUGH STATISTICS (ACTUAL NUMBERS)")
    print("="*80)
    
    borough_totals = df.groupby('borough')['Total_Hospitalization'].sum().sort_values(ascending=False)
    borough_avg = df.groupby('borough')['Total_Hospitalization'].mean().sort_values(ascending=False)
    
    if 'Asthma_Count' in df.columns:
        asthma_totals = df.groupby('borough')['Asthma_Count'].sum().sort_values(ascending=False)
    
    if 'Respiratory_Count' in df.columns:
        resp_totals = df.groupby('borough')['Respiratory_Count'].sum().sort_values(ascending=False)
    
    print("\n📊 BOROUGH RANKINGS:")
    print("-" * 80)
    
    for i, (borough, total) in enumerate(borough_totals.items(), 1):
        print(f"\n{i}. {borough.upper()}")
        print(f"   Total Cases: {int(total):,}")
        print(f"   Daily Average: {borough_avg[borough]:.1f} cases/day")
        
        if 'Asthma_Count' in df.columns:
            print(f"   Asthma Cases: {int(asthma_totals.get(borough, 0)):,}")
        
        if 'Respiratory_Count' in df.columns:
            print(f"   Respiratory Cases: {int(resp_totals.get(borough, 0)):,}")
        
        # Calculate percentage of total
        pct_of_total = (total / borough_totals.sum()) * 100
        print(f"   Share of Total: {pct_of_total:.1f}%")
        print("-" * 80)
    
    # ========================================================================
    # 5. MODEL PERFORMANCE (FROM RESULTS)
    # ========================================================================
    print("\n" + "="*80)
    print("5. MODEL PERFORMANCE METRICS")
    print("="*80)
    
    # Classification results
    test_class = class_results[class_results['Dataset'] == 'Test']
    best_class_idx = test_class['AUROC'].idxmax()
    best_class_model = test_class.loc[best_class_idx]
    
    print("\n📊 CLASSIFICATION (High-Risk Day Prediction):")
    print(f"   Best Model: {best_class_model['Model']}")
    print(f"   ➤ Accuracy: {best_class_model['Accuracy']*100:.1f}%")
    print(f"   ➤ AUROC: {best_class_model['AUROC']:.3f}")
    print(f"   ➤ Recall: {best_class_model['Recall']*100:.1f}%")
    print(f"   ➤ Precision: {best_class_model['Precision']*100:.1f}%")
    print(f"   ➤ F1-Score: {best_class_model['F1-Score']:.3f}")
    
    print("\n   All Models Performance:")
    for idx, row in test_class.iterrows():
        print(f"   • {row['Model']:25} | Accuracy: {row['Accuracy']*100:5.1f}% | AUROC: {row['AUROC']:.3f}")
    
    # Regression results
    test_reg = reg_results[reg_results['Dataset'] == 'Test']
    best_reg_idx = test_reg['R²'].idxmax()
    best_reg_model = test_reg.loc[best_reg_idx]
    
    print("\n📈 REGRESSION (Patient Volume Prediction):")
    print(f"   Best Model: {best_reg_model['Model']}")
    print(f"   ➤ R² Score: {best_reg_model['R²']:.3f}")
    print(f"   ➤ MAE: ±{best_reg_model['MAE']:.1f} patients")
    print(f"   ➤ RMSE: {best_reg_model['RMSE']:.1f} patients")
    print(f"   ➤ MAPE: {best_reg_model['MAPE']:.1f}%")
    
    print("\n   All Models Performance:")
    for idx, row in test_reg.iterrows():
        print(f"   • {row['Model']:30} | R²: {row['R²']:.3f} | MAE: ±{row['MAE']:5.1f}")
    
    # ========================================================================
    # 6. EXTREME WEATHER IMPACT BY BOROUGH
    # ========================================================================
    print("\n" + "="*80)
    print("6. EXTREME WEATHER IMPACT BY BOROUGH")
    print("="*80)
    
    if 'Temp_Max_C' in df.columns and 'Temp_Min_C' in df.columns and 'Humidity_Avg' in df.columns:
        # Define extreme weather conditions
        extreme_conditions = (
            (df['Temp_Max_C'] > 30) | 
            (df['Temp_Min_C'] < 0) | 
            (df['Humidity_Avg'] > 80)
        )
        
        df_extreme = df[extreme_conditions]
        df_normal = df[~extreme_conditions]
        
        if len(df_extreme) > 0 and len(df_normal) > 0:
            print(f"\nExtreme weather days identified: {len(df_extreme)}")
            print(f"Normal weather days: {len(df_normal)}")
            
            print("\n📊 IMPACT BY BOROUGH:")
            print("-" * 80)
            
            for borough in ['bronx', 'brooklyn', 'manhattan', 'queens', 'staten island']:
                extreme_borough = df_extreme[df_extreme['borough'] == borough]
                normal_borough = df_normal[df_normal['borough'] == borough]
                
                if len(extreme_borough) > 0 and len(normal_borough) > 0:
                    extreme_avg = extreme_borough['Total_Hospitalization'].mean()
                    normal_avg = normal_borough['Total_Hospitalization'].mean()
                    increase = ((extreme_avg - normal_avg) / normal_avg) * 100
                    
                    print(f"{borough.title():15} | Extreme weather: +{increase:5.1f}% admissions")
                    print(f"                | Avg on extreme days: {extreme_avg:.1f} patients")
                    print(f"                | Avg on normal days: {normal_avg:.1f} patients")
                    print("-" * 80)
    
    # ========================================================================
    # 7. DATA SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("7. DATA SUMMARY")
    print("="*80)
    
    print(f"\n📅 TIME PERIOD:")
    print(f"   Start Date: {df['Date'].min().strftime('%Y-%m-%d')}")
    print(f"   End Date: {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total Days: {len(df['Date'].unique())}")
    print(f"   Years Covered: {sorted(df['Date'].dt.year.unique())}")
    
    print(f"\n📊 TOTAL STATISTICS:")
    print(f"   Total Observations: {len(df):,}")
    print(f"   Total Hospitalizations: {int(df['Total_Hospitalization'].sum()):,}")
    if 'Asthma_Count' in df.columns:
        print(f"   Total Asthma Cases: {int(df['Asthma_Count'].sum()):,}")
    if 'Respiratory_Count' in df.columns:
        print(f"   Total Respiratory Cases: {int(df['Respiratory_Count'].sum()):,}")
    
    print(f"\n🏙️  BOROUGHS:")
    print(f"   Number of Boroughs: {df['borough'].nunique()}")
    print(f"   Boroughs: {', '.join([b.title() for b in sorted(df['borough'].unique())])}")
    
    # ========================================================================
    # SUMMARY FOR COPY-PASTE
    # ========================================================================
    print("\n" + "="*80)
    print("📋 QUICK SUMMARY FOR SLIDES (COPY THIS)")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    if pm25_cols and len(df_pm25) > 0:
        print("\n• PM2.5 Impact:")
        for borough in ['bronx', 'brooklyn', 'queens']:
            borough_data = df_pm25[df_pm25['borough'] == borough]
            if len(borough_data) > 0:
                high = borough_data[borough_data['PM25_avg'] >= pm25_threshold]
                normal = borough_data[borough_data['PM25_avg'] < pm25_threshold]
                if len(high) > 0 and len(normal) > 0:
                    pct = ((high['Total_Hospitalization'].mean() - normal['Total_Hospitalization'].mean()) / normal['Total_Hospitalization'].mean()) * 100
                    print(f"  - {borough.title()}: +{pct:.0f}% on high PM2.5 days")
    
    if 'Temp_Max_C' in df.columns:
        if len(df[df['Temp_Max_C'] > 30]) > 0:
            heat_pct = ((df[df['Temp_Max_C'] > 30]['Total_Hospitalization'].mean() - normal_temp_avg) / normal_temp_avg) * 100
            print(f"\n• Extreme Heat (>30°C): +{heat_pct:.0f}% admissions")
    
    if 'Temp_Min_C' in df.columns:
        if len(df[df['Temp_Min_C'] < 0]) > 0:
            cold_pct = ((df[df['Temp_Min_C'] < 0]['Total_Hospitalization'].mean() - normal_temp_avg) / normal_temp_avg) * 100
            print(f"• Extreme Cold (<0°C): +{cold_pct:.0f}% admissions")
    
    if 'Humidity_Avg' in df.columns and 'Asthma_Count' in df.columns:
        if len(df[df['Humidity_Avg'] > 80]) > 0:
            humid_pct = ((df[df['Humidity_Avg'] > 80]['Asthma_Count'].mean() - normal_asthma_avg) / normal_asthma_avg) * 100
            print(f"• High Humidity (>80%): +{humid_pct:.0f}% asthma cases")
    
    print(f"\n• Model Performance:")
    print(f"  - Classification Accuracy: {best_class_model['Accuracy']*100:.1f}%")
    print(f"  - Classification AUROC: {best_class_model['AUROC']:.2f}")
    print(f"  - Regression R²: {best_reg_model['R²']:.2f}")
    print(f"  - Regression MAE: ±{best_reg_model['MAE']:.0f} patients")
    
    print("\n" + "="*80)
    print("END OF STATISTICS")
    print("="*80)

def interactive_cli():
    """Interactive command-line interface for predictions."""
    print("\n" + "="*70)
    print("NYC HOSPITALIZATION PREDICTION SYSTEM")
    print("="*70)
    print("\nThis system predicts hospital admission counts and capacity needs")
    print("based on date and borough.")
    
    while True:
        print("\n" + "-"*70)
        print("OPTIONS:")
        print("  1. Make a prediction")
        print("  2. Exit")
        print("-"*70)
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '2':
            print("\n✓ Exiting prediction system. Goodbye!")
            break
        
        elif choice == '1':
            print("\n" + "-"*70)
            print("ENTER PREDICTION PARAMETERS")
            print("-"*70)
            
            # Get date input
            date_str = input("\nEnter date (YYYY-MM-DD, e.g., 2024-06-15): ").strip()
            
            # Get borough input
            print("\nAvailable boroughs:")
            print("  - Brooklyn")
            print("  - Bronx")
            print("  - Manhattan")
            print("  - Staten Island")
            print("  - Queens")
            borough = input("\nEnter borough: ").strip()
            
            # Get capacity (optional)
            capacity_input = input("\nEnter current bed capacity (default 80): ").strip()
            if capacity_input:
                try:
                    current_capacity = int(capacity_input)
                except:
                    print("Invalid capacity, using default (80)")
                    current_capacity = 80
            else:
                current_capacity = 80
            
            # Make prediction
            predict_for_date_borough(date_str, borough, current_capacity)
            
            # Ask if user wants to continue
            continue_choice = input("\nMake another prediction? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n✓ Exiting prediction system. Goodbye!")
                break
        
        else:
            print("\n❌ Invalid choice. Please enter 1 or 2.")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete pipeline."""
    print("\n" + "="*70)
    print("NYC HOSPITALIZATION PREDICTION PIPELINE")
    print("="*70)
    
    # Load data
    df_weather, df_resp, df_asthma, df_airq = load_all_data()
    
    # Clean each dataset
    weather_clean = prepare_weather_data(df_weather)
    health_clean = prepare_respiratory_asthma_data(df_resp, df_asthma)
    airq_clean = prepare_air_quality_data(df_airq)
    
    # Merge all datasets
    df_merged = merge_all_datasets(weather_clean, health_clean, airq_clean)
    
    # Save merged data for regional analysis (before feature engineering)
    df_merged_for_analysis = df_merged.copy()
    
    # Impute missing values
    df_imputed = impute_missing_data(df_merged)
    
    # Save merged data
    df_imputed.to_csv('merged_data_before_features.csv', index=False)
    print(f"\n✓ Merged data saved to 'merged_data_before_features.csv'")
    
    # Feature engineering
    df_featured = feature_engineering(df_imputed)
    
    # Create target
    df_final = create_target_variable(df_featured, threshold_percentile=75)
    
    # Prepare for modeling
    (X_train, X_val, X_test, 
     y_class_train, y_class_val, y_class_test,
     y_reg_train, y_reg_val, y_reg_test,
     features, scaler) = prepare_for_modeling(df_final)
    
    # Train classification models
    class_models, class_results = train_classification_models(
        X_train, X_val, X_test, y_class_train, y_class_val, y_class_test
    )
    
    # Train regression models
    reg_models, reg_results, reg_predictions = train_regression_models(
        X_train, X_val, X_test, y_reg_train, y_reg_val, y_reg_test
    )
    
    # Save results
    class_results.to_csv('classification_results.csv', index=False)
    reg_results.to_csv('regression_results.csv', index=False)
    print("\n✓ Results saved")
    
    # Display results
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    test_class = class_results[class_results['Dataset'] == 'Test']
    print(test_class.to_string(index=False))
    
    print("\n" + "="*70)
    print("REGRESSION RESULTS")
    print("="*70)
    test_reg = reg_results[reg_results['Dataset'] == 'Test']
    print(test_reg.to_string(index=False))
    
    # Best models
    best_class = test_class.loc[test_class['AUROC'].idxmax(), 'Model']
    best_reg = test_reg.loc[test_reg['R²'].idxmax(), 'Model']
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODELS:")
    print(f"   Classification: {best_class}")
    print(f"   Regression: {best_reg}")
    print(f"{'='*70}")
    
    # Visualize
    visualize_results(class_results, reg_results, class_models, reg_predictions,
                     X_test, y_class_test, y_reg_test)
    
    # Save models and artifacts for CLI
    save_models_and_artifacts(class_models, reg_models, scaler, features, df_final)
    
    # Create regional analysis plots
    create_regional_analysis_plots(df_merged_for_analysis, df_final)
    
    print("\n✓ Pipeline complete!")
    print("\nGenerated files:")
    print("  1. merged_data_before_features.csv - Combined raw data")
    print("  2. classification_results.csv - Model performance metrics")
    print("  3. regression_results.csv - Prediction accuracy metrics")
    print("  4. hospitalization_prediction_full_results.png - Model visualizations")
    print("  5. prediction_artifacts.pkl - Saved models for predictions")
    print("  6. selected_features.csv - List of features used in modeling")
    print("  7. regional_analysis_comprehensive.png - Regional analysis plots")
    print("  8. regional_summary.csv - Borough-wise statistics")
    calculate_presentation_stats(df_merged_for_analysis, class_results, reg_results)
    return df_final, class_models, reg_models, class_results, reg_results


if __name__ == "__main__":
    try:
        # Run the main pipeline
        final_data, class_models, reg_models, class_results, reg_results = main()
        
        # Launch interactive CLI
        print("\n" + "="*70)
        print("LAUNCHING INTERACTIVE PREDICTION SYSTEM")
        print("="*70)
        interactive_cli()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()