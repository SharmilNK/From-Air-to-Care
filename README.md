# From-Air-to-Care
Forecasting Hospital Admissions for Respiratory Distress using Environmental Indicators.

# Project Overview
Most hospitals today face a common challenge – not being able to predict patient surges that cause ER strain.
Tomorrow, when air pollution spikes or a heat wave hits, emergency rooms will be overwhelmed again. We are using data from air pollution + weather + climate to forecast hospital strain 3-7 days in advance. Air pollution, weather patterns and seasonality changes are highly correlated with increased ER visits. 

This predictive model using alternative data sources will help hospitals allocate resources and reduce costs.

# Objective 
-Build a time-stratified predictive model (pollution + weather lags) to forecast admissions in the next 3-7 days

-Use open alternative datasets (OpenAQ, EPA, NOAA, NYC DOHMH)

-Deliver the county wise probability for public health decision-making

# Models Evaluated

The following regression models were trained and evaluated:

Gradient Boosting Regressor

Random Forest Regressor

Lasso Regression

Ridge Regression

Gradient Boosting Regressor achieved the strongest overall performance across all evaluation metrics and was selected as the final model.

# Performance Metrics (Best Model)

R² Score: 0.906
→ Explains 90.6% of the variance in patient volumes

MAE (Mean Absolute Error): ±59.7 patients
→ Predictions are typically within ±60 patients of the actual count

# How to Run the Project

1. Clone the Repository
   
git clone https://github.com/SharmilNK/From-Air-to-Care.git
cd From-Air-to-Care

2. Create & Activate a Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run main.py

Slide Deck : https://www.canva.com/design/DAG5Pmr9zUE/KJS4vfdtRTmNggN_4O_lUQ/edit?ui=e30






