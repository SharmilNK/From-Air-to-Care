import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import numpy as np

def prepare_data():
    '''
    Read the csv files into dataframes and set the date as index.
    '''
    df_weather = pd.read_csv("nyc_weather_by_borough_2017-2024.csv", index_col=1, parse_dates=True)
    df_resp = pd.read_csv("Respiratory.csv", index_col=6, parse_dates=True)
    df_asthama = pd.read_csv("Asthama.csv", index_col=6, parse_dates=True)
    return df_weather,df_resp,df_asthama

def reset_index(df):

    # If it's an index named 'Date' but not a column, reset it
    if isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or df.columns[0]: 'Date'})
    # Ensure Date dtype
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    #print("After fixes, 'Date' in columns:", 'Date' in df.columns)
    #print(f"Columns now (first 10): {[repr(c) for c in df.columns[:10]]}")
    return df



def merge_er_data(df_resp,df_asthama):
    ''' Aggregate and Merge ER visits count for respiratory and asthama cases'''
   
    #In df_weather, column name = borough, in df_resp, df_asthama column = Dim1Value
    # standardise the column names for borough by renaming columns in df_resp,df_asthama
    df_resp.rename(columns={'Dim1Value': 'borough'}, inplace=True)
    df_asthama.rename(columns={'Dim1Value': 'borough'}, inplace=True)
    #Reset date column, since it was index
    df_resp    = reset_index(df_resp)
    #print(df_resp.head())
    #print(df_resp.shape)
    df_asthama  = reset_index(df_asthama)
    #print(df_asthama.head())
    #print(df_asthama.shape)

    #aggregating count of ER visits based on date & borough so there is one row with Count summed.
    resp_grouped = (
        df_resp.groupby(['Date', 'borough'], as_index=False)['Count'].sum().rename(columns={'Count': 'Total_Resp_Sum'})
    )    
    
    asthama_grouped = (
        df_asthama.groupby(['Date', 'borough'], as_index=False)['Count'].sum().rename(columns={'Count': 'Total_Astm_Sum'})
    ) 
    #print('grouped)')
    
    #print(resp_grouped.head(10))
    #print(resp_grouped.shape)
    #print(asthama_grouped.head(10))
    #print(asthama_grouped.shape)

    # merge respiratory and asthma df on Date and borough
    df_combined = pd.merge(
    resp_grouped,
    asthama_grouped,
    on=['Date', 'borough'],
    how='inner'   
)

# sort by date, borough
    df_combined = df_combined.sort_values(['Date', 'borough']).reset_index(drop=True)

    #print(df_combined_filtered.head())
    #print(df_combined_filtered.shape)

    #Filter data for yars =2017, 2018, 2019, 2023, 2024
    # Extract year
    df_combined['year'] = df_combined['Date'].dt.year

    # Keep only selected years
    keep_years = [2017, 2018, 2019, 2023, 2024]
    df_combined_filtered = df_combined[df_combined['year'].isin(keep_years)].copy()
    #print(df_combined_filtered.head())
    #print(df_combined_filtered.shape)
    return df_combined_filtered

def clean_weather_data (df_weather):

        #  Call function to reset date column
    df_weather = reset_index(df_weather)
    df_weather.rename(columns={'DATE':'Date'},inplace=True)
    #print(df_weather.head())
    #print(df_weather.shape)

    #Keeping relevant columns
    keep_cols = [
    'Date', 'AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN',
    'WDF2', 'WDF5', 'borough', 'year', 'RHAV', 'RHMN', 'RHMX',
    'ADPT', 'ASLP', 'ASTP', 'AWBT', 'AWDR']

    df_weather_trimmed = df_weather[keep_cols].copy()
    df_weather_trimmed = df_weather_trimmed.rename(columns={
    'TMAX': 'Temp_Max_C',
    'TMIN': 'Temp_Min_C',
    'PRCP': 'Precip_mm',
    'AWND': 'WindSpeed_mps',
    'RHAV': 'Humidity_Avg',
    'RHMX': 'Humidity_Max',
    'RHMN': 'Humidity_Min',
    'SNWD': 'Snow_Dept',
    'WDF2':'FastWind2m_deg', #Direction of fastest 2-minute wind
    'WDF5':'FastWind5s_deg', #Direction of fastest 5-second wind gust
    'ADPT':'DewPoint_Temp', #Air temperature at dew point (dew-point temperature)°C
    'ASLP': 'AirPres', #Air pressure at sea level, hectopascals (hPa)
    'ASTP':'AirPresStn', #Air pressure at station level, hPa
    'AWBT':'AirTempWB',#Air temperature at wet bulb degC used to understand human heat stress and evaporative cooling limits.When the wet-bulb temperature exceeds about 35 °C (95 °F), humans can no longer effectively cool themselves by sweating — it’s lethal after prolonged exposure.
    'AWDR':'AvgWindDir'#Average wind direction, degrees (0–360°)    
    }) 
 
    #print(df_weather_trimmed.head())
    #print("Shape:", df_weather_trimmed.shape)
    return df_weather_trimmed

def build_df(df_weather_trimmed,df_combined_filtered):
    df_final_merged = pd.merge(
    df_combined_filtered,             # left: health counts
    df_weather_trimmed,      # right: weather
    on=['Date', 'borough'],  # join keys
    how='left'              # or 'left' if you want to keep all health rows
)
    #print("Merged shape:", df_final_merged.shape)
    #print(df_final_merged.head(15))
    #output_file = "merged_data.csv"
    #df_final_merged.to_csv(output_file, index=False)
    return df_final_merged

def impute_data(df_final_merged):
    ''' Filter for NYC Boroughs, impute with ffill,bfill and drop column with all blanks'''
    drop_all_nan_cols=True
    valid_boroughs=None
    df_clean = df_final_merged.copy()
    
    # create borough filter
    if valid_boroughs is None:
        valid_boroughs = ['brooklyn', 'bronx', 'manhattan', 'staten island','queens']

    # Filter to specific boroughs
    if 'borough' in df_clean.columns:
        df_clean['borough'] = df_clean['borough'].astype(str).str.strip().str.lower()
        df_clean = df_clean[df_clean['borough'].isin(valid_boroughs)]
       
    # Drop columns that are entirely NaN (optional)
    if drop_all_nan_cols:
        all_nan_cols = df_clean.columns[df_clean.isna().all()].tolist()
        if all_nan_cols:
            df_clean = df_clean.drop(columns=all_nan_cols)
    
    #Sort based on date & borough
    df_clean = df_clean.sort_values(['borough', 'Date']).reset_index(drop=True)

    #Impute based on date & borough. This way the values are more reliable since they are closer in date & geography.
    df_clean = df_clean.ffill().bfill()
    #output_file = "cleaned_data.csv"
    #df_clean.to_csv(output_file, index=False)
    #df_filled = df_clean.groupby('borough', group_keys=False).apply(lambda g: g.ffill().bfill())
    #print(df_clean.head())
    print(df_clean[df_clean['borough'] == 'queens'].head())
    (df_clean[df_clean['borough'] == 'bronx'].head())
    return df_clean

def eda(df_clean):
    # Convert totals to numeric (some entries might be stored as text)
    df =df_clean.copy()
    df["Total_Resp_Sum"] = pd.to_numeric(df["Total_Resp_Sum"], errors="coerce")
    df["Total_Astm_Sum"] = pd.to_numeric(df["Total_Astm_Sum"], errors="coerce")

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include='number').dropna(subset=["Total_Resp_Sum", "Total_Astm_Sum"])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Get correlations with each target
    corr_resp = corr_matrix["Total_Resp_Sum"].sort_values(ascending=False)
    corr_asthma = corr_matrix["Total_Astm_Sum"].sort_values(ascending=False)

    # Display top correlations
    print("Top correlated features with Respiratory Cases:")
    print(corr_resp.head(10))
    print("\nTop correlated features with Asthma Cases:")
    print(corr_asthma.head(10))
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap: Weather vs Health Variables")
    plt.show()

    '''print("
    - Correlation (r) close to 1 or -1 means strong linear relationship.
    - r between 0.3 and 0.5 = moderate, <0.3 = weak.
    - Positive r = variable increases with cases.
    - Negative r = variable decreases with cases.")'''
    print("For overall respiratory cases, no weather feature shows a strong direct correlation (r > 0.3).Weather may have a delayed or non-linear effect — not captured by plain correlation.")
    print("Asthma cases have the strongest correlations (~0.4) with air pressure and wind-related variables.When air pressure is higher, or wind direction/speed change, asthma cases tend to rise slightly.Humidity and temperature are weak predictors in this dataset.")

''' 
#This was the orginal tim sris decomposition. However, we se that the data has inconsistency and outliers. So we will rewrite a new function including thresholds.
def resp_tsd(df_clean):
    df_cleaned=df_clean.copy()
    # Loop through each borough
    for borough in df_cleaned["borough"].unique():
        print(f" Decomposing time series for {borough}...")
        
        # Filter data for this borough
        df_b = df_cleaned[df_cleaned["borough"] == borough].copy()
        df_b = df_b.set_index("Date").sort_index()
        
        # Daily resample (sum cases per day)
        series = df_b["Total_Resp_Sum"].resample("D").sum()
        
        # Drop missing or all-NaN segments
        if series.isna().all() or len(series.dropna()) < 14:
            print(f" Skipping {borough}: not enough data.\n")
            continue
        
        # Seasonal decomposition
        try:
            decomp = seasonal_decompose(series.dropna(), model="additive", period=7)
            fig = decomp.plot()
            fig.suptitle(f"Time Series Decomposition — {borough}", fontsize=14)
            plt.show()
        except Exception as e:
            print(f"Error processing {borough}: {e}\n")
            continue

        # Extract components (optional)
        trend = decomp.trend
        seasonal = decomp.seasonal
        resid = decomp.resid

        # Example: save components back into dataframe
        df_b["trend"] = trend
        df_b["seasonal"] = seasonal
        df_b["resid"] = resid
'''
#Use parameters
OUTPUT_DIR = r"C:/Users/som/Desktop/CourseSem1/AlternativeData/Assignment/Project/csv/Upload/tsd_cleaned_plots"
MIN_DAYS = 14
RESAMPLE_FREQ = "D"
STL_PERIOD = 7
DO_LOG = True
OUTLIER_METHOD = "iqr"
PERCENTILE_LOW = 0.01
PERCENTILE_HIGH = 0.99
IQR_MULTIPLIER = 1.5

# Outlier handling function
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handle_outliers(s, method="iqr", p_low=0.01, p_high=0.99, iqr_mult=1.5):
    s_clean = s.copy()
    if method == "percentile":
        low = s.quantile(p_low)
        high = s.quantile(p_high)
        s_clean = s.clip(lower=low, upper=high)
    elif method == "iqr":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - iqr_mult * iqr
        high = q3 + iqr_mult * iqr
        s_clean = s.clip(lower=low, upper=high)
    else:
        raise ValueError("Unknown method for outlier handling.")
    return s_clean

def clean_resp_tsd(df_clean, date_col='Date'):
    """
    df_clean: pandas DataFrame containing at least ['borough','Total_Resp_Sum', date_col]
    date_col: name of the datetime column in df_clean (default 'date')
    Saves STL plots and a CSV of components per borough into OUTPUT_DIR.
    """
    df = df_clean.copy()                    # <-- fixed: add parentheses
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")
    df[date_col] = pd.to_datetime(df[date_col])
    if "borough" not in df.columns:
        raise ValueError("No 'borough' column found.")
    if "Total_Resp_Sum" not in df.columns:
        raise ValueError("No 'Total_Resp_Sum' column found.")
    df["Total_Resp_Sum"] = pd.to_numeric(df["Total_Resp_Sum"], errors="coerce")

    boroughs = df["borough"].dropna().unique()
    saved = []

    for borough in boroughs:
        print(f"Processing borough: {borough}")
        df_b = df[df["borough"] == borough].copy()
        df_b = df_b.set_index(date_col).sort_index()

        # Resample and sum
        series = df_b["Total_Resp_Sum"].resample(RESAMPLE_FREQ).sum()

        # Skip if insufficient data
        if series.dropna().shape[0] < MIN_DAYS:
            print(f"  Skipping {borough}: only {series.dropna().shape[0]} non-null days.")
            continue

        # Fill small gaps
        series = series.interpolate(limit=3, limit_direction='both')

        # Outlier handling
        series_clean = handle_outliers(series.fillna(0), method=OUTLIER_METHOD,
                                       p_low=PERCENTILE_LOW, p_high=PERCENTILE_HIGH,
                                       iqr_mult=IQR_MULTIPLIER)

        # Clip to non-negative
        series_clean = series_clean.clip(lower=0)

        # Optional log transform
        if DO_LOG:
            series_proc = np.log1p(series_clean)
        else:
            series_proc = series_clean

        series_proc = series_proc.interpolate(limit_direction='both')

        # Run STL
        try:
            stl = STL(series_proc.dropna(), period=STL_PERIOD, robust=True)
            res = stl.fit()
        except Exception as e:
            print(f"  STL failed for {borough}: {e}")
            continue

        # Plot and save
        fig = res.plot()
        fig.suptitle(f"STL Decomposition — {borough}", fontsize=14)
        plt.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, f"{borough.replace(' ','_')}_STL.png")
        fig.savefig(out_png)
        plt.close(fig)  # close to free memory

        # Save components CSV
        out_df = pd.DataFrame({
            "observed": series_proc,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid
        })
        out_csv = os.path.join(OUTPUT_DIR, f"{borough.replace(' ','_')}_components.csv")
        out_df.to_csv(out_csv, index=True)

        saved.append((borough, out_png, out_csv))
        print(f"  Saved plot => {out_png}")
        print(f"  Saved components => {out_csv}\n")

    print("Done. Files saved to:", OUTPUT_DIR)
    return saved


def astm_tsd(df_clean):
    df_cleaned=df_clean.copy()
    # Loop through each borough
    for borough in df_cleaned["borough"].unique():
        print(f" Decomposing time series for {borough}...")
        
        # Filter data for this borough
        df_b = df_cleaned[df_cleaned["borough"] == borough].copy()
        df_b = df_b.set_index("Date").sort_index()
        
        # Daily resample (sum cases per day)
        series = df_b["Total_Astm_Sum"].resample("D").sum()
        
        # Drop missing or all-NaN segments
        if series.isna().all() or len(series.dropna()) < 14:
            print(f" Skipping {borough}: not enough data.\n")
            continue
        
        # Seasonal decomposition
        try:
            decomp = seasonal_decompose(series.dropna(), model="additive", period=7)
            fig = decomp.plot()
            fig.suptitle(f"Time Series Decomposition — {borough}", fontsize=14)
            plt.show()
        except Exception as e:
            print(f"Error processing {borough}: {e}\n")
            continue

        # Extract components (optional)
        trend = decomp.trend
        seasonal = decomp.seasonal
        resid = decomp.resid

        # Example: save components back into dataframe
        df_b["trend"] = trend
        df_b["seasonal"] = seasonal
        df_b["resid"] = resid
        print(df_b.head())

def feature_corr(df_clean):
    df = df_clean.copy()



    # --- Create lagged weather features (1–3 day lags) ---
    weather_vars = ['Temp_Max_C', 'Humidity_Avg', 'WindSpeed_mps', 'AirPres']
    for var in weather_vars:
        for lag in range(1, 4):  # create 1-day, 2-day, 3-day lags
            df[f'{var}_lag{lag}'] = df.groupby('borough')[var].shift(lag)

    # --- Add same-day features (already present, just for naming consistency) ---
    for var in weather_vars:
        df[f'{var}_same_day'] = df[var]

    # --- Monthly aggregation (for each borough) ---
    monthly = (
        df.groupby(['borough', pd.Grouper(key='Date', freq='M')])
        .agg({'Total_Resp_Sum': 'sum',
            'Temp_Max_C': 'mean',
            'Humidity_Avg': 'mean',
            'WindSpeed_mps': 'mean',
            'AirPres': 'mean'})
        .reset_index()
    )
    monthly = monthly.rename(columns={'Total_Resp_Sum': 'Total_Resp_Sum_monthly'})

    # --- Merge monthly means back into daily df (aligned by borough and month) ---
    df['month'] = df['Date'].dt.to_period('M')
    monthly['month'] = monthly['Date'].dt.to_period('M')
    df = df.merge(monthly.drop(columns=['Date']), on=['borough', 'month'], how='left', suffixes=('', '_monthly'))

    # --- Compute correlations ---
    # Select numeric columns only
    df['Total_Resp_Sum'] = pd.to_numeric(df['Total_Resp_Sum'], errors='coerce')
    numeric_df = df.select_dtypes(include='number').dropna()

    # Correlations of respiratory cases with all weather-related columns
    corr_table = numeric_df.corr()['Total_Resp_Sum'].sort_values(ascending=False)

    # Display top correlations
    print("🔹 Correlation of weather features with Total_Resp_Sum:")
    print(corr_table.loc[[col for col in corr_table.index if 'Temp' in col or 'Humidity' in col or 'Wind' in col or 'AirPres' in col]])

    # --- Optional: correlation heatmap ---
    import seaborn as sns
    import matplotlib.pyplot as plt

    weather_cols = [c for c in numeric_df.columns if any(k in c for k in ['Temp', 'Humidity', 'Wind', 'AirPres'])]
    corr_matrix = numeric_df[['Total_Resp_Sum'] + weather_cols].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Correlation: Weather vs Respiratory Cases (Daily, Lagged, Monthly)")
    plt.show()

def plot_weather_vs_respiratory(df, target='Total_Resp_Sum'):
    """
    Creates scatter plots to test weather–respiratory hypotheses:
      1. Cold air (low temp) -> higher respiratory cases
      2. Calm, high pressure -> more cases
      3. Strong winds/humidity -> mild increase in cases
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain numeric columns for weather and respiratory counts
    target : str
        Column name for respiratory total (default='Total_Resp_Sum')
    """
    # Clean numeric data
    df = df.copy()
    for col in ['Temp_Max_C', 'Humidity_Avg', 'WindSpeed_mps', 'AirPres']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # --- Set up grid of plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1️⃣ Temperature vs Cases
    sns.regplot(
        x='Temp_Max_C', y=target, data=df,
        scatter_kws={'alpha':0.4, 'color':'steelblue'},
        line_kws={'color':'red'}, ax=axes[0]
    )
    #axes[0].set_title('Colder Air → More Respiratory Cases')
    axes[0].set_xlabel('Daily Maximum Temperature (°C)')
    axes[0].set_ylabel('Total Respiratory Cases')
    #axes[0].text(0.05, 0.9, 'Expected: Negative slope', transform=axes[0].transAxes, fontsize=10)

    # 2️⃣ Air Pressure vs Cases
    sns.regplot(
        x='AirPres', y=target, data=df,
        scatter_kws={'alpha':0.4, 'color':'darkorange'},
        line_kws={'color':'red'}, ax=axes[1]
    )
    #axes[1].set_title('High Pressure → Slightly Higher Cases (Pollution Trapping)')
    axes[1].set_xlabel('Air Pressure (hPa)')
    axes[1].set_ylabel('')
    #axes[1].text(0.05, 0.9, 'Expected: Slight positive slope', transform=axes[1].transAxes, fontsize=10)

    # 3️⃣ Wind Speed vs Cases
    sns.regplot(
        x='WindSpeed_mps', y=target, data=df,
        scatter_kws={'alpha':0.4, 'color':'green'},
        line_kws={'color':'red'}, ax=axes[2]
    )
   # axes[2].set_title('Stronger Winds → Mildly More Cases (Allergen Effect)')
    axes[2].set_xlabel('Wind Speed (m/s)')
    axes[2].set_ylabel('Total Respiratory Cases')
    #axes[2].text(0.05, 0.9, 'Expected: Weak positive slope', transform=axes[2].transAxes, fontsize=10)

    # 4️⃣ Humidity vs Cases
    sns.regplot(
        x='Humidity_Avg', y=target, data=df,
        scatter_kws={'alpha':0.4, 'color':'purple'},
        line_kws={'color':'red'}, ax=axes[3]
    )
    #axes[3].set_title('Humidity → Mild Positive / Mixed Effect')
    axes[3].set_xlabel('Average Humidity (%)')
    axes[3].set_ylabel('')
    #axes[3].text(0.05, 0.9, 'Expected: Weak positive slope', transform=axes[3].transAxes, fontsize=10)

    plt.tight_layout()
    plt.suptitle('Weather vs Reported Cases', fontsize=10, y=1.02)
    plt.show()
    sns.lmplot(x='Temp_Max_C', y='Total_Astm_Sum', data=df,
           lowess=True, scatter_kws={'alpha':0.4})
    plt.title('Temperature vs Asthma Cases (LOESS smoother)')
    plt.show()

def mnth_scatter(df_clean):
        # Monthly average (smooths daily noise)
        df =df_clean.copy()
         # Select numeric columns only
        df['Total_Resp_Sum'] = pd.to_numeric(df['Total_Resp_Sum'], errors='coerce')
        monthly = df.resample('M', on='Date').mean(numeric_only=True)

        # Scatter: Temperature vs Respiratory
        sns.regplot(x='Temp_Max_C', y='Total_Resp_Sum', data=monthly, lowess=True,
                    scatter_kws={'alpha':0.6, 'color':'steelblue'},
                    line_kws={'color':'red'})
        plt.title('Monthly Temperature vs Respiratory Cases')
        plt.show()

        # Scatter: Temperature vs Asthma
        sns.regplot(x='Temp_Max_C', y='Total_Astm_Sum', data=monthly, lowess=True,
                    scatter_kws={'alpha':0.6, 'color':'green'},
                    line_kws={'color':'red'})
        plt.title('Monthly Temperature vs Asthma Cases')
        plt.show()

        # 3-month rolling average (quarterly)
        three_month = df.resample('3M', on='Date').mean(numeric_only=True)

        # Plot both outcomes side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.regplot(x='Temp_Max_C', y='Total_Resp_Sum', data=three_month, lowess=True,
                    scatter_kws={'alpha':0.6, 'color':'steelblue'},
                    line_kws={'color':'red'}, ax=axes[0])
        axes[0].set_title('3-Month Avg: Temperature vs Respiratory')

        sns.regplot(x='Temp_Max_C', y='Total_Astm_Sum', data=three_month, lowess=True,
                    scatter_kws={'alpha':0.6, 'color':'green'},
                    line_kws={'color':'red'}, ax=axes[1])
        axes[1].set_title('3-Month Avg: Temperature vs Asthma')

        plt.tight_layout()
        plt.show()

def main():
    df_weather,df_resp,df_asthama = prepare_data()
    df_combined_filtered = merge_er_data(df_resp,df_asthama)
    df_weather_trimmed = clean_weather_data (df_weather)
    df_final_merged = build_df(df_weather_trimmed,df_combined_filtered)
    df_clean = impute_data(df_final_merged)
    #eda(df_clean)
    #resp_tsd(df_clean)
    #astm_tsd(df_clean)
    #clean_resp_tsd(df_clean)
    #feature_corr(df_clean)
    #plot_weather_vs_respiratory(df_clean, target='Total_Resp_Sum')
    #plot_weather_vs_respiratory(df_clean, target='Total_Astm_Sum')
    #mnth_scatter(df_clean)

if __name__ == "__main__":
    main()