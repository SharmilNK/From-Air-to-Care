import pandas as pd

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
    print(df_combined_filtered.head())
    print(df_combined_filtered.shape)
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
    print(df_final_merged.head(15))
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
        valid_boroughs = ['brooklyn', 'bronx', 'manhattan', 'staten island']

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
    return df_clean

def main():
    df_weather,df_resp,df_asthama = prepare_data()
    df_combined_filtered = merge_er_data(df_resp,df_asthama)
    df_weather_trimmed = clean_weather_data (df_weather)
    df_final_merged = build_df(df_weather_trimmed,df_combined_filtered)
    df_clean = impute_data(df_final_merged)

if __name__ == "__main__":
    main()