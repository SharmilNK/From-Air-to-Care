import requests
import pandas as pd
from datetime import datetime
import time
from io import StringIO

# NYC Borough approximate coordinates for finding nearby stations
NYC_BOROUGHS = {
    'Manhattan': {'lat': 40.7831, 'lon': -73.9712},
    'Brooklyn': {'lat': 40.6782, 'lon': -73.9442},
    'Queens': {'lat': 40.7282, 'lon': -73.7949},
    'Bronx': {'lat': 40.8448, 'lon': -73.8648},
    'Staten Island': {'lat': 40.5795, 'lon': -74.1502}
}

# Known major weather stations in NYC - multiple options per borough
NYC_STATIONS = {
    'Manhattan': ['USW00094728'],  # Central Park
    'Brooklyn': ['USW00094789'],   # JFK Airport area
    'Queens': ['USW00014732'],     # LaGuardia Airport
    'Bronx': ['USW00094728'],      # Central Park (closest)
    'Staten Island': ['USC00308962', 'USC00308946', 'USW00094789']  # Try multiple stations
}

# Years to include (excluding COVID years 2020-2022)
YEARS = [2017, 2018, 2019, 2023, 2024]

class NOAAWeatherFetcher:
    """Fetches weather data from NOAA NCEI API for NYC boroughs."""
    
    def __init__(self):
        self.data_base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
        
    def get_weather_data(self, station_id, start_date, end_date, dataset='daily-summaries', debug=False):
        """
        Fetch weather data for a specific station and date range.
        
        Args:
            station_id: Station identifier (WITHOUT GHCND: prefix)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dataset: Dataset type
            debug: If True, print detailed debug info
        """
        # DO NOT add GHCND: prefix - the API works without it!
        
        params = {
            'dataset': dataset,
            'stations': station_id,
            'startDate': start_date,
            'endDate': end_date,
            'format': 'csv',
            'units': 'standard'  # Fahrenheit, inches, mph
        }
        
        try:
            if debug:
                print(f"\n      DEBUG - Request URL: {self.data_base_url}")
                print(f"      DEBUG - Station ID: {station_id}")
                print(f"      DEBUG - Date range: {start_date} to {end_date}")
            
            response = requests.get(self.data_base_url, params=params, timeout=60)
            
            if debug:
                print(f"      DEBUG - Status code: {response.status_code}")
                print(f"      DEBUG - Response length: {len(response.text)} bytes")
            
            if response.status_code == 200:
                if len(response.text.strip()) < 200:  # Only headers, no data
                    if debug:
                        print("      DEBUG - Empty response (headers only)!")
                    return None
                
                # Parse CSV response
                df = pd.read_csv(StringIO(response.text))
                
                if len(df) == 0:
                    if debug:
                        print("      DEBUG - Parsed but 0 rows")
                    return None
                    
                if debug:
                    print(f"      DEBUG - Successfully parsed {len(df)} rows")
                return df
            else:
                if debug:
                    print(f"      DEBUG - Error: {response.text[:500]}")
                return None
                
        except Exception as e:
            if debug:
                print(f"      DEBUG - Exception: {type(e).__name__}: {e}")
            return None
    
    def get_borough_weather(self, borough_name, years=YEARS, debug_first=False):
        """
        Get weather data for a specific NYC borough across multiple years.
        
        Args:
            borough_name: Name of the borough
            years: List of years to fetch data for
            debug_first: If True, show debug info for first request
        """
        if borough_name not in NYC_BOROUGHS:
            print(f"Unknown borough: {borough_name}")
            return None
        
         
        # Get list of stations to try for this borough
        station_ids = NYC_STATIONS.get(borough_name, [])
        
        if not station_ids:
            print(f"  No stations available for {borough_name}")
            return None
        
        # Try each station until one works
        for station_id in station_ids:
            print(f"  Trying station: {station_id}")
            
            all_data = []
            success_count = 0
            
            # Fetch data for each year
            for idx, year in enumerate(years):
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                print(f"    {year}...", end=' ')
                
                # Enable debug for first request only
                debug = debug_first and idx == 0
                data = self.get_weather_data(station_id, start_date, end_date, debug=debug)
                
                if data is not None and not data.empty:
                    data['borough'] = borough_name
                    data['year'] = year
                    all_data.append(data)
                    success_count += 1
                    print(f"✓ ({len(data)} records)")
                else:
                    print("✗")
                
                # Be respectful of API - add small delay
                time.sleep(0.5)
            
            # If this station got data, use it and stop trying others
            if success_count > 0:
                print(f"  ✓ Success: {success_count}/{len(years)} years retrieved")
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df
            else:
                print(f"  ✗ No data from this station, trying next...")
        
        # If we get here, no station worked
        print(f"  ✗ Could not get data from any station")
        return None
    
    def get_all_boroughs_weather(self, years=YEARS):
        """Get weather data for all NYC boroughs."""
        all_borough_data = []
        
        for borough in NYC_BOROUGHS.keys():
            data = self.get_borough_weather(borough, years)
            if data is not None:
                all_borough_data.append(data)
            
            # Add delay between boroughs
            time.sleep(1)
        
        if not all_borough_data:
            print("\nNo data collected for any borough")
            return None
        
        # Combine all borough data
        combined_df = pd.concat(all_borough_data, ignore_index=True)
        return combined_df


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("NYC Weather Data Fetcher")
    print("Years: 2017-2019, 2023-2024 (excluding COVID years)")
    print("="*60)
    
    fetcher = NOAAWeatherFetcher()
    
    # Get data for all boroughs (debug enabled for first request)
    print("\nNote: Debug output will show for Manhattan's first request...\n")
    
    all_borough_data = []
    for idx, borough in enumerate(NYC_BOROUGHS.keys()):
        debug = (idx == 0)  # Debug first borough only
        data = fetcher.get_borough_weather(borough, debug_first=debug)
        if data is not None:
            all_borough_data.append(data)
        time.sleep(1)
    
    if not all_borough_data:
        print("\n✗ Failed to retrieve data")
        exit(1)
    
    df = pd.concat(all_borough_data, ignore_index=True)
    
    if df is not None:
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Total records: {len(df):,}")
        print(f"\nBoroughs covered:")
        for borough, count in df['borough'].value_counts().items():
            print(f"  - {borough}: {count:,} records")
        
        print(f"\nColumns: {df.columns.tolist()}")
        
        print(f"\nDate range:")
        if 'DATE' in df.columns:
            print(f"  From: {df['DATE'].min()}")
            print(f"  To:   {df['DATE'].max()}")
        
        print(f"\nFirst few rows:")
        print(df.head(10))
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        
        # Save to CSV
        output_file = "nyc_weather_by_borough_2017-2024.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Data saved to: {output_file}")
        print("="*60)
    else:
        print("\nFailed to retrieve data")