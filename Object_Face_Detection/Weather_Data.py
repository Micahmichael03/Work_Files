import pyodbc
import openmeteo_requests
from datetime import datetime
import pytz

# Database connection string (replace with your credentials)
connection_string = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=gjmfi7jmo2delewe55pp7ledge-q64xenopcmje5bzhzccghkr4bu.database.fabric.microsoft.com,1433;"
    "Database=warehouseDB-5606c843-5230-4432-9741-392553ea9fd5;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Authentication=ActiveDirectoryPassword;"
    "UID=Micahmichael@makoflash02gmail.onmicrosoft.com;"
    "PWD=@Chukwuemeka2025"
)

try:
    # Connect to the database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    # Drop the table if it exists and create it again
    cursor.execute("""
        IF EXISTS (SELECT * FROM sys.tables WHERE name = 'weather_data')
        DROP TABLE weather_data
    """)
    conn.commit()

    # Create the weather_data table with new columns
    cursor.execute("""
        CREATE TABLE weather_data (
            id INT IDENTITY(1,1) PRIMARY KEY,
            timestamp DATETIME,
            location VARCHAR(100),
            country VARCHAR(100),
            temperature FLOAT,
            wind_speed FLOAT,
            precipitation FLOAT,
            humidity FLOAT,
            pressure FLOAT,
            is_forecast BIT DEFAULT 0,
            CONSTRAINT unique_weather UNIQUE (timestamp, location, is_forecast)
        )
    """)
    conn.commit()

    # Define your location (e.g., Lagos, Nigeria)
    latitude = 6.4538
    longitude = 3.4067
    location_name = "Lagos"
    country_name = "Nigeria"

    # Fetch weather data from Open-Meteo
    openmeteo = openmeteo_requests.Client()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "wind_speed_10m", "precipitation", "relative_humidity_2m", "surface_pressure"],
        "current": ["temperature_2m", "wind_speed_10m", "precipitation", "relative_humidity_2m", "surface_pressure"],
        "forecast_days": 1,
        "timezone": "Africa/Lagos"  # Match local time zone
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        raise

    # Extract and insert current weather
    current = responses[0].Current()
    current_time = datetime.fromtimestamp(current.Time(), tz=pytz.timezone("Africa/Lagos"))
    # Set current weather time to 12:00
    current_time = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
    
    try:
        cursor.execute("""
            INSERT INTO weather_data (timestamp, location, country, temperature, wind_speed, precipitation, humidity, pressure, is_forecast)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_time, 
              location_name,
              country_name,
              float(current.Variables(0).Value()),  # temperature_2m
              float(current.Variables(1).Value()),  # wind_speed_10m
              float(current.Variables(2).Value()),  # precipitation
              float(current.Variables(3).Value()),  # relative_humidity_2m
              float(current.Variables(4).Value()),  # surface_pressure
              0))
        conn.commit()
    except pyodbc.IntegrityError as e:
        if "unique_weather" in str(e):
            print(f"Current weather data for {current_time} already exists, skipping insert.")
        else:
            print(f"Database error for current weather: {e}")
            raise
    except Exception as e:
        print(f"Error inserting current weather: {e}")
        raise

    # Extract and insert hourly forecast
    hourly = responses[0].Hourly()
    times = hourly.Time()
    temperatures = hourly.Variables(0).ValuesAsNumpy()  # temperature_2m
    wind_speeds = hourly.Variables(1).ValuesAsNumpy()   # wind_speed_10m
    precipitations = hourly.Variables(2).ValuesAsNumpy()  # precipitation
    humidities = hourly.Variables(3).ValuesAsNumpy()    # relative_humidity_2m
    pressures = hourly.Variables(4).ValuesAsNumpy()     # surface_pressure

    # Get the time array and convert to datetime objects
    time_array = []
    for i in range(len(temperatures)):
        dt = datetime.fromtimestamp(times + (i * 3600), tz=pytz.timezone("Africa/Lagos"))
        # Keep the original hour for forecast data
        time_array.append(dt)

    for dt, temp, wind, precip, hum, press in zip(time_array, temperatures, wind_speeds, precipitations, humidities, pressures):
        try:
            cursor.execute("""
                INSERT INTO weather_data (timestamp, location, country, temperature, wind_speed, precipitation, humidity, pressure, is_forecast)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (dt, 
                  location_name,
                  country_name,
                  float(temp), 
                  float(wind), 
                  float(precip), 
                  float(hum), 
                  float(press), 
                  1))
            conn.commit()
        except pyodbc.IntegrityError as e:
            if "unique_weather" in str(e):
                print(f"Forecast data for {dt} already exists, skipping insert.")
                continue
            else:
                print(f"Database error for forecast: {e}")
                raise
        except Exception as e:
            print(f"Error inserting forecast data: {e}")
            raise

    print("Weather data has been successfully fetched and stored in the database.")

except Exception as e:
    print(f"Unexpected error: {e}")
    raise

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()