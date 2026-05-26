import streamlit as st
import requests
import pandas as pd
import numpy as np
import pytz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error

from datetime import datetime, timedelta


# =========================================
# PAGE SETTINGS
# =========================================

st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Prediction App")
st.write("Machine Learning Weather Prediction")


# =========================================
# API DETAILS
# =========================================

API_KEY = "212e6e4959b9694425e3ac449a567829"

BASE_URL = "https://api.openweathermap.org/data/2.5/"


# =========================================
# USER INPUT
# =========================================

city = st.text_input("Enter City Name")


# =========================================
# BUTTON
# =========================================

if st.button("Predict Weather"):

    if city == "":

        st.warning("Please enter city name")

    else:

        # =========================================
        # API REQUEST
        # =========================================

        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"

        response = requests.get(url)

        data = response.json()

        if data["cod"] != 200:

            st.error("City not found")

        else:

            # =========================================
            # WEATHER DATA
            # =========================================

            current_temp = data["main"]["temp"]

            feels_like = data["main"]["feels_like"]

            temp_min = data["main"]["temp_min"]

            temp_max = data["main"]["temp_max"]

            humidity = data["main"]["humidity"]

            pressure = data["main"]["pressure"]

            description = data["weather"][0]["description"]

            country = data["sys"]["country"]

            wind_speed = data["wind"]["speed"]

            wind_deg = data["wind"].get("deg", 0)

            # =========================================
            # LOAD CSV
            # =========================================

            try:

                historical_data = pd.read_csv("weather.csv")

            except FileNotFoundError:

                st.error("weather.csv file not found")

                st.stop()

            # =========================================
            # CLEAN DATA
            # =========================================

            historical_data = historical_data.dropna()

            historical_data = historical_data.drop_duplicates()

            # =========================================
            # LABEL ENCODER
            # =========================================

            le = LabelEncoder()

            historical_data["WindGustDir"] = le.fit_transform(
                historical_data["WindGustDir"]
            )

            historical_data["RainTomorrow"] = le.fit_transform(
                historical_data["RainTomorrow"]
            )

            # =========================================
            # FEATURES
            # =========================================

            X = historical_data[
                [
                    "MinTemp",
                    "MaxTemp",
                    "WindGustDir",
                    "WindGustSpeed",
                    "Humidity",
                    "Pressure",
                    "Temp"
                ]
            ]

            y = historical_data["RainTomorrow"]

            # =========================================
            # TRAIN TEST SPLIT
            # =========================================

            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # =========================================
            # RAIN MODEL
            # =========================================

            rain_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            rain_model.fit(x_train, y_train)

            y_pred = rain_model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)

            # =========================================
            # WIND DIRECTION
            # =========================================

            compass_points = [
                ("N", 0, 22.5),
                ("NE", 22.5, 67.5),
                ("E", 67.5, 112.5),
                ("SE", 112.5, 157.5),
                ("S", 157.5, 202.5),
                ("SW", 202.5, 247.5),
                ("W", 247.5, 292.5),
                ("NW", 292.5, 337.5),
                ("N", 337.5, 360)
            ]

            compass_direction = "N"

            for point, start, end in compass_points:

                if start <= wind_deg < end:

                    compass_direction = point

                    break

            if compass_direction in le.classes_:

                compass_direction_encoded = le.transform(
                    [compass_direction]
                )[0]

            else:

                compass_direction_encoded = 0

            # =========================================
            # CURRENT DATAFRAME
            # =========================================

            current_df = pd.DataFrame([{

                "MinTemp": temp_min,
                "MaxTemp": temp_max,
                "WindGustDir": compass_direction_encoded,
                "WindGustSpeed": wind_speed,
                "Humidity": humidity,
                "Pressure": pressure,
                "Temp": current_temp

            }])

            # =========================================
            # RAIN PREDICTION
            # =========================================

            rain_prediction = rain_model.predict(
                current_df
            )[0]

            # =========================================
            # TEMPERATURE REGRESSION DATA
            # =========================================

            X_temp = []

            y_temp = []

            for i in range(len(historical_data) - 1):

                X_temp.append(
                    historical_data["Temp"].iloc[i]
                )

                y_temp.append(
                    historical_data["Temp"].iloc[i + 1]
                )

            X_temp = np.array(X_temp).reshape(-1, 1)

            y_temp = np.array(y_temp)

            # =========================================
            # TEMPERATURE MODEL
            # =========================================

            temp_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            temp_model.fit(X_temp, y_temp)

            # =========================================
            # HUMIDITY REGRESSION DATA
            # =========================================

            X_hum = []

            y_hum = []

            for i in range(len(historical_data) - 1):

                X_hum.append(
                    historical_data["Humidity"].iloc[i]
                )

                y_hum.append(
                    historical_data["Humidity"].iloc[i + 1]
                )

            X_hum = np.array(X_hum).reshape(-1, 1)

            y_hum = np.array(y_hum)

            # =========================================
            # HUMIDITY MODEL
            # =========================================

            hum_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            hum_model.fit(X_hum, y_hum)

            # =========================================
            # FUTURE TEMPERATURE
            # =========================================

            future_temp = [current_temp]

            for i in range(5):

                next_value = temp_model.predict(
                    np.array([[future_temp[-1]]])
                )

                future_temp.append(next_value[0])

            future_temp = future_temp[1:]

            # =========================================
            # FUTURE HUMIDITY
            # =========================================

            future_humidity = [humidity]

            for i in range(5):

                next_value = hum_model.predict(
                    np.array([[future_humidity[-1]]])
                )

                future_humidity.append(next_value[0])

            future_humidity = future_humidity[1:]

            # =========================================
            # FUTURE TIMES
            # =========================================

            timezone = pytz.timezone("Asia/Kolkata")

            now = datetime.now(timezone)

            future_times = []

            for i in range(1, 6):

                future_time = now + timedelta(hours=i)

                future_times.append(
                    future_time.strftime("%I:%M %p")
                )

            # =========================================
            # DISPLAY RESULTS
            # =========================================

            st.subheader("📍 Current Weather")

            st.write(f"### {city}, {country}")

            st.write(f"🌡️ Current Temperature: {current_temp} °C")

            st.write(f"🤗 Feels Like: {feels_like} °C")

            st.write(f"📉 Min Temperature: {temp_min} °C")

            st.write(f"📈 Max Temperature: {temp_max} °C")

            st.write(f"💧 Humidity: {humidity}%")

            st.write(f"🌥️ Weather: {description}")

            if rain_prediction == 1:

                st.success("☔ Rain Prediction: Yes")

            else:

                st.info("☀️ Rain Prediction: No")

            st.write(f"📊 Model MSE: {mse:.4f}")

            # =========================================
            # TEMPERATURE TABLE
            # =========================================

            st.subheader("🌡️ Future Temperature Prediction")

            temp_df = pd.DataFrame({

                "Time": future_times,
                "Temperature": future_temp

            })

            st.table(temp_df)

            st.line_chart(
                temp_df.set_index("Time")
            )

            # =========================================
            # HUMIDITY TABLE
            # =========================================

            st.subheader("💧 Future Humidity Prediction")

            hum_df = pd.DataFrame({

                "Time": future_times,
                "Humidity": future_humidity

            })

            st.table(hum_df)

            st.line_chart(
                hum_df.set_index("Time")
            )
