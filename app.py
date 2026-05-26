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
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Prediction App")
st.write("Predict weather, rain, temperature, and humidity")


# =========================================
# API DETAILS
# =========================================

API_KEY = "212e6e4959b9694425e3ac449a567829"
BASE_URL = "https://api.openweathermap.org/data/2.5/"


# =========================================
# GET CURRENT WEATHER
# =========================================

def get_current_weather(city):

    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data["cod"] != 200:
        return None

    weather_data = {
        "current_temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "temp_min": data["main"]["temp_min"],
        "temp_max": data["main"]["temp_max"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "wind_gust_speed": data["wind"]["speed"],
        "wind_gust_dir": data["wind"].get("deg", 0)
    }

    return weather_data


# =========================================
# READ HISTORICAL DATA
# =========================================

def read_historical_data(filename):

    df = pd.read_csv(filename)

    df = df.dropna()
    df = df.drop_duplicates()

    return df


# =========================================
# PREPARE DATA
# =========================================

def prepare_data(data):

    le = LabelEncoder()

    data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])

    X = data[
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

    y = data["RainTomorrow"]

    return X, y, le


# =========================================
# TRAIN RAIN MODEL
# =========================================

def train_rain_model(X, y):

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)

    return model, mse


# =========================================
# PREPARE REGRESSION DATA
# =========================================

def prepare_regression_data(data, feature):

    X = []
    y = []

    for i in range(len(data) - 1):

        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y


# =========================================
# TRAIN REGRESSION MODEL
# =========================================

def train_regression_model(X, y):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


# =========================================
# FUTURE PREDICTIONS
# =========================================

def predict_future(model, current_value):

    predictions = [current_value]

    for i in range(5):

        next_value = model.predict(
            np.array([[predictions[-1]]])
        )

        predictions.append(next_value[0])

    return predictions[1:]


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

        current_weather = get_current_weather(city)

        if current_weather is None:
            st.error("City not found")
        else:

            # =========================================
            # LOAD CSV
            # =========================================

            try:

                historical_data = read_historical_data(
                    "weather_data.csv"
                )

            except FileNotFoundError:

                st.error(
                    "weather_data.csv file not found"
                )

                st.stop()

            # =========================================
            # PREPARE DATA
            # =========================================

            X, y, le = prepare_data(historical_data)

            rain_model, mse = train_rain_model(X, y)

            # =========================================
            # WIND DIRECTION
            # =========================================

            wind_deg = current_weather["wind_gust_dir"] % 360

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
            # CURRENT DATA
            # =========================================

            current_data = {
                "MinTemp": current_weather["temp_min"],
                "MaxTemp": current_weather["temp_max"],
                "WindGustDir": compass_direction_encoded,
                "WindGustSpeed": current_weather["wind_gust_speed"],
                "Humidity": current_weather["humidity"],
                "Pressure": current_weather["pressure"],
                "Temp": current_weather["current_temp"]
            }

            current_df = pd.DataFrame([current_data])

            # =========================================
            # RAIN PREDICTION
            # =========================================

            rain_prediction = rain_model.predict(
                current_df
            )[0]

            # =========================================
            # TEMP MODEL
            # =========================================

            X_temp, y_temp = prepare_regression_data(
                historical_data,
                "Temp"
            )

            temp_model = train_regression_model(
                X_temp,
                y_temp
            )

            # =========================================
            # HUMIDITY MODEL
            # =========================================

            X_hum, y_hum = prepare_regression_data(
                historical_data,
                "Humidity"
            )

            hum_model = train_regression_model(
                X_hum,
                y_hum
            )

            # =========================================
            # FUTURE PREDICTIONS
            # =========================================

            future_temp = predict_future(
                temp_model,
                current_data["Temp"]
            )

            future_humidity = predict_future(
                hum_model,
                current_data["Humidity"]
            )

            # =========================================
            # FUTURE TIMES
            # =========================================

            timezone = pytz.timezone(
                "Asia/Kolkata"
            )

            now = datetime.now(timezone)

            future_times = []

            for i in range(1, 6):

                future_time = now + timedelta(
                    hours=i
                )

                future_times.append(
                    future_time.strftime(
                        "%I:%M %p"
                    )
                )

            # =========================================
            # DISPLAY RESULTS
            # =========================================

            st.subheader(
                "📍 Current Weather"
            )

            st.write(
                f"### {city}, "
                f"{current_weather['country']}"
            )

            st.write(
                f"🌡️ Current Temperature: "
                f"{current_weather['current_temp']} °C"
            )

            st.write(
                f"🤗 Feels Like: "
                f"{current_weather['feels_like']} °C"
            )

            st.write(
                f"📉 Min Temperature: "
                f"{current_weather['temp_min']} °C"
            )

            st.write(
                f"📈 Max Temperature: "
                f"{current_weather['temp_max']} °C"
            )

            st.write(
                f"💧 Humidity: "
                f"{current_weather['humidity']}%"
            )

            st.write(
                f"🌥️ Weather: "
                f"{current_weather['description']}"
            )

            if rain_prediction == 1:

                st.success(
                    "☔ Rain Prediction: Yes"
                )

            else:

                st.info(
                    "☀️ Rain Prediction: No"
                )

            st.write(
                f"📊 Model MSE: "
                f"{mse:.4f}"
            )

            # =========================================
            # TEMPERATURE TABLE
            # =========================================

            st.subheader(
                "🌡️ Future Temperature Prediction"
            )

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

            st.subheader(
                "💧 Future Humidity Prediction"
            )

            hum_df = pd.DataFrame({
                "Time": future_times,
                "Humidity": future_humidity
            })

            st.table(hum_df)

            st.line_chart(
                hum_df.set_index("Time")
            )
