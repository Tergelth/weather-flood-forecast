import os
from django.conf import settings
from django.shortcuts import render
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import VAR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from django.shortcuts import render, HttpResponse
import requests
import calendar

def train_and_save_models(N, river, var_model_file, poly_model_file, poly_features_file):
    df_var = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'data(eng).csv')).iloc[:87]
    df_var.set_index('number', inplace=True)
    df_var.index = pd.date_range(start='2017-01-01', periods=len(df_var), freq='MS')
    features = ['temperature', 'precipitation', 'dew point', 'humidity', river]
    df_var = df_var[features].iloc[:87]
    X_var = df_var.drop(columns=river)
    y_var = df_var[river]
    X_var_train, X_var_test, y_var_train, y_var_test = train_test_split(X_var, y_var, test_size=0.2, random_state=42)

    if len(X_var_train) >= N:
        model_var = VAR(X_var_train)
        results_var = model_var.fit(N)
        joblib.dump(results_var, os.path.join(settings.MEDIA_ROOT, 'model_data', var_model_file))
    else:
        raise ValueError(f"Not enough data to train VAR model. Required: {N}, available: {len(X_var_train)}")

    df_poly = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'data(eng).csv')).iloc[:87]
    X_poly = df_poly[['temperature', 'precipitation', 'dew point', 'humidity']]
    y_poly = df_poly[river]
    X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)
    degree = 1
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train_poly = poly_features.fit_transform(X_poly_train)
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train_poly, y_poly_train)

    joblib.dump(poly_model, os.path.join(settings.MEDIA_ROOT, 'model_data', poly_model_file))
    joblib.dump(poly_features, os.path.join(settings.MEDIA_ROOT, 'model_data', poly_features_file))


def forecast_river_level(river, date, var_model_file, poly_model_file, poly_features_file):
    var_model_path = os.path.join(settings.MEDIA_ROOT, 'model_data', var_model_file)
    poly_model_path = os.path.join(settings.MEDIA_ROOT, 'model_data', poly_model_file)
    poly_features_path = os.path.join(settings.MEDIA_ROOT, 'model_data', poly_features_file)

    if os.path.exists(var_model_path):
        results_var = joblib.load(var_model_path)
    poly_model = joblib.load(poly_model_path)
    poly_features = joblib.load(poly_features_path)
    df_var = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'data(eng).csv')).iloc[:87]
    df_var.set_index('number', inplace=True)
    df_var.index = pd.date_range(start='2017-01-01', periods=len(df_var), freq='MS')
    features = ['temperature', 'precipitation', 'dew point', 'humidity', river]
    df_var = df_var[features].iloc[:87]
    X_var = df_var.drop(columns=river)

    last_date_in_data = df_var.index[-1]
    start_date = datetime.strptime(date, '%Y-%m')

    N = (start_date.year - last_date_in_data.year) * 12 + (start_date.month - last_date_in_data.month)

    if N < 3:
        raise ValueError("The start date must be at least 3 months after the last date in the dataset.")

    if os.path.exists(var_model_path) and len(X_var) >= 5:
        forecast_var = results_var.forecast(X_var.values[-5:], steps=N)
        forecast_df = pd.DataFrame(forecast_var, columns=X_var.columns)
    else:
        X_var_train = X_var[-N:]
        forecast_var = []
        for _ in range(N):
            lr_model = LinearRegression()
            lr_model.fit(X_var_train, df_var[river][-N:])
            forecast_value = lr_model.predict(X_var_train[-1].values.reshape(1, -1))
            forecast_var.append(forecast_value[0])
            new_row = X_var_train[-1:].copy()
            new_row.iloc[0, 0] += 1  # Increment month or other logic to simulate future data
            X_var_train = pd.concat([X_var_train[1:], new_row])
        forecast_df = pd.DataFrame(forecast_var, columns=[river])

    forecast_poly_input = forecast_df[['temperature', 'precipitation', 'dew point', 'humidity']].values
    forecast_poly_input_poly = poly_features.transform(forecast_poly_input)

    feature_names = poly_features.get_feature_names_out(input_features=['temperature', 'precipitation', 'dew point', 'humidity'])
    forecast_poly_input_poly_df = pd.DataFrame(forecast_poly_input_poly, columns=feature_names)

    future_dates = pd.date_range(start=start_date, periods=N, freq='MS')
    forecast_df.index = future_dates

    forecasted_river_level = poly_model.predict(forecast_poly_input_poly)

    forecast_df[f'forecasted {river}'] = forecasted_river_level

    last_predicted_features = forecast_df.iloc[-1][['temperature', 'precipitation', 'dew point', 'humidity']].to_dict()
    last_predicted_river_level = forecast_df[f'forecasted {river}'].iloc[-1]

    return df_var, forecast_df, last_predicted_features, last_predicted_river_level


def index(request):
    last_predicted_features = None
    last_predicted_river_level = None
    error_message = None
    river = None
    date = None

    if request.method == "POST":
        river = request.POST.get("river")
        date = request.POST.get("date")

        if not date:
            error_message = "Please select a date."
        else:
            var_model_file = "var_model.pkl"
            poly_model_file = "poly_model.pkl"
            poly_features_file = "poly_features.pkl"

            try:
                train_and_save_models(3, river, var_model_file, poly_model_file, poly_features_file)
                df_var, forecast_df, last_predicted_features, last_predicted_river_level = forecast_river_level(river, date, var_model_file, poly_model_file, poly_features_file)
            except ValueError as e:
                error_message = str(e)

    url = 'https://api.openweathermap.org/data/2.5/forecast'
    params = {
        'lat': 47.9077,
        'lon': 106.8832,
        'units': 'metric',
        'appid': 'b61f845d06d23e8687922bce32020947'
    }

    response = requests.get(url, params=params)
    data = response.json()
    today = datetime.today().date()

    weather_data = []
    weather_today = []
    for weather in data['list']:
        dt_txt = datetime.strptime(weather['dt_txt'], "%Y-%m-%d %H:%M:%S")
        dt_date = dt_txt.date()
        month_name = calendar.month_name[dt_txt.month]
        date_str = f"{month_name} {dt_txt.day}"
        if dt_txt.hour == 15:
            weather_data.append({
                'temp': weather['main']['temp'],
                'description': weather['weather'][0]['description'],
                'icon': weather['weather'][0]['icon'],
                'time': date_str,
                'humidity': weather['main']['humidity'],
                'pressure': weather['main']['pressure'],
            })
        if dt_date == today and len(weather_today) < 1:
            weather_today.append({
                'temp_today': weather['main']['temp'],
                'description_today': weather['weather'][0]['description'],
                'icon_today': weather['weather'][0]['icon'],
                'time_today': date_str,
                'humidity_today': weather['main']['humidity'],
                'pressure_today': weather['main']['pressure'],
            })

    return render(request, 'index.html', {
        'forecasted_values': last_predicted_river_level,
        'error_message': error_message,
        'weather_data': weather_data,
        'weather_today': weather_today,
        'selected_river': river,
        'selected_date': date,
    })
