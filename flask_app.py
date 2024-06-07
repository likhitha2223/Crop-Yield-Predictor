from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import bz2
import requests

app = Flask(__name__)

label_encoders = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
dt_model = joblib.load('dt_model.pkl')

# Load data
data = pd.read_csv('crop_production.csv')
data['Season'] = data['Season'].str.strip()

states = sorted(data['State_Name'].unique())
crops = sorted(data['Crop'].unique())
seasons = sorted(data['Season'].unique())


def get_districts_by_state(state_name):
    return sorted(data[data['State_Name'] == state_name]['District_Name'].unique())


def predict_crop_yield(district_name, season, crop, area, production, area_unit):
    new_data = pd.DataFrame({
        'District_Name': [district_name],
        'Season': [season],
        'Crop': [crop],
        'Area': [area],
        'Production': [production]
    })

    for col in ['District_Name', 'Season', 'Crop']:
        new_data[col] = label_encoders[col].transform(new_data[col])

    new_data['Area'] = np.log1p(new_data['Area'])
    new_data['Production'] = np.log1p(new_data['Production'])

    new_data_scaled = scaler.transform(new_data)

    prediction = dt_model.predict(new_data_scaled)

    prediction = np.expm1(prediction)

    return prediction[0]


def recommend_best_crops(district_name, season, area, area_unit):
    best_crops = []

    for crop in crops:
        avg_production = data[(data['District_Name'] == district_name) &
                              (data['Season'] == season) &
                              (data['Crop'] == crop)]['Production'].mean()
        if np.isnan(avg_production):
            continue
        predicted_yield = predict_crop_yield(district_name, season, crop, area, avg_production, area_unit)
        best_crops.append({'Crop': crop, 'Predicted_Yield': predicted_yield, 'Avg_Production': avg_production})

    best_crops = sorted(best_crops, key=lambda x: x['Predicted_Yield'], reverse=True)[:5]

    return best_crops


@app.route('/')
def home():
    return render_template('index.html', states=states, crops=crops, seasons=seasons,
                           get_districts_by_state=get_districts_by_state)


@app.route('/get_districts', methods=['POST'])
def get_districts():
    state_name = request.json['state_name']
    districts = get_districts_by_state(state_name)
    return jsonify({'districts': districts})


@app.route('/predict_crop_yield', methods=['POST'])
def predict_crop_yield_route():
    district_name = request.form['district_name']
    season = request.form['season']
    crop = request.form['crop']
    area = float(request.form['area'])
    area_unit = request.form['area_unit']

    # Convert area to hectares if needed
    if area_unit == 'acres':
        area_in_hectares = area * 0.404686
    else:
        area_in_hectares = area

    avg_production = data[(data['District_Name'] == district_name) &
                          (data['Season'] == season) &
                          (data['Crop'] == crop)]['Production'].mean()

    if pd.isna(avg_production):
        return render_template('index.html', prediction_result="No production data available for the selected inputs.",
                               states=states, crops=crops, seasons=seasons,
                               selected_state=request.form['state_name'],
                               selected_district=district_name,
                               selected_season=season,
                               selected_crop=crop,
                               area=area,
                               area_unit=area_unit,
                               get_districts_by_state=get_districts_by_state)

    prediction = predict_crop_yield(district_name, season, crop, area_in_hectares, avg_production, area_unit)

    return render_template('index.html',
                           prediction_result=f"Predicted Crop Yield: {prediction:.2f} tons per hectare",
                           states=states, crops=crops, seasons=seasons,
                           selected_state=request.form['state_name'],
                           selected_district=district_name,
                           selected_season=season,
                           selected_crop=crop,
                           area=area,
                           area_unit=area_unit,
                           get_districts_by_state=get_districts_by_state)


@app.route('/recommend_crops', methods=['POST'])
def recommend_crops_route():
    district_name = request.form['district_name']
    season = request.form['season']
    area = float(request.form['area'])
    area_unit = request.form['area_unit']

    # Convert area to hectares if needed
    if area_unit == 'acres':
        area_in_hectares = area * 0.404686
    else:
        area_in_hectares = area

    recommended_crops = recommend_best_crops(district_name, season, area_in_hectares, area_unit)

    return render_template('index.html', recommended_crops=recommended_crops, states=states, crops=crops,
                           seasons=seasons,
                           selected_state=request.form['state_name'],
                           selected_district=district_name,
                           selected_season=season,
                           area=area,
                           area_unit=area_unit,
                           get_districts_by_state=get_districts_by_state)


if __name__ == '__main__':
    app.run(debug=True)
