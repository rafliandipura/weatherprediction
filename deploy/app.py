from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['wind_speed'])
        time_of_day = request.form['time_of_day']
        sky_condition = request.form['sky_condition']

        input_df = pd.DataFrame([{
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'time_of_day': time_of_day,
            'sky_condition': sky_condition
        }])

        temperature = model.predict(input_df)[0]
        prediction = round(temperature, 2)

    return render_template('index.html', prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True)
