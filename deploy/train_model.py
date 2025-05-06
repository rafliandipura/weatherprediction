import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Contoh data dummy
data = pd.DataFrame({
    'humidity': [40, 60, 75, 80, 50, 90, 85, 45],
    'pressure': [1012, 1010, 1005, 1003, 1011, 1000, 1002, 1013],
    'wind_speed': [2.0, 3.2, 4.5, 1.0, 2.2, 5.0, 3.0, 1.5],
    'time_of_day': ['pagi', 'siang', 'sore', 'malam', 'siang', 'malam', 'sore', 'pagi'],
    'sky_condition': ['cerah', 'mendung', 'hujan', 'mendung', 'cerah', 'hujan', 'cerah', 'mendung'],
    'temperature': [25, 30, 22, 20, 28, 19, 24, 26]
})

X = data.drop('temperature', axis=1)
y = data['temperature']

# Preprocessing untuk fitur kategorikal
categorical_features = ['time_of_day', 'sky_condition']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# Pipeline model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Training
pipeline.fit(X, y)

# Simpan model
joblib.dump(pipeline, 'model.pkl')
print("✅ Model trained and saved to model.pkl")
