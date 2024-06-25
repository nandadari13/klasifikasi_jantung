import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# app.py
import joblib
import numpy as np

# Load model and scaler
knn_classifier = joblib.load('model_klasifikais.pkl')
scaler = joblib.load('scaler.pkl')

# Define function for prediction
def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = knn_classifier.predict(features)
    return prediction[0]

# Streamlit app
st.title('Heart Disease Classification')

age = st.slider('Age', 20, 80, 50)
sex = st.selectbox('Sex', [0, 1])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
chol = st.slider('Serum Cholestoral (mg/dl)', 100, 400, 150)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.slider('Maximum Heart Rate Achieved', 70, 210, 150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
thal = st.selectbox('Thal', [0, 1, 2, 3])

features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

if st.button('Predict'):
    result = predict_heart_disease(features)
    if result == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")

