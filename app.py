import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Jantung")

# Input Data
st.sidebar.header("Input Parameter")
def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", (0, 1))
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0-2)", (0, 1, 2))
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", (0, 1))
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", (0, 1, 2))
    ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox("Thalassemia (0-3)", (0, 1, 2, 3))
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Tampilkan input pengguna
st.subheader('Input Parameters')
st.write(df)

# Load dataset
heart_data = pd.read_csv('heart.csv')  # Pastikan file dataset tersedia

# Preprocessing
X = heart_data.drop(columns='target')
y = heart_data['target']

# Standardisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)
df = scaler.transform(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
prediction = knn.predict(df)

# Tampilkan hasil prediksi
st.subheader('Hasil Prediksi')
st.write('Penyakit Jantung' if prediction[0] == 1 else 'Tidak Ada Penyakit Jantung')
