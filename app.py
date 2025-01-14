import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Jantung")

# Fungsi untuk mengumpulkan input pengguna
st.sidebar.header("Input Parameter")
def user_input_features():
    age = st.sidebar.number_input("Age (20-100)", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", (0, 1))
    cp = st.sidebar.selectbox("Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic)", (0, 1, 2, 3))
    trestbps = st.sidebar.number_input("Resting Blood Pressure (80-200 mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (100-400 mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", (0, 1))
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0 = Normal, 1 = Having ST-T Wave Abnormality, 2 = Showing Probable or Definite Left Ventricular Hypertrophy)", (0, 1, 2))
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (70-210 bpm)", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", (0, 1))
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (0.0-6.0)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)", (0, 1, 2))
    ca = st.sidebar.selectbox("Number of Major Vessels (0-4) Colored by Fluoroscopy", (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversable Defect)", (0, 1, 2, 3))
    
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
@st.cache_resource
def load_data():
    data = pd.read_csv('heart.csv')
    return data

heart_data = load_data()

# Preprocessing
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penskalaan fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
df_scaled = scaler.transform(df)

# Membuat model KNN
k = 5  # Jumlah tetangga terdekat yang akan digunakan
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Melatih model
knn_classifier.fit(X_train, y_train)

# Prediksi menggunakan input pengguna
user_prediction = knn_classifier.predict(df_scaled)

# Tampilkan hasil prediksi untuk input pengguna
st.subheader('Hasil Prediksi')
st.write('Penyakit Jantung' if user_prediction[0] == 1 else 'Tidak Ada Penyakit Jantung')
