import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
 
# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    lstm_model = load_model('lstm_model.h5')
    with open('svm_model.pkl', 'rb') as svm_file:
        svm_classifier = pickle.load(svm_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return lstm_model, svm_classifier, scaler

# Memuat model LSTM, SVM, dan Scaler
lstm_model, svm_classifier, scaler = load_models()

# Judul aplikasi
st.title("Aplikasi Prediksi Diabetes")

# Input dari pengguna
st.write("Masukkan nilai fitur untuk prediksi:")

# Membuat form input untuk fitur
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Menggabungkan input menjadi array
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Tombol prediksi
if st.button("Prediksi"):
    # Skala input menggunakan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_data)

    # Bentuk input untuk LSTM (samples, timesteps, features)
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    lstm_features = lstm_model.predict(input_lstm)
    
    # Pastikan bahwa output dari LSTM memiliki bentuk (1, 8)
    lstm_features = lstm_features.reshape(1, -1)

    # Prediksi menggunakan model SVM
    prediction = svm_classifier.predict(lstm_features)

    # Tampilkan hasil prediksi
    if prediction[0] == 1:
        st.write("Hasil Prediksi: Positif Diabetes")
    else:
        st.write("Hasil Prediksi: Negatif Diabetes")
