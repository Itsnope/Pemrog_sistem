import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Visualisasi Dataset")

# Input untuk mengunggah dataset
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

# Jika ada file yang diunggah, maka lanjutkan
if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset:")
    st.write(df.head())

    # Input untuk memilih kolom yang ingin ditampilkan
    selected_columns = st.multiselect("Pilih kolom yang ingin ditampilkan", df.columns)

    # Input untuk memilih jenis grafik
    chart_type = st.selectbox("Pilih jenis grafik", ["Line", "Bar", "Area"])

    # Tampilkan grafik sesuai pilihan
    if selected_columns:
        for column in selected_columns:
            st.write(f"Grafik untuk kolom: {column}")
            if chart_type == "Line":
                st.line_chart(df[column])
            elif chart_type == "Bar":
                st.bar_chart(df[column])
            elif chart_type == "Area":
                st.area_chart(df[column])
else:
    st.write("Silakan unggah file CSV untuk melihat visualisasi.")