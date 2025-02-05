# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model

# Muat dataset
url = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/download?datasetVersionNumber=1"
df = pd.read_csv("/workspaces/Tugas2_22220008/diabetes.csv")

# Pisahkan fitur dan label
X = df.drop(columns=['Outcome']).values  # Semua kolom kecuali 'Outcome'
y = df['Outcome'].values  # Kolom 'Outcome' sebagai label

# Skala data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Bagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ubah bentuk data latih untuk model LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Bangun model LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(1, 8), return_sequences=True))
lstm_model.add(Flatten())
lstm_model.add(Dense(8, activation='relu'))  # Output layer dengan 8 fitur
lstm_model.add(Dense(1, activation='sigmoid'))  # Output layer untuk klasifikasi biner

# Kompilasi model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Latih model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Simpan model LSTM sebagai lstm_model.h5
lstm_model.save('lstm_model.h5')

# Simpan scaler untuk digunakan kembali dalam aplikasi prediksi
import pickle
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model LSTM dan scaler telah disimpan.")