from flask import Flask, request, jsonify
import joblib
import pandas as pd

# inisial
app = Flask(__name__)

# memuat model
try:
    model = joblib.load('gbr_model.joblib')
except FileNotFoundError:
    print("model 'gbr_model.joblib' tidak ditemukan. Pastikan Anda sudah menjalankan 'data_training.py'")
    model = None

# pickle
# import pickle
#
# with open('gbr_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# membuat endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model tidak dapat dimuat'}), 500

    # mengambil data dari request json yang dikirim pengguna
    data = request.json['data']

    # melakukan prediksi menggunakan model yang sudah dimuat
    prediction = model.predict(data)

    #  mengembalikan hasil prediksi dalam format JSON
    return jsonify({'prediksi_harga': prediction.tolist()})

# menjalankan aplikasi
if __name__ == '__main__':

    app.run(debug=True)