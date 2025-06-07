from PIL import Image
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os  # Untuk mengecek keberadaan file
from datetime import datetime  # Untuk timestamp
from io import BytesIO # Diperlukan untuk download excel

# --- Konfigurasi Halaman & Fungsi Load Pipeline ---
st.set_page_config(
    page_title="Prediksi Waktu Pengantaran Makanan",
    page_icon="🍜",
    layout="wide"
)

# --- Menampilkan Logo Perusahaan ---
# Cek apakah file logo ada sebelum menampilkannya
if os.path.exists('logo.png'): # Pastikan nama file logo Anda benar (misal: logo.png)
    try:
        # Menggunakan PIL untuk membuka gambar
        image = Image.open('logo.png')
        # Menampilkan gambar dengan lebar yang ditentukan (misal: 150 piksel)
        st.image(image, width=150)
    except Exception as e:
        st.error(f"Error saat memuat atau menampilkan logo: {e}")
else:
    # Beri pesan jika logo tidak ditemukan, agar tidak error
    st.warning("File 'logo.png' tidak ditemukan. Logo tidak ditampilkan.")

# --- UI Utama ---
st.title("🍜 Prediksi Waktu Pengantaran Makanan")

# --- Menambahkan Project Overview & Business Objective ---
# Gunakan st.expander agar tampilan lebih rapi
with st.expander("Lihat Detail & Tujuan Proyek"):
    st.subheader("Problem Statement ")
    st.write("""
    Estimasi waktu pengiriman yang tidak akurat sering kali menyebabkan menurunnya kepuasan pelanggan dan inefisiensi alokasi kurir.
    Perusahaan perlu mengetahui faktor apa saja yang paling memengaruhi durasi pengiriman dan bagaimana cara memprediksinya secara tepat
    untuk mengatasi tantangan operasional ini.
    """)
    st.subheader("Business Objective")
    st.write("""
    Meningkatkan kepuasan dan kepercayaan pelanggan dengan memberikan ekspektasi yang lebih realistis,
    mengurangi jumlah keluhan terkait keterlambatan.
    """)

st.markdown("---")

# --- Fungsi untuk memuat pipeline ---
@st.cache_resource
def load_pipeline(pipeline_path):
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: File pipeline '{pipeline_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat pipeline: {e}")
        return None

PIPELINE_FILENAME = 'linear_regression_pipeline.pkl'
pipeline = load_pipeline(PIPELINE_FILENAME)

# --- Nama File dan Fungsi untuk Feedback ---
FEEDBACK_FILE = 'feedback_pengantaran.xlsx'

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_excel(FEEDBACK_FILE)
    else:
        columns = ['Timestamp', 'Predicted_Time_min', 'Actual_Time_min', 'Rating_Kurir', 'Komentar',
                   'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
                   'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        return pd.DataFrame(columns=columns)

def save_feedback_data(df_feedback):
    df_feedback.to_excel(FEEDBACK_FILE, index=False)

# --- Inisialisasi Session State ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# --- Bagian Input ---
st.subheader("Masukkan Detail Pengantaran")

col1, col2 = st.columns(2)

with col1:
    distance_km_input = st.number_input(
        "Distance (km)",
        min_value=0.1,
        max_value=20.0,  # <-- BATAS DARI df.describe() (dibulatkan dari 19.99)
        value=7.93,
        step=0.1,
        key="main_distance_km",
        help="Jarak pengantaran (KM). Berdasarkan data, maks 20 km."
    )
    preparation_time_min_input = st.number_input(
        "Preparation Time (minutes)",
        min_value=1,
        max_value=29,  # <-- BATAS DARI df.describe()
        value=12,
        step=1,
        key="main_preparation_time_min",
        help="Waktu persiapan makanan (menit). Berdasarkan data, maks 29 menit."
    )
    courier_experience_yrs_input = st.number_input(
        "Courier Experience (years)",
        min_value=0.0,
        max_value=9.0,   # <-- BATAS DARI df.describe()
        value=1.0,
        step=0.5,
        key="main_courier_experience_yrs",
        help="Pengalaman kurir (tahun). Berdasarkan data, maks 9 tahun."
    )

with col2:
    weather_input = st.selectbox("Weather Condition", options=['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'], key="main_weather", help="Kondisi cuaca.")
    traffic_level_input = st.selectbox("Traffic Level", options=['Low', 'Medium', 'High'], key="main_traffic_level", help="Tingkat kepadatan lalu lintas.")
    time_of_day_input = st.selectbox("Time of Day", options=['Afternoon', 'Evening', 'Night', 'Morning'], key="main_time_of_day", help="Bagian waktu dalam sehari.")
    vehicle_type_input = st.selectbox("Vehicle Type", options=['Scooter', 'Bike', 'Car'], key="main_vehicle_type", help="Jenis kendaraan.")

st.markdown("---")

# --- Tombol Prediksi dan Hasil ---
if st.button("Predict Delivery Time", key="main_predict_button", use_container_width=True):
    if pipeline is not None:
        input_features = {
            'Distance_km': distance_km_input,
            'Preparation_Time_min': preparation_time_min_input,
            'Courier_Experience_yrs': courier_experience_yrs_input,
            'Weather': weather_input,
            'Traffic_Level': traffic_level_input,
            'Time_of_Day': time_of_day_input,
            'Vehicle_Type': vehicle_type_input
        }
        input_df_raw = pd.DataFrame([input_features])

        try:
            prediction = pipeline.predict(input_df_raw)
            st.session_state.last_prediction = round(prediction[0], 2)
            st.session_state.last_inputs = input_features
            st.session_state.prediction_made = True

            st.subheader("⏳ Hasil Prediksi")
            st.metric(label="Estimasi Waktu Pengantaran", value=f"{st.session_state.last_prediction} menit")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.session_state.prediction_made = False
    else:
        st.error("Pipeline model tidak berhasil dimuat.")
        st.session_state.prediction_made = False

# --- Form Feedback ---
if st.session_state.get('prediction_made', False):
    st.markdown("---")
    st.subheader("📝 Berikan Feedback Anda")

    with st.form(key="feedback_form"):
        actual_time_input = st.number_input("Waktu Pengantaran Aktual (menit)", min_value=1, max_value=600, step=1, help="Berapa lama sebenarnya kurir tiba?")
        rating_kurir_input = st.slider("Rating untuk Kurir (1-5 Bintang)", min_value=1, max_value=5, value=3, step=1, help="Berikan rating untuk pelayanan kurir.")
        komentar_input = st.text_area("Komentar Tambahan", placeholder="Tulis komentar Anda di sini...", help="Saran, kritik, atau pujian.")

        submit_feedback_button = st.form_submit_button("Kirim Feedback")

        if submit_feedback_button:
            df_feedback = load_feedback_data()

            new_feedback_data = st.session_state.get('last_inputs', {})
            new_feedback_data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_feedback_data['Predicted_Time_min'] = st.session_state.get('last_prediction')
            new_feedback_data['Actual_Time_min'] = actual_time_input
            new_feedback_data['Rating_Kurir'] = rating_kurir_input
            new_feedback_data['Komentar'] = komentar_input

            df_new_entry = pd.DataFrame([new_feedback_data])
            df_feedback = pd.concat([df_feedback, df_new_entry], ignore_index=True)

            save_feedback_data(df_feedback)
            st.success("Terima kasih! Feedback Anda telah disimpan.")

            st.session_state.prediction_made = False
            st.rerun()

# --- Menampilkan dan Mengunduh Data Feedback ---
st.markdown("---")
st.subheader("📊 Data Feedback Terkumpul")

df_feedback_display = load_feedback_data()
if not df_feedback_display.empty:
    st.dataframe(df_feedback_display)

    @st.cache_data
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='FeedbackData')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(df_feedback_display)

    st.download_button(
        label="📥 Unduh Data Feedback (Excel)",
        data=excel_data,
        file_name='feedback_pengantaran_terkumpul.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.info("Belum ada data feedback yang terkumpul.")

st.markdown("---")
st.caption("Aplikasi Prediksi Waktu Pengantaran Makanan (Regresi Linear)")