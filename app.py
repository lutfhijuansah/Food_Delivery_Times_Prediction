import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Waktu Pengantaran Makanan (LR)",
    page_icon="🍜",
    layout="wide" # Anda bisa mencoba "centered" jika ingin konten tidak terlalu lebar
)

# --- Fungsi untuk Memuat Pipeline ---
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

# Muat pipeline Anda
PIPELINE_FILENAME = 'linear_regression_pipeline.pkl' # GANTI JIKA NAMA FILE ANDA BERBEDA
pipeline = load_pipeline(PIPELINE_FILENAME)

# --- Judul Aplikasi ---
st.title("🍜 Prediksi Waktu Pengantaran Makanan")
st.markdown("---")

# --- Business Objective ---
# Anda bisa memilih untuk tetap menampilkannya atau menghapusnya jika ingin halaman lebih fokus ke input
st.header("🎯 Business Objective")
st.write("""
Memberikan prediksi waktu pengantaran makanan yang tepat untuk pengalaman pelanggan terbaik dan operasional yang lebih efisien.
""")
st.markdown("---")


# --- Input Fitur untuk Prediksi di Area Utama ---
st.subheader("Masukkan Detail Pengantaran") # Judul seperti di gambar teman Anda

# Mendefinisikan input langsung di halaman utama
# Kita akan kumpulkan nilainya nanti saat tombol ditekan
distance_km_input = st.number_input(
    "Distance (km)", # Label seperti di gambar teman Anda
    min_value=0.1,
    max_value=50.0,
    value=7.93,
    step=0.1,
    key="main_distance_km",
    help="Masukkan jarak pengantaran dalam kilometer."
)
preparation_time_min_input = st.number_input(
    "Preparation Time (minutes)", # Label seperti di gambar teman Anda
    min_value=1, # Minimal 1 menit untuk persiapan
    max_value=120,
    value=12,
    step=1,
    key="main_preparation_time_min",
    help="Masukkan waktu persiapan makanan dalam menit."
)
courier_experience_yrs_input = st.number_input(
    "Courier Experience (years)", # Label seperti di gambar teman Anda
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.5,
    key="main_courier_experience_yrs",
    help="Masukkan pengalaman kurir dalam tahun."
)

st.markdown("---") # Pemisah visual kecil jika diinginkan

weather_input = st.selectbox(
    "Weather Condition", # Label seperti di gambar teman Anda
    options=['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'], # Dari unique values Anda
    index=0, # Default ke 'Windy'
    key="main_weather",
    help="Pilih kondisi cuaca saat ini."
)
traffic_level_input = st.selectbox(
    "Traffic Level", # Label seperti di gambar teman Anda
    options=['Low', 'Medium', 'High'], # Dari unique values Anda
    index=1, # Default ke 'Medium'
    key="main_traffic_level",
    help="Pilih tingkat kepadatan lalu lintas."
)
time_of_day_input = st.selectbox(
    "Time of Day", # Label seperti di gambar teman Anda
    options=['Afternoon', 'Evening', 'Night', 'Morning'], # Dari unique values Anda
    index=0, # Default ke 'Afternoon'
    key="main_time_of_day",
    help="Pilih bagian waktu dalam sehari."
)
vehicle_type_input = st.selectbox(
    "Vehicle Type", # Label seperti di gambar teman Anda
    options=['Scooter', 'Bike', 'Car'], # Dari unique values Anda
    index=0, # Default ke 'Scooter'
    key="main_vehicle_type",
    help="Pilih jenis kendaraan yang digunakan."
)

st.markdown("---") # Pemisah visual

# --- Tombol Prediksi dan Hasil ---
# Tombol sekarang ada di area utama, di bawah input
if st.button("Predict Delivery Time", key="main_predict_button"): # Label tombol seperti di gambar teman Anda
    if pipeline is not None:
        # Kumpulkan semua nilai input ke dalam dictionary
        # Nama kunci HARUS SAMA PERSIS dengan nama kolom yang diharapkan pipeline (X_train mentah)
        input_features = {
            'Distance_km': distance_km_input,
            'Preparation_Time_min': preparation_time_min_input,
            'Courier_Experience_yrs': courier_experience_yrs_input,
            'Weather': weather_input,
            'Traffic_Level': traffic_level_input,
            'Time_of_Day': time_of_day_input,
            'Vehicle_Type': vehicle_type_input
            # PASTIKAN SEMUA FITUR YANG DIGUNAKAN PIPELINE ANDA ADA DI SINI
            # Jika Anda memiliki fitur lain di X_train (misal: 'Delivery_person_Age', dll.),
            # Anda perlu menambahkan widget inputnya di atas dan memasukkannya ke dictionary ini.
        }
        
        input_df_raw = pd.DataFrame([input_features])

        # st.subheader("Input Mentah yang Dikirim ke Pipeline:")
        # st.dataframe(input_df_raw)

        try:
            prediction = pipeline.predict(input_df_raw)
            st.subheader("⏳ Hasil Prediksi")
            # Anda bisa menggunakan format yang lebih besar atau berbeda untuk hasil
            st.metric(label="Estimasi Waktu Pengantaran", value=f"{prediction[0]:.2f} menit")
            # Atau tetap menggunakan st.success jika lebih disukai:
            # st.success(f"Estimasi Waktu Pengantaran: **{prediction[0]:.2f} menit**")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.error("Pastikan semua fitur input telah diisi dengan benar dan pipeline Anda dilatih dengan fitur yang sesuai.")
    else:
        st.error("Pipeline model tidak berhasil dimuat. Tidak dapat melakukan prediksi.")

st.markdown("---")
st.caption("Aplikasi Prediksi Waktu Pengantaran Makanan (Regresi Linear)")