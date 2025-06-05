import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Untuk mengecek keberadaan file
from datetime import datetime # Untuk timestamp

# --- Konfigurasi Halaman & Fungsi Load Pipeline ---
st.set_page_config(
    page_title="Prediksi Waktu Pengantaran Makanan",
    page_icon="🍜",
    layout="wide"
)

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

# --- Nama File untuk Feedback ---
FEEDBACK_FILE = 'feedback_pengantaran.xlsx'

# --- Fungsi untuk mengelola data feedback ---
def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_excel(FEEDBACK_FILE)
    else:
        # Definisikan kolom-kolom yang Anda inginkan di Excel
        # Termasuk fitur input, prediksi, dan feedbacknya
        # (Ambil nama kolom fitur dari input_features.keys() atau definisikan secara eksplisit)
        # Contoh kolom:
        columns = ['Timestamp', 'Predicted_Time_min', 'Actual_Time_min', 'Rating_Kurir', 'Komentar',
                   'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 
                   'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        return pd.DataFrame(columns=columns)

def save_feedback_data(df_feedback):
    df_feedback.to_excel(FEEDBACK_FILE, index=False)

# --- Inisialisasi Session State (jika belum ada) ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# --- UI Utama ---
st.title("🍜 Prediksi Waktu Pengantaran Makanan")
# ... (Business Objective bisa ditaruh di sini) ...
st.markdown("---")

st.subheader("Masukkan Detail Pengantaran")

# --- Widget Input ---
# Pastikan semua input_features.keys() sesuai dengan kolom di load_feedback_data()
distance_km_input = st.number_input("Distance (km)", min_value=0.1, value=7.93, step=0.1, key="main_distance_km", help="Jarak pengantaran (KM).")
preparation_time_min_input = st.number_input("Preparation Time (minutes)", min_value=1, value=12, step=1, key="main_preparation_time_min", help="Waktu persiapan makanan (menit).")
courier_experience_yrs_input = st.number_input("Courier Experience (years)", min_value=0.0, value=1.0, step=0.5, key="main_courier_experience_yrs", help="Pengalaman kurir (tahun).")
st.markdown("---")
weather_input = st.selectbox("Weather Condition", options=['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'], key="main_weather", help="Kondisi cuaca.")
traffic_level_input = st.selectbox("Traffic Level", options=['Low', 'Medium', 'High'], key="main_traffic_level", help="Tingkat kepadatan lalu lintas.")
time_of_day_input = st.selectbox("Time of Day", options=['Afternoon', 'Evening', 'Night', 'Morning'], key="main_time_of_day", help="Bagian waktu dalam sehari.")
vehicle_type_input = st.selectbox("Vehicle Type", options=['Scooter', 'Bike', 'Car'], key="main_vehicle_type", help="Jenis kendaraan.")
st.markdown("---")


# --- Tombol Prediksi dan Hasil ---
if st.button("Predict Delivery Time", key="main_predict_button"):
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
            st.session_state.last_prediction = round(prediction[0], 2) # Simpan prediksi
            st.session_state.last_inputs = input_features # Simpan input terakhir
            st.session_state.prediction_made = True # Set flag bahwa prediksi sudah dibuat

            st.subheader("⏳ Hasil Prediksi")
            st.metric(label="Estimasi Waktu Pengantaran", value=f"{st.session_state.last_prediction} menit")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.session_state.prediction_made = False # Reset flag jika error
    else:
        st.error("Pipeline model tidak berhasil dimuat.")
        st.session_state.prediction_made = False # Reset flag

# --- Form Feedback (Muncul setelah prediksi dibuat) ---
if st.session_state.get('prediction_made', False): # Cek flag dari session_state
    st.markdown("---")
    st.subheader("📝 Berikan Feedback Anda")
    
    with st.form(key="feedback_form"):
        actual_time_input = st.number_input(
            "Waktu Pengantaran Aktual (menit)", 
            min_value=1, 
            max_value=600, # Sesuaikan
            step=1,
            help="Berapa lama sebenarnya kurir tiba?"
        )
        rating_kurir_input = st.slider(
            "Rating untuk Kurir (1-5 Bintang)", 
            min_value=1, 
            max_value=5, 
            value=3, 
            step=1,
            help="Berikan rating untuk pelayanan kurir."
        )
        komentar_input = st.text_area(
            "Komentar Tambahan",
            placeholder="Tulis komentar Anda di sini...",
            help="Saran, kritik, atau pujian."
        )
        
        submit_feedback_button = st.form_submit_button("Kirim Feedback")

        if submit_feedback_button:
            # Ambil data feedback yang sudah ada
            df_feedback = load_feedback_data()
            
            # Buat data feedback baru
            new_feedback_entry = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Predicted_Time_min': st.session_state.get('last_prediction'),
                'Actual_Time_min': actual_time_input,
                'Rating_Kurir': rating_kurir_input,
                'Komentar': komentar_input,
                **st.session_state.get('last_inputs', {}) # Menambahkan semua input fitur
            }
            
            # Tambahkan entry baru ke DataFrame
            # Pastikan semua kolom ada, jika tidak, perlu di-handle (misal, dengan reindex atau pastikan kolom sama)
            df_new_entry = pd.DataFrame([new_feedback_entry])
            df_feedback = pd.concat([df_feedback, df_new_entry], ignore_index=True)
            
            # Simpan kembali ke Excel
            save_feedback_data(df_feedback)
            st.success("Terima kasih! Feedback Anda telah disimpan.")
            
            # Reset agar form tidak langsung muncul lagi atau bisa di-submit ulang tanpa prediksi baru
            st.session_state.prediction_made = False 
            # (Opsional: jika Anda ingin form hilang setelah submit, uncomment baris di atas.
            #  Jika tidak, form akan tetap ada sampai prediksi baru dibuat atau halaman di-refresh)


# --- Menampilkan dan Mengunduh Data Feedback ---
st.markdown("---")
st.subheader("📊 Data Feedback Terkumpul")

df_feedback_display = load_feedback_data()
if not df_feedback_display.empty:
    st.dataframe(df_feedback_display)
    
    # Konversi DataFrame ke CSV (atau Excel) untuk diunduh
    # Untuk Excel, perlu engine openpyxl: pip install openpyxl
    @st.cache_data # Cache data agar tidak dikonversi ulang setiap render
    def convert_df_to_excel(df):
        output = pd.ExcelWriter('feedback_download.xlsx', engine='openpyxl')
        df.to_excel(output, index=False, sheet_name='FeedbackData')
        output.close() # Tutup writer untuk menyimpan file
        with open('feedback_download.xlsx', 'rb') as f:
            return f.read()

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