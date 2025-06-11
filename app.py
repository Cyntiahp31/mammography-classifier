import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from utils import (
    crop_and_apply_clahe_from_upload,
    F1Score,
    F2Score,
    Global_attention_block,
    Category_attention_block
)
from tensorflow.keras.models import Model

st.set_page_config(page_title="Klasifikasi Mamografi", layout="centered")

@st.cache_resource
def load_feature_model():
    base_model = load_model("densenet201_adam_0001_128_am_best (3).keras", custom_objects={
        'F1Score': F1Score,
        'F2Score': F2Score,
        'Global_attention_block': Global_attention_block,
        'Category_attention_block': Category_attention_block
    })
    feature_layer = base_model.layers[-2].output
    feature_model = Model(inputs=base_model.input, outputs=feature_layer)
    return feature_model

@st.cache_resource
def load_svm_and_scaler():
    svm = joblib.load("best_densenet_svm_am (1).pkl")
    scaler = joblib.load("scaler_densenet_svm_am.pkl")
    return svm, scaler

feature_model = load_feature_model()
svm_model, scaler = load_svm_and_scaler()

# Navigasi samping
st.sidebar.markdown("## 🔍 Navigasi")
if st.sidebar.button("🏠 Unggah & Prediksi"):
    st.session_state.page = "Home"
if st.sidebar.button("📘 Info & Panduan"):
    st.session_state.page = "Info"
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Konten Utama
if st.session_state.page == "Home":
    st.title("🩺 Klasifikasi Citra Mamografi")
    st.markdown("Unggah gambar mamografi dan klik **Prediksi** untuk melihat hasilnya:")

    uploaded_file = st.file_uploader("Pilih gambar (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_container_width=True)

        st.markdown(" ")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            predict_clicked = st.button("🔍 Prediksi")

        if predict_clicked:
            with st.spinner("⏳ Mengolah gambar dan melakukan prediksi..."):
                try:
                    image_arr = crop_and_apply_clahe_from_upload(uploaded_file)
                    cnn_ready = np.expand_dims(image_arr, axis=0)

                    features = feature_model.predict(cnn_ready)
                    features_scaled = scaler.transform(features)

                    prediction = svm_model.predict_proba(features_scaled)[0]
                    label = "Kanker" if np.argmax(prediction) == 1 else "Non-Kanker"

                    st.subheader("🔍 Hasil Prediksi")
                    st.success(f"Terdeteksi: **{label}**")

                    st.markdown("**Probabilitas Kelas:**")
                    st.markdown(f"- 🟢 **Non-Kanker**: `{prediction[0]:.4%}`")
                    st.markdown(f"- 🔴 **Kanker**: `{prediction[1]:.4%}`")

                except Exception as e:
                    st.error("❌ Terjadi kesalahan saat memproses gambar. Silakan coba gambar lain.")
                    st.exception(e)

elif st.session_state.page == "Info":
    st.title("ℹ️ Tentang Aplikasi Ini")
    st.markdown("""
    Situs web ini menggunakan pendekatan machine learning untuk mengklasifikasikan citra mamografi ke dalam kelas Kanker dan Non-Kanker:
    - **Pra-pemrosesan**: Peningkatan kontras gambar menggunakan filter CLAHE (Contrast Limited Adaptive Histogram Equalization), pemotongan gambar, dan resize.
    - **Ekstraksi Fitur**: Fitur citra diekstrak menggunakan arsitektur DenseNet201 yang dikombinasikan dengan attention mechanism.
    - **Klasifikasi**: Fitur yang telah diekstrak akan diklasifikasikan menggunakan SVM (Support Vector Machine).
    """)

    st.subheader("📘 Cara Menggunakan")
    st.markdown("""
    1. Klik **🏠 Unggah & Prediksi** pada sidebar.
    2. Unggah gambar mamografi dengan format JPG, PNG, atau JPEG.
    3. Klik **🔍 Prediksi** untuk melihat hasil.
    4. Lihat hasil klasifikasi dan probabilitasnya.
    5. Kembali ke halaman ini jika membutuhkan bantuan.
    """)

    st.subheader("👤 Lima Faktor Manusia Terukur")
    st.markdown("""
    - **Waktu Belajar**: Antarmuka sederhana dan ikon intuitif.
    - **Kecepatan Kinerja**: Prediksi dilakukan dengan cepat.
    - **Tingkat Kesalahan**: Validasi format file dan pesan kesalahan yang jelas.
    - **Daya Ingat**: Struktur tetap dan mudah diingat.
    - **Kepuasan Subjektif**: Kontrol penuh pengguna dan umpan balik instan.
    """)
