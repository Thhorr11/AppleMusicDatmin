import streamlit as st
import pandas as pd

# Import prediksi dari masing-masing algoritma
from xgb_boost import predict_single as predict_xgb
from app import predict_single as predict_svm

# ===========================
# Konfigurasi halaman
# ===========================
st.set_page_config(
    page_title="Perbandingan SVM vs XGBoost",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Perbandingan Algoritma SVM vs XGBoost")
st.markdown(
    "Masukkan satu ulasan Apple Music, kemudian sistem akan "
    "menampilkan hasil prediksi dari kedua algoritma."
)

st.divider()

# ===========================
# Input
# ===========================
text = st.text_area(
    "Masukkan ulasan:",
    height=180,
    placeholder="Contoh: Harga langganannya mahal tapi kualitas audionya sangat bagus."
)

# ===========================
# Prediksi
# ===========================
if st.button("🔍 Bandingkan Hasil"):

    if not text.strip():
        st.warning("Masukkan ulasan terlebih dahulu.")
        st.stop()

    with st.spinner("Menjalankan kedua model..."):

        hasil_svm = predict_svm(text)
        hasil_xgb = predict_xgb(text)

    # Cek error
    if "error" in hasil_svm:
        st.error(f"SVM Error: {hasil_svm['error']}")
        st.stop()

    if "error" in hasil_xgb:
        st.error(f"XGBoost Error: {hasil_xgb['error']}")
        st.stop()

    # ===========================
    # Tabel Perbandingan
    # ===========================
    data = {
        "Kategori": [
            "Sentimen",
            "Audio & Fitur",
            "Performa Sistem",
            "Harga & Layanan"
        ],
        "SVM": [
            hasil_svm.get("Sentimen_label", "-"),
            hasil_svm.get("Aspek_Audio_Fitur_label", "-"),
            hasil_svm.get("Aspek_Performa_Sistem_label", "-"),
            hasil_svm.get("Aspek_Harga_Layanan_label", "-"),
        ],
        "XGBoost": [
            hasil_xgb.get("Sentimen_label", "-"),
            hasil_xgb.get("Aspek_Audio_Fitur_label", "-"),
            hasil_xgb.get("Aspek_Performa_Sistem_label", "-"),
            hasil_xgb.get("Aspek_Harga_Layanan_label", "-"),
        ],
    }

    df = pd.DataFrame(data)

    st.subheader("📋 Hasil Perbandingan")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ===========================
    # Confidence (opsional)
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 SVM")

        if "proba_Sentimen" in hasil_svm:
            st.metric(
                "Confidence",
                f"{hasil_svm['proba_Sentimen']:.2%}"
            )

    with col2:
        st.subheader("🚀 XGBoost")

        if "proba_Sentimen" in hasil_xgb:
            st.metric(
                "Confidence",
                f"{hasil_xgb['proba_Sentimen']:.2%}"
            )

    # ===========================
    # Kesimpulan
    # ===========================
    st.subheader("📝 Kesimpulan")

    if hasil_svm.get("Sentimen_label") == hasil_xgb.get("Sentimen_label"):
        st.success(
            "✅ Kedua algoritma menghasilkan prediksi sentimen yang sama."
        )
    else:
        st.warning(
            "⚠️ Prediksi sentimen SVM dan XGBoost berbeda."
        )

    # ===========================
    # Detail
    # ===========================
    with st.expander("🔍 Detail Hasil SVM"):
        st.json(hasil_svm)

    with st.expander("🔍 Detail Hasil XGBoost"):
        st.json(hasil_xgb)