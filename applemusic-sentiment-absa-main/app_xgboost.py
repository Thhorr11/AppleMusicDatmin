import streamlit as st
from xgb_boost import predict_single

# ===========================
# Konfigurasi Halaman
# ===========================
st.set_page_config(
    page_title="Apple Music Analyzer - XGBoost",
    page_icon="🎵",
    layout="wide"
)

# ===========================
# Header
# ===========================
st.title("🎵 Apple Music Analyzer")
st.markdown("### Aspect-Based Sentiment Analysis menggunakan XGBoost")

st.write("""
Masukkan ulasan Apple Music untuk dianalisis.
Model akan mendeteksi:

- 😊 Sentimen (Positif / Negatif)
- 🎧 Aspek Audio & Fitur
- ⚙️ Aspek Performa Sistem
- 💳 Aspek Harga & Layanan
""")

st.divider()

# ===========================
# Input
# ===========================
user_input = st.text_area(
    "Masukkan ulasan",
    height=180,
    placeholder="Contoh: Aplikasinya bagus tetapi sering force close."
)

# ===========================
# Tombol Analisis
# ===========================
if st.button("🔍 Analisis"):

    if user_input.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
        st.stop()

    with st.spinner("Sedang melakukan prediksi..."):

        try:
            hasil = predict_single(user_input)

        except Exception as e:
            st.exception(e)
            st.stop()

    if "error" in hasil:
        st.error(hasil["error"])
        st.stop()

    st.success("Prediksi berhasil!")

    # ===========================
    # Sentimen
    # ===========================
    st.subheader("😊 Hasil Sentimen")

    sentimen = hasil.get("Sentimen_label", "-")

    if "positif" in sentimen.lower():
        st.success(sentimen)
    else:
        st.error(sentimen)

    if "proba_Sentimen" in hasil:
        st.metric(
            "Confidence",
            f"{hasil['proba_Sentimen'] * 100:.2f}%"
        )

    st.divider()

    # ===========================
    # Aspek
    # ===========================
    st.subheader("📌 Deteksi Aspek")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎧 Audio & Fitur",
            hasil.get("Aspek_Audio_Fitur_label", "-")
        )

    with col2:
        st.metric(
            "⚙️ Performa Sistem",
            hasil.get("Aspek_Performa_Sistem_label", "-")
        )

    with col3:
        st.metric(
            "💳 Harga & Layanan",
            hasil.get("Aspek_Harga_Layanan_label", "-")
        )

    st.divider()

    # ===========================
    # Detail
    # ===========================
    with st.expander("🔎 Detail Prediksi"):

        st.write("### Teks Asli")
        st.write(hasil.get("teks_asli", ""))

        st.write("### Hasil Preprocessing")
        st.write(hasil.get("clean_text", ""))

        st.write("### Output Lengkap")
        st.json(hasil)