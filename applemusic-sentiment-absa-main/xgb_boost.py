import warnings
warnings.filterwarnings("ignore")

import re
import string
import joblib
import pandas as pd
import numpy as np

# Sastrawi — NLP Bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("[PERINGATAN] Sastrawi tidak ditemukan. Stemming & stopword removal dinonaktifkan.")
    print("             Install: pip install PySastrawi")

# Slang dictionary (pastikan file slang_dictionary.py ada di folder yang sama)
try:
    from slang_dictionary import slang_dict
except ImportError:
    slang_dict = {}
    print("[PERINGATAN] slang_dictionary.py tidak ditemukan. Normalisasi slang dinonaktifkan.")


# =============================================================================
# KONFIGURASI PATH MODEL
# =============================================================================

MODEL_PATHS = {
    "Sentimen"            : "xgb_model_sentimen.pkl",
    "Aspek_Audio_Fitur"   : "xgb_model_aspek_audio_fitur.pkl",
    "Aspek_Performa_Sistem": "xgb_model_aspek_performa_sistem.pkl",
    "Aspek_Harga_Layanan" : "xgb_model_aspek_harga_layanan.pkl",
}

VECTORIZER_PATH = "tfidf_vectorizer_xgb.pkl"

# Label mapping untuk output yang mudah dibaca
LABEL_MAP = {
    "Sentimen"             : {1: "Positif 😊", 0: "Negatif 😠"},
    "Aspek_Audio_Fitur"    : {1: "Terdeteksi ✅", 0: "Tidak Terdeteksi ❌"},
    "Aspek_Performa_Sistem": {1: "Terdeteksi ✅", 0: "Tidak Terdeteksi ❌"},
    "Aspek_Harga_Layanan"  : {1: "Terdeteksi ✅", 0: "Tidak Terdeteksi ❌"},
}

ASPEK_NAMES = {
    "Aspek_Audio_Fitur"    : "🎧 Audio & Fitur",
    "Aspek_Performa_Sistem": "⚙️  Performa Sistem",
    "Aspek_Harga_Layanan"  : "💳 Harga & Layanan",
}


# =============================================================================
# PREPROCESSING PIPELINE
# (identik dengan preprocessing.py agar tidak ada train-serve skew)
# =============================================================================

if SASTRAWI_AVAILABLE:
    _stemmer          = StemmerFactory().create_stemmer()
    _stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


def clean_text(text: str) -> str:
    """
    Pipeline preprocessing teks Bahasa Indonesia.
    Urutan tahapan harus identik dengan yang digunakan saat pelatihan.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Case folding
    text = text.lower()

    # 2. Hapus URL, mention, hashtag, angka, tag HTML
    text = re.sub(r"http\S+|www\S+|<.*?>|@\w+|#\w+|\d+", "", text)

    # 3. Normalisasi slang (multi-word support via regex boundary)
    for slang, formal in slang_dict.items():
        text = re.sub(r"\b" + re.escape(slang) + r"\b", formal, text)

    # 4. Hapus tanda baca & whitespace berlebih
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()
    text = re.sub(r"\s+", " ", text)

    # 5. Stopword removal
    if SASTRAWI_AVAILABLE:
        text = _stopword_remover.remove(text)

    # 6. Stemming
    if SASTRAWI_AVAILABLE:
        text = _stemmer.stem(text)

    return text.strip()


# =============================================================================
# LOADER — MUAT MODEL SEKALI, SIMPAN DI MEMORI
# =============================================================================

_vectorizer = None
_models     = {}


def load_models():
    """Muat vectorizer dan semua model XGBoost ke memori."""
    global _vectorizer, _models

    if _vectorizer is not None:
        return  # sudah dimuat sebelumnya

    print("[INFO] Memuat vectorizer dan model XGBoost...")
    _vectorizer = joblib.load(VECTORIZER_PATH)

    for target, path in MODEL_PATHS.items():
        _models[target] = joblib.load(path)

    print("[INFO] Semua model berhasil dimuat.")


# =============================================================================
# FUNGSI PREDIKSI UTAMA
# =============================================================================

def predict_single(text: str) -> dict:
    """
    Analisis sentimen & aspek untuk satu teks ulasan.

    Parameters
    ----------
    text : str
        Teks ulasan mentah dari pengguna.

    Returns
    -------
    dict dengan key:
        - clean_text      : teks setelah preprocessing
        - Sentimen        : int (0/1)
        - Sentimen_label  : str ('Positif'/'Negatif')
        - Aspek_*         : int (0/1) untuk setiap aspek
        - Aspek_*_label   : str ('Terdeteksi'/'Tidak Terdeteksi')
        - proba_*         : float probabilitas kelas 1 (jika tersedia)
    """
    load_models()

    clean  = clean_text(text)
    if not clean:
        return {"error": "Teks tidak valid setelah preprocessing."}

    X_vec = _vectorizer.transform([clean])

    result = {
        "teks_asli"  : text,
        "clean_text" : clean,
    }

    for target, model in _models.items():
        pred = int(model.predict(X_vec)[0])
        result[target] = pred
        result[f"{target}_label"] = LABEL_MAP[target][pred].split()[0]  # tanpa emoji

        # XGBoost mendukung predict_proba — tampilkan probabilitas
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)[0]
            result[f"proba_{target}"] = round(float(proba[1]), 4)  # proba kelas 1

    return result


def predict_batch(df: pd.DataFrame, text_column: str = "content",
                  already_clean: bool = False) -> pd.DataFrame:
    """
    Analisis sentimen & aspek untuk seluruh baris DataFrame.

    Parameters
    ----------
    df            : DataFrame dengan kolom teks
    text_column   : nama kolom teks
    already_clean : set True jika kolom sudah melalui preprocessing

    Returns
    -------
    DataFrame asli + kolom prediksi tambahan
    """
    load_models()

    df = df.copy()

    if already_clean:
        df["clean_teks"] = df[text_column]
    else:
        print(f"[INFO] Preprocessing kolom '{text_column}' ...")
        df["clean_teks"] = df[text_column].apply(clean_text)

    # Hapus baris kosong setelah preprocessing
    mask    = df["clean_teks"].str.strip() != ""
    n_drop  = (~mask).sum()
    if n_drop:
        print(f"[INFO] {n_drop} baris dibuang karena teks kosong setelah preprocessing.")
    df = df[mask].reset_index(drop=True)

    X_vec = _vectorizer.transform(df["clean_teks"])

    for target, model in _models.items():
        df[f"pred_{target}"] = model.predict(X_vec).astype(int)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)[:, 1]
            df[f"proba_{target}"] = np.round(proba, 4)

    return df


# =============================================================================
# UTILITAS TAMPILAN
# =============================================================================

def print_result(result: dict):
    """Cetak hasil prediksi single ke terminal dengan format rapi."""
    if "error" in result:
        print(f"\n  ❌ ERROR: {result['error']}")
        return

    print("\n" + "=" * 60)
    print("  HASIL ANALISIS ABSA — XGBoost")
    print("=" * 60)
    print(f"  Teks Asli     : {result['teks_asli'][:80]}...")
    print(f"  Teks Bersih   : {result['clean_text'][:80]}...")
    print("-" * 60)

    # Sentimen
    s_label = result.get("Sentimen_label", "?")
    s_proba = result.get("proba_Sentimen", None)
    s_icon  = "✅ POSITIF" if result.get("Sentimen") == 1 else "❌ NEGATIF"
    proba_str = f"  (probabilitas: {s_proba:.2%})" if s_proba is not None else ""
    print(f"\n  Sentimen Keseluruhan : {s_icon}{proba_str}")

    # Aspek
    print("\n  Deteksi Aspek:")
    for key, name in ASPEK_NAMES.items():
        pred  = result.get(key, 0)
        proba = result.get(f"proba_{key}", None)
        icon  = "✅ Terdeteksi    " if pred == 1 else "❌ Tidak Terdeteksi"
        p_str = f" (p={proba:.2%})" if proba is not None else ""
        print(f"    {name:<26}: {icon}{p_str}")

    print("=" * 60)


# =============================================================================
# DEMO — JALANKAN LANGSUNG
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DEMO PREDIKSI — XGBoost ABSA Apple Music")
    print("=" * 60)

    # --- Contoh teks ulasan ---
    sample_texts = [
        # Positif — menyebut kualitas audio
        "Kualitas suaranya luar biasa, Spatial Audio dan Dolby Atmos-nya bikin dengerin musik jadi pengalaman yang beda banget!",

        # Negatif — keluhan performa
        "Aplikasinya sering banget force close dan lag parah. Udah update ke versi terbaru tetap aja masalah yang sama.",

        # Negatif — keluhan harga
        "Harganya mahal banget buat ukuran Indonesia, apalagi kursnya naik terus. Mending pindah ke Spotify yang lebih murah.",

        # Positif campuran dengan keluhan minor
        "Fitur liriknya bagus sih, tapi sayang algoritmanya masih sering salah rekomendasiin lagu. Semoga segera diperbaiki.",

        # Ulasan sangat pendek
        "Bagus!",
    ]

    for i, text in enumerate(sample_texts, 1):
        print(f"\n  [Contoh {i}]")
        result = predict_single(text)
        print_result(result)

    # --- Demo prediksi batch (dari CSV) ---
    import os
    CSV_PATH = "data_preprocessed.csv"
    if os.path.exists(CSV_PATH):
        print("\n" + "=" * 60)
        print("  DEMO BATCH PREDICTION (5 baris pertama dari dataset)")
        print("=" * 60)

        df_sample = pd.read_csv(CSV_PATH).head(5)
        df_result = predict_batch(df_sample, text_column="clean_teks", already_clean=True)

        cols_show = (
            ["content"]
            + [f"pred_{t}" for t in MODEL_PATHS.keys()]
            + [f"proba_{t}" for t in MODEL_PATHS.keys() if f"proba_{t}" in df_result.columns]
        )
        print(df_result[[c for c in cols_show if c in df_result.columns]].to_string(index=False))