import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import numpy as np
import joblib

# --- NLP & Feature Engineering ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# --- XGBoost ---
from xgboost import XGBClassifier

# --- Imbalanced handling ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Hyperparameter Tuning ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[INFO] Optuna tidak ditemukan. Menggunakan parameter default XGBoost.")
    print("       Install dengan: pip install optuna")

# --- Visualisasi (opsional) ---
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend (cocok di server)
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

DATA_PATH       = "data_preprocessed.csv"
TEXT_COLUMN     = "clean_teks"
TARGETS         = [
    "Sentimen",
    "Aspek_Audio_Fitur",
    "Aspek_Performa_Sistem",
    "Aspek_Harga_Layanan",
]
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
N_SPLITS_CV     = 5          # lipatan cross-validation
OPTUNA_TRIALS   = 30         # jumlah trial Optuna (naikkan untuk hasil lebih optimal)
OUTPUT_DIR      = "."        # folder penyimpanan model


# =============================================================================
# UTILITAS
# =============================================================================

def separator(title="", width=60, char="="):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char * side} {title} {char * side}")
    else:
        print(char * width)


def print_class_distribution(y: pd.Series, label: str):
    counts = y.value_counts().sort_index()
    total  = len(y)
    print(f"\n  Distribusi kelas — {label}:")
    for cls, cnt in counts.items():
        bar = "█" * int(cnt / total * 30)
        print(f"    Kelas {cls}: {cnt:5d} ({cnt/total*100:.1f}%)  {bar}")


# =============================================================================
# 1. MUAT DATA
# =============================================================================

separator("MEMUAT DATA")
print(f"  Membaca: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TEXT_COLUMN])
df = df[df[TEXT_COLUMN].str.strip() != ""]
df = df.reset_index(drop=True)

print(f"  Total data valid  : {len(df):,} baris")
print(f"  Kolom tersedia    : {df.columns.tolist()}")


# =============================================================================
# 2. EKSTRAKSI FITUR TF-IDF
# =============================================================================

separator("EKSTRAKSI FITUR TF-IDF")

TFIDF_PARAMS = {
    "ngram_range" : (1, 2),   # unigram + bigram (sama seperti SVM)
    "max_features": 50_000,   # batasi vocab agar XGBoost tidak kehabisan memori
    "sublinear_tf": True,     # log-normalisasi TF — membantu distribusi fitur
    "min_df"      : 2,        # abaikan token yang muncul < 2 dokumen
}

print(f"  Parameter TF-IDF  : {TFIDF_PARAMS}")

vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
X_tfidf   = vectorizer.fit_transform(df[TEXT_COLUMN])

# Simpan vectorizer — HARUS sama dengan yang dipakai saat inferensi
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_xgb.pkl")
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"  Matriks TF-IDF    : {X_tfidf.shape[0]:,} dokumen × {X_tfidf.shape[1]:,} fitur")
print(f"  Vectorizer disimpan → {VECTORIZER_PATH}")


# =============================================================================
# 3. PARAMETER DEFAULT XGBOOST
#    (dipakai jika Optuna tidak tersedia atau OPTUNA_TRIALS = 0)
# =============================================================================

DEFAULT_XGB_PARAMS = {
    "n_estimators"      : 300,
    "max_depth"         : 6,
    "learning_rate"     : 0.1,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "min_child_weight"  : 3,
    "gamma"             : 0.1,
    "reg_alpha"         : 0.1,   # regularisasi L1
    "reg_lambda"        : 1.0,   # regularisasi L2
    "tree_method"       : "hist",  # cepat untuk data sparse
    "eval_metric"       : "logloss",
    "use_label_encoder" : False,
    "random_state"      : RANDOM_STATE,
    "n_jobs"            : -1,
}


# =============================================================================
# 4. OPTUNA — HYPERPARAMETER TUNING (per target)
# =============================================================================

def optuna_tune(X_train_arr, y_train_arr, n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Mencari hyperparameter XGBoost terbaik menggunakan Optuna
    dengan StratifiedKFold cross-validation (f1_macro sebagai objective).
    """
    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method"      : "hist",
            "eval_metric"      : "logloss",
            "use_label_encoder": False,
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
        }

        model  = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train_arr, y_train_arr,
            cv=skf, scoring="f1_macro", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Tambahkan parameter tetap yang tidak di-tune
    best.update({
        "tree_method"      : "hist",
        "eval_metric"      : "logloss",
        "use_label_encoder": False,
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
    })

    print(f"    ✓ Optuna selesai | Best F1-macro (CV): {study.best_value:.4f}")
    print(f"    ✓ Parameter terbaik: {study.best_params}")
    return best


# =============================================================================
# 5. TRAINING LOOP — SATU MODEL PER TARGET
# =============================================================================

results_summary = {}   # menyimpan ringkasan hasil semua model
smote = SMOTE(random_state=RANDOM_STATE)

for target in TARGETS:
    separator(f"TARGET: {target}")

    y = df[target].astype(int)
    print_class_distribution(y, target)

    # -------------------------------------------------------------------------
    # 5a. SMOTE Oversampling pada data training saja
    # -------------------------------------------------------------------------
    # Split terlebih dahulu → SMOTE hanya pada training set
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        
        X_tfidf, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y          # pertahankan proporsi kelas
    )

    print(f"\n  Ukuran data latih (sebelum SMOTE) : {X_train_raw.shape[0]:,}")
    print(f"  Ukuran data uji                   : {X_test_raw.shape[0]:,}")

    try:
        X_train_res, y_train_res = smote.fit_resample(X_train_raw, y_train)
        print(f"  Ukuran data latih (setelah SMOTE) : {X_train_res.shape[0]:,}")
        print_class_distribution(pd.Series(y_train_res), "setelah SMOTE")
    except ValueError as e:
        print(f"  [PERINGATAN] SMOTE dilewati: {e}")
        X_train_res, y_train_res = X_train_raw, y_train

    # -------------------------------------------------------------------------
    # 5b. Hitung scale_pos_weight (fallback jika SMOTE dilewati)
    #     XGBoost bisa menangani imbalance via scale_pos_weight
    # -------------------------------------------------------------------------
    neg_count = int((y_train_res == 0).sum())
    pos_count = int((y_train_res == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # -------------------------------------------------------------------------
    # 5c. Hyperparameter Tuning
    # -------------------------------------------------------------------------
    t0 = time.time()

    if OPTUNA_AVAILABLE and OPTUNA_TRIALS > 0:
        print(f"\n  [Optuna] Menjalankan {OPTUNA_TRIALS} trial...")
        best_params = optuna_tune(X_train_res, y_train_res, n_trials=OPTUNA_TRIALS)
    else:
        print("\n  [Info] Menggunakan parameter default XGBoost.")
        best_params = DEFAULT_XGB_PARAMS.copy()

    # Tambahkan scale_pos_weight ke parameter akhir
    best_params["scale_pos_weight"] = scale_pos_weight

    # -------------------------------------------------------------------------
    # 5d. Latih model final dengan parameter terbaik
    # -------------------------------------------------------------------------
    print(f"\n  Melatih model final XGBoost untuk: {target} ...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train_res, y_train_res,
        eval_set              = [(X_test_raw, y_test)],
        verbose               = False,
    )

    elapsed = time.time() - t0
    print(f"  ✓ Pelatihan selesai dalam {elapsed:.1f} detik")

    # -------------------------------------------------------------------------
    # 5e. Evaluasi pada data uji
    # -------------------------------------------------------------------------
    y_pred = final_model.predict(X_test_raw)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wei = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    separator("HASIL EVALUASI", width=50, char="-")
    print(f"  Akurasi          : {acc * 100:.2f}%")
    print(f"  F1-Score (Macro) : {f1_mac:.4f}")
    print(f"  F1-Score (Weighted): {f1_wei:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # -------------------------------------------------------------------------
    # 5f. Cross-Validation pada data training untuk estimasi generalisasi
    # -------------------------------------------------------------------------
    print("  Cross-Validation (5-Fold, F1-Macro) pada data latih:")
    skf_eval = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        XGBClassifier(**best_params),
        X_train_res, y_train_res,
        cv      = skf_eval,
        scoring = "f1_macro",
        n_jobs  = -1,
    )
    print(f"  CV Scores  : {cv_scores.round(4)}")
    print(f"  Mean ± Std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -------------------------------------------------------------------------
    # 5g. Simpan confusion matrix sebagai gambar (opsional)
    # -------------------------------------------------------------------------
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Negatif/Tidak", "Positif/Ya"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {target}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, f"cm_xgb_{target.lower()}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix disimpan → {cm_path}")

    # -------------------------------------------------------------------------
    # 5h. Simpan model
    # -------------------------------------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, f"xgb_model_{target.lower()}.pkl")
    joblib.dump(final_model, model_path)
    print(f"  Model disimpan   → {model_path}")

    # Simpan ringkasan hasil
    results_summary[target] = {
        "accuracy"   : round(acc * 100, 2),
        "f1_macro"   : round(f1_mac, 4),
        "f1_weighted": round(f1_wei, 4),
        "cv_mean"    : round(cv_scores.mean(), 4),
        "cv_std"     : round(cv_scores.std(), 4),
        "best_params": {k: v for k, v in best_params.items()
                        if k not in ("tree_method", "eval_metric",
                                     "use_label_encoder", "random_state", "n_jobs")},
    }


# =============================================================================
# 6. RINGKASAN AKHIR
# =============================================================================

separator("RINGKASAN HASIL SEMUA MODEL")
print(f"  {'Target':<28} {'Accuracy':>10} {'F1-Macro':>10} {'CV Mean':>10}")
print("  " + "-" * 62)
for target, res in results_summary.items():
    print(f"  {target:<28} {res['accuracy']:>9.2f}% {res['f1_macro']:>10.4f} {res['cv_mean']:>10.4f}")

print("\n  File yang dihasilkan:")
print(f"    • {VECTORIZER_PATH}")
for target in TARGETS:
    print(f"    • xgb_model_{target.lower()}.pkl")
if MATPLOTLIB_AVAILABLE:
    for target in TARGETS:
        print(f"    • cm_xgb_{target.lower()}.png")

separator()
print("  Semua model XGBoost berhasil dilatih dan disimpan!")
print("  Gunakan file xgb_predict.py untuk prediksi data baru.")
separator()