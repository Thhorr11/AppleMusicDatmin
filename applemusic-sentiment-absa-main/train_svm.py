import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Memuat dataset...")
# 1. Memuat Data
df = pd.read_csv('data_preprocessed.csv')

# Memastikan tidak ada nilai kosong pada kolom teks yang sudah dibersihkan
df = df.dropna(subset=['clean_teks'])

# Teks Input
X_raw = df['clean_teks']

# 2. Ekstraksi Fitur TF-IDF
print("\nMelakukan ekstraksi fitur dengan TF-IDF (Unigram & Bigram)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X_raw)

# Simpan Vectorizer untuk dipakai saat prediksi nanti
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("TF-IDF Vectorizer berhasil disimpan sebagai 'tfidf_vectorizer.pkl'")

# Target yang ingin diprediksi
targets = ['Aspek_Audio_Fitur', 'Aspek_Performa_Sistem', 'Aspek_Harga_Layanan', 'Sentimen']

# Parameter yang akan diuji oleh GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear'], # Tetap linear untuk NLP, tapi kita cari nilai C terbaik
    'class_weight': ['balanced', None]
}

for target in targets:
    print("="*60)
    print(f"Melatih Model SVM dengan GridSearchCV untuk target: {target}")
    print("="*60)
    
    y = df[target]
    
    # 3. Pembagian Data Latih dan Uji (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # 4. Inisialisasi dan Pelatihan Model SVM dengan GridSearch
    base_model = SVC(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Parameter Terbaik ({target}): {grid_search.best_params_}")
    
    # 5. Evaluasi Model Terbaik
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Terbaik ({target}): {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 6. Menyimpan Model
    model_filename = f'svm_model_{target.lower()}.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model berhasil disimpan sebagai '{model_filename}'\n")

print("Semua proses pelatihan dan Tuning selesai!")
