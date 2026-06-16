import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from slang_dictionary import slang_dict

print("1. Inisialisasi Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<.*?>|@\w+|#\w+|\d+', '', text)
    
    # Slang replacement (Multi-word support)
    for slang, formal in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', formal, text)
        
    text = text.translate(str.maketrans('', '', string.punctuation)).strip()
    text = re.sub(r'\s+', ' ', text)
    text = stopword_remover.remove(text)
    return stemmer.stem(text)

print("2. Memuat dataset mentah...")
df_raw = pd.read_csv('labeled_ulasan_applemusic_completed.csv')

print("3. Memproses ulang teks (Ini mungkin butuh waktu beberapa menit)...")
df_raw['clean_teks'] = df_raw['content'].apply(clean_text)

# Drop rows where clean_teks is empty
df_raw = df_raw[df_raw['clean_teks'].str.strip() != '']

# Simpan data preprocessed yang baru
df_raw.to_csv('data_preprocessed.csv', index=False)
print("Data preprocessed berhasil di-update!")

X_raw = df_raw['clean_teks']

print("4. Ekstraksi Fitur TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X_raw)

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

targets = ['Aspek_Audio_Fitur', 'Aspek_Performa_Sistem', 'Aspek_Harga_Layanan', 'Sentimen']
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear'],
    'class_weight': ['balanced', None]
}

smote = SMOTE(random_state=42)

for target in targets:
    print("="*60)
    print(f"Melatih Model SVM untuk: {target}")
    y = df_raw[target]
    
    # SMOTE oversampling
    print("Melakukan SMOTE Oversampling...")
    try:
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    except ValueError as e:
        print(f"SMOTE gagal untuk {target} (mungkin sampel terlalu sedikit). Melewati SMOTE...")
        X_resampled, y_resampled = X_tfidf, y
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    print("Mencari parameter terbaik dengan GridSearchCV...")
    base_model = SVC(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Parameter Terbaik ({target}): {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Terbaik ({target}): {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    model_filename = f'svm_model_{target.lower()}.pkl'
    joblib.dump(best_model, model_filename)

print("Selesai!")
