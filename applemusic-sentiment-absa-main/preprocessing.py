import pandas as pd
import re
import string

# IMPORT LANGSUNG DARI MODUL UTAMA (Lebih aman dari beda versi)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from slang_dictionary import slang_dict

# Jika baris di atas masih rewel, jalankan alternatif instan ini:
# import Sastrawi

# 1. Load Data hasil labelan kamu
# Sesuaikan 'data_labeled.csv' dengan nama file kamu
df = pd.read_csv('labeled_ulasan_applemusic_completed.csv') 

# Pastikan nama kolom teks dan labelnya sesuai. 
# Contoh di bawah mengasumsikan nama kolomnya adalah 'teks' dan 'label'
nama_kolom_teks = 'content' 

print("Menginisialisasi Sastrawi...")
# Inisialisasi Stemmer dan Stopword Sastrawi
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # a. Case Folding: Mengubah ke huruf kecil semua
    text = text.lower()
    
    # b. Menghapus URL, Mention (@), dan Hashtag (#)
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # c. Menghapus angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # d. Menghapus whitespace berlebih
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # e. Menghapus karakter non-ASCII (seperti emotikon/emoji)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # f. Normalisasi Slang Words
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    
    # e. Stopword Removal (Menghapus kata dasar yang kurang bermakna)
    text = stopword_remover.remove(text)
    
    # f. Stemming (Mengubah kata berimbuhan menjadi kata dasar)
    # Proses ini agak memakan waktu tergantung jumlah baris data
    text = stemmer.stem(text)
    
    return text

# 2. Jalankan Proses Preprocessing
print("Memulai proses preprocessing teks (mohon tunggu, proses stemming sedang berjalan)...")
df['clean_teks'] = df[nama_kolom_teks].apply(clean_text)

# 3. Hapus baris yang kosong setelah dibersihkan (jika ada)
df = df[df['clean_teks'] != '']

print("Proses selesai! Menampilkan 5 data teratas:")
print(df[[nama_kolom_teks, 'clean_teks', 'Sentimen']].head())

# 4. Simpan ke file CSV baru untuk modal Training SVM besok
df.to_csv('data_preprocessed.csv', index=False)
print("\nData berhasil disimpan dengan nama: 'data_preprocessed.csv'")