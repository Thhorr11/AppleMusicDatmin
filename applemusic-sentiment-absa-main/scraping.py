import pandas as pd
from google_play_scraper import Sort, reviews

print("====================================================")
print("Memulai Proses Scraping Mandiri Ulasan Apple Music Android...")
print("====================================================")

# ID Paket Resmi Apple Music di Google Play Store
app_id = 'com.apple.android.music'

all_reviews = []
nxt_token = None

# Kita tarik 50 paket (@200 data = 10.000 data ulasan terbaru berbahasa Indonesia)
# Jumlah yang jauh lebih ideal untuk kebutuhan data mining dan machine learning
for i in range(50):
    print(f"-> Mengambil paket ulasan Apple Music ke-{i+1}...")
    try:
        result, nxt_token = reviews(
            app_id,
            lang='id',         # Mengunci ulasan khusus Bahasa Indonesia
            country='id',      # Region Indonesia
            sort=Sort.NEWEST,  # Tren ulasan paling gres Juni 2026
            count=200,
            continuation_token=nxt_token
        )
        all_reviews.extend(result)
        if not nxt_token:
            break
    except Exception as e:
        print(f"Kendala pada batch ke-{i+1}: {e}")
        break

# Simpan hasil scraping murni kelompokmu
if len(all_reviews) > 0:
    df = pd.DataFrame(all_reviews)
    
    # Filter kolom penting untuk data understanding kalian
    df_filtered = df[['userName', 'score', 'at', 'content']].copy()
    
    # Simpan ke file lokal .csv
    df_filtered.to_csv('raw_ulasan_applemusic.csv', index=False)
    
    print("\n====================================================")
    print("🎉 BOOM! SCRAPING MANDIRI APPLE MUSIC SUKSES TOTAL!")
    print(f"Berhasil mengamankan {len(df_filtered)} baris data ulasan riil.")
    print("File 'raw_ulasan_applemusic.csv' resmi lahir di foldermu!")
    print("====================================================")
    print(df_filtered[['score', 'content']].head(3))
else:
    print("\nGagal menarik data. Pastikan koneksi internet di laptopmu lancar.")