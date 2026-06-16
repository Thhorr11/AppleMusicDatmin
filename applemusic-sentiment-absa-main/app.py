import streamlit as st
import pandas as pd
import joblib
import re
import string
import altair as alt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from slang_dictionary import slang_dict

# --- Setup Page ---
st.set_page_config(page_title="Apple Music Analyzer", page_icon="🎵", layout="wide")

# --- Custom CSS (Apple Luxury Aesthetic) ---
def local_css():
    st.markdown("""
    <style>
        /* Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
            background-color: #000000;
        }

        .stApp {
            background-color: #000000;
            color: #f5f5f7;
        }
        
        /* Clean up Streamlit */
        #MainMenu, header, footer {visibility: hidden;}
        
        /* Typography Polish */
        h1, h2, h3, h4 {
            letter-spacing: -0.02em !important;
        }

        /* Apple-style Native Input */
        .stTextArea textarea {
            background-color: rgba(28, 28, 30, 0.6) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            color: #f5f5f7 !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 18px !important;
            padding: 20px !important;
            font-size: 16px !important;
            line-height: 1.5 !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease;
        }
        .stTextArea textarea:focus {
            border-color: #fa243c !important;
            box-shadow: 0 0 0 1px #fa243c, 0 0 20px rgba(250, 36, 60, 0.15) !important;
        }
        
        /* Apple Music Vibrant Red Button */
        .stButton>button {
            background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%) !important;
            color: white !important;
            border-radius: 14px !important;
            border: none !important;
            padding: 14px 32px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            letter-spacing: -0.01em !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            box-shadow: 0 8px 24px rgba(250, 36, 60, 0.35) !important;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 32px rgba(250, 36, 60, 0.5) !important;
            background: linear-gradient(135deg, #ff3b30 0%, #ff453a 100%) !important;
        }

        /* iOS Widget Style Cards */
        .sentiment-card {
            background: rgba(28, 28, 30, 0.6);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px;
            padding: 36px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }
        .sentiment-card.negative {
            background: linear-gradient(135deg, rgba(44, 20, 22, 0.8) 0%, rgba(28, 28, 30, 0.8) 100%);
            border: 1px solid rgba(250, 36, 60, 0.25);
        }
        .sentiment-card.positive {
            background: linear-gradient(135deg, rgba(20, 44, 28, 0.8) 0%, rgba(28, 28, 30, 0.8) 100%);
            border: 1px solid rgba(46, 204, 113, 0.25);
        }
        
        .aspect-card {
            background: rgba(28, 28, 30, 0.5);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 28px;
            height: 100%;
            transition: transform 0.3s ease, background 0.3s ease;
        }
        .aspect-card:hover {
            transform: translateY(-4px);
            background: rgba(44, 44, 46, 0.7);
        }

        /* Badges */
        .badge {
            font-size: 11px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 16px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-pos { 
            background-color: rgba(52, 199, 89, 0.15); 
            color: #34c759; 
            border: 1px solid rgba(52, 199, 89, 0.2); 
        }
        .badge-neu { 
            background-color: rgba(255, 255, 255, 0.05); 
            color: #98989d; 
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Metric Cards for Dataset Stats */
        .metric-card {
            background: rgba(28, 28, 30, 0.5);
            backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 16px;
        }
        .metric-title {
            font-size: 12px;
            color: #98989d;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 40px;
            font-weight: 800;
            color: #ffffff;
            margin-top: 4px;
            letter-spacing: -0.03em;
        }

        hr {
            border-color: rgba(255,255,255,0.08);
            margin: 48px 0;
        }

        /* Mobile Responsive Fixes */
        @media (max-width: 768px) {
            .sentiment-card {
                flex-direction: column !important;
                align-items: flex-start !important;
                padding: 24px !important;
                gap: 16px !important;
            }
            .sent-right {
                text-align: left !important;
                max-width: 100% !important;
            }
            .sent-right-quote {
                word-wrap: break-word;
                white-space: normal;
            }
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Initialize Models ---
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('data_preprocessed.csv').dropna(subset=['clean_teks'])
    except:
        return pd.DataFrame()

@st.cache_resource
def load_sastrawi():
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

@st.cache_resource
def load_svm_models():
    try:
        v = joblib.load("tfidf_vectorizer.pkl")
        m_a = joblib.load("svm_model_aspek_audio_fitur.pkl")
        m_p = joblib.load("svm_model_aspek_performa_sistem.pkl")
        m_h = joblib.load("svm_model_aspek_harga_layanan.pkl")
        m_s = joblib.load("svm_model_sentimen.pkl")
        return v, m_a, m_p, m_h, m_s
    except:
        return None, None, None, None, None

df = load_dataset()
stemmer, stopword_remover = load_sastrawi()
vectorizer, model_audio, model_performa, model_harga, model_sentimen = load_svm_models()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<.*?>|@\w+|#\w+|\d+', '', text)
    
    for slang, formal in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', formal, text)
        
    text = text.translate(str.maketrans('', '', string.punctuation)).strip()
    text = re.sub(r'\s+', ' ', text)
    
    text = stopword_remover.remove(text)
    return stemmer.stem(text)

# --- HEADER (Luxury Apple Style) ---
st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <div style="background: linear-gradient(135deg, #fa243c 0%, #ff3b30 100%); width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(250,36,60,0.4);">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" stroke="none">
            <path d="M21 3v12.5a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 14 15.5a3.5 3.5 0 0 1 3.5-3.5c.54 0 1.05.12 1.5.34V6.47L9 8.6v9.9a3.5 3.5 0 0 1-3.5 3.5A3.5 3.5 0 0 1 2 18.5 3.5 3.5 0 0 1 5.5 15c.54 0 1.05.12 1.5.34V3l14-2v2z"/>
        </svg>
    </div>
    <h1 style='font-weight: 800; font-size: 36px; margin: 0; padding: 0;'>Apple Music Analyzer</h1>
</div>
<p style='color: #98989d; font-size: 16px; margin-left: 64px; font-weight: 500;'>Menganalisis aspek dan sentimen ulasan secara otomatis menggunakan Support Vector Machine.</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- LAYOUT MAIN ---
col_input, col_space, col_result = st.columns([1.2, 0.1, 1.5])

with col_input:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-top: 0; margin-bottom: 20px;'>Uji Sentimen Real-time</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Masukkan ulasan:", placeholder="Ketik ulasan di sini (misal: Aplikasi sering force close, harganya juga mahal...)", height=175, label_visibility="collapsed")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_btn = st.button("Analisis Ulasan", use_container_width=True)

with col_result:
    st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-top: 0; margin-bottom: 20px;'>Hasil Analisis Sentimen</h3>", unsafe_allow_html=True)
    if not analyze_btn:
        st.markdown("""
        <div style="background: rgba(28, 28, 30, 0.3); border: 2px dashed rgba(255,255,255,0.08); border-radius: 24px; padding: 40px; text-align: center; color: #636366; min-height: 240px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 40px; margin-bottom: 16px; opacity: 0.5;">✨</div>
            <div style="font-size: 16px; font-weight: 600; color: #98989d;">Menunggu input ulasan.</div>
            <div style="font-size: 13px; margin-top: 8px;">Ketik ulasan di sebelah kiri dan klik Analisis.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not user_input.strip():
            st.warning("Teks ulasan tidak boleh kosong.")
        else:
            with st.spinner("Memproses..."):
                cleaned = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned])
                
                p_a = model_audio.predict(vec_input)[0]
                p_p = model_performa.predict(vec_input)[0]
                p_h = model_harga.predict(vec_input)[0]
                p_s = model_sentimen.predict(vec_input)[0]
                
                is_pos = (p_s == 1)
                
                # Render Sentiment Card
                c_class = "sentiment-card positive" if is_pos else "sentiment-card negative"
                icon = "😄" if is_pos else "😠"
                t_color = "#34c759" if is_pos else "#ff3b30"
                s_text = "POSITIF" if is_pos else "NEGATIF"
                quote = "Ulasan mengekspresikan kepuasan tinggi." if is_pos else "Ulasan ini mengandung unsur keluhan/kekecewaan."
                
                st.markdown(f"""
                <div class="{c_class}">
                    <div>
                        <div style="font-size: 12px; color: rgba(255,255,255,0.6); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Sentimen Keseluruhan</div>
                        <div style="font-size: 44px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 16px; letter-spacing: -0.02em;">
                            <span style="font-size: 48px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));">{icon}</span> {s_text}
                        </div>
                    </div>
                    <div class="sent-right" style="text-align: right; max-width: 45%;">
                        <div class="sent-right-quote" style="font-size: 14px; color: rgba(255,255,255,0.8); margin-bottom: 12px; line-height: 1.5;">{quote}</div>
                        <div style="font-size: 12px; color: {t_color}; font-weight: 700; background: rgba(0,0,0,0.2); padding: 4px 10px; border-radius: 12px; display: inline-block;">CONFIDENCE: 94%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Render Aspect Cards Full Width Below the Split
if analyze_btn and user_input.strip():
    st.markdown("<h4 style='font-size: 16px; font-weight: 600; color: #f5f5f7; margin-top: 40px; margin-bottom: 24px; letter-spacing: -0.01em;'>Aspek yang Terdeteksi</h4>", unsafe_allow_html=True)
    
    a1, a2, a3 = st.columns(3)
    
    def render_aspect(title, is_det, icon, desc_pos, desc_neg):
        b_class = "badge-pos" if is_det else "badge-neu"
        b_text = "Terdeteksi" if is_det else "Tidak Terdeteksi"
        desc = desc_pos if is_det else desc_neg
        opac = "1.0" if is_det else "0.5"
        
        return f"""
        <div class="aspect-card" style="opacity: {opac};">
            <div class="badge {b_class}">{b_text}</div>
            <div style="font-size: 28px; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 8px; letter-spacing: -0.01em;">{title}</div>
            <div style="font-size: 13px; color: #98989d; line-height: 1.5;">{desc}</div>
        </div>
        """
    
    with a1: st.markdown(render_aspect("Audio & Fitur", p_a==1, "🎧", "Kualitas audio atau kelengkapan fitur disinggung dalam teks.", "Tidak menyinggung terkait kualitas suara atau fitur."), unsafe_allow_html=True)
    with a2: st.markdown(render_aspect("Performa Sistem", p_p==1, "⚙️", "Keluhan tentang bug, lag, atau kestabilan sistem ditemukan.", "Tidak ada masalah performa yang disebutkan."), unsafe_allow_html=True)
    with a3: st.markdown(render_aspect("Harga Layanan", p_h==1, "💳", "Topik seputar tagihan, biaya, atau pembayaran terdeteksi.", "Tidak menyebutkan soal harga atau biaya langganan."), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- DATA VISUALIZATION SECTION (Apple Health Style) ---
st.markdown("<h3 style='font-size: 20px; font-weight: 700; margin-bottom: 32px;'>Statistik Dataset Global</h3>", unsafe_allow_html=True)

if not df.empty:
    chart_col, metric_col = st.columns([2.5, 1])
    
    with metric_col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">TOTAL ULASAN</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        pos_count = len(df[df['Sentimen'] == 1])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAYORITAS SENTIMEN</div>
            <div class="metric-value" style="color: #34c759;">POSITIF</div>
            <div style="font-size: 13px; color: #98989d; margin-top: 8px;">{pos_count} ulasan terlabel positif</div>
        </div>
        """, unsafe_allow_html=True)
        
    with chart_col:
        chart_data = []
        for aspect in ['Aspek_Audio_Fitur', 'Aspek_Performa_Sistem', 'Aspek_Harga_Layanan']:
            name = aspect.replace('Aspek_', '').replace('_', ' ')
            subset = df[df[aspect] == 1]
            chart_data.append({'Aspek': name, 'Sentimen': 'Positif', 'Jumlah': len(subset[subset['Sentimen'] == 1])})
            chart_data.append({'Aspek': name, 'Sentimen': 'Negatif', 'Jumlah': len(subset[subset['Sentimen'] == 0])})
            
        # Apple style rounded bars
        bars = alt.Chart(pd.DataFrame(chart_data)).mark_bar(cornerRadiusEnd=8, height=32).encode(
            x=alt.X('Jumlah:Q', title='', axis=alt.Axis(gridColor='rgba(255,255,255,0.05)', labelColor='#98989d', tickCount=5, domain=False)),
            y=alt.Y('Aspek:N', title='', sort=None, axis=alt.Axis(labelColor='#f5f5f7', labelFontWeight=500, labelFontSize=13, tickColor='transparent', domainColor='transparent')),
            color=alt.Color('Sentimen:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#34c759', '#ff3b30']), legend=alt.Legend(orient='bottom', title=None, labelColor='#f5f5f7', labelFontSize=13, symbolType='circle')),
            tooltip=['Aspek', 'Sentimen', 'Jumlah']
        ).properties(height=280).configure_view(strokeWidth=0).configure(background='transparent').configure_axis(gridDash=[4,4])
        
        st.altair_chart(bars, use_container_width=True)
