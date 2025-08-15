import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import gdown
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import re

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="🎬 Film Öneri Sistemi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri - IMDb temasına uygun, sarı ve siyah tonlarda
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: #F5C518;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #FFFFFF;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    h1, h2, h3, h4, h5, h6, .st-b5, .st-b6, .st-b7, .st-b8 {
        color: #F5C518;
    }
    
    /* Yan menü arkaplanı */
    [data-testid="stSidebar"] {
        background-color: #121212;
    }
    
    /* Metin kutuları ve düğmeler için daha belirgin stiller */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stButton>button {
        background-color: #2c2c2c;
        border: 1px solid #444444;
        color: #F5C518;
    }
    
    .stButton>button:hover {
        background-color: #3e3e3e;
        border-color: #F5C518;
    }
    
    /* Öneri kartları - daha şık bir tasarım */
    .recommendation-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-left: 5px solid #F5C518;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(245, 197, 24, 0.1);
        display: flex;
        align-items: center;
    }
    
    .rec-rank {
        font-size: 2.5rem;
        font-weight: bold;
        color: #F5C518;
        margin-right: 1rem;
        width: 40px;
        text-align: center;
    }
    
    .rec-content {
        flex-grow: 1;
    }
    
    .rec-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    
    .rec-details {
        font-size: 0.9rem;
        color: #D3D3D3;
        margin-top: 0.5rem;
    }

    .st-emotion-cache-1px0v2t {
        background-color: #1a1a1a;
    }

    .st-emotion-cache-12w0qpk {
        color: #F5C518;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def download_data_from_drive(file_id):
    """Google Drive'dan veri setini indir ve önbelleğe al"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        output_file = "movies_imdb_2.csv"

        with st.spinner('📥 Veri seti indiriliyor... Bu işlem birkaç dakika sürebilir.'):
            if not os.path.exists(output_file):
                gdown.download(url, output_file, quiet=False)
        
        st.success("✅ Veri seti başarıyla hazırlandı!")
        return output_file

    except Exception as e:
        st.error(f"❌ Veri seti indirilemedi: {str(e)}")
        return None

def weighted_rating(rating, votes, min_votes, mean_rating):
    """Ağırlıklı derecelendirme hesapla (IMDb formatında)"""
    denominator = votes + min_votes
    if denominator == 0:
        return 0
    return (votes / denominator) * rating + (min_votes / denominator) * mean_rating

def normalize_title(title):
    """Film başlıklarını normalleştir - gelişmiş versiyon"""
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', title.lower())
        if unicodedata.category(c) != 'Mn'
    )
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def advanced_movie_search(query, movie_titles, top_n=5):
    """Gelişmiş film arama sistemi - TF-IDF tabanlı"""
    if not query.strip() or len(query) < 2:
        return []
    
    normalized_titles = [normalize_title(title) for title in movie_titles]
    normalized_query = normalize_title(query)
    
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words=None,
        lowercase=True,
        max_features=10000
    )
    
    try:
        all_texts = [normalized_query] + normalized_titles
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        query_vector = tfidf_matrix[0:1]
        title_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, title_vectors).flatten()
        
        similar_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] > 0.2:  # Minimum benzerlik eşiği artırıldı
                results.append({
                    'title': movie_titles[idx],
                    'similarity': similarities[idx],
                    'normalized': normalized_titles[idx]
                })
        
        return results[:top_n]
    
    except Exception as e:
        st.warning(f"Arama hatası: {e}. Basit eşleşmeye dönülüyor.")
        query_lower = normalized_query
        matches = []
        for i, title in enumerate(normalized_titles):
            if query_lower in title:
                matches.append({
                    'title': movie_titles[i],
                    'similarity': len(query_lower) / len(title),
                    'normalized': title
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)[:top_n]


@st.cache_data(ttl=3600)
def prepare_data(filepath, vote_threshold=1000, min_votes=2500):
    """Veri setini hazırla ve öneri sistemi için işle"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text('📊 Veri seti yükleniyor...')
        progress_bar.progress(10)
        
        df = pd.read_csv(filepath)
        
        status_text.text('🔧 Veri temizleniyor...')
        progress_bar.progress(30)
        
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        df["YEAR"] = df["YEAR"].astype(int)
        df["RATING_10"] = df["RATING"] * 2
        
        status_text.text('📈 İstatistikler hesaplanıyor...')
        progress_bar.progress(50)
        
        vote_counts = df.groupby("TITLE")["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        
        mean_rating = df["RATING_10"].mean()
        
        movie_stats = df.groupby("TITLE").agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max"
        }).reset_index()
        
        movie_stats["IMDB_SCORE"] = movie_stats.apply(
            lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
            axis=1
        )
        
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        
        status_text.text('🎯 Öneri matrisi oluşturuluyor...')
        progress_bar.progress(70)
        
        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID",
            columns="TITLE",
            values="RATING_10",
            aggfunc='mean'
        ).fillna(0)
        
        status_text.text('🔄 Benzerlik matrisi hesaplanıyor...')
        progress_bar.progress(90)
        
        movie_similarity_df = pd.DataFrame(
            cosine_similarity(user_movie_matrix.T),
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        progress_bar.progress(100)
        status_text.text('✅ Veri hazırlama tamamlandı!')
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return df, df_filtered, user_movie_matrix, movie_similarity_df
        
    except Exception as e:
        st.error(f"❌ Veri hazırlanırken hata oluştu: {str(e)}")
        return None, None, None, None

def recommend_by_title(title, similarity_df, df, top_n=5):
    """Film başlığına göre öneri yap - gelişmiş arama ile"""
    available_movies = list(similarity_df.columns)
    
    search_results = advanced_movie_search(title, available_movies, top_n=1)
    
    if not search_results:
        return None, []
    
    best_match = search_results[0]['title']
    
    if best_match not in similarity_df.columns:
        return None, []
    
    scores = similarity_df[best_match].drop(labels=[best_match], errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(top_n)
    
    rec_data = []
    for movie, similarity_score in recommendations.items():
        movie_info = df[df["TITLE"] == movie].iloc[0]
        rec_data.append({
            "Film": movie,
            "Benzerlik Skoru": f"{similarity_score:.3f}",
            "IMDb Skoru": f"{movie_info['IMDB_SCORE']:.2f}",
            "Yıl": int(movie_info["YEAR"]),
            "Türler": movie_info["GENRES"]
        })
    
    return rec_data, best_match

def get_top_movies_by_year(df, year, top_n=10):
    """Yıla göre en iyi filmleri getir"""
    year_movies = df[df['YEAR'] == year]
    if year_movies.empty:
        return []
    
    top = year_movies.groupby(['TITLE', 'GENRES', 'YEAR'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def get_top_movies_by_genre(df, genre, top_n=10):
    """Türe göre en iyi filmleri getir"""
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    
    top = genre_movies.groupby(['TITLE', 'YEAR', 'GENRES'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def display_recommendations(rec_list):
    """Öneri listesini şık bir kart formatında gösterir"""
    for i, rec in enumerate(rec_list):
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="rec-rank">{i+1}</div>
            <div class="rec-content">
                <div class="rec-title">{rec['Film']} ({rec['Yıl']})</div>
                <div class="rec-details">
                    <b>IMDb Skoru:</b> {rec['IMDb Skoru']} | 
                    <b>Türler:</b> {rec['Türler']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🎬 Film Öneri Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">26M+ film verisi ile kişiselleştirilmiş öneriler</p>', unsafe_allow_html=True)
    
    FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
    
    if 'data_loaded' not in st.session_state:
        filepath = download_data_from_drive(FILE_ID)
        
        if filepath is not None:
            df, df_filtered, user_movie_matrix, movie_similarity_df = prepare_data(filepath)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.df_filtered = df_filtered
                st.session_state.user_movie_matrix = user_movie_matrix
                st.session_state.movie_similarity_df = movie_similarity_df
                st.session_state.data_loaded = True
            else:
                st.error("❌ Veri hazırlanamadı. Lütfen sayfayı yenileyin.")
                return
        else:
            st.error("❌ Veri indirilemedi. Lütfen internet bağlantınızı kontrol edin.")
            return
    
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    movie_similarity_df = st.session_state.movie_similarity_df
    
    with st.sidebar:
        st.header("📊 Veri Seti İstatistikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
            st.metric("Toplam Kullanıcı", f"{df['USERID'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
            st.metric("Ortalama IMDb Skoru", f"{df['IMDB_SCORE'].mean():.2f}")
        
        st.subheader("📅 Yıllara Göre Film Sayısı")
        year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
        fig = px.line(year_counts, x='YEAR', y='TITLE',
                      title='Yıllara Göre Film Sayısı',
                      labels={'YEAR': 'Yıl', 'TITLE': 'Film Sayısı'})
        fig.update_layout(height=300, plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#D3D3D3')
        fig.update_traces(line_color='#F5C518')
        st.plotly_chart(fig, use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Film Bazlı Öneriler",
        "📅 Yıla ve Türe Göre Öneriler",
        "🔍 Veri Keşfi"
    ])
    
    with tab1:
        st.header("🎯 Film Bazlı Öneriler")
        st.write("Sevdiğiniz bir filmi yazın, benzer filmleri önereceğiz! Gelişmiş arama sistemi ile tam eşleşme gerekmiyor.")
        
        movie_input = st.text_input("Film Adı:", placeholder="Örnek: Shawshank, Matrix, Batman...", key="movie_input_1")
        
        if movie_input:
            search_results = advanced_movie_search(movie_input, list(movie_similarity_df.columns), top_n=5)
            
            if search_results:
                st.subheader("Bunu mu kastettiniz?")
                selected_movie_option = st.radio(
                    "Lütfen listeden bir film seçin:",
                    [result['title'] for result in search_results]
                )
                
                num_recommendations = st.selectbox("Öneri Sayısı:", [5, 10, 15, 20], index=0)
                
                if st.button("🎬 Öneri Al", type="primary"):
                    recommendations, best_match = recommend_by_title(
                        selected_movie_option, movie_similarity_df, df, num_recommendations
                    )
                    
                    if recommendations:
                        st.markdown(f'<h3 style="color:#F5C518;">"{best_match}" filmine göre önerileriniz:</h3>', unsafe_allow_html=True)
                        display_recommendations(recommendations)
                    else:
                        st.warning(f"⚠️ '{selected_movie_option}' için benzer film bulunamadı.")
            else:
                st.warning("⚠️ Aradığınız filmi bulamadık. Lütfen farklı bir arama terimi deneyin.")
                
    with tab2:
        st.header("📅 Yıla ve Türe Göre En İyi Filmler")
        st.write("Özel kriterlerinize göre en iyi filmleri keşfedin.")
        
        years = sorted(df['YEAR'].unique(), reverse=True)
        all_genres = set()
        for genres in df['GENRES'].dropna():
            all_genres.update([g.strip() for g in genres.split('|')])
        genre_options = sorted(list(all_genres))
        
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Yıl seçin:", years, key="year_select")
        with col2:
            selected_genre = st.selectbox("Tür seçin:", ["Tümü"] + genre_options, key="genre_select")
        
        num_rec_tab2 = st.selectbox("Kaç film gösterilsin:", [5, 10, 15, 20, 25], index=1, key="num_rec_tab2")
        
        if st.button("🌟 En İyileri Göster", type="primary", key="show_best_button"):
            results_df = df_filtered.copy()
            
            # Yıl filtresi
            results_df = results_df[results_df['YEAR'] == selected_year]
            
            # Tür filtresi
            if selected_genre != "Tümü":
                results_df = results_df[results_df["GENRES"].str.contains(selected_genre, case=False, na=False)]
            
            if results_df.empty:
                st.error(f"❌ Seçilen kriterlere uygun film bulunamadı.")
            else:
                # IMDb skorlarına göre sırala ve en iyileri al
                top_movies_list = results_df.groupby(['TITLE', 'YEAR', 'GENRES'])['IMDB_SCORE'].mean().reset_index()
                top_movies_list = top_movies_list.sort_values('IMDB_SCORE', ascending=False).head(num_rec_tab2)
                
                # Sütun isimlerini düzenle
                top_movies_list = top_movies_list.rename(columns={'IMDB_SCORE': 'IMDb Skoru'})
                
                st.markdown(f'<h3 style="color:#F5C518;">Seçilen kriterlere göre en iyi filmler:</h3>', unsafe_allow_html=True)
                
                # Tablo yerine kartları kullan
                rec_data_list = top_movies_list.to_dict('records')
                display_recommendations(rec_data_list)
                
                # Grafik oluştur
                fig = px.bar(top_movies_list, x='TITLE', y='IMDb Skoru',
                             title='En İyi Filmlerin IMDb Skorları',
                             labels={'TITLE': 'Film Adı'},
                             color='IMDb Skoru', color_continuous_scale='YlOrRd')
                fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#D3D3D3')
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🔍 Veri Keşfi ve Analiz")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
        with col3:
            st.metric("Ortalama Rating", f"{df['RATING'].mean():.2f}")
        with col4:
            st.metric("En Son Yıl", f"{df['YEAR'].max()}")
        
        st.subheader("Veri Görselleştirmeleri")
        chart_type = st.selectbox("Gösterilecek grafik:", [
            "En Çok Değerlendirilen Filmler",
            "Yıllara Göre Film Sayısı",
            "En Popüler Türler",
            "Rating Dağılımı"
        ])
        
        if chart_type == "En Çok Değerlendirilen Filmler":
            top_rated = df['TITLE'].value_counts().head(20)
            fig = px.bar(x=top_rated.values, y=top_rated.index,
                         title='En Çok Değerlendirilen 20 Film',
                         labels={'x': 'Değerlendirme Sayısı', 'y': 'Film'},
                         orientation='h', color_discrete_sequence=['#F5C518'])
            fig.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#D3D3D3')
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Yıllara Göre Film Sayısı":
            year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
            fig = px.area(year_counts, x='YEAR', y='TITLE',
                          title='Yıllara Göre Film Sayısı Trend',
                          labels={'YEAR': 'Yıl', 'TITLE': 'Film Sayısı'})
            fig.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#D3D3D3')
            fig.update_traces(line_color='#F5C518')
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "En Popüler Türler":
            genre_counts = {}
            for genres in df['GENRES'].dropna():
                for genre in genres.split('|'):
                    genre = genre.strip()
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Tür', 'Sayı'])
            genre_df = genre_df.sort_values('Sayı', ascending=False).head(15)
            
            fig = px.bar(genre_df, x='Sayı', y='Tür',
                         title='En Popüler 15 Film Türü',
                         orientation='h', color_discrete_sequence=['#F5C518'])
            fig.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#D3D3D3')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Rating Dağılımı":
            fig = px.histogram(df, x='RATING', nbins=50,
                               title='Rating Dağılımı (1-5 Skala)',
                               color_discrete_sequence=['#F5C518'])
            fig.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#D3D3D3')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Veri Seti Örneği")
        st.dataframe(df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
