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

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #7D8590;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def download_data_from_drive(file_id):
    """Google Drive'dan veri setini indir ve önbelleğe al"""
    try:
        # Google Drive URL'si
        url = f"https://drive.google.com/uc?id={file_id}"
        output_file = "movies_imdb_2.csv"
        
        with st.spinner('📥 Veri seti indiriliyor... Bu işlem birkaç dakika sürebilir.'):
            # Dosya zaten varsa, tekrar indirme
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
    # Unicode karakterleri kaldır
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', title.lower())
        if unicodedata.category(c) != 'Mn'
    )
    
    # Özel karakterleri kaldır ve temizle
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def advanced_movie_search(query, movie_titles, top_n=5):
    """Gelişmiş film arama sistemi - TF-IDF tabanlı"""
    if not query.strip():
        return []
    
    # Film başlıklarını normalleştir
    normalized_titles = [normalize_title(title) for title in movie_titles]
    normalized_query = normalize_title(query)
    
    # TF-IDF vektörleştirme
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words=None,
        lowercase=True,
        max_features=10000
    )
    
    try:
        # Tüm metinleri (sorgu + başlıklar) vektörleştir
        all_texts = [normalized_query] + normalized_titles
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Sorgu vektörü ile film başlıkları arasında benzerlik hesapla
        query_vector = tfidf_matrix[0:1]
        title_vectors = tfidf_matrix[1:]
        
        # Cosine similarity hesapla
        similarities = cosine_similarity(query_vector, title_vectors).flatten()
        
        # En benzer filmleri bul
        similar_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # Minimum benzerlik eşiği
                results.append({
                    'title': movie_titles[idx],
                    'similarity': similarities[idx],
                    'normalized': normalized_titles[idx]
                })
        
        return results
        
    except Exception as e:
        st.warning(f"Arama hatası: {e}")
        # Fallback: basit string matching
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
        
        # CSV'yi oku
        df = pd.read_csv(filepath)
        
        status_text.text('🔧 Veri temizleniyor...')
        progress_bar.progress(30)
        
        # Başlık ve yıl bilgisini ayır
        df[["TITLE", "YEAR"]] = df["TITLE"].str.extract(r"^(.*) \((\d{4})\)$")
        
        # Zaman bilgisini datetime'a çevir
        df["TIME"] = pd.to_datetime(df["TIME"], dayfirst=True, errors='coerce')
        
        # Eksik verileri temizle
        df.dropna(subset=["TITLE", "YEAR", "TIME", "RATING"], inplace=True)
        
        # Yılı integer'a çevir
        df["YEAR"] = df["YEAR"].astype(int)
        
        # Derecelendirmeyi 10'luk sisteme çevir
        df["RATING_10"] = df["RATING"] * 2
        
        status_text.text('📈 İstatistikler hesaplanıyor...')
        progress_bar.progress(50)
        
        # Film başına oy sayılarını hesapla
        vote_counts = df.groupby("TITLE")["RATING"].count()
        df["NUM_VOTES"] = df["TITLE"].map(vote_counts)
        
        # Ortalama derecelendirme
        mean_rating = df["RATING_10"].mean()
        
        # Film istatistiklerini topla
        movie_stats = df.groupby("TITLE").agg({
            "RATING_10": "mean",
            "NUM_VOTES": "max"
        }).reset_index()
        
        # Ağırlıklı IMDb skorlarını hesapla
        movie_stats["IMDB_SCORE"] = movie_stats.apply(
            lambda x: weighted_rating(x["RATING_10"], x["NUM_VOTES"], min_votes, mean_rating),
            axis=1
        )
        
        # Skorları ana veri çerçevesine ekle
        df["IMDB_SCORE"] = df["TITLE"].map(movie_stats.set_index("TITLE")["IMDB_SCORE"])
        
        status_text.text('🎯 Öneri matrisi oluşturuluyor...')
        progress_bar.progress(70)
        
        # Popüler filmleri filtrele
        popular_titles = vote_counts[vote_counts >= vote_threshold].index
        df_filtered = df[df["TITLE"].isin(popular_titles)].copy()
        
        # Kullanıcı-film derecelendirme matrisi
        user_movie_matrix = df_filtered.pivot_table(
            index="USERID",
            columns="TITLE",
            values="RATING_10",
            aggfunc='mean'
        ).fillna(0)
        
        status_text.text('🔄 Benzerlik matrisi hesaplanıyor...')
        progress_bar.progress(90)
        
        # Film benzerlik matrisi (cosine similarity)
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
    
    # Gelişmiş arama kullan
    search_results = advanced_movie_search(title, available_movies, top_n=10)
    
    if not search_results:
        return None, []
    
    # En iyi eşleşmeyi al
    best_match = search_results[0]['title']
    
    # Bu filmin benzerlik skorlarını al
    if best_match not in similarity_df.columns:
        return None, [result['title'] for result in search_results[:5]]
    
    scores = similarity_df[best_match].drop(labels=[best_match], errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(top_n)
    
    # Film bilgilerini ekle
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
    
    top = year_movies.groupby(['TITLE', 'GENRES'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def get_top_movies_by_genre(df, genre, top_n=10):
    """Türe göre en iyi filmleri getir"""
    genre_movies = df[df["GENRES"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []
    
    top = genre_movies.groupby(['TITLE', 'YEAR'])['IMDB_SCORE'].mean().reset_index()
    top = top.sort_values('IMDB_SCORE', ascending=False).head(top_n)
    
    return top.to_dict('records')

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🎬 Film Öneri Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">26M+ film verisi ile kişiselleştirilmiş öneriler</p>', unsafe_allow_html=True)
    
    # Google Drive dosya ID'si
    FILE_ID = "1gl_iJXRyEaSzhHlgfBUdTzQZMer4gdsS"
    
    # Veri setini indir ve hazırla
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
    
    # Veri setini session state'den al
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
    user_movie_matrix = st.session_state.user_movie_matrix
    movie_similarity_df = st.session_state.movie_similarity_df
    
    # Sidebar - İstatistikler
    with st.sidebar:
        st.header("📊 Veri Seti İstatistikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
            st.metric("Toplam Kullanıcı", f"{df['USERID'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
            st.metric("Ortalama IMDb Skoru", f"{df['IMDB_SCORE'].mean():.2f}")
        
        # Yıl dağılımı grafiği
        st.subheader("📅 Yıllara Göre Film Sayısı")
        year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
        fig = px.line(year_counts, x='YEAR', y='TITLE', 
                     title='Yıllara Göre Film Sayısı')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Ana içerik
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Film Bazlı Öneriler", 
        "📅 Yıla Göre En İyiler",
        "🎭 Türe Göre En İyiler",
        "🔍 Veri Keşfi"
    ])
    
    with tab1:
        st.header("🎯 Film Bazlı Öneriler")
        st.write("Sevdiğiniz bir filmi yazın, benzer filmleri önereceğiz! Gelişmiş arama sistemi ile tam eşleşme gerekmiyor.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_input = st.text_input("Film Adı:", placeholder="Örnek: Shawshank, Matrix, Batman...")
        with col2:
            num_recommendations = st.selectbox("Öneri Sayısı:", [5, 10, 15, 20], index=0)
        
        # Anlık arama önerileri
        if movie_input and len(movie_input) >= 2:
            search_results = advanced_movie_search(movie_input, list(movie_similarity_df.columns), top_n=5)
            if search_results:
                st.write("**Bulunan filmler:**")
                for i, result in enumerate(search_results[:5]):
                    similarity_percent = int(result['similarity'] * 100)
                    st.write(f"{i+1}. {result['title']} (Benzerlik: {similarity_percent}%)")
        
        if st.button("🎬 Öneri Al", type="primary"):
            if movie_input:
                recommendations, match_or_alternatives = recommend_by_title(
                    movie_input, movie_similarity_df, df, num_recommendations
                )
                
                if recommendations is None:
                    if match_or_alternatives:
                        st.error("❌ Tam eşleşme bulunamadı. Şunları kastetmiş olabilir misiniz?")
                        for alt in match_or_alternatives:
                            st.write(f"• {alt}")
                    else:
                        st.error("❌ Hiç eşleşen film bulunamadı. Lütfen farklı bir arama terimi deneyin.")
                else:
                    st.success(f"✅ '{match_or_alternatives}' filmine göre öneriler:")
                    
                    # Önerileri tablo olarak göster
                    rec_df = pd.DataFrame(recommendations)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Benzerlik skoru grafiği
                    fig = px.bar(rec_df, x='Film', y='Benzerlik Skoru', 
                               title=f'{match_or_alternatives} - Benzerlik Skorları',
                               color='IMDb Skoru', color_continuous_scale='viridis')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Lütfen bir film adı girin.")
    
    with tab2:
        st.header("📅 Yıla Göre En İyi Filmler")
        
        years = sorted(df['YEAR'].unique(), reverse=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_year = st.selectbox("Yıl seçin:", years)
        with col2:
            year_num_rec = st.selectbox("Kaç film gösterilsin:", [5, 10, 15, 20, 25], index=1, key="year_num")
        
        if st.button("📅 Yılın En İyilerini Göster", type="primary"):
            top_movies = get_top_movies_by_year(df_filtered, selected_year, year_num_rec)
            
            if not top_movies:
                st.error(f"❌ {selected_year} yılı için film bulunamadı.")
            else:
                st.success(f"✅ {selected_year} yılının en iyi {len(top_movies)} filmi:")
                
                # Tablo olarak göster
                movies_df = pd.DataFrame(top_movies)
                movies_df = movies_df.rename(columns={
                    'TITLE': 'Film',
                    'IMDB_SCORE': 'IMDb Skoru',
                    'GENRES': 'Türler'
                })
                st.dataframe(movies_df, use_container_width=True)
                
                # Grafik
                fig = px.bar(movies_df, x='Film', y='IMDb Skoru', 
                           title=f'{selected_year} Yılının En İyi {len(top_movies)} Filmi',
                           color='IMDb Skoru', color_continuous_scale='blues')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🎭 Türe Göre En İyi Filmler")
        
        # Mevcut türleri al
        all_genres = set()
        for genres in df['GENRES'].dropna():
            all_genres.update([g.strip() for g in genres.split('|')])
        
        popular_genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 
                         'Horror', 'Adventure', 'Animation', 'Crime', 'Mystery']
        
        available_popular = [g for g in popular_genres if g in all_genres]
        other_genres = sorted([g for g in all_genres if g not in popular_genres])
        
        genre_options = available_popular + other_genres
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_genre = st.selectbox("Tür seçin:", genre_options)
        with col2:
            genre_num_rec = st.selectbox("Kaç film gösterilsin:", [5, 10, 15, 20, 25], index=1, key="genre_num")
        
        if st.button("🎭 Türün En İyilerini Göster", type="primary"):
            top_movies = get_top_movies_by_genre(df_filtered, selected_genre, genre_num_rec)
            
            if not top_movies:
                st.error(f"❌ {selected_genre} türü için film bulunamadı.")
            else:
                st.success(f"✅ {selected_genre} türünün en iyi {len(top_movies)} filmi:")
                
                # Tablo olarak göster
                movies_df = pd.DataFrame(top_movies)
                movies_df = movies_df.rename(columns={
                    'TITLE': 'Film',
                    'YEAR': 'Yıl',
                    'IMDB_SCORE': 'IMDb Skoru'
                })
                st.dataframe(movies_df, use_container_width=True)
                
                # Grafik
                fig = px.scatter(movies_df, x='Yıl', y='IMDb Skoru', 
                               size='IMDb Skoru', hover_name='Film',
                               title=f'{selected_genre} Türü - En İyi {len(top_movies)} Film (Yıl ve IMDb Skoru)',
                               color='IMDb Skoru', color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔍 Veri Keşfi ve Analiz")
        
        # Genel istatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Film", f"{df['TITLE'].nunique():,}")
        with col2:
            st.metric("Toplam Değerlendirme", f"{len(df):,}")
        with col3:
            st.metric("Ortalama Rating", f"{df['RATING'].mean():.2f}")
        with col4:
            st.metric("En Son Yıl", f"{df['YEAR'].max()}")
        
        # Grafik seçenekleri
        chart_type = st.selectbox("Grafik türü seçin:", [
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
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Yıllara Göre Film Sayısı":
            year_counts = df.groupby('YEAR')['TITLE'].nunique().reset_index()
            fig = px.area(year_counts, x='YEAR', y='TITLE', 
                         title='Yıllara Göre Film Sayısı Trend')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "En Popüler Türler":
            # Türleri ayır ve say
            genre_counts = {}
            for genres in df['GENRES'].dropna():
                for genre in genres.split('|'):
                    genre = genre.strip()
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Tür', 'Sayı'])
            genre_df = genre_df.sort_values('Sayı', ascending=False).head(15)
            
            fig = px.bar(genre_df, x='Sayı', y='Tür', 
                        title='En Popüler 15 Film Türü',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Rating Dağılımı":
            fig = px.histogram(df, x='RATING', nbins=50, 
                             title='Rating Dağılımı (1-5 Skala)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Veri seti örneği
        st.subheader("📋 Veri Seti Örneği")
        st.dataframe(df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
