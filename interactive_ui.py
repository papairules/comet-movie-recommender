import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from movie_model import (
    movie_similarity_df,
    recommend_movies,
    predict_sentiment,
    mock_reviews,
    rf_model,
    xgb_model,
    mlb,
    feature_cols
)

# -------------------------
# 🎨 Page config and style
# -------------------------
st.set_page_config(layout="wide")

def get_base64_img(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_img("background.png")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baguet+Script&display=swap');

    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    .custom-title {{
        font-family: 'Baguet Script', cursive;
        font-size: 65px;
        color: #975923;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
    }}

    .recommend-header {{
        font-weight: 700;
        color: white;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }}

    .stSelectbox, .stSlider, .stMultiSelect, .stCheckbox, .stTextInput {{
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 10px;
        padding: 10px;
    }}

    label, .stMarkdown, .stTextInput > div > div, .stMultiSelect label {{
        color: #000000 !important;
        font-weight: 600;
    }}

    .stButton>button {{
        background-color: #f63366;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }}

    .stDataFrame {{
        background-color: white;
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 🎬 App Title
# -------------------------
st.markdown("<div class='custom-title'>Comet Movie Recommendation</div>", unsafe_allow_html=True)

# -------------------------
# 📦 Load Data
# -------------------------
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings_small.csv", usecols=["movieId", "rating"])

# Genre preprocessing
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|') if pd.notnull(x) else [])
all_genres = sorted(set(g for genre_list in movies_df['genres'] for g in genre_list))

# -------------------------
# 🎛️ Filters (Main layout)
# -------------------------
col1, col2 = st.columns([1.4, 2.5])

with col1:
    st.markdown("""
    <div style='font-size:24px; font-weight:700; color:#975923; margin-bottom:10px;'>
        🎛️ Filter Your Preferences
    </div>
""", unsafe_allow_html=True)
    selected_genres = st.multiselect("Preferred Genres", options=all_genres)
    year_range = st.slider("Release Year Range", 1950, 2025, (1990, 2020))
    min_rating = st.slider("Minimum Average Rating", 0.0, 5.0, 3.0, step=0.1)
    rewatch_pref = st.checkbox("Include only rewatchable classics (≥ 4.0 rating)")

with col2:
    st.markdown("<div class='recommend-header'>🎯 Pick a Movie to Get Recommendations</div>", unsafe_allow_html=True)
    selected_movie = st.selectbox("Select a movie:", movie_similarity_df.columns.sort_values())
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend 🎉"):
        recommendations = recommend_movies(selected_movie, movie_similarity_df, top_n=top_n)

        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            rows = []
            for movie, score in recommendations.items():
                meta = movies_df[movies_df['title'] == movie]
                movie_genres = meta['genres'].values[0] if not meta.empty else []
                release_year = int(re.search(r'\((\d{4})\)', movie).group(1)) if re.search(r'\((\d{4})\)', movie) else 2000
                movie_id = meta['movieId'].values[0] if not meta.empty else None

                # Filtering logic
                if selected_genres and not any(g in movie_genres for g in selected_genres):
                    continue
                if not (year_range[0] <= release_year <= year_range[1]):
                    continue

                avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean() if movie_id else None
                if rewatch_pref and (avg_rating is None or avg_rating < 4.0):
                    continue
                if not rewatch_pref and (avg_rating is None or avg_rating < min_rating):
                    continue

                # Sentiment analysis
                review = mock_reviews.get(movie, "This movie was okay.")
                sentiment = predict_sentiment(review)

                # Correct feature vector using mlb.classes_
                user_activity = 30  # Static mock values
                user_avg_rating = 3.8
                genre_vector = [1 if g in movie_genres else 0 for g in mlb.classes_]
                like_features = genre_vector + [release_year, user_activity, user_avg_rating]

                rf_like = rf_model.predict([like_features])[0]
                xgb_like = xgb_model.predict([like_features])[0]

                rows.append({
                    "Movie": movie,
                    "Similarity Score": round(score, 3),
                    "Avg Rating": round(avg_rating, 2) if avg_rating else "N/A",
                    "Genres": ', '.join(movie_genres),
                    "Sentiment": sentiment,
                    "Sample Review": review,
                    "RF Like Prediction": "👍" if rf_like else "👎",
                    "XGB Like Prediction": "👍" if xgb_like else "👎"
                })

            if rows:
                st.markdown("### 📝 Filtered Recommendations")
                st.dataframe(pd.DataFrame(rows))
            else:
                st.warning("⚠️ No movies matched your filters. Try relaxing them.")
