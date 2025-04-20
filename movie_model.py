import pandas as pd
import numpy as np
import re

# Collaborative Filtering
from sklearn.metrics.pairwise import cosine_similarity

# Sentiment Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Like Prediction Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

# -----------------------------
# 📦 Load & Process Movie Data
# -----------------------------
ratings = pd.read_csv("ratings_small.csv")
movies = pd.read_csv("movies.csv")

# Collaborative filtering filtering
df = pd.merge(ratings, movies, on='movieId')
popular_movies = df['title'].value_counts().head(300).index
active_users = df['userId'].value_counts().head(1000).index
df_small = df[df['title'].isin(popular_movies) & df['userId'].isin(active_users)]

user_movie_matrix = df_small.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix_filled = user_movie_matrix.fillna(0)

movie_ratings_matrix = user_movie_matrix_filled.T
movie_similarity = cosine_similarity(movie_ratings_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_ratings_matrix.index, columns=movie_ratings_matrix.index)

# -----------------------------
# 🧠 Sentiment Analysis (Logistic Regression)
# -----------------------------
imdb_df = pd.read_csv("IMDB_small.csv")
imdb_df['sentiment'] = imdb_df['sentiment'].map({'positive': 1, 'negative': 0})
imdb_df['clean_review'] = imdb_df['review'].apply(lambda x: re.sub(r'<.*?>', '', x.lower()))

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(imdb_df['clean_review'])
y_text = imdb_df['sentiment']

X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_text_train, y_text_train)

def predict_sentiment(review_text):
    cleaned = re.sub(r'<.*?>', '', review_text.lower())
    vector = vectorizer.transform([cleaned])
    pred = lr_model.predict(vector)[0]
    return "positive" if pred == 1 else "negative"

# -----------------------------
# 💬 Mock Reviews
# -----------------------------
mock_reviews = {
    "Toy Story 2 (1999)": "Even better than the first! Heartwarming and fun.",
    "Forrest Gump (1994)": "A timeless classic. Forrest is such a lovable character.",
    "Back to the Future (1985)": "An overrated mess. I couldn’t finish it.",
    "Jurassic Park (1993)": "Scary, thrilling, and visually stunning.",
    "Star Wars: Episode IV - A New Hope (1977)": "Way too slow and outdated for today's audience."
}

# -----------------------------
# 🔍 Recommendation Function
# -----------------------------
def recommend_movies(movie_title, similarity_df, top_n=5):
    if movie_title not in similarity_df.columns:
        return f"❌ Movie '{movie_title}' not found in similarity matrix."
    scores = similarity_df[movie_title].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# -----------------------------
# 🧪 Like Prediction Models (Random Forest & XGBoost)
# -----------------------------
# Sample and preprocess
ratings_sampled = ratings.sample(n=30000, random_state=42)
ratings_sampled = ratings_sampled.merge(movies, on='movieId', how='left')
ratings_sampled['liked'] = (ratings_sampled['rating'] >= 4).astype(int)

# Extract year
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else np.nan

ratings_sampled['year'] = ratings_sampled['title'].apply(extract_year)
ratings_sampled['user_activity'] = ratings_sampled.groupby('userId')['movieId'].transform('count')
ratings_sampled['user_avg_rating'] = ratings_sampled.groupby('userId')['rating'].transform('mean')

# Genre encoding
ratings_sampled['genres'] = ratings_sampled['genres'].apply(lambda x: str(x).split('|'))
mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(ratings_sampled['genres'])
genre_df = pd.DataFrame(genre_dummies, columns=mlb.classes_)
ratings_sampled = pd.concat([ratings_sampled.reset_index(drop=True), genre_df], axis=1)

# Final features
feature_cols = mlb.classes_.tolist() + ['year', 'user_activity', 'user_avg_rating']
ratings_sampled.dropna(subset=feature_cols, inplace=True)
X = ratings_sampled[feature_cols]
y = ratings_sampled['liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("✅ Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("✅ XGBoost Accuracy:", round(xgb_accuracy * 100, 2), "%")

# -----------------------------
# Export for Streamlit
# -----------------------------
__all__ = [
    "movie_similarity_df",
    "recommend_movies",
    "predict_sentiment",
    "mock_reviews",
    "rf_model",
    "xgb_model"
]
