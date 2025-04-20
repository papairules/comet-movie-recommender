import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------
# 🎥 Collaborative Filtering
# ---------------------
ratings = pd.read_csv("ratings_small.csv")
movies_df = pd.read_csv("movies.csv")
df = pd.merge(ratings, movies_df, on="movieId")

# Aggressive Filtering
popular_movies = df["title"].value_counts().head(300).index
active_users = df["userId"].value_counts().head(1000).index
df_small = df[df["title"].isin(popular_movies) & df["userId"].isin(active_users)]

# Pivot and similarity matrix
user_movie_matrix = df_small.pivot_table(index='userId', columns='title', values='rating').fillna(0)
movie_ratings_matrix = user_movie_matrix.T
movie_similarity = cosine_similarity(movie_ratings_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_ratings_matrix.index, columns=movie_ratings_matrix.index)

# Recommendation Function
def recommend_movies(movie_title, similarity_df, top_n=5):
    if movie_title not in similarity_df.columns:
        return f"❌ Movie '{movie_title}' not found."
    scores = similarity_df[movie_title].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# ---------------------
# 💬 Sentiment Analysis (Logistic Regression)
# ---------------------
imdb_df = pd.read_csv("IMDB_small.csv").dropna()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return text.lower()

imdb_df["clean_review"] = imdb_df["review"].apply(clean_text)
imdb_df["sentiment"] = imdb_df["sentiment"].map({"positive": 1, "negative": 0})

# TF-IDF + Logistic Regression
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_sent = tfidf_vectorizer.fit_transform(imdb_df["clean_review"])
y_sent = imdb_df["sentiment"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_s, y_train_s)

def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    vec = tfidf_vectorizer.transform([cleaned])
    pred = log_model.predict(vec)[0]
    return "positive" if pred == 1 else "negative"

# ---------------------
# 🔁 Like Prediction (Random Forest)
# ---------------------
sample_df = ratings.sample(n=30000, random_state=42).merge(movies_df, on="movieId", how="left")
sample_df["liked"] = (sample_df["rating"] >= 4).astype(int)

# Extract year
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else np.nan

sample_df["year"] = sample_df["title"].apply(extract_year)
sample_df["user_activity"] = sample_df.groupby("userId")["movieId"].transform("count")
sample_df["user_avg_rating"] = sample_df.groupby("userId")["rating"].transform("mean")

# Genre encoding
sample_df["genres"] = sample_df["genres"].apply(lambda x: str(x).split("|"))
mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb.fit_transform(sample_df["genres"]), columns=mlb.classes_)

sample_df = pd.concat([sample_df.reset_index(drop=True), genre_df], axis=1)
feature_cols = mlb.classes_.tolist() + ["year", "user_activity", "user_avg_rating"]
sample_df.dropna(subset=feature_cols, inplace=True)

X_like = sample_df[feature_cols]
y_like = sample_df["liked"]
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_like, y_like, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train_l, y_train_l)

def predict_like(user_features):
    """user_features should be a DataFrame with the same structure as X_like"""
    prediction = rf_model.predict(user_features)[0]
    return "liked" if prediction == 1 else "not liked"

# ---------------------
# 🎯 Mock Reviews
# ---------------------
mock_reviews = {
    "Toy Story 2 (1999)": "Even better than the first! Heartwarming and fun.",
    "Forrest Gump (1994)": "A timeless classic. Forrest is such a lovable character.",
    "Back to the Future (1985)": "An overrated mess. I couldn’t finish it.",
    "Jurassic Park (1993)": "Scary, thrilling, and visually stunning.",
    "Star Wars: Episode IV - A New Hope (1977)": "Way too slow and outdated for today's audience."
}

# Exported symbols
__all__ = [
    "movie_similarity_df",
    "recommend_movies",
    "predict_sentiment",
    "predict_like",
    "mock_reviews"
]
