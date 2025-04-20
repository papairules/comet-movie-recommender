import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# -----------------------------
# 📦 Load & Clean Ratings Data
# -----------------------------
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df = pd.merge(ratings, movies, on='movieId')

print("Unique users:", df['userId'].nunique())
print("Unique movies:", df['title'].nunique())
print("Total rows in merged dataframe:", len(df))

# ✅ Aggressive filtering to avoid memory issues
popular_movies = df['title'].value_counts().head(300).index
active_users = df['userId'].value_counts().head(1000).index
df_small = df[df['title'].isin(popular_movies) & df['userId'].isin(active_users)]
print("Shape after aggressive filtering:", df_small.shape)

# Create pivot table
user_movie_matrix = df_small.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# -----------------------------
# 🎯 Build Recommender
# -----------------------------
movie_ratings_matrix = user_movie_matrix_filled.T
movie_similarity = cosine_similarity(movie_ratings_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_ratings_matrix.index, columns=movie_ratings_matrix.index)

def recommend_movies(movie_title, similarity_df, top_n=5):
    if movie_title not in similarity_df.columns:
        return f"❌ Movie '{movie_title}' not found."
    scores = similarity_df[movie_title].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# -----------------------------
# 🧠 Load IMDb Reviews & Train Sentiment Model
# -----------------------------
imdb_df = pd.read_csv('IMDB Dataset.csv')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return text.lower()

imdb_df['clean_review'] = imdb_df['review'].apply(clean_text)
label_encoder = LabelEncoder()
imdb_df['sentiment_encoded'] = label_encoder.fit_transform(imdb_df['sentiment'])

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(imdb_df['clean_review'])
y = imdb_df['sentiment_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = nb_model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predict sentiment for a custom review
def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    vector = vectorizer.transform([cleaned])
    pred = nb_model.predict(vector)[0]
    return label_encoder.inverse_transform([pred])[0]

# -----------------------------
# 💬 Mock Reviews for Testing
# -----------------------------
mock_reviews = {
    "Toy Story 2 (1999)": "Even better than the first! Heartwarming and fun.",
    "Forrest Gump (1994)": "A timeless classic. Forrest is such a lovable character.",
    "Back to the Future (1985)": "An overrated mess. I couldn’t finish it.",
    "Jurassic Park (1993)": "Scary, thrilling, and visually stunning.",
    "Star Wars: Episode IV - A New Hope (1977)": "Way too slow and outdated for today's audience."
}

top_movies = recommend_movies("Toy Story (1995)", movie_similarity_df, top_n=5)
recommendation_with_sentiment = []

for movie, score in top_movies.items():
    review = mock_reviews.get(movie, "This movie was okay.")
    sentiment = predict_sentiment(review)
    recommendation_with_sentiment.append((movie, round(score, 3), sentiment, review))

df_vis = pd.DataFrame(recommendation_with_sentiment, columns=['Movie', 'Similarity Score', 'Sentiment', 'Sample Review'])

# -----------------------------
# 📊 Bar Chart Visualization
# -----------------------------
color_map = {'positive': 'green', 'negative': 'red'}
label_map = {'positive': 'Positive', 'negative': 'Negative'}
df_vis['Color'] = df_vis['Sentiment'].map(color_map)
df_vis['Label'] = df_vis['Sentiment'].map(label_map)

plt.figure(figsize=(10, 6))
bars = plt.barh(df_vis['Movie'], df_vis['Similarity Score'], color=df_vis['Color'])
plt.xlabel('Similarity Score')
plt.title('Top Recommended Movies with Sentiment')

for bar, label in zip(bars, df_vis['Label']):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=10)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# ☁️ WordClouds
# -----------------------------
positive_reviews = ' '.join(df_vis[df_vis['Sentiment'] == 'positive']['Sample Review'])
negative_reviews = ' '.join(df_vis[df_vis['Sentiment'] == 'negative']['Sample Review'])

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
wc_pos = WordCloud(background_color='white', colormap='Greens').generate(positive_reviews)
plt.imshow(wc_pos, interpolation='bilinear')
plt.title('Positive Sentiment WordCloud', fontsize=14)
plt.axis('off')

plt.subplot(1, 2, 2)
wc_neg = WordCloud(background_color='white', colormap='Reds').generate(negative_reviews)
plt.imshow(wc_neg, interpolation='bilinear')
plt.title('Negative Sentiment WordCloud', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()
