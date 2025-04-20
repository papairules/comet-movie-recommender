import pandas as pd

# Output files
ratings_output = "ratings_small.csv"
imdb_output = "IMDB_small.csv"

# Parameters
chunk_size = 500_000
target_rows = 100_000
sampled_chunks = []

# Load in chunks and sample progressively
reader = pd.read_csv("ratings.csv", chunksize=chunk_size)
for chunk in reader:
    sampled_chunk = chunk.sample(frac=0.02, random_state=42)  # 2% from each chunk
    sampled_chunks.append(sampled_chunk)
    if sum(len(c) for c in sampled_chunks) >= target_rows:
        break

# Concatenate and write to file
ratings_sample = pd.concat(sampled_chunks).head(target_rows)
ratings_sample.to_csv(ratings_output, index=False)
print(f"✅ Saved: {ratings_output}")

# Sample IMDb Dataset
imdb = pd.read_csv("IMDB Dataset.csv")
imdb_sample = imdb.sample(10_000, random_state=42)
imdb_sample.to_csv(imdb_output, index=False)
print(f"✅ Saved: {imdb_output}")
