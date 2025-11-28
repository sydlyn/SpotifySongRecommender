# Playlist-Based Music Recommendation Using Song Metadata and User Playlists

**Author:** Sydney Lynch

---

## 1. Introduction

My goal for this project is to create a music recommender system that predicts songs to add to an existing playlist by leveraging both song metadata and user playlists. The core idea is to capture user preferences via playlists while also incorporating audio and track-level features to improve recommendations.

I used two datasets from Kaggle:

1. **Song Metadata** — [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (song_metadata.csv): 114,000 songs × 21 columns, including artist, track name, genre, tempo, and features like danceability, energy, valence, and acousticness.
2. **Playlists** — [Spotify Playlists Dataset](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists) (playlists_clean.csv): ~12.8 million playlist entries × 4 columns, linking users, tracks, and playlist names.

This combination supports a recommender system where I can use historical playlist compositions to predict songs for other playlists. While metadata captures content features, playlists provide implicit collaborative signals that help capture user taste.

---

## 2. Data Exploration & Preprocessing

### Playlist Dataset

Loading the 12M-row playlist dataset into pandas initially failed due to nested quotes breaking the parser. Only 8 rows were misaligned, so I dropped them. Missing values were minimal: the `artist` column had ~0.26% missing (~3,350 rows), which I dropped as artist is important for merging datasets. Most columns with zeros were ordinal; `popularity` and `instrumentalness` had many zeros but were fine as-is.

Playlist lengths were highly skewed:

Most playlists were <100 songs, but a few extremely large playlists (up to 47k songs) corresponded to ‘Starred’ or favorites playlists. Top tracks had simple titles like “You”, “Stay”, and “Heaven,” matching expectations, and top artists matched the metadata dataset.

### Song Metadata Dataset

Metadata had 114k songs × 21 columns. Some track names were repeated heavily in niche genres (e.g., 50 variations of “Run Rudolph Run” in Christmas-themed genres), which made sense when splitting by genre (`1000 songs per genre`). Feature distributions were mostly right-skewed, except tempo and danceability, which were closer to normal. Surprisingly, correlations between danceability and tempo were near zero, even when splitting by genre.

Listening to a few songs, the features mostly made intuitive sense. For example, a danceability score of 0.9 for _I Wanna Be Your Ghost_ seemed reasonable, while _Memories Acoustic_ scored 0.8 despite a very different vibe. Overall, these features seemed more useful as latent signals rather than primary predictors.

### Merging Datasets

Merging playlists with metadata was tricky due to duplicate tracks with different genres or album releases. I aggregated genres into a `track_genres` list to capture all associated labels. Handling duplicates remains an open question: differences may reflect noise or potentially meaningful variation.

---

## 3. Modeling & Feature Development

### Evaluation Strategy

I created an evaluator to consistently measure model performance:

1. Hide 2 songs per playlist (`n_samples=2000`).
2. Feed remaining playlist tracks into the model.
3. Get top k=5 recommendations.
4. Calculate `recall@5` and `MRR` for the hidden songs.

Recall@5 was chosen because the goal is to recover hidden songs; precision is capped at 0.4 (5 recommendations for 2 hidden songs) and would artificially lower F1. MRR helps gauge how highly ranked the hidden songs are among the predictions.

---

### Baselines

1. **RandomBaseline** — randomly recommends a song. Recall@5 ≈ 0.
2. **RandomMaxGenre** — randomly selects a song with the playlist’s most common genre. Recall@5 ≈ 0.002. Quick to run and slightly personalized but still generic.

---

### KNNCentroid (Content + Metadata)

I used KNN to find songs nearest to a playlist centroid (averaged track features). Features were selected based on metadata and embeddings:

- Metadata features improving recall: `danceability`, `energy`, `speechiness`, `valence`.
- Features that decreased recall: `popularity`, `artist_popularity`, `tempo` (likely noisy).

#### Playlist-Based Word2Vec Embeddings

To leverage playlists more directly, I trained three separate Word2Vec embeddings:

1. **Track ID embeddings** — captures co-occurrence of tracks in playlists.
2. **Genre embeddings** — captures stylistic/theme co-occurrence.
3. **Artist embeddings** — captures artists frequently appearing together.

Including any single embedding increased recall@5 from ~0.003 to 0.08, and all three together achieved ~0.14. Scaling embeddings further optimized recall: `track_id * 0.8`, `track_genre * 1.2`, `artist * 2` → recall@5 ≈ 0.168.

> ![recall@5 by song metadata feature combinations](./plots/sm_feature_comp.png)  
> Ranking of recall@5 by song metadata feature combinations

> ![recall@5 by track/genre/artist word2vec embedding scaling](./plots/word2vec_scaling_comp.png)  
> Ranking of recall@5 by track/genre/artist embedding scaling

---

### PerTrackKNN

Instead of condensing playlists to a centroid, PerTrackKNN recommends based on individual tracks, aggregating the most frequent suggestions. With the same tuned features and embeddings, recall@5 increased to 0.1822, capturing more playlist nuance.

> ![final_scores](./plots/final_scores.png)  
> Evaluator recall@5 and MRR across models

---

## 4. Sample Recommendations

To check qualitative performance, I created two small playlists:

**Moody Teen Pop** (7 tracks) and **Basic Rap** (4 tracks). Top 10 recommendations for each using the final model included reasonable choices, with tracks matching genres and artists, though some surprising inclusions occurred (e.g., Olivia Rodrigo songs in Rap playlists).

---

## 5. Discussion

**Strengths**

- Embeddings capture user taste better than raw metadata.
- Per-track aggregation preserves playlist-level nuance.
- Strong improvement over baselines (~x85).

**Weaknesses / Limitations**

- Struggles with niche or very short playlists.
- Metadata features like popularity/tempo can add noise.
- Handling duplicates and varying genres remains unresolved.

---

## 6. Future Work

- Hyperparameter tuning: embedding dimensions, KNN neighbors.
- Alternative algorithms: Random Forest, matrix factorization, deep learning.
- Expanded evaluation: hide more songs per playlist, other metrics.
- Better handling of duplicates and genre ambiguity.

---

## 7. Conclusion

A hybrid recommender using playlist-based embeddings plus selected metadata features forms a strong baseline. This system achieves reasonable recall@5 while capturing both playlist co-occurrence patterns and track-level content features. Code and datasets are included in the public repository for reproducibility.
