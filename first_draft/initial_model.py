import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import load_datasets, print_dataset_info
from collections import defaultdict
from gensim.models import Word2Vec

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ~~~ RECOMMENDATION MODELS ~~~ #

class RandomBaseline:
    """Predict a random song from the entire catalog."""
    def __init__(self, SM):
        self.songs = SM.index.values
    
    def predict(self, _, n=10, random_seed=12):
        np.random.seed(random_seed)
        return np.random.choice(self.songs, n, replace=False)
    
class PopularBaseline:
    """Predict the most popular, not-saved song."""
    
class RandomMaxGenre:
    """Predict a random song from the most frequent genre in the playlist."""
    

class SimilarityBaseline:
    """Recommend the song closest to the playlist centroid in feature space."""
    def __init__(self, SM, feature_cols):
        self.SM = SM
        self.feature_cols = feature_cols
        self.features = SM[feature_cols].values
        self.song_ids = SM.index.values
    
    def predict(self, playlist_songs, n=10):
        # compute centroid of given playlist songs
        centroid = self.SM.loc[playlist_songs, self.feature_cols].mean().values
        # candidates = all songs not in playlist
        mask = ~self.SM.index.isin(playlist_songs)
        candidates = self.features[mask]
        candidate_ids = self.song_ids[mask]
        # compute Euclidean distance
        dist = np.linalg.norm(candidates - centroid, axis=1)
        # return closest song(s)
        return candidate_ids[np.argsort(dist)[:n]]
    
class KNearestNeighbors:
    def __init__(self, SM, feature_cols):
        self.SM = SM
        self.feature_cols = feature_cols
        self.song_features = SM[feature_cols].values
        self.song_ids = SM.index.values
        # build kNN index
        self.knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.knn.fit(self.song_features)
    
    def predict(self, playlist_songs, n=10):
        # compute centroid
        centroid = self.SM.loc[playlist_songs, self.feature_cols].mean().values.reshape(1, -1)
        # find nearest neighbor to centroid
        distances, indices = self.knn.kneighbors(centroid, n_neighbors=n)
        recommended = self.song_ids[indices[0]]
        # exclude songs already in playlist if needed
        recommended = [s for s in recommended if s not in playlist_songs]
        return recommended[:n]


# ~~~ DATA PREPARATION ~~~ #

def prepare_playlist_data(DB):
    """Merge playlists and song metadata, compute audio features per track."""
    SM = DB['SM']
    PL = DB['PL']

    # TRACK_GENRE one-hot encoded
    mlb = MultiLabelBinarizer()
    genre_ohe = mlb.fit_transform(SM["track_genre"])
    
    genre_df = pd.DataFrame(
        genre_ohe,
        columns=[f"genre__{g}" for g in mlb.classes_],
        index=SM.index
    )

    SM = pd.concat([SM, genre_df], axis=1)

    # TRACK_NAME
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_mat = tfidf.fit_transform(SM["track_name"])

    svd = TruncatedSVD(n_components=32, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_mat)

    tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_reduced.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_reduced, columns=tfidf_cols, index=SM.index)

    SM = pd.concat([SM, tfidf_df], axis=1)

    # ARTISTS
    artist_counts = SM["artists"].value_counts()
    SM["artist_popularity"] = SM["artists"].map(artist_counts)

    # MERGE
    merged = PL.merge(SM, how='inner', on=['track_name', 'artists'])
    merged.to_parquet("merged.parquet")

    # COLS
    numeric_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "valence", "tempo", "artist_popularity", "popularity"
    ]
    genre_cols = [col for col in SM.columns if col.startswith("genre__")]
    tfidf_cols = [col for col in SM.columns if col.startswith("tfidf_")]

    feature_cols = numeric_cols + genre_cols + tfidf_cols

    # standardize / scale features
    scaler = StandardScaler()
    merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols])
    merged[genre_cols] = merged[genre_cols] * 0.2

    # aggregate playlists: each row is a unique (user_id, playlist_name)
    playlist_agg = merged.groupby(['user_id', 'playlist_name']).agg({
        'track_uid': lambda x: list(x)
    }).rename(columns={'track_uid': 'track_uids'}).reset_index()
    
    print("playlist_agg:\n", playlist_agg)

    # === Build simple track -> index mapping ===
    track_ids = SM['track_uid'].tolist()
    track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}

    try:
        # playlists as lists of string track_uids
        sentences = [
            [str(t) for t in track_list]
            for track_list in playlist_agg["track_uids"]
        ]

        w2v_model = Word2Vec(
            sentences,
            vector_size=32,
            window=3,
            min_count=1,
            workers=1,
            sg=1
        )

        w2v_embeds = np.zeros((len(track_ids), 32))
        for tid in track_ids:
            key = str(tid)
            if key in w2v_model.wv:
                w2v_embeds[track_id_to_index[tid]] = w2v_model.wv[key]
            else:
                w2v_embeds[track_id_to_index[tid]] = np.zeros(32)

        # add W2V to SM
        w2v_cols = [f"w2v_{i}" for i in range(w2v_embeds.shape[1])]
        w2v_df = pd.DataFrame(w2v_embeds, index=track_ids, columns=w2v_cols)

        SM = SM.join(w2v_df)
        feature_cols += w2v_cols

    except Exception as e:
        print("Word2Vec failed:", e)

    return SM.set_index('track_uid'), playlist_agg, feature_cols

# ~~~ EVALUATION ~~~ #

def evaluate_recommender(recommender, playlist_agg, n_samples=5000, k=5):
    """Evaluate a recommender using multiple metrics and produce a ranking plot."""

    sampled = playlist_agg.sample(min(n_samples, len(playlist_agg)), random_state=42)

    total_hidden = 0
    hits = 0
    reciprocal_ranks = []

    rank_histogram = defaultdict(int)

    for _, row in sampled.iterrows():
        songs = row['track_uids']
        if len(songs) < 4:
            continue
        
        # hide 2 songs
        hidden = list(np.random.choice(songs, 2, replace=False))
        others = [s for s in songs if s not in hidden]
        total_hidden += len(hidden)
        
        preds = recommender.predict(others, n=50)  # request large list for metrics

        # loop through each hidden song
        for h in hidden:
            if h in preds[:k]:
                hits += 1
            
            # compute reciprocal rank
            if h in preds:
                preds = list(preds) 
                rank = preds.index(h) + 1  # 1-indexed
                reciprocal_ranks.append(1 / rank)
                rank_histogram[min(rank, 50)] += 1
            else:
                reciprocal_ranks.append(0)

    hit_rate = hits / total_hidden
    mrr = np.mean(reciprocal_ranks)

    # precision@k and recall@k are same here (always 2 hidden items)
    precision_at_k = hit_rate
    recall_at_k = hit_rate

    return {
        "hit_rate@k": hit_rate,
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "MRR": mrr,
    }

# def evaluate_recommender(recommender, playlist_agg, n_samples=10000):
#     """Evaluate a recommender by hiding one song per playlist and checking hit rate."""
#     hits = 0
#     sampled = playlist_agg.sample(min(n_samples, len(playlist_agg)), random_state=42)

#     for _, row in sampled.iterrows():
#         songs = row['track_uids']
#         if len(songs) < 4 or row['playlist_name'] == "Starred":
#             continue
#         hidden = np.random.choice(songs, 2)
#         others = [s for s in songs if s not in hidden]
#         pred = recommender.predict(others, n=5)
#         for h in hidden:
#             if h in pred:
#                 hits += 1

#     hit_rate = hits / len(sampled)
#     return hit_rate

# ~~~ WORKFLOW ~~~ #

def run_models(DB):
    SM, playlist_agg, feature_cols = prepare_playlist_data(DB)

    print(f"Total songs: {len(SM)}, Total playlists: {len(playlist_agg)}\n")

    # ~~~ BASELINE MODELS ~~~
    random_baseline = RandomBaseline(SM)
    similarity_baseline = SimilarityBaseline(SM, feature_cols)
    nearest_neighbors = KNearestNeighbors(SM, feature_cols)

    # ~~~ EVALUATION ~~~
    print("Evaluating Random Baseline...")
    random_hit = evaluate_recommender(random_baseline, playlist_agg)
    print(f"Random Baseline hit rate: {random_hit['hit_rate@k']:.6f}")

    print("Evaluating Similarity Baseline...")
    similarity_hit = evaluate_recommender(similarity_baseline, playlist_agg)
    print(f"Similarity Baseline hit rate: {similarity_hit['hit_rate@k']:.6f}")

    # print("Evaluating Nearest Neighbors...")
    # neighbors_hit = evaluate_recommender(nearest_neighbors, playlist_agg)
    # print(f"Nearest Neighbors hit rate: {neighbors_hit:.6f}")

    return {
        "RandomBaseline": random_hit,
        "SimilarityBaseline": similarity_hit,
        # "NearestNeighbors": neighbors_hit,
    }

# ~~~ MAIN ~~~ #

def main():
    DB = load_datasets(fresh=False)
    print_dataset_info(DB)

    results = run_models(DB)

    print("\nModel comparison:")
    for name, hit in results.items():
        print(f"{name}: {hit}")

if __name__ == "__main__":
    main()

