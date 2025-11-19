import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import load_datasets, print_dataset_info
from collections import Counter, defaultdict
from gensim.models import Word2Vec

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ~~~ RECOMMENDATION MODELS ~~~ #

class RandomBaseline:
    """Predict a random song from the entire catalog."""
    def __init__(self, SM, random_state=12):
        self.songs = SM.index.values
        self.random_state = random_state
    
    def predict(self, _, n=5):
        np.random.seed(self.random_state)
        return np.random.choice(self.songs, n, replace=False)
    
class RandomMaxGenre:
    """Predict random songs from the playlist's most frequent genre."""
    def __init__(self, SM, random_state=12):
        self.SM = SM
        self.random_state = random_state
        self.genre_to_songs = defaultdict(list)

        for uid, genre_list in zip(SM.index, SM["track_genre"]):
            for g in genre_list:
                self.genre_to_songs[g].append(uid)
    
    def predict(self, playlist_songs, n=5):
        genres = []
        for s in playlist_songs:
            genres.extend(self.SM.loc[s, "track_genre"])

        most_common = pd.Series(genres).mode()[0]
        candidates = self.genre_to_songs.get(most_common, [])

        np.random.seed(self.random_state)
        return list(np.random.choice(candidates, min(n, len(candidates)), replace=False))

class KNNCentroid:
    """Recommend the song closest to the playlist centroid in feature space."""
    def __init__(self, SM, feature_cols):
        self.SM = SM
        self.feature_cols = feature_cols
        self.song_features = SM[feature_cols].values
        self.song_ids = SM.index.values
        # build kNN index
        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.knn.fit(self.song_features)
    
    def predict(self, playlist_songs, n=5):
        # compute centroid
        centroid = self.SM.loc[playlist_songs, self.feature_cols].mean().values.reshape(1, -1)
        # find nearest neighbor to centroid
        distances, indices = self.knn.kneighbors(centroid, n_neighbors=n)
        recommended = self.song_ids[indices[0]]
        # exclude songs already in playlist if needed
        recommended = [s for s in recommended if s not in playlist_songs]
        return recommended[:n]

class PerTrackKNN:
    """Playlist-aware recommender:
    1. For each playlist track, get k nearest neighbors.
    2. Aggregate neighbors with weighted voting.
    3. Rank by frequency + distance."""
    
    def __init__(self, SM, feature_cols, k=20):
        self.SM = SM
        self.feature_cols = feature_cols
        self.song_features = SM[feature_cols].values
        self.song_ids = SM.index.values
        self.k = k

        # kNN index
        self.knn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"
        )
        self.knn.fit(self.song_features)

    def predict(self, playlist_songs, n=5, distance_weight=0.7):
        vote_counter = Counter()

        for song in playlist_songs:
            if song not in self.SM.index:
                continue
            
            # feature vector for this track
            vec = self.SM.loc[song, self.feature_cols].values.reshape(1, -1)
            
            # k neighbors
            distances, indices = self.knn.kneighbors(vec, n_neighbors=self.k)

            distances = distances[0]
            neighbor_ids = self.song_ids[indices[0]]

            # Add weighted votes
            for dist, neigh in zip(distances, neighbor_ids):
                if neigh in playlist_songs:
                    continue
                # inverse distance weight
                score = 1.0 / (1.0 + distance_weight * dist)
                vote_counter[neigh] += score

        # final sorted result
        ranked = [song for song, _ in vote_counter.most_common()]
        return ranked[:n]

# ~~~ DATA PREPARATION ~~~ #

def prepare_playlist_data(DB):
    """Merge playlists and song metadata, compute audio features per track."""
    SM = DB['SM']
    PL = DB['PL']

    # MERGE
    merged = PL.merge(SM, how='inner', on=['track_name', 'artists'])
    merged.to_parquet("merged.parquet")

    # COLS
    numeric_cols = [
        "danceability", 
        "energy", 
        "speechiness",
        "valence", 
    ]
    feature_cols = numeric_cols

    # standardize / scale features
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

    # aggregate playlists: each row is a unique (user_id, playlist_name)
    playlist_agg = merged.groupby(['user_id', 'playlist_name']).agg({
        'track_uid': lambda x: list(x)
    }).rename(columns={'track_uid': 'track_uids'}).reset_index()

    # =====================================================================
    # 1. TRACK WORD2VEC (playlist co-occurrence)
    # =====================================================================
    try:
        # playlists as sequences of track IDs
        track_sentences = [
            [str(t) for t in track_list]
            for track_list in playlist_agg["track_uids"]
        ]

        if len(track_sentences) == 0:
            raise ValueError("No playlist sentences found for track Word2Vec.")

        track_w2v = Word2Vec(
            sentences=track_sentences,
            vector_size=32,
            window=3,
            min_count=1,
            workers=1,
            sg=1
        )

        track_vecs = np.zeros((len(SM), 32))
        for i, tid in enumerate(SM.index):
            key = str(tid)
            if key in track_w2v.wv:
                track_vecs[i] = track_w2v.wv[key]

        track_cols = [f"track_w2v_{i}" for i in range(32)]
        track_df = pd.DataFrame(track_vecs, index=SM.index, columns=track_cols)

        SM = SM.join(track_df)
        feature_cols += track_cols
        print("Added Track Word2Vec.")

    except Exception as e:
        print(f"Track Word2Vec failed: {e}")

    # =====================================================================
    # 2. GENRE WORD2VEC 
    # =====================================================================
    try:
        # each track has a list of genres → Word2Vec sentence
        genre_sentences = []
        for _, row in playlist_agg.iterrows():
            playlist_genres = []
            for tid in row["track_uids"]:
                playlist_genres.extend(SM.loc[tid, "track_genre"])
            genre_sentences.append([str(g) for g in playlist_genres])

        genre_w2v = Word2Vec(
            sentences=genre_sentences, 
            vector_size=32, 
            window=3, 
            min_count=1, 
            sg=1
        )

        genre_vecs = np.zeros((len(SM), 32))
        for i, genre_list in enumerate(SM["track_genre"]):
            vecs = []
            for g in genre_list:
                if str(g) in genre_w2v.wv:
                    vecs.append(genre_w2v.wv[str(g)])
            if len(vecs) > 0:
                genre_vecs[i] = np.mean(vecs, axis=0)

        genre_cols = [f"genre_w2v_{i}" for i in range(32)]
        genre_df = pd.DataFrame(genre_vecs, index=SM.index, columns=genre_cols)

        SM = SM.join(genre_df)
        feature_cols += genre_cols
        print("Added Genre Word2Vec.")

    except Exception as e:
        print(f"Genre Word2Vec failed: {e}")

    # =====================================================================
    # 3. ARTIST WORD2VEC
    # =====================================================================
    try:
        artist_sentences = []
        for _, row in playlist_agg.iterrows():
            artists = SM.loc[row["track_uids"], "artists"].astype(str).tolist()
            artist_sentences.append(artists)

        if len(artist_sentences) == 0:
            raise ValueError("No artist sentences found.")

        artist_w2v = Word2Vec(
            sentences=artist_sentences,
            vector_size=32,
            window=3,
            min_count=1,
            workers=1,
            sg=1
        )

        artist_vecs = np.zeros((len(SM), 32))
        for i, artist in enumerate(SM["artists"].astype(str)):
            if artist in artist_w2v.wv:
                artist_vecs[i] = artist_w2v.wv[artist]

        artist_cols = [f"artist_w2v_{i}" for i in range(32)]
        artist_df = pd.DataFrame(artist_vecs, index=SM.index, columns=artist_cols)

        SM = SM.join(artist_df)
        feature_cols += artist_cols
        print("Added Artist Word2Vec.")

    except Exception as e:
        print(f"Artist Word2Vec failed: {e}")

    # scale word2vec embeddings
    SM[track_cols] = SM[track_cols] * 0.8
    SM[genre_cols] = SM[genre_cols] * 1.2
    SM[artist_cols] = SM[artist_cols] * 2

    return SM.set_index('track_uid'), playlist_agg, feature_cols

# ~~~ EVALUATION ~~~ #

def evaluate_recommender(recommender, playlist_agg, n_samples=2000, k=5):
    """Evaluate a recommender using multiple metrics and produce a ranking plot."""

    sampled = playlist_agg.sample(min(n_samples, len(playlist_agg)), random_state=42)

    total_hidden = 0
    hits = 0
    reciprocal_ranks = []

    for _, row in sampled.iterrows():
        songs = row['track_uids']
        if len(songs) < 4 or len(songs) > 300:
            continue
        
        # hide 2 songs
        np.random.seed(12)
        hidden = list(np.random.choice(songs, 2, replace=False))
        others = [s for s in songs if s not in hidden]
        total_hidden += len(hidden)
        
        preds = recommender.predict(others, n=50)

        # loop through each hidden song
        for h in hidden:
            if h in preds[:k]:
                hits += 1
            
            # compute reciprocal rank
            if h in preds:
                preds = list(preds) 
                rank = preds.index(h) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

    recall = hits / total_hidden
    mrr = np.mean(reciprocal_ranks)

    return {
        "recall@k": recall,
        "MRR": mrr,
    }

# ~~~ WORKFLOW ~~~ #

def eval_models(DB):
    SM, playlist_agg, feature_cols = prepare_playlist_data(DB)

    print(f"Total songs: {len(SM)}, Total playlists: {len(playlist_agg)}\n")

    # ~~~ BASELINE MODELS ~~~
    random_baseline = RandomBaseline(SM, random_state=35)
    random_max_genre = RandomMaxGenre(SM)
    nearest_neighbors = KNNCentroid(SM, feature_cols)
    knn_per_track = PerTrackKNN(SM, feature_cols)

    # ~~~ EVALUATION ~~~
    print("Evaluating RandomBaseline...")
    random_scores = evaluate_recommender(random_baseline, playlist_agg, n_samples=10000)
    print(f"RandomBaseline recall: {random_scores['recall@k']:.6f}\n")

    print("Evaluating RandomMaxGenre...")
    random_max_genre_score = evaluate_recommender(random_max_genre, playlist_agg)
    print(f"RandomMaxGenre recall: {random_max_genre_score['recall@k']:.6f}\n")

    print("Evaluating KNNCentroid...")
    knn_centroid_scores = evaluate_recommender(nearest_neighbors, playlist_agg)
    print(f"KNNCentroid recall: {knn_centroid_scores['recall@k']:.6f}\n")

    print("Evaluating PerTrackKNN...")
    per_track_knn_scores = evaluate_recommender(knn_per_track, playlist_agg)
    print(f"PerTrackKNN recall: {per_track_knn_scores['recall@k']:.6f}\n")

    return {
        "RandomBaseline": random_scores,
        "RandomMaxGenre": random_max_genre_score,
        "NearestNeighbors": knn_centroid_scores,
        "PerTracKNN": per_track_knn_scores,
    }

# ~~~ TESTING ~~~ #

def test_custom_playlist(DB, model=RandomBaseline, num_recs=3):
    SM, playlist_agg, feature_cols = prepare_playlist_data(DB)

    print(f"Total songs: {len(SM)}, Total playlists: {len(playlist_agg)}\n")

    my_model = model(SM, feature_cols)

    # your custom playlists
    my_songs = [70831, 73006, 10807, 24744, 11160, 691, 1419]
    my_songs_rap = [32637, 21835, 57561, 21839]

    nonrap_recs = my_model.predict(my_songs, n=num_recs)
    rap_recs = my_model.predict(my_songs_rap, n=num_recs)

    def show(title, recs):
        print(f"\n=== {title} ===")
        rec_df = SM.loc[recs][['track_name', 'artists', 'track_genre', 'popularity']]
        print(rec_df)

    show("Moody Teen Pop", my_songs)
    show("My song recommendations", nonrap_recs)

    show("Basic Rap", my_songs_rap)
    show("Rap recommendations", rap_recs)

    return nonrap_recs, rap_recs    

def make_plots():
    # --- Chart 1: Feature combinations ---
    feature_results = {
        "[BASELINE] Track/Genre/Artist Embeddings Only": 0.145271,
        "Energy": 0.145766,
        "Energy+acousticness": 0.144695,
        "Energy+danceability": 0.146302,
        "Energy+danceability+loudness": 0.109861,
        "Energy+danceability+speechiness": 0.146838,
        "Energy+danceability+speechiness+acousticness": 0.145766,
        "Energy+danceability+speechiness+valence": 0.148446,
        "Energy+danceability+speechiness+valence+tempo": 0.020900,
        "Energy+danceability+speechiness+valence+artist_popularity": 0.134512,
        "Energy+danceability+speechiness+valence+popularity": 0.054662,
        "Filtered Random": 0.0021436,
        "Pure Random": 0.000109
    }
    feature_results = dict(sorted(feature_results.items(), key=lambda x: x[1], reverse=True))

    features = list(feature_results.keys())
    recalls = list(feature_results.values())

    plt.figure(figsize=(10,6))
    plt.barh(features, recalls, color='skyblue')
    plt.xlabel("Recall@5")
    plt.title("Recall@5 by Feature Combination")
    plt.gca().invert_yaxis()  # Best at top
    plt.show()

    # --- Chart 2: Track/Genre/Artist scaling experiments ---
    scaling_results = {
        "[BASELINE] Track*1, Genre*1, Artist*1": 0.147910,
        "Track*2, Genre*1, Artist*1": 0.085745,
        "Track*1, Genre*2, Artist*1": 0.150054,
        "Track*1, Genre*1, Artist*2": 0.164523,
        "Track*1, Genre*2, Artist*2": 0.162379,
        "Track*1, Genre*1.5, Artist*2": 0.162915,
        "Track*1, Genre*1.2, Artist*2": 0.165595,
        "Track*0.5, Genre*1.2, Artist*2": 0.166667,
        "Track*0, Genre*1.2, Artist*2": 0.117899,
        "Track*0.8, Genre*1.2, Artist*2": 0.168274,
        "Filtered Random": 0.0021436,
        "Random": 0.000109
    }
    scaling_results = dict(sorted(scaling_results.items(), key=lambda x: x[1], reverse=True))

    configs = list(scaling_results.keys())
    recalls2 = list(scaling_results.values())

    plt.figure(figsize=(12,6))
    plt.barh(configs, recalls2, color='lightgreen')
    plt.xlabel("Recall@5")
    plt.title("Recall@5 by Track/Genre/Artist Word2Vec Embedding Scaling")
    plt.gca().invert_yaxis()
    plt.show()

# ~~~ MAIN ~~~ #

def main():
    DB = load_datasets(fresh=False)
    # print_dataset_info(DB)

    # make_plots() # make plots with data from manual feature testing

    test_custom_playlist(DB, model=PerTrackKNN, num_recs=10)

    # results = eval_models(DB)
    # print("\nModel comparison:")
    # for name, scores in results.items():
    #     print(f"{name}: {scores}")

if __name__ == "__main__":
    main()
