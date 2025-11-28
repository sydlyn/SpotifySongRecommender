import pandas as pd
import os

# print headers of each dataset
def print_dataset_info(DB):
    for name, df in DB.items():
        print(f"Dataset: {name}")
        print("Shape:", df.shape)
        print(f"{df.columns}")
        print(df.head())
        print("\n")

# return the percentage of rows that are missing
def get_perc_empty(col, zero_as_null=False):
    num = len(col)
    if num == 0:
        return 0.0

    num_empty = col.isna().sum()

    if col.dtype == object or col.dtype.name == "string":
        num_empty += (col == "").sum()
    elif zero_as_null and ("int" in col.dtype.name or "float" in col.dtype.name):
        num_empty += (col == 0).sum()

    return (num_empty / num) * 100

# clean the playlists csv and save as a parquet
def clean_playlist(input_path="../data/playlists.csv", output_path="../data/playlists_clean.parquet"):
    """Load the playlist CSV, clean, and save as parquet."""
    bad_count = 0

    def bad_count_tracker(_):
        nonlocal bad_count
        bad_count += 1
        return None

    df = pd.read_csv(
        input_path,
        engine="python",
        on_bad_lines=bad_count_tracker
    )
    print(f"[playlists] {bad_count} bad rows skipped")

    df.columns = ["user_id", "artists", "track_name", "playlist_name"]

    # drop empty
    df = df.dropna(subset=["user_id", "artists", "track_name", "playlist_name"])
    df = df[(df["artists"] != "") & (df["track_name"] != "")]

    df.to_parquet(output_path)
    print(f"[playlists] cleaned file saved -> {output_path}")
    return df

# clean the song metadata csv and save as a parquet
def clean_song_metadata(input_path="../data/song_metadata.csv", output_path="../data/song_metadata_clean.parquet"):
    """Load song_metadata.csv, clean, dedupe by track_id, and save."""
    df = pd.read_csv(input_path, index_col=0)

    # Drop rows missing core fields
    essential_cols = ["track_id", "track_name", "artists", "track_genre"]
    df = df.dropna(subset=essential_cols)

    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # Remove special columns from each list
    protected = ["track_id", "track_name", "artists", "track_genre", "popularity"]
    numeric_cols = [c for c in numeric_cols if c not in protected]
    non_numeric_cols = [c for c in non_numeric_cols if c not in protected]

    def merge_genres(g):
        return list(set(g))  # convert to python list of unique genres

    df = (
        df.groupby(['artists', 'track_name'], as_index=False)
          .agg({
              "track_name": "first",
              "artists": "first",
              "track_genre": merge_genres,
              "popularity": "max" if "popularity" in df.columns else "first",
                **{col: "mean" for col in numeric_cols},
                **{col: "first" for col in non_numeric_cols},
          })
    )
    df["track_uid"] = df.groupby(["artists", "track_name"]).ngroup()

    df.to_parquet(output_path)
    print(f"[song_metadata] cleaned file saved -> {output_path}")
    return df

# load the datasets, clean if needed
def load_datasets(fresh=False, base_path="../data"):
    """Returns { 'SM': song_metadata_df, 'PL': playlists_df }
    If fresh=True: clean CSVs and overwrite the _clean.parquet files. 
    Otherwise: load the parquet files."""

    pl_clean_path = os.path.join(base_path, "playlists_clean.parquet")
    sm_clean_path = os.path.join(base_path, "song_metadata_clean.parquet")

    if fresh:
        playlists = clean_playlist(
            os.path.join(base_path, "playlists.csv"),
            pl_clean_path
        )
        song_metadata = clean_song_metadata(
            os.path.join(base_path, "song_metadata.csv"),
            sm_clean_path
        )
        return {"PL": playlists, "SM": song_metadata}

    # otherwise load the pre-cleaned files
    if not os.path.exists(pl_clean_path) or not os.path.exists(sm_clean_path):
        return load_datasets(fresh=True)

    print("Loading cleaned parquet files...")
    return {
        "PL": pd.read_parquet(pl_clean_path),
        "SM": pd.read_parquet(sm_clean_path),
    }
