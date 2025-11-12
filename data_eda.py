import pandas as pd

def clean_playlist_dataset():
    clean = pd.read_csv("data/playlists.csv", engine="python", on_bad_lines='skip')
    clean.to_csv("data/playlists_clean.csv")

def load_data():
    DB = {
        "SM": pd.read_csv("data/song_metadata.csv"),
        "PL": pd.read_csv("data/playlists_clean.csv"),
    }
    return DB

def print_dataset_info(DB):
    for name, df in DB.items():
        print(f"Dataset: {name}")
        print("Shape:", df.shape)
        print(df.head())
        print("\n")

def main():
    # clean_playlist_dataset()
    DB = load_data()
    print_dataset_info(DB)

if __name__ == "__main__":
    main()
