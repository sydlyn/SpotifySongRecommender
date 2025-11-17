import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ~~~ INITIAL DATA LOADING ~~~ #

def clean_playlist_dataset():
    '''Load up the playlist dataset and do some inital column cleaning.'''
    
    bad_count = 0
    def bad_count_tracker(_):
        nonlocal bad_count
        bad_count += 1
        return None

    # skip bad lines and count how many get skipped
    clean = pd.read_csv("../data/playlists.csv", engine="python", on_bad_lines=bad_count_tracker) 
    print(f"{bad_count} lines skipped in the playlist dataset due to bad formatting")

    clean.columns = ['user_id', 'artists', 'track_name', 'playlist_name']
    clean.to_parquet("../data/playlists_clean.parquet")

def load_data(save_samples=False):
    DB = {
        "SM": pd.read_csv("../data/song_metadata.csv", index_col=0),
        "PL": pd.read_parquet("../data/playlists_clean.parquet"),
    }

    if save_samples:
        for name, df in DB.items():
            df.head(1000).to_parquet(f"../data/{name}.parquet")

    return DB

def print_dataset_info(DB):
    for name, df in DB.items():
        print(f"Dataset: {name}")
        print("Shape:", df.shape)
        print(df.head())
        print("\n")


# ~~~ CHECK + DROP EMPTY ROWS ~~~ #

basic_col_check = pd.DataFrame(columns=['table', 'column', 'num_unique', 'perc_empty'])

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

# print some summary numbers for each given column in the given df
def check_cols(cols, db, table_name='', add_to_table=True, p=False, drop=True):
    global basic_col_check

    print(f"Total Number of Rows in {table_name}: {len(db)}")
    for col in cols:
        if p: print(f"\n{col.upper()}:")
        col_data = db[col]
        num_unique = col_data.nunique()
        perc_empty = get_perc_empty(col_data)
        perc_zero = get_perc_empty(col_data, zero_as_null=True)
        if p:
            print("Unique values:\t\t", num_unique)
            print("% of empty rows:\t", perc_empty)
            print(col_data.value_counts())

        if add_to_table:
            add_row(table_name, col, num_unique, perc_empty, perc_zero)

    if drop:
        # drop rows where any of the specified columns are NaN or empty strings
        before = len(db)
        db = db.dropna(subset=cols)
        for c in cols:
            if db[c].dtype == object or db[c].dtype.name == "string":
                db = db[db[c] != ""]
        after = len(db)
        if (before-after > 0): 
            print(f"Dropped {before - after} rows with empty values in {table_name}")

def add_row(table_name, col, num_unique, perc_empty, perc_zero):
    global basic_col_check
    new_row = pd.DataFrame({
        'table': [table_name], 
        'column': [col], 
        'num_unique': [num_unique],
        'perc_empty': [perc_empty],
        'perc_zero': [perc_zero] 
    })
    basic_col_check = pd.concat([basic_col_check, new_row], ignore_index=True)


# ~~~ EXPLORE + VISUALS ~~~ #

def explore_sm(DB):
    SM = DB['SM']

    # for col in SM.columns[1:4]:
    #     print(SM[col].value_counts()[:30])

    features = ["danceability", "energy", "loudness", "speechiness", "acousticness", "valence", "tempo"]
    corr = SM[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation between Audio Features")
    plt.show()

    cols = SM.columns[7:-1]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        SM[col].hist(ax=axes[i], bins=30, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
    # hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("Distributions of Audio Features", fontsize=16, y=1.02)
    plt.show()

    genre_summary = (
        SM.groupby('track_genre')[['danceability', 'energy', 'loudness', 'speechiness', 'tempo']]
        .mean()
        .sort_values('energy', ascending=False)
    )

    print(genre_summary.head(30))

    print("counts of counts of track_genre (how many genres have x amount of counts):")
    print(SM['track_genre'].value_counts().value_counts())


def explore_pl(DB):
    PL = DB['PL']

    # # loop through value counts for each col
    # for col in PL.columns:
    #     print(PL[col].value_counts()[:30])

    # # loop through the value counts of rows that had no artist
    # empty_artist = PL[PL['artists'].isna()]
    # print("number of rows with no artist listed: ")
    # print(len(empty_artist))
    # for col in empty_artist:
    #     print(empty_artist[col].value_counts())


    ## PLOT PLAYLIST LENGTHS
    # Group playlists by name and user, count tracks per playlist
    playlist_lengths = PL.groupby(['playlist_name', 'user_id']).size()

    # Show summary stats
    print(playlist_lengths.describe())

    # Filter only playlists with ≤100 tracks for plotting
    filtered_lengths = playlist_lengths[playlist_lengths <= 100]

    # Basic stats for context
    print("Number of playlists (≤100 tracks):", len(filtered_lengths))
    print("Average playlist length (≤100 tracks):", filtered_lengths.mean())
    print("Max playlist length (≤100 tracks):", filtered_lengths.max())

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_lengths, bins=50, color='mediumseagreen', edgecolor='black')
    plt.title("Distribution of Playlist Lengths (0-100 Tracks)", fontsize=16)
    plt.xlabel("Number of Tracks per Playlist")
    plt.ylabel("Count of Playlists")
    plt.tight_layout()
    plt.show()

    # Find the playlist with 47,309 tracks
    max_len = playlist_lengths.max()
    largest_playlist = playlist_lengths[playlist_lengths == max_len]

    print("\nPlaylist(s) with the maximum number of tracks:")
    print(largest_playlist)

    # If you want to extract the corresponding user/playlist name directly:
    for (playlist_name, user_id), length in largest_playlist.items():
        print(f"User ID: {user_id}, Playlist Name: {playlist_name}, Length: {length}")


# ~~~ ATTEMPTs TO MERGE ~~~ #

def merge(DB):
    song_metadata_unique = DB['SM'].drop_duplicates(subset=["track_name", "artists"])
    playlists_unique = DB['PL'].drop_duplicates(subset=["track_name", "artists", "playlist_name"])

    merged = pd.merge(playlists_unique, song_metadata_unique, on=['track_name', 'artists'], how='inner')
    merged.head(1000).to_parquet("merged.parquet")
    return merged

# ~~~ MAIN ~~~ #

def main():
    # clean_playlist_dataset()
    DB = load_data(save_samples=False)
    # print_dataset_info(DB)

    check_cols(DB['SM'].columns, DB['SM'], 'song_metadata')
    check_cols(DB['PL'].columns, DB['PL'], 'playlists')
    print(basic_col_check)


    explore_sm(DB)
    # explore_pl(DB)

    # merged = merge(DB)
    # print(merged)
    # print(merged.describe())

if __name__ == "__main__":
    main()
