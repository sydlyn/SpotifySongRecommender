Sydney Lynch

# Playlist-Based Music Recommendation Using Song Metadata and User Playlists

This is a music recommender system that predicts songs to add to an existing playlist by leveraging both song metadata and user playlists. The core idea is to capture user preferences via playlists while also incorporating audio and track-level features to improve recommendations.

## Running the Code

In order to run this code, these two datastets must be downloaded from Kaggle:

- (renamed) song_metadata.csv — from [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

- (renamed) playlists.csv — from [Spotify Playlists Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)

Once downloaded and renamed, place them in the `data/` folder.

## Code Organization

There are 3 folders that each serve a different purpose for this project.

1. **Data Exploration**: The `eda/` folder holds code and plots for the initial code exploration and cleaning.

2. **First Draft and Initial Models**: The `first_draft/` folder holds code that cleans the data according to what was found from the EDA, along with evaluation comparing different initial models with simple random baselines.

3. **Final Draft**: The `final_draft/` folder holds all materials and code relevant to the final model, with an evaluation comparison against the baselines.
