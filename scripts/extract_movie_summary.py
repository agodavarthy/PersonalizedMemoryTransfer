"""
Given the movie_match.csv which contains the mapping of movielens and redial dataset
we search our db for plots. This does not do any preprocessing, just writes it out.
"""
import argparse
import pandas as pd
import pymongo
from tqdm import tqdm

# import os
# import sys
# sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
# import util
# from movie_data import TextEncoder, tokenizer, get_key_in_order

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--movie_match', help='Path to  movie_match.csv', type=str,
                    default="data/movie_match.csv")
parser.add_argument('-o', '--output', help='Output filename', type=str,
                    default="data/movie_plot.csv")


def get_key_in_order(item, keys, default=None):
    """Return the first key if present in keys else return
    default, if default is None raises exception"""
    for key in keys:
        if key in item:
            return item[key]
    if default is None:
        raise Exception("Could not find any keys: ", keys)
    return default


if __name__ == '__main__':
    opt = parser.parse_args()
    # df = pd.read_csv("data/movie_match.csv")
    df = pd.read_csv(opt.movie_match).rename(
            columns={'id': 'movie_id',
                     'title': 'redial_title',
                     'year': 'redial_year'})
    # Remove missing movies
    df = df[df.ml_id != -1]
    df = df[df.movie_id != -1]
    print(f"Loaded {len(df):,} movies")

    # Connect to db
    client = pymongo.MongoClient(port=27017)
    db = client.get_database('movies')
    collection = db.get_collection('movies')

    # Add some columns
    df['plot'] = None
    df['title'] = None
    df['imdb_id'] = None
    df['tmdb_id'] = None
    imdb_count = 0
    tmdb_count = 0
    plot_short = 0
    missing_entry = 0
    # tmdb_movies = []

    # Find them
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        item = collection.find_one({"_id": int(row['ml_id'])})
        if item is None:
            missing_entry += 1
            continue

        title = item.get('title', row['title'])
        plot = ""

        if 'imdb' in item and len(item['imdb']):
            imdb = item['imdb']
            title = get_key_in_order(imdb, ['smart canonical title', 'canonical title', 'title'], title)
            plot = get_key_in_order(imdb, ['plot', 'plot outline', 'synopsis', 'summary', 'plot summary'], "")
            if isinstance(plot, list):
                best_plot = plot[0]
                # Assume the longest plot is the best
                for p in plot[1:]:
                    if len(p) > len(best_plot):
                        best_plot = p
                plot = best_plot
        # Check if we did not find the plot
        if 'tmdb' in item and len(plot) == 0:
            title = item['tmdb']['title']
            plot = item['tmdb']['overview']
            tmdb_count += 1
            # tmdb_movies.append(item)
        elif len(plot):
            # Found so increment imdb count
            imdb_count += 1

        # Empty plot
        if len(plot) == 0:
            plot_short += 1

        df.loc[idx, 'imdb_id'] = item['imdb_id']
        df.loc[idx, 'tmdb_id'] = item['tmdb_id']
        df.loc[idx, 'title'] = title
        df.loc[idx, 'plot'] = plot

    print(f"Saving to {opt.output}")
    df.to_csv(opt.output, index=False)

    print(f"""
            IMDB Count:   {imdb_count:,}
            TMDB Count:   {tmdb_count:,}
            No Plot:      {plot_short:,}
            
            Missing:      {missing_entry}            
            """)
