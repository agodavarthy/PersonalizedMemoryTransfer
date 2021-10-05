import pandas as pd
import json
import regex
from tqdm import tqdm
import numpy as np
from collections import Counter
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--movie_merged", required=True, type=str,
                    help='Movie merged output from script')
parser.add_argument("--ml_movies", required=True, type=str,
                    help='path to movies.csv from movielens')
parser.add_argument("--output", required=True, help='Output filename', type=str)


def extract_year(title):
    year = -1
    match = regex.search('\((\d{4})\)$', title.strip())
    if match:
        year = int(match.group(1).strip())
        title = title[:match.start()].strip()
    return title, year


if __name__ == '__main__':
    opt = parser.parse_args()
    movies = pd.read_csv(opt.movie_merged)

    ml_movies = pd.read_csv(opt.ml_movies, usecols=['movieId', 'title']).rename(columns={'movieId': 'ml_id'})
    matches = []

    for idx, row in tqdm(movies.iterrows()):
        ml_id = row['id']
        ml_title, ml_year = extract_year(ml_movies.loc[ml_movies.ml_id==ml_id].iloc[0]['title'])

        matches.append({
            'ml_id': ml_id,
            'ml_year': ml_year,
            'ml_title': ml_title,
            'id': row['id'],
            'title': ml_title,
            'year': ml_year
        })

    matches = pd.DataFrame.from_dict(matches)
    matches.to_csv(opt.output, index=False, encoding='utf-8')
