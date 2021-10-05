"""
match_movies.py from https://github.com/RaymondLi0/conversational-recommendations/blob/master/scripts/match_movies.py
from paper: "Towards Deep Conversational Recommendations"
See https://github.com/RaymondLi0/conversational-recommendations for full instructions


wget -O redial_dataset.zip https://github.com/ReDialData/website/raw/data/redial_dataset.zip
wget -O ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip

python scripts/match_movies.py --redial data/redial_dataset/ --ml_movies data/ml-latest/movies.csv --output data/movie_match.csv
"""
import pandas as pd
import json
import regex
from tqdm import tqdm
import numpy as np
from collections import Counter
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--redial", required=True, type=str,
                    help='directory to redial dataset, loads test/train splits')
parser.add_argument("--ml_movies", required=True, type=str,
                    help='path to movies.csv from movielens')
parser.add_argument("--output", required=True, help='Output filename', type=str)

"""
Note: Movie: 'It' has many releases with different years
 
Redial dataset Id: ==> MovieLens IDs that are manually checked
"""
corrections = {
    # LittleMan Use ID: 46865, second ID that is matched is a tv show with the same year
    # See: https://movielens.org/movies/46865 and https://movielens.org/movies/172427
    # '135895': 46865,

    # 179401,Jumanji: Welcome to the Jungle (2017),Action|Adventure|Children
    # 83552,Jumanji (2017),502
    # Missing the full title to match
    '83552': 179401
}


def extract_year(title):
    year = -1
    match = regex.search('\((\d{4})\)$', title.strip())
    if match:
        year = int(match.group(1).strip())
        title = title[:match.start()].strip()
    return title, year


def preprocess_titles(df):
    """
    We use two different checking for titles,

    First exact match including 'The '
    then we add edit_tile which removes 'The' and 'A'

    :param df: pd.Dataframe
    :return: pd.Dataframe
    """
    # Remove end ", The"
    match_index = df['title'].str.endswith(", The")
    df.loc[match_index, 'title'] = df.title[match_index].str.slice(stop=-5)

    # Add back "The " to the start if we found it
    df.loc[match_index, 'title'] = df.title[match_index].map(lambda title: "The " + title)

    # Replace & with and
    df.title = df.title.str.replace("&", "and")

    df['edit_title'] = df['title']

    # Remove start "The"
    match_index = df['title'].str.startswith("The ")
    df.loc[match_index, 'edit_title'] = df.title[match_index].str.slice(start=4)

    # Remove start "A"
    match_index = df['edit_title'].str.startswith("A ")
    df.loc[match_index, 'edit_title'] = df.edit_title[match_index].str.slice(start=2)

    return df


def write_movie_mentions(data: pd.DataFrame):
    """
    Write out - Movie_id, Title, Number of mentions

    :param data:
    :return:
    """
    # Get all movie mentions and extract years
    movie_mentions = Counter()
    for d in data.movieMentions.values:
        for k, v in d.items():
            if v:
                movie_mentions[(k, v.replace("  ", " ").strip())] += 1
    movie_mentions = pd.DataFrame.from_dict([(key, name, count) for (key, name), count in movie_mentions.most_common()])
    movie_mentions.columns = ['movie_id', 'title', 'count']
    movie_mentions.to_csv('../data/movie_mentions.csv', index=False)


if __name__ == '__main__':
    opt = parser.parse_args()#"--redial ../data/redial_dataset/ --ml_movies ../data/ml-latest/movies.csv --output /tmp/tmp.csv".split())
    ###########################################################################
    # Redial dataset loading/preprocessing
    ###########################################################################
    # with open(os.path.join(opt.redial, "train_data.jsonl"), 'r', encoding='utf-8') as f:
    #     data = [json.loads(line) for line in f]
    #
    # with open(os.path.join(opt.redial, "test_data.jsonl"), 'r', encoding='utf-8') as f:
    #     data.extend([json.loads(line) for line in f])
    # data = pd.DataFrame.from_dict(data)
    #
    # # Drop row with no movie mentions
    # data = data[data.movieMentions.str.len() > 0]
    #
    # # Get all movie mentions and extract years
    # movie_mentions = {}
    # for d in data.movieMentions.values:
    #     movie_mentions.update({k: v.replace("  ", " ").strip() for k, v in d.items() if v})

    mentions = pd.read_csv(opt.redial)#"../data/redial_dataset/movies_with_mentions.csv")
    movie_mentions_ls = []
    # for movie_id, title in movie_mentions.items():
    for movie_id, title in zip(mentions.movieId.values, mentions.movieName.values):
        title, year = extract_year(title)
        item = {'movie_id': movie_id, 'title': title,
                'year': year}
        movie_mentions_ls.append(item)

    # 'movie_id', 'title', 'year', 'edit_title'
    movies_df = pd.DataFrame.from_dict(movie_mentions_ls)
    movies_df = preprocess_titles(movies_df)

    ###########################################################################
    # Start MovieLens preprocessing
    ###########################################################################
    movielens_df = pd.read_csv(opt.ml_movies, usecols=['movieId', 'title'])
    # Extract years from MovieLens
    years = []
    titles = []
    for title in movielens_df.title.values:
        title, year = extract_year(title)
        years.append(year)
        titles.append(title)

    movielens_df['raw_title'] = movielens_df['title']
    movielens_df['year'] = years
    movielens_df['title'] = titles
    movielens_df = movielens_df.rename(columns={'movieId': 'ml_id'})
    # Preprocess titles
    movielens_df = preprocess_titles(movielens_df)

    ###########################################################################
    # Start Matching of movies
    ###########################################################################
    matches = []
    multiple_matches = []
    year_miss = []
    missing = []

    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
        # Manually matched/fixed
        if row['movie_id'] in corrections:
            match_index = movielens_df['ml_id'] == corrections[row['movie_id']]
        else:
            # Match by title
            match_index = movielens_df.title == row.title
            # Try preprocessed one removing The/A
            if not match_index.any():
                match_index = movielens_df.edit_title == row.edit_title

        # Check if we found any
        if not match_index.any():
            missing.append(row)
            continue

        match = movielens_df[match_index]

        # Check if we should check year
        if row['year'] != -1 and (match.year != -1).any():
            # Exact match first
            match_index = match.year == row['year']
            if match_index.sum() > 1:
                # We have exact year match, there are duplicate movies
                # pointing to the same movie for some reason just take the first one as in paper
                match = match[match_index].iloc[:1]
                match_index = match_index.iloc[:1]
            else:
                # Check if year differs by 1, noticed this occurs sometimes due
                # to missmatch with dbpedia and movielens
                if not match_index.any():
                    match_index = (match.year >= row['year'] - 1) & (match.year <= row['year'] + 1)

                # Check if we do not have the year
                if not match_index.any():
                    match_index = match.year == -1

                if not match_index.any():
                    year_miss.append((row, match))
                    continue
                match = match[match_index]

        # No matches
        if match_index.sum() == 0:
            continue

        # More than 1 match....
        if match_index.sum() > 1:
            multiple_matches.append((row, match))
            # Following paper take the first occurence
            match = match.sort_values('year').iloc[:-1]

        match = match.to_dict('list')
        matches.append({
            'ml_id': match['ml_id'][0],
            'ml_year': match['year'][0],
            'ml_title': match['title'][0],
            'id': row['movie_id'],
            'title': row['title'],
            'year': row['year']
        })
    print("Matched {:,} / {:,};  {:,} Matched multiple movie titles, {:,} Years missmatched, {:,} no match".format(
            len(matches), len(movies_df), len(multiple_matches), len(year_miss), len(missing)))

    matches = pd.DataFrame.from_dict(matches)
    matches.to_csv(opt.output, index=False, encoding='utf-8')
