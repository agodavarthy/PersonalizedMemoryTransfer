import pickle
from collections import deque
import json
from typing import List
import argparse
import os
import sys
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))


from tqdm import tqdm
import regex
import pandas as pd
import numpy as np

from util.preprocess_archana import tokenizer, TextEncoder
from util.helper import parser_add_str2bool
from util.data import Movie


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_add_str2bool(parser)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str,
                    default="0")
parser.add_argument('-ds', '--dataset', help='dataset', type=str,
                    default="redial")

parser.add_argument('-o', '--output', help='output filename', type=str,
                    required=True)
parser.add_argument('-enc', '--encoder', help='Type of text encoder to use, '
                                              'choices: transformer, elmo, dan, '
                                              'nnlm, bert or a custom path',
                    type=str, required=True)
parser.add_argument('-movie_map', '--movie_map', help='movie_map.csv path', type=str,
                    default="data/full/movie_map.csv", required=True)
parser.add_argument('-movie_plot', '--movie_plot', help='movie_plot.csv path', type=str,
                    default="data/movie_plot.csv")
parser.add_argument('-redial', '--redial_path', help='path to redial dataset folder', type=str,
                    default="data/redial_dataset/")

parser.add_argument('-tok', '--tokenizer', help='tokenizer type', type=str,
                    default="split", choices=['spacy', 'split', 'nltk'])
# parser.add_argument('-m', '--max_examples', help='Max number of examples to '
#                                                  'store in pickle file, eg for '
#                                                  'validation set (-1 disables)',
#                     type=int, default=-1)
parser.add_argument('-hl', '--history_length', help='Number of exchanges in the '
                                                    'conversation to keep for '
                                                    'text encoder',
                    type=int, default=2)
parser.add_argument('-test', '--test', help='Use testing dataset', type='bool',
                    default=False)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)
parser.add_argument('-debug', '--debug', help='Use debug mode?', type='bool',
                    default=False)

parser.add_argument('-min', '--min_movies', help='Minimum number of movies the '
                                                 'user must mention',
                    type=int, default=-1)

parser.add_argument('-bert_num', '--bert_num', help='bert num ',
                    type=str, default='0')

#CONV_SPLITS_FILENAME = '/home/ywang/convmovie/data/gorecdial/conv_split_ids.pkl'
CONV_SPLITS_FILENAME = 'data/gorecdial/conv_split_ids.pkl'

def extract_year(title: str):
    """
    Remove the year from the Title and return the cleaned up version and year
    :param title:
    :return:
    """
    year = -1
    match = regex.search('\((\d{4})\)$', title.strip())
    if match:
        year = int(match.group(1).strip())
        title = title[:match.start()].strip()
    return title, year


def fix_title(title: str):
    """
    Movies sometimes are
    "NAME, The"  ==>  "The NAME"

    Change & ==> and
    :param title:
    :return:
    """
    if title.endswith(", The"):
        title = "The " + title[:-len(", The")]
    title, _ = extract_year(title)
    title = title.replace("&", "and").replace("  ", " ")
    return title


def preprocess_title(title: str, tokenizer_method: str='spacy'):
    """

    :param title:
    :return:
    """
    return " ".join(tokenizer(fix_title(title), tokenizer_method))


def preprocess_plot(plot: str, tokenizer_method: str='spacy'):
    """
    Perform preprocessing

    :param plot:
    :return:
    """
    # remove_list = ["::Claudio Carvalho, Rio de Janeiro, Brazil",
    #                "::Sujit R. Varma"]
    #
    # # Remove plot withs crediting the authors
    # # EG ".... Who will win?::Kris Hopson"
    # match = regex.search(r"::\w+\s{0,1}\w+$", plot)
    #
    # # If not then find some with ending with an email
    # # EG "::<jhailey@hotmail.com >"
    # # < and > are optional
    # if not match:
    #     match = regex.search(r"::<?\w+@\w+\.\w{,3}\s?>?$", plot)
    #
    # # .::Otaku - sempai
    # if not match:
    #     match = regex.search(r"::\w+ - \w+$", plot)
    #
    # if match:
    #     plot = plot[:match.start()]
    # else:
    #     for s in remove_list:
    #         plot = plot.replace(s, "")
    #
    idx = plot.find("::")

    if idx != -1:
        plot = plot[:idx]

    return " ".join(tokenizer(plot, tokenizer_method))


def test_preprocess():
    """Check preprocessing results"""
    s = "Who will win?::Kris Hopson"
    assert preprocess_plot(s) == "Who will win ?"

    s = "an asset?::<jhailey@hotmail.com >"
    assert preprocess_plot(s) == "an asset ?"

    s = "place she saw.::rcs0411@yahoo.com"
    assert preprocess_plot(s) == "place she saw ."

    assert preprocess_plot("Rough Cut.::Otaku - sempai") == "Rough Cut ."

    assert preprocess_plot("zombies in Las Vegas.::Claudio Carvalho, Rio de Janeiro, Brazil") == "zombies in Las Vegas ."
    assert preprocess_plot("relationships.::Sujit R. Varma") == "relationships ."


def replace_movie(s: str, row):
    """
    Replaces the movie id with the corresponding movie name

    :param s: utterance with movie ids
    :param row: pd.Series from the redial dataset, containing moviementions
    :return: string with replaced movie names, list of movie ids
    """
    # TODO: Fix how we do this...
    mentions = row['movieMentions']
    questions = row['initiatorQuestions']
    match = regex.search(r"(@\d+)", s)
    ids = []

    while match:
        start, end = match.span()
        movie_id = match.group(0)[1:]
        # ConvMovie: ID
        ids.append(movie_id)
        if movie_id not in mentions:
            print("Could not find movie in mentions...", movie_id)
            movie_name = "movie"
        else:
            movie_name = mentions[movie_id].strip()
        # Seeker mentioned
        s = f"{s[:start]} {movie_name} {s[end:]}"
        match = regex.search(r"(@\d+)", s)
    # Remove double whitespace
    s = regex.sub(r"\s{2,}", " ", s).strip()
    return s, ids


# def construct_episodes_old(save_filename: str, movies_list: List[Movie],
#                        encoder: TextEncoder, opt):
#     """
#
#     movies_list -> List[Movies], Attributes: movie_id, ml_id, matrix_id, title, plot
#
#     :param save_filename:
#     :param movies_list:
#     :param encoder:
#     :return:
#     """
#     HISTORY_LEN = opt.history_length
#     if opt.test:
#         print("Using test dataset....")
#
#     filename = os.path.join(opt.redial_path,
#                             "test_data.jsonl" if opt.test else "train_data.jsonl")
#
#     with open(filename, 'r') as f:
#         data = [json.loads(line) for line in f]
#
#     if opt.max_examples != -1:
#         print(f"Truncating to {opt.max_examples} episodes")
#         np.random.shuffle(data)
#     # if BERT use [SEP] token else .
#     SEP_TOKEN = " [SEP] " if opt.encoder == 'bert' else " . "
#     data = pd.DataFrame.from_dict(data)
#     print("Encoding movie plots")
#
#     # Encode Movie Title/Plot
#     for movie in tqdm(movies_list, total=len(movies_list)):
#         movie.title = preprocess_title(movie.title, opt.tokenizer)
#         movie.plot = preprocess_plot(movie.plot, opt.tokenizer)
#         movie.text = movie.title + SEP_TOKEN + movie.plot
#         # Encode Movie Title/Plot
#         movie.vector = encoder.encode(movie.text)[0]
#
#     # Movie Mapping
#     movieid_to_movie = {str(movie.movie_id): movie for movie in movies_list}
#
#     # Replace included movieMention titles with our preprocessed version
#     for idx, mentions in data.movieMentions.iteritems():
#         if len(mentions):
#             data.at[idx, 'movieMentions'] = {
#                 str(k): movieid_to_movie[k].title
#                 if k in movieid_to_movie else fix_title(v)
#                 for k, v in mentions.items() if v}
#
#     # Preprocess our examples
#     episodes = []
#     print("Starting preprocessing....")
#
#     for _, row in tqdm(data.iterrows(), total=opt.max_examples):
#         # No movies mentioned...
#         if len(row.movieMentions) == 0:
#             continue
#
#         example = []
#         convId = row.conversationId
#         # Seeker ID
#         seeker_id = row.initiatorWorkerId
#         # Recommender ID
#         rec_id = row.respondentWorkerId
#         prev_id = None
#         # Questions = {'203371': {'suggested': 1, 'seen': 0, 'liked': 1}}
#         questions = row['initiatorQuestions']
#         seen = set()
#         for i, msg in enumerate(row.messages):
#             is_seeker = msg['senderWorkerId'] == seeker_id
#
#             # Replace movie ids with names, return ids
#             text, ids = replace_movie(msg['text'], row)
#
#             movie_matches = []
#             for _id in ids:
#                 # If we have the movie and they like it or did not say
#                 if _id in movieid_to_movie:
#                     # If seeker and liked the movie, sometimes missing in questions
#                     # We only ignore if its a strong preference eg they do not like it
#                     if len(questions) and is_seeker \
#                             and _id in questions \
#                             and questions[_id]['liked'] != 0:
#                         continue
#
#                     # We do not have this movie in the item matrix
#                     if movieid_to_movie[_id].matrix_id == -1:
#                         raise Exception("We should have fixed this in preprocessing, double checking here")
#
#                     ml_id = movieid_to_movie[_id].ml_id
#                     if ml_id not in seen:
#                         movie_matches.append(ml_id)
#                         seen.add(ml_id)
#
#             # Perform tokenization then rejoin since the model performs
#             # whitespace tokenization
#             text = " ".join(tokenizer(text, opt.tokenizer))
#
#             # add to previous, join together if same speaker
#             if msg['senderWorkerId'] == prev_id:
#                 example[-1]['text'] += SEP_TOKEN + text
#                 example[-1]['movie_id'] += ids
#                 example[-1]['ml_id'] += movie_matches
#             else:
#                 # New Speaker
#                 # Text, Movie IDs, MovieLens IDS
#                 example.append(
#                     dict(seeker=is_seeker, text=text, movie_id=ids,
#                          ml_id=movie_matches, convId=convId))
#             prev_id = msg['senderWorkerId']
#
#         # Number of movie mentions
#         n_movies = sum([len(ex['ml_id']) for ex in example])
#
#         # Skip empty or does not meet criteria
#         if n_movies == 0 or n_movies <= opt.min_movies:
#             continue
#
#         # Truncate the end if no other recommendations are performed
#         while len(example[-1]['ml_id']) == 0:
#             example.pop()
#
#         if len(example) == 0:
#             raise Exception(f"Example has 0 entries... ConvId: {convId}")
#
#         # Keep a history length of dialog to 2
#         history = deque(maxlen=HISTORY_LEN)
#         # Encode each text
#         for i, ex in enumerate(example):
#             history.append(ex['text'])
#             # Query vector is seen after we perform evaluation important!
#             output = encoder.encode(SEP_TOKEN.join(history), as_dict=True)
#
#             # If we have sentiment
#             if 'logistic' in output:
#                 example[i]['query_vec'] = output['embeddings'][0]
#                 example[i]['sentiment'] = output['logistic'][0]
#             else:
#                 example[i]['query_vec'] = output['default'][0]
#
#         if len(example):
#             episodes.append(example)
#
#         if len(episodes) == opt.max_examples:
#             break
#
#     print(f"Total Episodes: {len(episodes)}")
#     print("Encoding movies")
#     movies_list = [movie.to_dict() for movie in movies_list]
#     print(f"DONE!\nSaving to file: {save_filename}")
#     pickle.dump({'episodes': episodes,
#                  'movies': movies_list,
#                  }, open(save_filename, 'wb'))


def construct_episodes(save_filename: str, movies_list: List[Movie],
                       encoder: TextEncoder, opt, test_convids):
    """

    movies_list -> List[Movies], Attributes: movie_id, ml_id, matrix_id, title, plot

    :param save_filename:
    :param movies_list:
    :param encoder:
    :return:
    """
    HISTORY_LEN = opt.history_length
    if opt.test:
        print("Using test dataset....")

    data = []

    # load testing data
    with open(os.path.join(opt.redial_path, "test_data.jsonl"), 'r') as f:
        data.extend([json.loads(line) for line in f])

    # Load training data
    with open(os.path.join(opt.redial_path, "train_data.jsonl"), 'r') as f:
        data.extend([json.loads(line) for line in f])

    # if BERT use [SEP] token else .
    SEP_TOKEN = " [SEP] " if opt.encoder == 'bert' else " . "
    data = pd.DataFrame.from_dict(data)
    print("Encoding movie plots")

    # Encode Movie Title/Plot
    for movie in tqdm(movies_list, total=len(movies_list)):
        movie.title = preprocess_title(movie.title, opt.tokenizer)
        movie.plot = preprocess_plot(movie.plot, opt.tokenizer)
        movie.text = movie.title + SEP_TOKEN + movie.plot
        # Encode Movie Title/Plot
        movie.vector = encoder.encode(movie.text)[0]

    # Movie Mapping
    movieid_to_movie = {str(movie.movie_id): movie for movie in movies_list}

    # Replace included movieMention titles with our preprocessed version
    for idx, mentions in data.movieMentions.iteritems():
        if len(mentions):
            data.at[idx, 'movieMentions'] = {
                str(k): movieid_to_movie[k].title
                if k in movieid_to_movie else fix_title(v)
                for k, v in mentions.items() if v}

    test_convids = set(test_convids)
    # Preprocess our examples
    episodes = []
    print("Starting preprocessing....")
    for _, row in tqdm(data.iterrows(), total=len(data)):
        # No movies mentioned...
        if len(row.movieMentions) == 0:
            continue

        example = []
        convId = row.conversationId
        if convId not in test_convids:
            continue

        # Seeker ID
        seeker_id = row.initiatorWorkerId
        # Recommender ID
        rec_id = row.respondentWorkerId
        prev_id = None
        # Questions = {'203371': {'suggested': 1, 'seen': 0, 'liked': 1}}
        questions = row['initiatorQuestions']
        seen = set()
        for i, msg in enumerate(row.messages):
            is_seeker = msg['senderWorkerId'] == seeker_id

            # Replace movie ids with names, return ids
            text, ids = replace_movie(msg['text'], row)

            movie_matches = []
            for _id in ids:
                # If we have the movie and they like it or did not say
                if _id in movieid_to_movie:
                    # If seeker and liked the movie, sometimes missing in questions
                    # We only ignore if its a strong preference eg they do not like it
                    if len(questions) and is_seeker \
                            and _id in questions \
                            and questions[_id]['liked'] != 0:
                        continue

                    # We do not have this movie in the item matrix
                    if movieid_to_movie[_id].matrix_id == -1:
                        raise Exception("We should have fixed this in preprocessing, double checking here")

                    ml_id = movieid_to_movie[_id].ml_id
                    if ml_id not in seen:
                        movie_matches.append(ml_id)
                        seen.add(ml_id)

            # Perform tokenization then rejoin since the model performs
            # whitespace tokenization
            text = " ".join(tokenizer(text, opt.tokenizer))

            # add to previous, join together if same speaker
            if msg['senderWorkerId'] == prev_id:
                example[-1]['text'] += SEP_TOKEN + text
                example[-1]['movie_id'] += ids
                example[-1]['ml_id'] += movie_matches
            else:
                # New Speaker
                # Text, Movie IDs, MovieLens IDS
                example.append(
                        dict(seeker=is_seeker, text=text, movie_id=ids,
                             ml_id=movie_matches, convId=convId))
            prev_id = msg['senderWorkerId']

        # Number of movie mentions
        n_movies = sum([len(ex['ml_id']) for ex in example])

        # Skip empty or does not meet criteria
        if n_movies == 0 or n_movies <= opt.min_movies:
            continue

        # Truncate the end if no other recommendations are performed
        while len(example[-1]['ml_id']) == 0:
            example.pop()

        if len(example) == 0:
            raise Exception(f"Example has 0 entries... ConvId: {convId}")

        # Keep a history length of dialog to 2
        history = deque(maxlen=HISTORY_LEN)
        # Encode each text
        for i, ex in enumerate(example):
            history.append(ex['text'])
            # Query vector is seen after we perform evaluation important!
            output = encoder.encode(SEP_TOKEN.join(history), as_dict=True)

            # If we have sentiment
            if 'logistic' in output:
                example[i]['query_vec'] = output['embeddings'][0]
                example[i]['sentiment'] = output['logistic'][0]
            else:
                example[i]['query_vec'] = output['default'][0]

        if len(example):
            episodes.append(example)

    print(f"Total Testing Episodes: {len(episodes)}")

    movies_list = [movie.to_dict() for movie in movies_list]
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    pickle.dump({'movies': movies_list, 'data': episodes},
                open(save_filename, 'wb'))
    print("DONE!")


def print_statistics(episodes, movies_list):
    """
    Compute some statistics for data

    :param episodes: List
    :param movies_list: List of movies
    """
    avg_episode_len = np.mean([len(ex) for ex in episodes])
    print("{:<50} {:.4f}".format("Avg Episode Length", avg_episode_len))

    valid_movies = [sum([len(ex['ml_id']) for ex in episode]) for episode in episodes]
    print("\nValid Movies")
    print("{:<50} {:.4f}".format("Avg Movies/Episode", np.mean(valid_movies)))
    print("{:<50} {:.4f}".format("Min Movies/Episode", np.min(valid_movies)))
    print("{:<50} {:.4f}".format("Max Movies/Episode", np.max(valid_movies)))

    n_tokens = [len(movie.plot_tokens) + len(movie.title_tokens) for movie in movies_list]
    print("\nMovie Summary Statistics")
    print("{:<50} {:.4f}".format("Avg Tokens", np.mean(n_tokens)))
    print("{:<50} {:.4f}".format("Min Tokens", np.min(n_tokens)))
    print("{:<50} {:.4f}".format("Max Tokens", np.max(n_tokens)))


class MovieInspector(object):

    def __init__(self, movies):
        self.movies = movies
        self.movies.movie_id = self.movies.movie_id.astype(str)
        self.movies_list = [Movie.from_dict(m) for m in movies.to_dict('records')]
        self.movies_dict = {str(m.movie_id): m for m in movies_list}

    def get_row(self, movie_id):
        return self.movies.loc[self.movies['movie_id'] == movie_id].iloc[0]


def main():
    test_preprocess()
    opt = parser.parse_args()
    print(opt)
    print(f"Loading {opt.movie_map}")
    df = pd.read_csv(opt.movie_map)

    print(f"Loading {opt.movie_plot}")
    movies = pd.read_csv(opt.movie_plot)
    merged = df.merge(movies, on='movie_id').rename(columns={'ml_title': 'title',
                                                             'title': 'imdb_title',
                                                             'ml_id_x': 'ml_id'})
    assert len(df) == len(merged), 'miss match on merging'
    print(f"Final amount of matched movies {len(movies)}")
    movies_list = [Movie.from_dict(m) for m in merged.to_dict('records')]

    # print("Reading in movies.csv for genres")
    # movielens = pd.read_csv("data/ml-latest/movies.csv")
    #
    # movielens['genres'] = movielens['genres'].str.split("|")
    # genre_map = {mlid: genres for mlid, genres in
    #              zip(movielens['movieId'].values, movielens['genres'].values)}
    #
    # for movie in movies_list:
    #     movie.genres = genre_map[movie.ml_id]

    if opt.debug:
        SEP_TOKEN = " [SEP] " if opt.encoder == 'bert' else " . "
        print()
        np.random.shuffle(movies_list)
        show = 5
        # Encode Movie Title/Plot
        for movie in movies_list:
            show -= 1
            movie.title = preprocess_title(movie.title, opt.tokenizer)
            movie.plot = preprocess_plot(movie.plot, opt.tokenizer)
            print(movie)
            print("PLOT: " + movie.plot)
            print("=" * 80)

            if show == 0:
                cont = input("Any key to continue (q to quit): ").strip()
                if cont == 'q':
                    sys.exit()
                show = 5
                print()
                print('-' * 80)
                print()
        print()
    else:
        np.random.seed(opt.seed)
        # Initialize Text Encoder

        with open(CONV_SPLITS_FILENAME, 'rb') as f:
            splits = pickle.load(f)

        for fold_index in range(len(splits)):
            print("Loading pretrained models....")
            encoder = TextEncoder(opt.gpu, opt.bert_num, opt.encoder + "{}/module".format(fold_index),
                                  is_bert='bert' in opt.encoder)
            path = os.path.join(opt.output, str(fold_index) + ".pkl")
            print(f"Preprocessing episodes to {path}")
            construct_episodes(path, movies_list, encoder, opt,
                               splits[fold_index]['test'])
        # fold_index=4
        # print("Loading pretrained models....")
        # encoder = TextEncoder(opt.gpu, opt.encoder + "{}/module".format(fold_index),
        #                       is_bert='bert' in opt.encoder)
        # path = os.path.join(opt.output, str(fold_index) + ".pkl")
        # print(f"Preprocessing episodes to {path}")
        # construct_episodes(path, movies_list, encoder, opt,
        #                    splits[fold_index]['test'])


if __name__ == '__main__':
    main()

