"""
Split ratings data, keeping only movies we matched up and write out the
movie_map which includes the mapping to matrix_ids

inputs: movie_match.csv -- from match_movies.py
        ratings.csv -- from movielens dataset

outputs: movie_map.csv -- Entire mapping between movies, movielens and matrix ids


    train.npz, test.npz -- These are split ratings with matched movie items only 90/10 split

    cv/train.npz, cv/test.npz -- this train is 80% of ratings and test is 10%,
                                 essentially making it a 80/10/10 train/valid/test split

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import coo_matrix
import os
# import sys
# sys.path.append("/home/tebesu/notebooks/src/")
from collections import Counter
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output', help='what directory to write split in', type=str,
                    required=True)
parser.add_argument('-movie_match', '--movie_match', help='matched up movies - movie_match.csv path', type=str,
                    default="data/movie_match.csv")
parser.add_argument('-ratings', '--ratings', help='path to ratings.csv from movielens',
                    type=str, required=True)


def create_mapping(values):
    """
    Given a list of values we create a dictionary/vocabulary mapping from 0 to n
    ie [a, b, c, c]
    results in mapping {a: 0, b: 1, c: 2}

    Basically create a vocabulary from a set of hashable list of values

    :param values: list of values (converted to a set to be unique)
    :returns: dictionary of mapping
    """
    values = list(set(values))
    return {v: i for i, v in enumerate(values)}


def save_sparse_matrix(fname, matrix):
    """
    Saves a sparse matrix row, col and data as a npz, faster than
    market matrix format

    :param fname:
    :param matrix:
    :return:
    """
    np.savez(fname,
             row=matrix.row, col=matrix.col, data=matrix.data,
             shape=matrix.shape)


def check_coldstart(triplets, user_count: int, item_count: int):
    """
    Check if we have a cold-start scenario, if the number of unique users/items
    dont match up to our initial counts

    :param triplets: array of triplets [user, item, rating]
    :param user_count: number of users expected
    :param item_count: number of items expected
    :return: if its cold-start or not
    """
    items = set(triplets[:, 1])
    if len(items) != item_count:
        print("Missing {:,} Items".format(item_count - len(items)))
        return True
    users = set(triplets[:, 0])
    if len(users) != user_count:
        print("Missing {:,} Users".format(user_count - len(users)))
        # print "Users Missing"
        return True
    return False


if __name__ == '__main__':
    opt = parser.parse_args()
    data = pd.read_csv(opt.ratings)
    print(f"Loaded {len(data):,} Ratings")

    movies = pd.read_csv(opt.movie_match, usecols=[0, 1]).rename(
            columns={'id': 'movie_id'})
    print(f"Loaded {len(movies):,} matched movies")
    save_dir = opt.output
    os.makedirs(save_dir, exist_ok=True)
    test_size = 0.1
    valid_size = 0.1

    # Filter out users with less than xxx ratings
    min_ratings_per_user = 5

    # Min ratings per a user/item required to have it in testing set
    min_item_ratings_test = 5
    min_user_ratings_test = 5

    # reproduce split
    random_state = 2555
    np.random.seed(29)

    # MovieLens Ratings Keys
    user_key = 'userId'
    item_key = 'movieId'
    time_key = 'timestamp'
    date_args = dict(unit='s')

    # Here we can match up movies together
    movies = movies[(movies.movie_id != -1) & (movies.ml_id != -1)]
    movies.sort_values('ml_id', inplace=True)

    valid_movie_ids = set(movies.ml_id.values)

    # IF we want to keep the movies in the ReDial dataset
    data = data[data.movieId.isin(valid_movie_ids)]

    # Optional Remove if < x ratings
    remove_users = set([g[1][user_key].values[0] for g in data.groupby(user_key)
                        if len(g[1]) < min_ratings_per_user])
    data = data[~data[user_key].isin(remove_users)]

    # Map users/items to unique ids
    user_mapping = create_mapping(data[user_key].values)
    item_mapping = create_mapping(data[item_key].values)
    user_count = len(user_mapping)
    item_count = len(item_mapping)

    data['userid'] = data[user_key].apply(lambda x: user_mapping[x])
    data['itemid'] = data[item_key].apply(lambda x: item_mapping[x])

    print(
        "{:<20} {:>12,}\n"
        "{:<20} {:>12,}\n"
        "{:<20} {:>12,}".format("User Count", user_count, "Item Count", item_count, "Ratings", len(data)))
    _counts = data.groupby("itemid").count()
    print("Item Count Min/Max: {:,} / {:,}".format(_counts.min().values[0], _counts.max().values[0]))
    _counts = data.groupby("userid").count()
    print("User Count Min/Max: {:,} / {:,}".format(_counts.min().values[0], _counts.max().values[0]))
    # _full_data = data

    ratings = coo_matrix((data.rating, (data.userid, data.itemid))).astype(np.float16)
    cv_count = int(ratings.nnz * valid_size)

    print(ratings.shape, f"{ratings.getnnz():,}")

    rating_triplets = np.array(list(zip(ratings.row, ratings.col, ratings.data)))
    training_only = []
    item_training_only = set()
    user_training_only = set()
    print("\nFiltering out Users/Items....")
    # Filter out users/items with only a small number of ratings keep in training only
    for i in range(10):
        # Item-Based Filtering
        classes, y_indices = np.unique(rating_triplets[:, 1], return_inverse=True)
        class_counts = np.bincount(y_indices)
        item_training_only.update(set(np.where(class_counts < min_item_ratings_test)[0]))

        # User-Based Filtering
        _user_classes, _user_indices = np.unique(rating_triplets[:, 0], return_inverse=True)
        _user_counts = np.bincount(_user_indices)
        user_training_only.update(set(np.where(_user_counts < min_user_ratings_test)[0]))

        # If they occur only n time, keep in training set
        # If not keep in rating_triplets
        print("Iteration/Items/Users: {} / {:,} / {:,} ".format(i, len(item_training_only), len(user_training_only)))
        temp = []
        for triplet in rating_triplets:
            if triplet[0] in user_training_only or triplet[1] in item_training_only:
                training_only.append(triplet)
            else:
                temp.append(triplet)

        rating_triplets = np.array(temp)
        # Check if we have only a single item occurence then we have a problem
        counts = Counter(rating_triplets[:, 1])
        number_singles = [k for k, v in counts.most_common()[::-1] if v == 1]
        print(len(number_singles))

        if len(number_singles) == 0:
            print("SUCCESS")
            break

    print("Items only in Training: {:,}".format(len(item_training_only)))
    print("Users only in Training: {:,}".format(len(user_training_only)))
    print("Ratings: {:,}".format(len(rating_triplets)))
    print("\nDONE!")
    print("\nSpliting Train/Test")
    # Perform actual train/test split
    count = 0
    while True:
        splitter = StratifiedShuffleSplit(1, random_state=random_state,
                                          test_size=test_size)
        # Item Split is 1
        for train_index, test_index in splitter.split(
                np.zeros(len(rating_triplets)), rating_triplets[:, 1]):
            break
        train = rating_triplets[train_index]
        # add training only data
        if training_only:
            train = np.concatenate([train, training_only], axis=0)

        test = rating_triplets[test_index]
        assert len(train) + len(test) == ratings.nnz
        train_items = set(train[:, 1])
        train_users = set(train[:, 0])
        print("\nIteration", count)
        # Check Cold-Start on training set
        if not check_coldstart(train, len(user_mapping), len(item_mapping)):
            break
        count += 1
        if count > 10:
            raise Exception("COULD NOT FIND A SPLIT!")
        random_state += 1

    print("FOUND!!!!")
    # Create train/test split
    print("Saving train/test split")
    test = coo_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(user_count, item_count)).astype(np.float16)
    train = coo_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(user_count, item_count)).astype(np.float16)

    save_sparse_matrix(save_dir + "/test", test)
    save_sparse_matrix(save_dir + "/train", train)
    # Create a validation split
    # May have cold-start
    indices = np.arange(train.nnz)
    np.random.shuffle(indices)
    valid_index = indices[:cv_count]
    train_index = indices[cv_count:]
    print("Saving cross-validation split")
    cv_valid = coo_matrix((train.data[valid_index], (train.row[valid_index], train.col[valid_index])), shape=(user_count, item_count)).astype(np.float16)
    cv_train = coo_matrix((train.data[train_index], (train.row[train_index], train.col[train_index])), shape=(user_count, item_count)).astype(np.float16)
    os.makedirs(save_dir + "/cv/", exist_ok=True)
    save_sparse_matrix(save_dir + "/cv/test", cv_valid)
    save_sparse_matrix(save_dir + "/cv/train", cv_train)
    print("DONE!")

    print("Saving Matrix Ids...")
    # Save matrix ids
    movies['matrix_id'] = movies.ml_id.apply(lambda x: item_mapping.get(x, -1))
    # movies = movies.rename({'databaseId': 'movie_id', 'movielensId': 'ml_id'}, axis=1)
    movies = movies[movies['matrix_id'] != -1].sort_values('matrix_id')
    movies.to_csv(save_dir + "/movie_map.csv", index=False)
    print("DONE!")
