#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   Travis A. Ebesu
@created:
@summary:  Dataset provider
"""
import numpy as np
from collections import defaultdict
import bottleneck
import torch.utils.data
import torch
import torch.nn.functional as F
from scipy.sparse.coo import coo_matrix


class Movie(object):

    def __init__(self, movie_id: str, ml_id, matrix_id: int,
                 title: str = None, plot: str = None,
                 imdb_id: str = None, tmdb_id: str = None,
                 vector=None,
                 title_tokens=None,
                 plot_tokens=None, text=None, genres=None):
        """
        Data Structure to map various elements of a given movie

        :param movie_id: ReDial Movie ID
        :param ml_id: MovieLens identifier
        :param matrix_id: int, Index into our item embeddings
        :param title: str, title of the film
        :param plot: str, plot/summary of movie
        :param imdb_id:  IMDB identifier
        :param tmdb_id: TMDB identifier
        """
        self.movie_id = movie_id
        self.ml_id = ml_id
        self.imdb_id = imdb_id
        self.tmdb_id = tmdb_id
        self.matrix_id = matrix_id

        self.title = title  # Title from Redial
        self.ml_title = None  # Movie Lens Title
        self.plot = plot  # Plot as string
        self.genres = genres

        self.vector = vector  # Vector representation
        self.title_tokens = title_tokens  # Title tokenized
        self.plot_tokens = plot_tokens  # Plot tokenized
        self.source = None  # Plot source eg IMDB or TMDB
        self.text = text  # Full title concatenated with plot text
        self._keys = ['movie_id', 'ml_id', 'imdb_id', 'tmdb_id', 'matrix_id',
                      'title', 'ml_title', 'plot', 'vector', 'title_tokens',
                      'plot_tokens', 'source', 'text', 'genres']

    def to_dict(self):
        """
        Convert this Movie object to a dictionary
        """
        item = {}
        for key in self._keys:
            item[key] = getattr(self, key)
        return item

    @staticmethod
    def from_dict(item: dict):
        """
        Create a Movie object from a dictionary

        :param item: Dictionary to load, requires movie_id, ml_id, matrix_id
        :return: Movie
        """
        movie = Movie(item['movie_id'], item['ml_id'], item['matrix_id'])
        for key in movie._keys:
            setattr(movie, key, item.get(key))
        return movie

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return f"<Movie: \"{self.title}\" - ID:{self.movie_id}, ML_ID:{self.ml_id}, Matrix:{self.matrix_id}>"


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


def load_sparse_matrix(fname):
    """
    Load sparse matrix from npz and return as coo_matrix

    Use save_sparse_matrix
    :param fname:
    :return:
    """
    data = np.load(fname)
    matrix = coo_matrix((data['data'], (data['row'], data['col'])),
                        shape=data['shape'])
    return matrix


class Dataset(object):

    def __init__(self, datadir: str):
        """
        Wraps dataset and produces batches for the model to consume
        """
        print("[Loading Ratings]")
        # Took: 42.38213324546814
        self.train = load_sparse_matrix(f"{datadir}/train.npz")
        self.test = load_sparse_matrix(f"{datadir}/test.npz")

        # Mtx slow! Took: 89.56421852111816
        # self.train = mmread(f"{datadir}/train.mtx")
        # self.test = mmread(f"{datadir}/test.mtx")

        self._n_users, self._n_items = self.train.shape

        # Index to access each item
        self._train_index = np.arange(self.train.nnz)

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)

        for u, i in zip(self.train.row, self.train.col):
            self.user_items[u].add(i)
            self.item_users[i].add(u)

        self.train_relevance = self.user_items

        # Remove defaultdict for sampling <- Can cause problems!!
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

        # Get a list version so we do not need to perform type casting
        # set version is for checking membership in constant while list version
        # is for sampling
        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        self.user_items_list = {k: list(v) for k, v in self.user_items.items()}

        # Compute the max number of neighbors for our computational graph
        self.max_user_neighbors = max([len(x) for x in self.item_users.values()])
        self.max_item_neighbors = max([len(x) for x in self.user_items.values()])

        # Index to compute relevance. user_id: [rated items]
        self.test_relevance = defaultdict(set)

        for u, i in zip(self.test.row, self.test.col):
            self.test_relevance[u].add(i)

    @property
    def train_size(self) -> int:
        """
        number of examples in training set
        """
        return self.train.nnz

    @property
    def test_size(self) -> int:
        """
        number of examples in test set
        """
        return self.test.nnz

    @property
    def user_count(self) -> int:
        """
        Number of users in dataset
        """
        return self._n_users

    @property
    def item_count(self) -> int:
        """
        Number of items in dataset
        """
        return self._n_items

    def _sample_item(self) -> int:
        """
        Draw an item item index uniformly
        """
        return np.random.randint(0, self.item_count)

    def _sample_negative_item(self, user_id: int):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        n = self._sample_item()
        positive_items = self.user_items[user_id]

        if len(positive_items) >= self.item_count:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.item_count))

        while n in positive_items or n not in self.item_users:
            n = self._sample_item()
        return n

    def get_data_uniform(self, batch_size: int, samples_per_user: int):
        """Samples num users * samples per a user,
        uniformly selects a user
        uniformly select item and neg item
        """
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        idx = 0
        for i in range(self.user_count*samples_per_user):
            user_idx = np.random.randint(0, self.user_count)
            item_idx = np.random.randint(0, len(self.user_items_list[user_idx]))
            neg_item_idx = self._sample_negative_item(user_idx)
            batch[idx, :] = [user_idx, item_idx, neg_item_idx]
            idx += 1
            if idx == batch_size:
                yield batch
                idx = 0  # Reset

    def get_xent_data(self, batch_size: int, neg_count: int):
        """
        Batch data together as (user, item, negative item), pos_neighborhood,
        length of neighborhood, negative_neighborhood, length of negative neighborhood

        if neighborhood is False returns only user, item, negative_item so we
        can reuse this for non-neighborhood-based methods.

        :param batch_size: size of the batch
        :param neg_count: number of negative samples to uniformly draw per a pos
                          example
        :return: generator
        """
        # Allocate inputs
        batch = np.zeros((batch_size, 2), dtype=np.int64)
        y = np.zeros(batch_size, dtype=np.float32)
        # Shuffle index
        np.random.shuffle(self._train_index)

        idx = 0
        for user_idx, item_idx in zip(self.train.row[self._train_index], self.train.col[self._train_index]):
            batch[idx, :] = [user_idx, item_idx]
            y[idx] = 1.0
            idx += 1
            # Yield batch if we filled queue
            if idx == batch_size:
                yield batch, y
                # Reset
                idx = 0

            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, neg_item_idx]
                y[idx] = 0.0
                idx += 1
                # Yield batch if we filled queue
                if idx == batch_size:
                    yield batch, y
                    # Reset
                    idx = 0

    def get_data(self, batch_size: int, neighborhood: bool, neg_count: int):
        """
        Batch data together as (user, item, negative item), pos_neighborhood,
        length of neighborhood, negative_neighborhood, length of negative neighborhood

        if neighborhood is False returns only user, item, negative_item so we
        can reuse this for non-neighborhood-based methods.

        :param batch_size: size of the batch
        :param neighborhood: return the neighborhood information or not
        :param neg_count: number of negative samples to uniformly draw per a pos
                          example
        :return: generator
        """
        # Allocate inputs
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        if neighborhood:
            pos_neighbor = np.zeros((batch_size, self.max_user_neighbors), dtype=np.int32)
            pos_length = np.zeros(batch_size, dtype=np.int32)
            neg_neighbor = np.zeros((batch_size, self.max_user_neighbors), dtype=np.int32)
            neg_length = np.zeros(batch_size, dtype=np.int32)

        # Shuffle index
        np.random.shuffle(self._train_index)

        idx = 0
        for user_idx, item_idx in zip(self.train.row[self._train_index], self.train.col[self._train_index]):
            # TODO: set positive values outside of for loop
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                if neighborhood:
                    if len(self.item_users[item_idx]) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(self.item_users[neg_item_idx]) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx

                idx += 1
                # Yield batch if we filled queue
                if idx == batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                              neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                      neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]


class WeightedContentDataset(Dataset, torch.utils.data.Dataset):
    def __init__(self, datadir: str, content_file: str, neg_count: int,
                 top_similar: int, weighted_count: int, apply_softmax: bool=True):
        """

        :param datadir:
        :param content_file:
        :param neg_count:
        :param top_similar: The number of top and bottom similar items to consider
        :param weighted_count: Number of items to use for weighted latent factor
        """
        torch.utils.data.Dataset.__init__(self)
        Dataset.__init__(self, datadir)
        self.neg_count = neg_count
        self.top_similar = top_similar
        self.weighted_count = weighted_count
        self.apply_softmax = apply_softmax

        print("[Loading Content Similarity Matrix]")
        # from util.data_handler import ITEM_ID_LIMIT
        # content_vec = pickle.load(open(content_file, 'rb'))
        # content_vec = np.vstack([content_vec[idx] for idx in range(len(content_vec))])
        self.content_vec = np.load(content_file).astype(np.float32)
        # MovieLens has additional items but no feedback
        # self.content_vec = content_vec[:ITEM_ID_LIMIT].astype(np.float32)
        # print(f"[Using Subset of Ratings]")
        # self.total = 5000000
        # similarity = content.dot(content.T)
        # data = np.load(content_file)
        # self.sim = data['sim'].astype(np.float32)
        # self.pos_index = data['pos_index']
        # self.neg_index = data['neg_index']
        # cur_top = self.pos_index.shape[1]
        # if top_similar > cur_top:
        #     print("Requested Top Similarity is greater than size of provided "
        #           "file with max of {}".format(cur_top))
        # self.pos_index = self.pos_index[:, :self.top_similar].astype(np.int)
        # self.neg_index = self.neg_index[:, :self.top_similar].astype(np.int)
        print("[DONE!]")
        self._sim = None
        self._pos_index = None

    @property
    def sim(self):
        if self._sim is None:
            self._sim = self.content_vec.dot(self.content_vec.T).astype(np.float32)
        return self._sim

    @property
    def pos_index(self):
        if self._pos_index is None:
            # Item Indices for most similar and least similar
            self._pos_index = \
                bottleneck.argpartition(self.sim, self.top_similar)[:, :self.top_similar].astype(np.int)

        return self._pos_index

    def __len__(self):
        # return self.total
        return self.train.getnnz()

    def __getitem__(self, index):
        """
        Returns user index, item indices, item indices similarity,
        negative item indices, neg item similarity

        :param index:
        :return:
        """
        index = np.random.randint(self.train.getnnz())
        user_idx = self.train.row[index]
        item_idx = self.train.col[index]
        item_vec = self.content_vec[item_idx].reshape(-1, 1)
        # Other positive items
        item_list = self.user_items_list[user_idx]
        sample_indices = np.random.choice(item_list, size=self.weighted_count-1,
                                          replace=len(item_list) < self.weighted_count)

        item_indices = np.append(sample_indices, [item_idx])
        item_indices = torch.from_numpy(item_indices).long()

        # [n, 512] [512, 1]
        item_sim = torch.from_numpy(self.content_vec[sample_indices].dot(item_vec).ravel())

        # Normalize, if not linear start
        if self.apply_softmax:
            item_sim = F.softmax(item_sim, -1)

        # Sample some random negatives
        neg_indices = np.asarray([self._sample_negative_item(user_idx)
                                  for _ in range(self.weighted_count)])

        # Normalize
        neg_sim = torch.from_numpy(self.content_vec[neg_indices[1:]].dot(self.content_vec[neg_indices[0]].reshape(-1, 1)).ravel())
        if self.apply_softmax:
            neg_sim = F.softmax(neg_sim, -1)

        neg_indices = torch.from_numpy(neg_indices).long()

        # TODO: Should we add more negatives here?
        return torch.LongTensor([user_idx]), item_indices, item_sim, neg_indices, neg_sim

    # def __getitem__(self, index):
    #     """
    #     Returns user index, item indices, item indices similarity,
    #     negative item indices, neg item similarity
    #
    #     :param index:
    #     :return:
    #     """
    #     # TODO: Profile why is this so slow?????
    #     user_idx = self.train.row[index]
    #     item_idx = self.train.col[index]
    #     sample_index = np.random.choice(self.top_similar, size=self.weighted_count-1, replace=False)
    #     item_indices = np.append(self.pos_index[item_idx][sample_index], [item_idx])
    #     item_indices = torch.from_numpy(item_indices)
    #
    #     # Normalize, if not linear start
    #     item_sim = torch.from_numpy(self.sim[item_idx][item_indices])
    #     if self.apply_softmax:
    #         item_sim = F.softmax(item_sim, -1)
    #
    #     # Sample some random negatives
    #     # TODO: Check that these are not observed for the user
    #     sample_neg_index = np.random.choice(self.top_similar, size=self.weighted_count-1, replace=False)
    #     # Ideally, we would want to make sure all of them are not observed not
    #     # just true negative sample
    #     neg_indices = np.append(self.neg_index[item_idx][sample_neg_index],
    #                             [self._sample_negative_item(user_idx)])
    #
    #     # Normalize
    #     neg_sim = torch.from_numpy(self.sim[item_idx][neg_indices])
    #     if self.apply_softmax:
    #         neg_sim = F.softmax(neg_sim, -1)
    #
    #     # TODO: Should we add more negatives here?
    #     return torch.LongTensor([user_idx]), item_indices, item_sim, neg_indices, neg_sim


if __name__ == '__main__':
    print("HELLO WORLD!")
    import time
    start = time.time()
    dataset = Dataset("../data/")
    end = time.time()
    print(f"Took: {end-start}")
    """
    %timeit np.random.choice(300, size=10, replace=False)
    81.9 µs ± 1.39 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    130 µs ± 7.84 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    
    samples = np.arange(0, 300, dtype=np.int)
    %timeit np.random.choice(samples, size=10, replace=False)
    107 µs ± 3.52 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    126 µs ± 16.2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    
    np.random.randint(0, 300, size=10)
    import random
    x = np.arange(300)

    %timeit random.sample(range(300), 10)
    
    index = np.random.choice(58098, size=50)
    %timeit content[index]
    33.8 ms ± 9.19 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        
    %timeit np.take(content, index, axis=0, out=temp)
    152 ms ± 16.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """

    # datatset = WeightedContentDataset("../data/", "../data/ml_content_sim.npz", 4,
    #                                   300, 10)
    # print(datatset[0])
    # loader = torch.utils.data.DataLoader(datatset, batch_size=4)
    # for example in loader:
    #     user_idx, item_idx, item_sim, neg_idx, neg_sim = example
    #     print(example)
    #     print(user_idx.shape)
    #     break

    # matrix = mmread("data/test.mtx")
    # save_sparse_matrix("data/test.npz", matrix)
    datadir = "data/full/"
    import numpy as np
    import pandas as pd

    data = np.load(f"{datadir}/train.npz")
    data = np.column_stack([data['row'], data['col'], data['data']])
    df = pd.DataFrame(data, columns=['user', 'item', 'rating'])
    df.user = df.user.astype(np.int)
    df.item = df.item.astype(np.int)
    df.to_csv(f"{datadir}/train.csv", header=None, index=False, sep="\t")

    data = np.load(f"{datadir}/test.npz")
    data = np.column_stack([data['row'], data['col'], data['data']])
    df = pd.DataFrame(data, columns=['user', 'item', 'rating'])
    df.user = df.user.astype(np.int)
    df.item = df.item.astype(np.int)
    df.to_csv(f"{datadir}/test.csv", header=None, index=False, sep="\t")
