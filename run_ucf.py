import numpy as np
import tqdm
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd

import json

from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from tqdm import tqdm
# import libraries for content based agent
# import sys
# import os
# import json
import argparse
#
# # Add one level up
# sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
# from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface
# from util.latent_factor import MovieSelector
#
# import util.results
# from util.data import Movie
# import pickle

def compute_metrics(item_ranks):
    """
    dcg = sum[(2^rel[i] - 1) / log(i + 2) for i range(K)]
    ndcg = dcg / ideal dcg
    Recall = Hits@K / n_relvance
    MRR = 1/ Min Hit Position

    :return: Dictionary of NDCG, MRR and Recall
    """

    eval_at = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20, 21, 22, 23, 24, 25]

    total_recall = {k: [] for k in eval_at}
    total_ndcg = {k: [] for k in eval_at}
    total_mrr = {k: [] for k in eval_at}
    # genre_recall_count = {k: {} for k in [1, 3, 5, 10, 25]}
    # genre_recall = {k: [] for k in [1, 3, 5, 10, 25]}

    if len(item_ranks):
        max_cutoff = max(eval_at)
        # Compute the ideal dcg for each user
        ideal_dcg = np.ones(max_cutoff, dtype=np.float32)
        # Compute Ideal NDCG
        ideal_dcg = ((np.power(2, ideal_dcg) - 1.0) / np.log2(np.arange(2, max_cutoff + 2)))

        # We can view this as for each user which is actually a single utterance
        for ranks, movies in item_ranks:
            n_relv = len(movies)
            recall = {k: 0 for k in eval_at}
            dcg = {k: 0.0 for k in eval_at}
            # For each item we ranked
            for item_rank in ranks:
                # For each cut off we want to calculate for
                for k in eval_at:
                    # If the item rank is < K
                    if item_rank < k:
                        recall[k] += 1
                        dcg[k] += 1.0 / np.log2(item_rank + 2)  # + 2 since we start at 0

            # Compute MRR
            if len(ranks):
                min_rank = min(ranks) + 1
                mrr = 1 / min_rank
            else:
                min_rank = -1
                mrr = 0

            for i, k in enumerate(eval_at):
                total_mrr[k].append(mrr if min_rank < k else 0.0)
                # Divide by the ideal ndcg
                total_ndcg[k].append(dcg[k] / ideal_dcg[:n_relv].sum())
                # Compute recall, N Relv / Total Relv
                total_recall[k].append(recall[k] / n_relv)
            #
            # for k in [1, 3, 5, 10, 25]:
            #     # We set recall with item_rank < k, but genre recall is set
            #     # when index = k-1
            #     val = genre[k - 1]
            #     genre_recall[k].append(val)
            #
            #     if val in genre_recall_count[k]:
            #         genre_recall_count[k][val] += 1
            #     else:
            #         genre_recall_count[k][val] = 1

    metrics = {}
    metrics.update({"r@%s" % k: np.mean(v) if len(v) else 0.0 for k, v in total_recall.items()})
    metrics.update({"mrr@%s" % k: np.mean(v) if len(v) else 0.0 for k, v in total_mrr.items()})
    metrics.update({"ndcg@%s" % k: np.mean(v) if len(v) else 0.0 for k, v in total_ndcg.items()})

    return metrics


def load_train_test_dataset(folder_path):

    seeker_id = 0

    seekers = []
    train_data = []
    test_data = []
    train_convs = []

    for fold_index in range(5):
        data = pickle.load(open(os.path.join(folder_path, "{}.pkl".format(fold_index)), 'rb'))
        episodes = data['data']
        for episode in episodes:
            convs = []
            for conv in episode:
                # add first mentioned movie
                convs.append(conv)

                if conv['ml_id']:
                    if seeker_id not in seekers:
                        for m in conv['ml_id']:
                            r = conv['sentiment']
                            train_data.append([seeker_id, m, r])

                        seekers.append(seeker_id)
                        train_convs.append(convs)
                    else:
                        for m in conv['ml_id']:
                            r = conv['sentiment']
                            test_data.append([seeker_id, m, r])

            seeker_id += 1


    # train_data = _preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6_kfold/splits.pkl')


    train_df = pd.DataFrame(data=train_data, columns=['userID', 'itemID', 'rating'])
    test_df = pd.DataFrame(data=test_data, columns=['userID', 'itemID', 'rating'])

    movies = set(list(train_df.itemID.values) + list(test_df.itemID.values))

    return train_df, test_df, list(movies), train_convs






def main(opt):
    if opt.data == 'gorecdial':
        # gorecdial
        folder_path = 'data/gorecdial/transformer/gorecdial_flr1e-6_l21e-5/test.pkl'
    elif opt.data == 'redial':
        # redial
        folder_path = 'data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl'
    else:
        raise ValueError

    train_df, test_df, movies, train_convs = load_train_test_dataset(folder_path)

    test_df_grouped = test_df.groupby('userID')

    if opt.model == 'svd':
        model = SVD()
    elif opt.model == 'nmf':
        model = NMF()
    elif opt.model == 'knn':
        model = KNNBasic()
    else:
        raise ValueError

    reader = Reader(rating_scale=(0, 1))
    train_data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader)

    d = train_data.build_full_trainset()

    model.fit(d)
    item_ranks = []

    for userID, rows in tqdm(test_df_grouped):
        # target_movies = list(rows.itemID.values)
        #
        # temp_train_df = pd.concat([train_df, rows])
        # temp_test_df = pd.DataFrame({'userID': [userID] * len(target_movies),
        #               'itemID': target_movies,
        #               'rating': [0] * len(target_movies)})
        #
        # train_data = Dataset.load_from_df(temp_train_df[['userID', 'itemID', 'rating']], reader)
        # model.fit(d)

        r_pred = []

        for m in movies:
            pred = model.predict(userID, m, verbose=False)
            r_pred.append(pred.est)

        rank = pd.DataFrame.from_dict({'itemID':movies, 'rating':r_pred})
        rank['rank'] = rank['rating'].rank(method='min', ascending=False)

        for i, row in rows.iterrows():
            m = row['itemID']
            r = rank[rank['itemID'] == m]['rank'].values[0]
            if row['rating'] >= 0.5:
                item_ranks.append(([r], [m]))
    print('item rank length: {}'.format(len(item_ranks)))


    # test_data = Dataset.load_from_df(test_df[['userID', 'itemID', 'rating']], reader)
    out_f = './eval_results/0906/{}_{}.txt'.format(opt.data, opt.model)

    with open(out_f, 'w') as f:
        json.dump(compute_metrics(item_ranks), f)




    # for u in tqdm.tqdm(test_user_ids):
    #     m_r = test_ratings.getrow(u)
    #     nns = model.kneighbors(m_r, 20, return_distance=False)[0]
    #     ratings = []
    #
    #     for n in nns[1:]:
    #         ratings.append(train_ratings.getrow(n).toarray())
    #
    #     if ratings:
    #         ratings = np.vstack(ratings)
    #     else:
    #         print(1)
    #     # ratings[ratings==0] = np.nan
    #
    #     ratings_means = np.mean(ratings, axis=0)
    #     ranked_movies = np.argsort(ratings_means)[::-1]
    #
    #     _, target_movies = test_ratings.getrow(u).nonzero()
    #     for m in target_movies:
    #         r = test_ratings[u, m]
    #         if r > 0.5:
    #             rank = np.where(ranked_movies == m)[0][0]
    #             item_ranks.append(([rank], [m]))

    # print(compute_metrics(item_ranks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='nmf')
    parser.add_argument('-d', '--data', type=str, default='gorecdial')


    opt = parser.parse_args()
    main(opt)



