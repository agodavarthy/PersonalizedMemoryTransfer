import numpy as np
import tqdm
import pickle
import os
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

def compute_metrics(item_ranks):
    """
    dcg = sum[(2^rel[i] - 1) / log(i + 2) for i range(K)]
    ndcg = dcg / ideal dcg
    Recall = Hits@K / n_relvance
    MRR = 1/ Min Hit Position

    :return: Dictionary of NDCG, MRR and Recall
    """

    eval_at = [10, 25]

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


def load_train_test_dataset():
    seeker_id = 0
    def _preprocess_data(folder_path):
        nonlocal seeker_id
        full_data = []

        for fold_index in range(5):
            data = pickle.load(open(os.path.join(folder_path, "{}.pkl".format(fold_index)), 'rb'))
            episodes = data['data']
            for episode in episodes:
                for conv in episode:
                    if conv['ml_id']:
                        for m in conv['ml_id']:
                            r = 1 if conv['sentiment'] >= 0.5 else 0
                            full_data.append([seeker_id, m, r])
                seeker_id += 1

        return np.array(full_data)

    train_data = _preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6_kfold/splits.pkl')
    test_data = _preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6/test.pkl')

    full_data = np.vstack((train_data, test_data))

    field_dims = np.max(full_data[:, :2].astype(np.int), axis=0) + 1

    train_csr = csr_matrix((train_data[:, 2], (train_data[:, 1], train_data[:, 0])),
                           shape=[field_dims[1], field_dims[0]])
    test_csr = csr_matrix((test_data[:, 2], (test_data[:, 1], test_data[:, 0])),
                           shape=[field_dims[1], field_dims[0]])

    return train_csr, test_csr

def main(model_name='cosine'):

    train_ratings, test_ratings = load_train_test_dataset()
    train_ratings.eliminate_zeros()
    test_ratings.eliminate_zeros()

    if model_name == "lmf":
        model = LogisticMatrixFactorization()

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender(B=0.2)

    else:
        raise NotImplementedError("TODO: model %s" % model_name)

    model.fit(train_ratings)

    test_rows, test_cols = test_ratings.nonzero()
    test_user_items = test_ratings.T.tocsr()
    test_user_ids = np.unique(test_cols)

    train_rows, _ = train_ratings.nonzero()

    movie_ids = np.unique(np.hstack((test_rows, train_rows)))
    item_ranks = []

    for u in test_user_ids:
        ranked_movies = model.rank_items(u, test_user_items, movie_ids)
        target_movies, _ = test_ratings.getcol(u).nonzero()
        for m in target_movies:
            rank = np.where(np.array(ranked_movies)[:, 0] == m)[0][0]
            if rank != 0:
                item_ranks.append(([rank], [m]))

    print(compute_metrics(item_ranks))


if __name__ == '__main__':
    main()



