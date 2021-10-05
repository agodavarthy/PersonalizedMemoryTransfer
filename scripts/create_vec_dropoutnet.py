import sys
import os
sys.path.append(os.getcwd())
import DropoutNet.dn_utils

from sklearn.utils.extmath import randomized_svd
import scipy.sparse
from tqdm import tqdm
import numpy as np
from sklearn import datasets
import pickle
import os
import scipy.sparse as sp
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--fold', help='Fold number', type=int, default=0)


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


if __name__ == '__main__':
    opt = parser.parse_args()
    fold_index = opt.fold
    # fold_index = 0

    print("Fold index", fold_index)

    count_vec = pickle.load(open(f'data/ratings/{fold_index}/count_vec.pkl', 'rb'))
    # We use this because dropoutnet does svd on frequency
    user_content, _ = datasets.load_svmlight_file(f'data/ratings/{fold_index}/user_features',
                                                  zero_based=True, dtype=np.float16)

    user_content = user_content[:50]
    assert user_content.nnz
    path = os.path.join("data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl", "{}.pkl".format(fold_index))

    # data = pickle.load(open(os.path.join(opt.data, "{}.pkl".format(fold_index)), 'rb'))
    print("Reading", path)
    data = pickle.load(open(path, 'rb'))

    for episode in tqdm(data['data']):
        for state in episode:
            content = count_vec.transform([state['text']])
            content = tfidf(content).astype(np.float16)
            u, s, _ = randomized_svd(scipy.sparse.vstack([user_content, content]).astype(np.float16),
                                     n_components=50, n_iter=3)
            content = u * s
            _, content = DropoutNet.dn_utils.prep_standardize(content)
            state['dropout_net'] = content[-1:].astype(np.float16)
    print("Dumping back....")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
