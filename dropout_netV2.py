import time
import sys
import os
import json
import argparse
import numpy as np
from collections import Counter
# Add one level up
# sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
sys.path.append(os.getcwd())
sys.path.append("DropoutNet")
from util.simulator import MovieConversationSimulator, ModelRunner
import sys
import scipy.sparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import DropoutNet.dn_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn import datasets
import DropoutNet.data
import DropoutNet.model
import scipy.sparse as sp
import json
from tqdm import tqdm
import argparse
import os


import util.results
from util.data import Movie
import pickle
import sys
import os
import json
import numpy as np
import argparse
from collections import Counter
# Add one level up
# sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface

import util.results
from util.data import Movie
import pickle



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', help='Episodes data', type=str)
parser.add_argument('-mf', '--model_file', help='directory to checkpoints', type=str)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)
n_users = 8769 + 1
n_items = 2065 + 1



from sklearn.utils.extmath import randomized_svd


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


def load_content(filename):
    content, _ = datasets.load_svmlight_file(filename, zero_based=True, dtype=np.float32)
    content = tfidf(content)
    u, s, _ = randomized_svd(content, n_components=50, n_iter=5)
    content = u * s
    _, content = DropoutNet.dn_utils.prep_standardize(content)

    if sp.issparse(content):
        content = content.tolil(copy=False)

    return content



class DropoutNetAgent(BaseConvMovieInterface):

    def __init__(self, fold_idx, checkpt_path):
        # self.movies_list = movies_list
        # self.mlid_to_movie = {movie.ml_id: movie for movie in self.movies_list}
        # self.matrixid_to_movie = {movie.matrix_id: movie for movie in self.movies_list}

        self.item_content = load_content('data/gorec_ratings/item_features')
        # self.count_vec = pickle.load(open(f'data/redial/ratings/{fold_idx}/count_vec.pkl', 'rb'))
        # We use this because dropoutnet does svd on frequency
        # self.user_content, _ = datasets.load_svmlight_file(f'data/redial/ratings/{fold_idx}/user_features',
        #                                                    zero_based=True, dtype=np.float32)
        # self.user_content = tfidf(self.user_content)[:300]
        opt = json.load(open(checkpt_path + "/config.json"))

        self.v_pref = np.loadtxt(f'data/gorec_ratings/{fold_idx}/30/V.txt')
        self.u_coldstart = self.v_pref.mean(0).reshape(1, -1)

        print("Loading DropoutNet:", json.dumps(opt, sort_keys=True, indent=2))
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model = DropoutNet.model.DeepCF(latent_rank_in=30,
                                              user_content_rank=opt['user_content_rank'],
                                              item_content_rank=opt['item_content_rank'],
                                              model_select=opt['model_select'][0],
                                              rank_out=opt['rank_out'])
        # with tf.device('/cpu:0'):
        self.model.build_model()

        # with tf.device('/cpu:0'):
        self.model.build_predictor([300], 300)

        self.sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(checkpt_path))


    def reset(self):
        """nothing to reset"""
        pass

    def _predict(self, user_content):
        output = self.sess.run(
                self.model.preds,
                feed_dict={
                    self.model.Uin: self.u_coldstart,
                    self.model.Vin: self.v_pref,
                    self.model.Vcontent: self.item_content,
                    self.model.Ucontent: user_content,
                    self.model.phase: 0
                })
        return output

    def observe(self, prev_state, is_seeker: bool, msg_length):
        """Nothing to observe"""
        response = {}
        if prev_state is not None and 'dropout_net' in prev_state:
            prediction = self._predict(prev_state['dropout_net'].reshape(1, -1))
            response['rec'] = np.argsort(-prediction)[:300].tolist()
        return response


if __name__ == '__main__':
    opt = parser.parse_args()
    #opt.data = 'data/gorecdial/transformer/gorecdial_flr1e-6_l21e-5/test.pkl'
    #opt.data = 'data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl'
    opt.data = opt.data 

    #opt.model_file = 'results/dropoutnet'
    opt.model_file = opt.model_file 

    opt.baseline = True
    opt.model = 'dropoutnet'
    opt.cross_validation = False
    all_results = []
    # keeper = util.results.MongoDBTracker([key for key, _ in opt._get_kwargs()],
    #                                      collection_name='redial')


    for fold_index in range(5):
        p = os.path.join(opt.data, "{}.pkl".format(fold_index))

        data = pickle.load(open(os.path.join(opt.data, "{}.pkl".format(fold_index)), 'rb'))
        episodes = data['data']
        movies_list = [Movie.from_dict(m) for m in data['movies']
                       if m['matrix_id'] != -1]
        chkpt = "{}/{}/{}".format(opt.model_file, fold_index, 30)
        print("check point = ", chkpt)
        time.sleep(1)
        simulator = MovieConversationSimulator(episodes, movies_list)
        agent = DropoutNetAgent(fold_index, chkpt)
        runner = ModelRunner(opt, agent, simulator)
        results = runner.run()
        print()
        print("results = ", results)
        time.sleep(5)
        all_results.append(results.copy())

        p = p.split('/')
        out_f = './eval_results/dropout_{}_{}.txt'.format(p[1], fold_index)
        print("Saving to output = ", out_f)
        time.sleep(1)

        with open(out_f, 'w') as f:
            json.dump(results, f)




