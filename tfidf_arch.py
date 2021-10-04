import os
import sys
import time
import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
# Add one level up
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
sys.path.append(os.getcwd())
from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface

import util.results
from util.data import Movie
import pickle
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', help='Episodes data', type=str,
                    required=True)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)


class TFIDFSimBaseline(BaseConvMovieInterface):

    def __init__(self, movies_list: list):
        self.movies = movies_list
        self.tfidf_vectorizer = TfidfVectorizer(dtype=np.float32)
                                                #analyzer=str.split)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                [m.title + " " + m.plot for m in movies_list])

    def reset(self):
        """nothing to reset"""
        return

    def observe(self, prev_state, is_seeker: bool, text: str):
        #if is_seeker :
        """Produce recommendations"""
        # vectorizer
        text_vec = self.tfidf_vectorizer.transform([text])
        print("text = ", text)
        print("text_vec = ", text_vec)
        # compute similarity
        sim = self.tfidf_matrix.dot(text_vec.T).astype(np.float32)
        sim = sim.toarray().ravel()

        indices = np.argsort(-sim)
        retrieved = [self.movies[i] for i in indices]
        predicted_ids = [m.matrix_id for m in retrieved]
        print("Is seeker is ", is_seeker)
        print("returning ", len(predicted_ids))
        #time.sleep(3)
        #return {'rec': predicted_ids}
        #else :
        #    print("Is seeker is ", is_seeker)
        #    print("returning empty results")
        #    time.sleep(4)
        #    return {'rec': []}



if __name__ == '__main__':
    opt = parser.parse_args()

    opt.baseline = True
    opt.model = 'tfidf'
    keeper = util.results.MongoDBTracker([key for key, _ in opt._get_kwargs()],
                                         collection_name='redial')
    #for fold_index in range(5):
    #ifilename = "data/redial_dataset/test_data.jsonl"
    #ifd = open(ifilename, "r")

    #p = os.path.join(opt.data, "{}.pkl".format(fold_index))
    #p = os.path.join(opt.data, "{}.pkl".format(0))
    p = "data/redial/tfifd_test_episode.pkl"
    data = pickle.load(open(p, 'rb'))
    episodes = data['data']
    movies_list = [Movie.from_dict(m) for m in data['movies']
                   if m['matrix_id'] != -1]
    
    simulator = MovieConversationSimulator(episodes, movies_list)

    print("Loading movie simulator")

    agent = TFIDFSimBaseline(simulator.movies_list)

    runner = ModelRunner(opt, agent, simulator)
    print("Running....")
    results = runner.run()
    metrics = simulator.metrics()
    # print()
    # print(json.dumps(metrics))
    # print()
    # all_results.append(metrics.copy())

    p = opt.data.split('/')
    out_f = './eval_results/oct2/tfidf_{}_{}.txt'.format(p[1], fold_index)
    with open(out_f, 'w') as f:
        json.dump(results, f)
    # summary = {
    #     'results': all_results
    # }
    #
    # summary.update(opt.__dict__)
    #
    # # Write
    # keeper.report(summary)
