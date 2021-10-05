import os
import re
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

    def observe(self, prev_state, is_seeker: bool, text: str, movie_id: list):
            """Produce recommendations"""
            # vectorizer
            print("text = ", text)
            #ind_ucase = 
            for mid in movie_id:
                mname = movie_id_name_dict[mid]
                if mid == '182022': continue
                if mname in text:
                    #text = re.sub(m, str(mid), text)
                    #print("Matched ", m)
                    text = text.replace(mname, "@"+mid)
                    text = text.lower()
                    #break
            text = text.lower()
            print("replaced text = ", text)
            print("--------------------------------------------------")
            text_vec = self.tfidf_vectorizer.transform([text])
            # compute similarity
            sim = self.tfidf_matrix.dot(text_vec.T).astype(np.float32)
            sim = sim.toarray().ravel()

            indices = np.argsort(-sim)
            retrieved = [self.movies[i] for i in indices]
            predicted_ids = [m.matrix_id for m in retrieved]
            #print("Is seeker is ", is_seeker)
            #time.sleep(3)
            return {'rec': predicted_ids}

def movie_id_names():
    movie_name_id_dict = {}
    movie_id_name_dict = {}
    ifilename = "data/redial_dataset/movies_with_mentions.csv"
    ifd = open(ifilename, "r")
    linestart = True
    for line in ifd:
            if linestart == True: 
                linestart = False
                continue
            tokens = line.split(",")
            if len(tokens) == 3:
                    mid = tokens[0]
                    mname = tokens[1]
            #else:
            #        mid = tokens[0]
            #        mname = tokens[1]+", "+tokens[2]
            mname = re.sub('\([^()]*\)', '', mname)
            mname = mname.strip(" ")
            if mname.strip(" ") != " " or mname.strip(" ") != "":
                movie_name_id_dict[mname] = mid
                movie_id_name_dict[mid] = mname
    return movie_id_name_dict, movie_name_id_dict

movie_id_name_dict, movie_name_id_dict = movie_id_names()
for mname in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
        if mname in movie_name_id_dict:
                del movie_name_id_dict[mname]
if 'Shutter Island' in movie_name_id_dict:
        print("Found Shutter Island")
if __name__ == '__main__':
    opt = parser.parse_args()

    opt.baseline = True
    opt.model = 'tfidf'
    keeper = util.results.MongoDBTracker([key for key, _ in opt._get_kwargs()],
                                         collection_name='redial')
    for fold_index in range(5):
        p = os.path.join(opt.data, "{}.pkl".format(fold_index))
        data = pickle.load(open(p, 'rb'))
        episodes = data['data'][90:100]
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
