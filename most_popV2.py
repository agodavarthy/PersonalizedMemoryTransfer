import sys
import os
import json
import numpy as np
import argparse
from collections import Counter
# Add one level up
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface

import util.results
from util.data import Movie
import pickle
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', help='Episodes data', type=str,
                    required=True)
parser.add_argument('-ratings', '--ratings', help='Ratings npz filename', type=str,
                    required=True)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)


class MostPopBaseline(BaseConvMovieInterface):

    def __init__(self, matrix_ids):
        self.matrix_ids = matrix_ids

    def reset(self):
        """nothing to reset"""
        pass

    def observe(self, prev_state, is_seeker: bool, msg_length):
        """Nothing to observe"""
        return {'rec': self.matrix_ids}


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.baseline = True
    opt.model = 'mostpop'

    print("Loading ratings...")
    data = np.load(opt.ratings)

    # Compute the most popular
    print("Computing most popular...")
    most_popular = Counter(data['col'])
    most_pop_ids, _ = zip(*most_popular.most_common(300))
    most_pop_ids = list(set(most_pop_ids))

    keeper = util.results.MongoDBTracker([key for key, _ in opt._get_kwargs()],
                                         collection_name='redial')
    all_results = []
    for fold_index in range(5):
        p = os.path.join(opt.data, "{}.pkl".format(fold_index))

        data = pickle.load(open(os.path.join(opt.data, "{}.pkl".format(fold_index)), 'rb'))
        episodes = data['data']
        movies_list = [Movie.from_dict(m) for m in data['movies']
                       if m['matrix_id'] != -1]

        simulator = MovieConversationSimulator(episodes, movies_list)
        print("Loading movie simulator")
        agent = MostPopBaseline(most_pop_ids.copy())
        runner = ModelRunner(opt, agent, simulator)
        print("Running....")
        results = runner.run()
        metrics = simulator.metrics()
        print()
        # print(json.dumps(metrics))
        print()
        all_results.append(metrics.copy())

        p = p.split('/')
        out_f = './eval_results/pop_{}_{}.txt'.format(p[1], fold_index)

        with open(out_f, 'w') as f:
            json.dump(results, f)

    summary = {
        'results': all_results
    }

    summary.update(opt.__dict__)

    # Write
    # keeper.report(summary)
