import sys
import os
import json
import numpy as np
import argparse
from util.data import Movie
import pickle

# Add one level up
# sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', help='Episodes data', type=str,
                    required=True)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)


class RandomBaseline(BaseConvMovieInterface):

    def __init__(self, matrix_ids):
        self.matrix_ids = matrix_ids

    def reset(self):
        """nothing to reset"""
        pass

    def _recommend(self):
        """Perform a recommendation for all the items as a 1d array"""
        return np.random.choice(self.matrix_ids, size=300, replace=False).tolist()

    def observe(self, prev_state, is_seeker: bool, msg_length):
        """Nothing to observe"""
        return {'rec': self._recommend()}


if __name__ == '__main__':
    opt = parser.parse_args()
    print("Loading movie simulator")
    data = pickle.load(open(opt.data, 'rb'))
    episodes = data['data']
    movies_list = [Movie.from_dict(m) for m in data['movies']
                   if m['matrix_id'] != -1]

    simulator = MovieConversationSimulator(episodes, movies_list)
    matrix_ids = [movie.matrix_id for movie in simulator.movies_list]
    agent = RandomBaseline(matrix_ids)
    runner = ModelRunner(opt, agent, simulator)
    print("Running....")
    results = runner.run()
    print()
    print(json.dumps(simulator.metrics()))
    print()
