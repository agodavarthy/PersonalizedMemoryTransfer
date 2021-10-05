import sys
import os
import json
import argparse
import numpy as np
# Add one level up
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__), "../"))
from util.simulator import MovieConversationSimulator, ModelRunner, BaseConvMovieInterface
from util.latent_factor import MovieSelector

import util.results
from util.data import Movie
import pickle
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', help='Episodes data', type=str,
                    required=True)
parser.add_argument('-seed', '--seed', help='Random seed, used to create '
                                            'additional bootstrap samples',
                    type=int, default=42)


class ContentBaseline(BaseConvMovieInterface):

    def __init__(self, movies_list: list):
        self.selector = MovieSelector(movies_list, 'cpu')

    def reset(self):
        """nothing to reset"""
        return

    def observe(self, prev_state, is_seeker: bool, msg_length):
        """Produce recommendations"""
        if prev_state:
            prev_vector = prev_state.get('query_vec')
            if prev_vector is not None:
                scores, retrieved = self.selector.query_vector(prev_vector)
                predicted_ids = [m.matrix_id for m in retrieved]
                return {'rec': predicted_ids}
        return {}


if __name__ == '__main__':
    opt = parser.parse_args()

    opt.baseline = True
    opt.model = 'content'
    keeper = util.results.MongoDBTracker([key for key, _ in opt._get_kwargs()],
                                         collection_name='redial')
    all_results = []

    r10 = []
    r25 = []
    ndcg10 = []
    ndcg25 = []
    mrr10 = []
    mrr25 = []
    for fold_index in range(5):
        p = os.path.join(opt.data, "{}.pkl".format(fold_index))
        data = pickle.load(open(p, 'rb'))
        episodes = data['data']
        movies_list = [Movie.from_dict(m) for m in data['movies']
                       if m['matrix_id'] != -1]
        simulator = MovieConversationSimulator(episodes, movies_list)

        print("Loading movie simulator")

        agent = ContentBaseline(simulator.movies_list)

        runner = ModelRunner(opt, agent, simulator)
        print("Running....")
        results = runner.run()
        metrics = simulator.metrics()

        # r10.append(metrics['r@10'])
        # r25.append(metrics['r@25'])
        # ndcg10.append(metrics['ndcg@10'])
        # ndcg25.append(metrics['ndcg@25'])
        # mrr10.append(metrics['mrr@10'])
        # mrr25.append(metrics['mrr@25'])
        p = p.split('/')
        out_f = './eval_results/content_{}_{}.txt'.format(p[1], fold_index)

        with open(out_f, 'w') as f:
            json.dump(results, f)



        print()
        # print(json.dumps(metrics))
        print()
        all_results.append(metrics.copy())

    summary = {
        'results': all_results
    }

    # print('r@10: {} \n'
    #       'r@25: {} \n'
    #       'ndcg@10: {} \n'
    #       'ndcg@25: {} \n'
    #       'mrr@10: {} \n'
    #       'mrr@25: {} \n'.format(np.mean(r10),
    #                              np.mean(r25),
    #                              np.mean(ndcg10),
    #                              np.mean(ndcg25),
    #                              np.mean(mrr10),
    #                              np.mean(mrr25)))

    summary.update(opt.__dict__)

    # Write
    # keeper.report(summary)

    # summary = metrics.copy()
    # summary.update(opt.__dict__)
    # keeper.report(summary)

    # np.random.seed(42)
    # print("Starting.......")
    # all_matrix_ids = [movie.matrix_id for movie in simulator.movies_list]
    # for episode_idx in tqdm(range(len(simulator))):
    #     while True:
    #         predicted_ids = np.random.choice(all_matrix_ids, size=300, replace=False).tolist()
    #         episode = simulator.get_episode(predicted_ids)
    #         if episode['done']:
    #             break
    #     if episode_idx > 0 and episode_idx % 500 == 0:
    #         print(simulator.metrics())
    #         print()
    # print()
    # for k, v in simulator.metrics().items():
    #     print(f"{k}: {v:.4}")
    # print()
    # print(json.dumps(simulator.metrics(), indent=2))
    # print()
