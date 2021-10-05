import time
import datetime as dt
import pickle
import sys
from typing import List

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from util.data import Movie
from collections import defaultdict


class MovieConversationSimulator(object):

    def __init__(self, episodes, movies_list,
                 eval_at=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20, 21, 22, 23, 24, 25)):
        """
        * initiatorWorkerId: an integer identifying to the worker initiating the conversation (the recommendation seeker)
        * respondentWorkerId: an integer identifying the worker responding to the initiator (the recommender)

        1. suggested: Whether the movie was mentioned by the seeker, or was a suggestion from the recommender
        2. seen: if the seeker has seen the movie: one of Seen it, Haven’t seen it, or Didn’t say
        3. liked: seeker liked the movie or the suggestion: one of Liked, Didn’t like, Didn’t say.
        """
        # if episodes is None and movies_list is None:
        #     data = pickle.load(open(filename, 'rb'))
        #     self.episodes = data['episodes']
        #     self.movies_list = [Movie.from_dict(m) for m in data['movies']
        #                         if m['matrix_id'] != -1]
        # else:
        #     assert filename is None, 'Cannot pass both filename and episodes/movies_list'
        self.episodes = episodes
        self.movies_list = movies_list

        self.mlid_to_movie = {movie.ml_id: movie for movie in self.movies_list}
        self.matrixid_to_movie = {movie.matrix_id: movie for movie in self.movies_list}

        # Insert episodes to have the movies data structure
        for episode in self.episodes:
            for e in episode:
                e['ml_id'] = [self.mlid_to_movie[mlid] for mlid in e['ml_id']]

        self._eval_at = eval_at

        # Single Episode Specific State
        self._sub_episode_index = 0  # Index into the current episode
        self._already_seen = set()  # Keep track of movies seen

        # Global States for entire dataset
        self._index = 0  # Index to the episodes
        self._item_ranks = []  # List[List[Item Rank], N Total Relevant]
        # Item ranks by turns
        self._episode_item_ranks = []  # List[List[List[Item Rank], N Total Relevant]]
        self._turn = 0

        self.reset_episodes()
        print("In MovieConversationSimulator")

    def _reset(self):
        """Reset internal state per an episode"""
        # Reset sub index
        self._sub_episode_index = 0
        self._turn = 0
        # Clear movies we saw
        self._already_seen = set()

    def reset_episodes(self):
        """Reset entire state and start from the first episode"""
        # Current relevance positions for each utterance
        self._item_ranks = []
        self._episode_item_ranks = []
        # Global episode index
        self._index = 0
        self._reset()

    def __getitem__(self, item: int):
        return self.episodes[item]

    def __len__(self):
        return len(self.episodes)

    def get_episode(self, predicted_matrix_ids: List[int]):
        """
        Evaluates predicted_matrix_ids only on seeker

        The predicted_matrix_ids are evaluated for the current utterance which
        we have not shown to the model yet. This way we ensure we are not
        looking ahead.

        :param predicted_matrix_ids: Ranked list of predicted matrix ids
        :return: dict, state info
        """
        if self._index >= len(self.episodes):
            raise Exception("Completed all episodes. Call reset_episodes()")

        # Get a response within the conversation
        if self._sub_episode_index < len(self.episodes[self._index]):
            print("Archana1...in IF")
            # Perform a copy since the agent may modify it
            episode = self.episodes[self._index][self._sub_episode_index].copy()
            episode['done'] = False  # Last response?
            self._sub_episode_index += 1  # Increment response

            # If we have potential recs in this conversation
            if len(episode['ml_id']) and predicted_matrix_ids is not None:
                # Truncate
                predicted_matrix_ids = predicted_matrix_ids[:max(self._eval_at)]
                ranks = []
                # Dict[MovieId, Rank]
                predicted_movie_map = {_id: r for r, _id in enumerate(predicted_matrix_ids)}
                # ground_truth_genres = set()
                # For each movie check if we predicted correctly
                for movie in episode['ml_id']:
                    # Check if we have not already seen it, -1 means we do not have it
                    if movie.matrix_id in self._already_seen or movie.matrix_id == -1:
                        continue
                    # ground_truth_genres.update(movie.genres)
                    # Check if we predicted correctly, if we truncate to top 300 id may not exist
                    if movie.matrix_id in predicted_matrix_ids:
                        position = predicted_movie_map[movie.matrix_id]
                        ranks.append(position)
                    # Count only first occurence
                    self._already_seen.add(movie.matrix_id)

                # genres = set()
                genres_recall = []

                # for matrix_id in predicted_matrix_ids[:25]:
                #     genres.update(self.matrixid_to_movie[matrix_id].genres)
                #     # num of genres correctly predicted @k / movies genres
                #     genres_recall.append(float(len(genres.intersection(ground_truth_genres)))
                #                          / len(ground_truth_genres))
                # We append the rank of all possible relevant for this utterance
                # List[List[Item Rank], N Total Relevant]
                self._item_ranks.append((ranks, episode['ml_id'], genres_recall))

                # Keep track of results per a turn in the episode
                if len(self._episode_item_ranks) <= self._turn:
                    self._episode_item_ranks.append([(ranks, episode['ml_id'], genres_recall)])
                else:
                    # Append to existing episode
                    self._episode_item_ranks[self._turn].append((ranks, episode['ml_id'], genres_recall))
                self._turn += 1
        else:
            print("Archana1...in ELSE")
            # Return the computed metrics
            episode = {
                # Signal we are done with this episode
                'done': True,
                # Add Conversation Id
                'convId': self.episodes[self._index][-1]['convId']
            }
            self._index += 1
            self._reset()
        print("Archana returning episode = ", episode)
        time.sleep(2)
        return episode

    def _compute_metrics(self, item_ranks) -> dict:
        """
        dcg = sum[(2^rel[i] - 1) / log(i + 2) for i range(K)]
        ndcg = dcg / ideal dcg
        Recall = Hits@K / n_relvance
        MRR = 1/ Min Hit Position

        :return: Dictionary of NDCG, MRR and Recall
        """
        total_recall = {k: [] for k in self._eval_at}
        total_ndcg = {k: [] for k in self._eval_at}
        total_mrr = {k: [] for k in self._eval_at}
        # genre_recall_count = {k: {} for k in [1, 3, 5, 10, 25]}
        # genre_recall = {k: [] for k in [1, 3, 5, 10, 25]}

        if len(item_ranks):
            max_cutoff = max(self._eval_at)
            # Compute the ideal dcg for each user
            ideal_dcg = np.ones(max_cutoff, dtype=np.float32)
            # Compute Ideal NDCG
            ideal_dcg = ((np.power(2, ideal_dcg) - 1.0) / np.log2(np.arange(2, max_cutoff + 2)))

            # We can view this as for each user which is actually a single utterance
            for ranks, movies, genre in item_ranks:
                n_relv = len(movies)
                recall = {k: 0 for k in self._eval_at}
                dcg = {k: 0.0 for k in self._eval_at}
                # For each item we ranked
                for item_rank in ranks:
                    # For each cut off we want to calculate for
                    for k in self._eval_at:
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

                for i, k in enumerate(self._eval_at):
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
        # metrics.update({"genre_count_r@%s" % k: str(v) for k, v in genre_recall_count.items() if len(v)})
        # metrics.update({"genre_r@%s" % k: np.mean(v) if len(v) else 0.0 for k, v in genre_recall.items()})
        # metrics.update({"genre_rstd@%s" % k: np.std(v) if len(v) else 0.0 for k, v in genre_recall.items()})
        return metrics

    def metrics(self) -> dict:
        """Get recall over the dataset passed through so far"""
        metrics = self._compute_metrics(self._item_ranks)
        # results_by_turn = [
        #                    ]
        metrics["turn"] = [self._compute_metrics(self._episode_item_ranks[i])
                           for i in range(len(self._episode_item_ranks))]
        # for turn in results_by_turn:
        #     for k, v in turn.items():
        #         turn[k] = np.mean(v)
        #
        return metrics


class BaseConvMovieInterface(object):

    def reset(self):
        """Reset the state for this model eg optimizers, randomly init weights"""
        raise NotImplementedError

    def observe(self, prev_state: dict, is_seeker: bool, text: str) -> dict:
        """Show the current conversation data to model"""
        raise NotImplementedError


class ModelRunner(object):

    def __init__(self, opt, model: BaseConvMovieInterface, simulator: MovieConversationSimulator):
        """
        Execute the experiment

        :param opt:
        :param model:
        :param simulator:
        """
        self.model = model
        self.opt = opt
        self.simulator = simulator
        self.verbose = getattr(opt, 'verbose', False)
        # self.verbose = True

        self.is_training = getattr(opt, 'train', False)
        self.train_labels = None
        self.full_debug = getattr(opt, 'debug', False)
        # self.full_debug = True

        self.debug_info = {}

        if hasattr(opt, 'update_length') and opt.update_length > 0:
            raise NotImplementedError("Update Length not supported")

        # If we set the training flag
        if self.is_training:
            raise Exception("NO TRAINING PLEASE")
            # Set each conversation id with all the movies mentioned for training
            self.train_labels = defaultdict(set)
            for episode in self.simulator.episodes:
                for msg in episode:
                    self.train_labels[msg['convId']].update(
                            [movie.matrix_id for movie in msg['ml_id']])
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)
        print("In Model Runner")


    def run(self) -> dict:
        """
        Run experiment for a model


        :return: metrics
        """
        self.simulator.reset_episodes()
        _count = len(self.simulator)
        skip_count = 0
        start = dt.datetime.now()
        print("Start Time: %s\n" % start)
        
        # Loop through each episode
        for episode_idx in tqdm(range(_count)):
            prev_state = None
            episode = None
            # if episode_idx > 100:
            #     break
            # Reset Model State
            self.model.reset()
            prev_movies = []

            if self.verbose:
                print(f"\nEpisode ID: {episode_idx:,}/{_count:,}  {episode_idx/_count*100.:.2f}%")

            while True:
                if self.verbose and skip_count > 0:
                    # Skip Episodes
                    skip_count -= 1
                    episode = self.simulator.get_episode(None)
                    while not episode['done']:
                        episode = self.simulator.get_episode(None)
                    break



                if self.is_training:
                    raise Exception("NO THNAK YOU")
                    observed = {}
                    # TODO: Check if we should actually do this, since we are
                    #  only updating when a movie is mentioned? Maybe wont be
                    #  robust to noise?
                    if episode and len(episode['ml_id']):
                        # Perform recommendations based on previous conversation
                        observed = self.model.observe(prev_state,
                                                      episode['seeker'],
                                                      self.train_labels[episode['convId']])
                else:
                    # Perform recommendations based on previous conversation
                    observed = self.model.observe(prev_state,
                                                  episode['seeker'] if episode else None,
                                                  episode['text'] if episode else "")
                predicted_ids = observed.get('rec', None)
                print("predicted_ids = ", predicted_ids)

                if self.verbose:
                    if episode:

                        if len(episode['ml_id']):
                            # print(colored("{:<4}Ground Truth: {}".format(
                            #         "", " | ".join(["{}:{}".format(m.title,
                            #                                        predicted_ids.index(m.matrix_id)) for m in
                            #                         episode['ml_id']])), "red"))

                            print(colored("{:<4}Ground Truth: {}".format(
                                    "", " | ".join(["{}".format(m.title) for m in
                                                    episode['ml_id']])), "red"))
                            # if len(prev_movies):
                            #     print(" " * 7, colored("Previous: {}".format(
                            #             " | ".join(["{}:{}".format(m.title,
                            #                                        predicted_ids.index(m.matrix_id)) for m in
                            #                         prev_movies])), "magenta"))

                        # if len(episode['ml_id']):
                            # Keep track of previous movies to show rank change
                            prev_movies.extend(episode['ml_id'])
                            # Matrix IDs
                            print(colored("    Pred:", 'cyan'), " | ".join([self.simulator.matrixid_to_movie[_id].title
                                                                           for _id in predicted_ids[:10]]))
                # Evaluate the predicted ids on the utterance at t
                # then reveal the utterance at t to the model
                episode = self.simulator.get_episode(predicted_ids)

                # Done
                if episode['done']:
                    if self.verbose:
                        # print(" " * 7, colored("Previous: {}".format(
                        #     " | ".join(["{}:{}".format(m.title,
                        #                                predicted_ids.index(m.matrix_id)) for m in
                        #                 prev_movies])), "magenta"))

                        print(" " * 7, colored("Previous: {}".format(
                            " | ".join(["{}".format(m.title) for m in
                                        prev_movies])), "magenta"))
                    break

                # Update our semantic vector
                prev_state = episode

                if self.verbose:
                    #  red, green, yellow, blue, magenta, cyan, white
                    if episode['seeker']:
                        s = colored(f"Seek: {episode['text']}", 'green')
                    else:
                        s = colored(f"\nRec: {episode['text']}", 'yellow')
                    print(s)
                    if predicted_ids and episode['seeker'] and not len(episode['ml_id']) and len(prev_movies):
                        # print(" " * 7, colored("Previous: {}".format(
                        #         " | ".join(["{}:{}".format(m.title,
                        #                                    predicted_ids.index(m.matrix_id)) for m in
                        #                     prev_movies])), "magenta"))
                        print(" " * 7, colored("Previous: {}".format(
                                " | ".join(["{}".format(m.title) for m in
                                            prev_movies])), "magenta"))


            # Status Report
            if self.verbose and skip_count == 0:
                print(f"\n\nConvID: {episode['convId']}\n")
                # print(self.simulator.metrics())
                print(self.simulator.metrics()['ndcg@10'])
                print()


                # if episode['convId'] == '1000878':
                #     cont = input("Any key to continue (q to quit or int to skip): ").strip()
                #     if cont == 'q':
                #         sys.exit()
                #     if cont.isdigit():
                #         skip_count = int(skip_count)
                if self.simulator.metrics()['ndcg@10'] > 0.03:
                    cont = input("Any key to continue (q to quit or int to skip): ").strip()
                    if cont == 'q':
                        sys.exit()
                    if cont.isdigit():
                        skip_count = int(skip_count)
                # if not getattr(sys, 'gettrace', None)():
                #     cont = input("Any key to continue (q to quit or int to skip): ").strip()
                #     if cont == 'q':
                #         sys.exit()
                #     if cont.isdigit():
                #         skip_count = int(skip_count)
                print()
                print("="*120)
                print("\n")
            # Full debug get metrics for each conversation and reset the metric
            if self.full_debug:
                self.debug_info[episode['convId']] = {
                    # aggregated metrics for this conversation
                    'total': self.simulator.metrics(),
                    # Metrics calculated based on each valid turn
                    'turn': [self.simulator._compute_metrics(self.simulator._item_ranks[i])
                             for i in range(len(self.simulator._item_ranks))]
                }

                # NOTE: We clear out the values!!!
                self.simulator._item_ranks = []

        print("\n\nTook %s" % (dt.datetime.now()-start))
        print("End Time: %s\n" % dt.datetime.now())
        print(self.simulator.metrics())
        time.sleep(2)
        return self.simulator.metrics()


