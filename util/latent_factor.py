from typing import List
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.data import Movie
from util.simulator import BaseConvMovieInterface
from util.models.gmf import GeneralizedMatrixFactorization
from util.models.mlp import MLPMatrixFactorization
from util.models.mf import MatrixFactorization
from util.models.neumf import NeuralMatrixFactorization


class MovieSelector(object):

    def __init__(self, movies: List[Movie], device='cpu'):
        """

        :param movies: List of movies
        :param device: cpu/cuda for pytorch, defaults to CPU
        """
        self.device = device
        self.movies = movies
        # Extract movie content vectors
        vectors = np.stack([movie.vector for movie in movies])
        self.vectors = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
        self.embed_dim = self.vectors.shape[1]

    def query_vector(self, query_vec: np.ndarray, top_k: int = None, reverse: bool = False):
        """
        Performs a simple inner product between query and all movies then
        returns the top-k

        :param query_vec: Single vector of the same dim as encoded vectors
        :param top_k: int, top_k to return if none return all
        :param reverse: bool, if we should sort ascending
        :return: similarity scores, movies
        """
        # Inner product similarity
        # [N, D] [D, 1]
        if isinstance(query_vec, np.ndarray):
            query_vec = torch.from_numpy(query_vec.astype(np.float32))

        query_vec = query_vec.to(self.device).view(self.embed_dim, 1)
        scores = torch.matmul(self.vectors, query_vec).view(-1)

        # Sorted by similarity score
        if reverse:
            indices = torch.argsort(scores)
        else:
            indices = torch.argsort(-scores)
        scores = scores[indices]

        # Truncate
        if top_k is not None:
            scores = scores[:top_k]
            indices = indices[:top_k]
        return scores, [self.movies[i] for i in indices]


class AvgLatentFactorAgent(BaseConvMovieInterface):

    def __init__(self, opt: Namespace, movies_list: List[Movie]):
        """
        Use averaged user latent factor rather than learning one

        :param opt:
        :param movies_list:
        """
        states = torch.load(opt.model_file, 'cpu')
        model_type = opt.model
        self.model_params = states['params']
        if model_type == 'avggmf':
            model = GeneralizedMatrixFactorization(states['params'])
        elif model_type == 'avgmf':
            model = MatrixFactorization(states['params'])
        elif model_type == 'avgnmf':
            model = NeuralMatrixFactorization(states['params'])
        else:
            raise ValueError(f"Unknown.... model: {model_type}")
        model.load_state_dict(states['model'])

        model.eval()
        with torch.no_grad():
            if model_type in ['avggmf', 'avgmf']:
                # Set the new user latent rep to averaged version
                model.new_user.data = model.user_memory.weight.data.mean(0).squeeze()
            else:
                # NMF has two set, although they can be tied
                model.mf_new_user.data = model.mf_user.weight.data.mean(0).squeeze()
                model.mlp_new_user.data = model.mlp_user.weight.data.mean(0).squeeze()
            scores = model.new_user_recommend().cpu().numpy()
            self.predicted_ids = np.argsort(-scores).tolist()

    def reset(self):
        """Nothing to reset"""
        return

    def observe(self, prev_state, is_seeker: bool, msg_length) -> dict:
        return {'rec': self.predicted_ids}


class LatentFactorConvMovieAgent(BaseConvMovieInterface):

    def __init__(self, opt: Namespace, movies_list: List[Movie]):
        """

        :param opt: options/config
        :param movies_list: List of the movies for querying
        """
        self.opt = opt
        self.selector = MovieSelector(movies_list, 'cpu')
        self.optimizer = None
        self.model = None
        self.verbose = getattr(opt, 'verbose', False)
        self.is_training = getattr(opt, 'train', False)
        self._sentiment_style = getattr(self.opt, 'sentiment', 'scale')
        if getattr(self.opt, 'replay', False):
            raise NotImplementedError("Reimbelement it if you want to use it")
        assert getattr(self.opt, 'loss', 'bpr') == 'bpr', 'xent not supported'
        assert opt.weight_rescale == 'none', "Meh no weight rescaling..."
        states = torch.load(opt.model_file, 'cpu')
        model_type = opt.model
        self.model_params = states['params']

        if model_type == 'gmf':
            self.model = GeneralizedMatrixFactorization(states['params'])
        elif model_type == 'mf':
            self.model = MatrixFactorization(states['params'])
        elif model_type == 'mlp':
            self.model = MLPMatrixFactorization(states['params'])
        elif model_type == 'nmf':
            self.model = NeuralMatrixFactorization(states['params'])
        else:
            raise ValueError(f"Unknown.... model: {model_type}")
        self.model.load_state_dict(states['model'])

        if self.is_training:
            print("[Fine tunning model on dataset]\n")
            # self.model.v.requires_grad = False
            self.model.user_memory.weight.requires_grad = False
        else:
            self.model.freeze_params()
        self.reset()
        self.total_hits = 0
        # Rescale all weights by l2 norm
        # if opt.weight_rescale == 'l2':
        #     for p in self.model.parameters():
        #         if len(p.shape) == 2:
        #             # [N, E] => [N, 1]
        #             p.data /= p.data.norm(dim=1).unsqueeze(-1)
        # elif opt.weight_rescale == 'l2_item':
        #     # Normalize only the item embeddings
        #     self.model.item_memory.weight.data /= self.model.item_memory.weight.data.norm(dim=1).unsqueeze(-1)

    def reset(self):
        """
        Reset the model's new user parameters and reinit the optimizer
        :return:
        """
        # Init new user
        self.model.new_user_reset(self.opt.init)
        # eps = self.model.user_memory.weight.data.var()
        # self.model.new_user.data.uniform_(-eps, eps)

        if self.is_training:
            # Init only once, we zero out the cached momentum for new users
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=self.opt.lr, momentum=self.opt.momentum,
                                                 weight_decay=self.opt.l2)
                # self.optimizer = torch.optim.Adam(self.model.parameters(),
                #                                   weight_decay=self.opt.l2)
            # Reset momentum for new user parameters
            new_user_params = set(self.model.new_user_parameters())
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in new_user_params:
                        param_state = self.optimizer.state[p]
                        # This is for SGD Momentum
                        if 'momentum_buffer' in param_state:
                            param_state['momentum_buffer'] = torch.zeros_like(p.data)

                        # This is for Adam
                        if 'step' in param_state:
                            param_state['step'] = 0
                            # Exponential moving average of gradient values
                            param_state['exp_avg'] = torch.zeros_like(p.data)
                            # Exponential moving average of squared gradient values
                            param_state['exp_avg_sq'] = torch.zeros_like(p.data)

        else:
            self.optimizer = torch.optim.SGD(self.model.new_user_parameters(),
                                             lr=self.opt.lr, momentum=self.opt.momentum,
                                             weight_decay=self.opt.l2)

    def _recommend(self):
        self.model.eval()
        with torch.no_grad():
            scores = self.model.new_user_recommend().cpu().numpy()
            predicted_ids = np.argsort(-scores).tolist()
        return predicted_ids

    def _sample_neg(self, retrieved, pos_matrix_ids=None):
        """
        Sample negative items from the retrieved movies which assumed to be
        sorted

        For hard negative mining we require the ranking scores

        if pos_matrix_ids are passed we make sure these are not in the negative
        sample

        :param retrieved:
        :param pos_matrix_ids:
        :return:
        """
        # Randomly select some negatives from the least similar
        neg_movies = retrieved[-self.opt.weighted_count:]
        if pos_matrix_ids:
            negs = []
            # Sample true negatives for training
            for _ in range(self.opt.neg_count):
                n = neg_movies[np.random.randint(self.opt.weighted_count)]
                while n.matrix_id in pos_matrix_ids:
                    if n.matrix_id in pos_matrix_ids:
                        self.total_hits += 1
                        print(f"[Neg Samples Hit: {self.total_hits}]")
                    n = neg_movies[np.random.randint(self.opt.weighted_count)]
                negs.append(n)
        else:
            # Random sample
            negs = np.random.choice(neg_movies, replace=False if len(neg_movies) >=
                                                                 self.opt.neg_count else True,
                                      size=self.opt.neg_count)
        return negs

    def query_item_lf(self, query_vec: np.ndarray, item_memory: torch.FloatTensor,
                      reverse: bool=False, eps: float=1e-7):
        """
        Given the query vector it computes the similarity against all movies from the
        selector, uses the model's item_memory weights to weight the latent factors

        :param query_vec: Conversation vector to query against the movies
        :param item_memory: [n_items, embed size], item latent factor values
        :param reverse: reverse the sorting from ascending/descending
        :param eps: For L2 normalization stability
        :return:
        """
        scores, retrieved = self.selector.query_vector(query_vec, reverse=reverse)

        # Take only valid ones we have in our matrix
        # Valid score indices where we have the movie id

        # ignore missing movies
        # matrix_ids = [movie.matrix_id for movie in retrieved if movie.matrix_id < item_memory.shape[0]]
        matrix_ids = [movie.matrix_id for movie in retrieved]

        if self.opt.weighted_count is not None:
            scores = scores[:self.opt.weighted_count]
            matrix_ids = matrix_ids[:self.opt.weighted_count]

        # Get all item latent factors that are relevant
        raw_item_lf = item_memory[matrix_ids]

        # attention, softmax normalize
        if self.opt.norm == 'softmax':
            attn = nn.functional.softmax(scores, dim=-1)
        elif self.opt.norm == 'l2':
            # w / ||w||_2 + epsilon
            attn = scores / (torch.norm(scores) + eps)
        else:
            raise ValueError("Unknown normalization scheme")

        # Broadcast and weighted sum
        item_latent_factor = (attn.unsqueeze(-1) * raw_item_lf).sum(0)

        return item_latent_factor, retrieved

    def observe(self, prev_state, is_seeker: bool, train_labels: set=None):
        """

        :param prev_state:
        :param is_seeker:
        :param train_labels: only passed if training
        :return:
        """
        obs = {}

        # Update style: always, seeker or recommender
        if prev_state is not None and self.opt.update == 'all' \
                or (self.opt.update == 'seeker' and is_seeker) \
                or (self.opt.update == 'rec' and not is_seeker):
            # if msg_length > self.opt.update_length:
            self.model.train()

            item_lf, retrieved = self.query_item_lf(prev_state['query_vec'],
                                                    self.model.item_memory.weight)
            # Compute the loss
            neg_movies = self._sample_neg(retrieved)  # Sample negatives

            # If we have sentiment scale else does nothing
            sentiment = torch.tensor(prev_state.get('sentiment', 1.0))
            obs['loss'] = 0.0

            # True SGD updates
            for neg_movie in neg_movies:
                self.optimizer.zero_grad()
                neg_ids = torch.LongTensor([neg_movie.matrix_id])
                neg_item_lf = self.model.item_memory(neg_ids)  # Lookup neg item lf

                pos = self.model.new_user_score(item_lf)
                neg = self.model.new_user_score(neg_item_lf)
                if self._sentiment_style == 'xent':
                    diff = pos-neg
                    loss = torch.nn.BCEWithLogitsLoss()(diff.reshape(-1), sentiment.reshape(-1))
                elif self._sentiment_style == 'scale':
                    loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12) * sentiment
                # elif self._sentiment_style == 'gate':
                #     pos = sentiment * pos
                #     neg = (1.0-sentiment) * neg
                #     loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12)
                elif self._sentiment_style == 'none':
                    loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12)
                else:
                    raise ValueError("Unknown sentiment style.....")

                # retain_graph=True is for item_lf/retrieved
                loss.backward(retain_graph=True)
                # Update
                self.optimizer.step()
                obs['loss'] += loss.item()

            if self.verbose:
                print(f"    Loss: {loss.item():.4f}")
                print("{:<4}Pos: {}".format("", " | ".join([m.title for m in retrieved[:10]])))
                print("{:<4}Neg: {}".format("", " | ".join([m.title for m in neg_movies[:10]])))

        obs['rec'] = self._recommend()
        return obs


class NeuMFAgent(LatentFactorConvMovieAgent):

    def observe(self, prev_state, is_seeker: bool, train_labels: set=None):
        """
        Recommendation is slightly different for NeuMF since we have
        two embeddings instead of one, so we implement a new observe
        """
        obs = {}

        # Update always, seeker or recommender
        # if prev_state is not None and self.opt.update == 'all' \
        #         or (self.opt.update == 'seeker' and is_seeker) \
        #         or (self.opt.update == 'rec' and not is_seeker):

            # if train_labels:
            #     print(train_labels)
            #     if train_labels > self.opt.update_length:
            #         self.model.train()
            #         # We have two item embeddings to learn since they are not shared
            #         mf_item_lf, retrieved = self.query_item_lf(prev_state, self.model.mf_item.weight)
            #         # TODO: this is doing redundant computation, we can optimize it
            #         mlp_item_lf, _ = self.query_item_lf(prev_state, self.model.mlp_item.weight)
            #
            #         # Compute the loss
            #         neg_movies = self._sample_neg(retrieved)  # Sample negatives
            #         obs['loss'] = 0.0
            #         # SGD updates
            #         for neg_movie in neg_movies:
            #             self.optimizer.zero_grad()
            #             neg_ids = torch.LongTensor([neg_movie.matrix_id])
            #             # Lookup neg item lf
            #             neg_mf_item = self.model.mf_item(neg_ids)
            #             neg_mlp_item = self.model.mlp_item(neg_ids)
            #
            #             pos = self.model.new_user_score(mf_item_lf, mlp_item_lf)
            #             neg = self.model.new_user_score(neg_mf_item, neg_mlp_item)
            #
            #             loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12)
            #             loss.backward(retain_graph=True)
            #             # Update
            #             self.optimizer.step()
            #             obs['loss'] += loss.item()
            #
            #         if self.verbose:
            #             print(f"    Loss: {loss.item():.4f}")
            #             print("{:<4}Pos: {}".format("", " | ".join([m.title for m in retrieved[:10]])))
            #             print("{:<4}Neg: {}".format("", " | ".join([m.title for m in neg_movies[:10]])))

        obs['rec'] = self._recommend()
        return obs
