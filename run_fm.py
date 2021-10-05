import pickle
import numpy as np
import torch.utils.data
import pandas as pd

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os

# from torchfm.dataset.avazu import AvazuDataset
# from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
# from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork


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


class GoRecDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.seeker_id = 0
        train_data = self.__preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6_kfold/splits.pkl')
        test_data = self.__preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6/test.pkl')

        full_data = np.vstack((train_data, test_data))
        if mode == 'train':
            data = train_data
        elif mode == 'test':
            data = test_data
        else:
            raise ValueError

        self.data = data

        self.items = data[:, :2].astype(np.int)
        self.full_items = full_data[:, :2].astype(np.int)

        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.full_items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    def __preprocess_data(self, folder_path):
        full_data = []

        for fold_index in range(5):
            data = pickle.load(open(os.path.join(folder_path, "{}.pkl".format(fold_index)), 'rb'))
            episodes = data['data']
            for episode in episodes:
                for conv in episode:
                    if conv['ml_id']:
                        for m in conv['ml_id']:
                            r = 5 if conv['sentiment'] >= 0.5 else 0
                            full_data.append([self.seeker_id, m, r])
                self.seeker_id += 1

        return np.array(full_data)


class GoRecEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, items):
        # test data
        # data = self.__preprocess_data('data/gorecdial/transformer/gorecdial_flr1e-6_l21e-6/test.pkl')
        self.ranked_items = np.unique(items[:, 1])

        items = data[:, :2].astype(np.int)
        self.items = self.__extend_test(items)

        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        return self.items[index]

    def __extend_test(self, items):
        extended_items = []
        user_id = np.unique(items[:, 0])
        print('saving the evaluation dataset')
        for u in tqdm.tqdm(user_id):
            for m in self.ranked_items:
                extended_items.append([u, m])
        return np.array(extended_items)

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    # def __preprocess_data(self, folder_path):
    #     seeker_id = 0
    #     full_data = []
    #
    #     for fold_index in range(5):
    #         data = pickle.load(open(os.path.join(folder_path, "{}.pkl".format(fold_index)), 'rb'))
    #         episodes = data['data']
    #         for episode in episodes:
    #             for conv in episode:
    #                 if conv['ml_id']:
    #                     for m in conv['ml_id']:
    #                         r = 5 if conv['sentiment'] >= 0.5 else 0
    #                         full_data.append([seeker_id, m, r])
    #             seeker_id += 1
    #
    #     return np.array(full_data)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    # elif name == 'hofm':
    #     return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def eval_metrics(model, test_dataloader, test_extend_loader, device):
    model.eval()
    item_ranks = []
    predicts = []
    users = []
    items = []

    with torch.no_grad():
        for fields in tqdm.tqdm(test_extend_loader, smoothing=0, mininterval=1.0):
            users.extend(fields[:, 0].tolist())
            items.extend(fields[:, 1].tolist())

            fields = fields.to(device)
            y = model(fields)
            predicts.extend(y.tolist())

    df = pd.DataFrame({'user': users,
                       'item': items,
                       'rating': predicts})
    df_grouped = df.groupby('user')

    ranking = []
    for g, rows in df_grouped:
        ranking.extend(rows['rating'].rank(method='dense', ascending=False).values.tolist())

    df['ranking'] = ranking

    for fields, _ in test_dataloader:
        for field in fields:
            field = field.tolist()
            m, u = field[0], field[1]
        # for m, u in fields:
            r = df[(df['user'] == m) & (df['item'] == u)].ranking.values[0]
            item_ranks.append(([r], [fields[1]]))

    return compute_metrics(item_ranks)



def main(model_name,
         epoch,
         batch_size,
         learning_rate,
         weight_decay,
         device):
    train_dataset = GoRecDataset(mode='train')
    test_dataset = GoRecDataset(mode='test')
    test_extend_dataset = GoRecEvalDataset(test_dataset.data, test_dataset.full_items)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    test_extend_loader = DataLoader(test_extend_dataset, batch_size=batch_size, num_workers=8)


    model = get_model(model_name, train_dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, test_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)

    print(eval_metrics(model, test_data_loader, test_extend_loader, device))



if __name__ == '__main__':
    import argparse
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    args = parser.parse_args()
    main(args.model_name,
         args.epoch,
         args.batch_size,
         args.learning_rate,
         args.weight_decay,
         device)

