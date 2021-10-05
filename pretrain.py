#!/usr/bin/env python
<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
@author:   Travis Ebesu
@summary:  Pretrain a latent factor model
"""
=======
>>>>>>> be21c49 (adding code and data)
import sys
import argparse
import os
import numpy as np
from datetime import datetime as dt
from tqdm import tqdm
from util.helper import parser_add_str2bool
from util.models.gmf import GeneralizedMatrixFactorization
from util.models.mlp import MLPMatrixFactorization
from util.models.mf import MatrixFactorization
from util.models.neumf import NeuralMatrixFactorization
from util.data import Dataset
from rutil.metric.ir import precision_recall_score
import torch
import json
import torch.nn.functional as F
# import rutil.util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_add_str2bool(parser)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str,
                    default="0")
parser.add_argument('-m', '--model', type=str, choices=['mf', 'gmf', 'nmf', 'mlp'],
                    help='model name/version', default='gmf', required=True)

parser.add_argument('-i', '--iters', help='Max iters', type=int, default=3)
parser.add_argument('-bs', '--batch_size', help='Batch Size', type=int, default=128)

parser.add_argument('-n', '--neg_count', help='Negative Samples Count', type=int, default=4)
parser.add_argument('-d', '--dataset', help='directory to train/test npz file',
                    type=str, required=True)
parser.add_argument('-l2', '--l2', help='l2 Regularization', type=float, default=1e-6)
parser.add_argument('-mf', '--model_file', help='path to model directory',
                    type=str)
parser.add_argument('-ft', '--fine_tune', help='path to model to finetune',
                    type=str)
parser.add_argument('-opt', '--optimizer', help='Optimizer type', type=str,
                    default='adam', choices=['adam', 'sgd', 'rmsprop'])
parser.add_argument('-lr', '--learning_rate', help='Learning Rate', type=float, default=1e-3)
parser.add_argument('-mom', '--momentum', help='Momentum for SGD/RMSProp', type=float, default=0.9)
parser.add_argument('-decay', '--decay', help='Decay for RMSProp', type=float, default=0.99)
parser.add_argument('-loss', '--loss', help='Loss type', type=str,
                    default='bpr', choices=['bpr', 'xent'])

parser.add_argument('-e', '--embed_size', help='Embedding Size', type=int, default=16)
parser.add_argument('-nl', '--n_layers', help='Number of layers, // 2 of embed size',
                    type=int, default=3)
parser.add_argument('-tie', '--tie', help='Tie the MF and MLP embedding layers',
                    type='bool', default=True)
parser.add_argument('-gs', '--grid_search', help='Perform full grid search',
                    type='bool', default=False)
parser.add_argument('-init', '--init', help='Rescale embeddings weights?',
                    type=str, default='default', choices=['default', 'he', 'xavier'])
parser.set_defaults(start_time=dt.now().strftime("%Y-%m-%d %H:%M:%S"))

<<<<<<< HEAD



"""
MODEL_OPTIONS = {
    'mf': MatrixFactorization,
    'gmf': GeneralizedMatrixFactorization,
    'nmf': NeuralMatrixFactorization
}


def parse_args(args=None):

    # Find the model to add
    argv = sys.argv[1:]

    # We are debugging...
    # if getattr(sys, 'gettrace', None)():
    #     print("\nDebugging\n")
    #     model_name = 'x'
    #     argv = ['-m', model_name]
    #     parser.set_defaults(model=model_name)

    model_name = None
    for i in range(len(argv)):
        if argv[i] == "-m" or argv[i] == "--model":
            model_name = argv[i + 1].strip()
            model = MODEL_OPTIONS[model_name]
            model.add_arguments(parser)
            break

    # Get the default model and add its arguments
    if model_name is None:
        MODEL_OPTIONS[parser.__dict__['model']].add_arguments(parser)

    return parser.parse_args(args)
"""


=======
>>>>>>> be21c49 (adding code and data)
def run(opt, dataset):
    # rutil.util.setup_exp(opt, copy_dirs=['util'])
    if opt.model == 'mf':
        model = MatrixFactorization(opt)
    elif opt.model == 'nmf':
        opt.layers = []
        size = opt.embed_size
        if opt.tie:
            size = opt.embed_size * 2
        for _ in range(opt.n_layers + 1):
            opt.layers.append(size)
            # Half
            size = size // 2
        model = NeuralMatrixFactorization(opt)
    elif opt.model == 'gmf':
        model = GeneralizedMatrixFactorization(opt)
    elif opt.model == 'mlp':
        model = MLPMatrixFactorization(opt)
    else:
        raise ValueError("Unknown model type... %s" % opt.model)
    print(model)
    print(opt)
    model.cuda()

    opt_params = dict(params=model.parameters(),
                      # weight_decay=opt.l2,
                      lr=opt.learning_rate)
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(**opt_params)
    elif opt.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(momentum=opt.momentum, alpha=opt.decay,
                                        **opt_params)
    else:
        optimizer = torch.optim.SGD(momentum=opt.momentum, **opt_params)

    epoch = 0
    if opt.model_file:
        save_model_filename = os.path.join(opt.model_file, 'model.pth')
        if os.path.exists(save_model_filename):
            print(f"Loading Existing Model From {save_model_filename}")
            states = torch.load(open(save_model_filename, 'rb'), map_location='cpu')
            opt.loss = states.get('loss', 'bpr')
            model.load_state_dict(states['model'], strict=False)
            optimizer.load_state_dict(states['opt'])
            epoch = states.get('epoch', 0)
            print(f"Resuming from epoch {epoch}")
        elif opt.fine_tune:
            print(f"Fine tuning Model From {opt.fine_tune}")
            states = torch.load(open(opt.fine_tune, 'rb'), map_location='cpu')
            opt.loss = states.get('loss', 'bpr')
            print(states.keys())
            # model.item_memory.load_state_dict(states['model'])
            model.load_state_dict({'item_memory.weight': states['model']['item_memory.weight']},
                                  strict=False)

    # Train Loop
    while epoch < opt.iters:
        epoch += 1
        model.train()
        loss = []
        if opt.loss == 'bpr':
            _data = dataset.get_data(opt.batch_size, False, opt.neg_count)
        else:
            _data = dataset.get_xent_data(opt.batch_size, opt.neg_count)

        progress = tqdm(
                enumerate(_data),
                total=(dataset.train_size * opt.neg_count) // opt.batch_size,
                # enumerate(dataset.get_data_uniform(opt.batch_size, opt.neg_count)),
                # total=(dataset.user_count * opt.neg_count) // opt.batch_size,
                dynamic_ncols=True, leave=False)
        for k, example in progress:
            if k % 100000 == 0:
                # Weight statistics
                print("-" * 80)
                for name, param in model.named_parameters():
                    if ('user' in name or 'item' in name) and len(param.shape) == 2:
                        l2 = param.norm(dim=1)
                        l2_std = l2.std()
                        l2 = l2.mean()
                        min_val = param.min(dim=1)[0].mean()
                        max_val = param.max(dim=1)[0].mean()
                    else:
                        l2 = param.norm()
                        l2_std = torch.tensor([0.0])
                        min_val = param.min()
                        max_val = param.max()
                    mean = param.mean()
                    std = param.std()
                    print(f"{name:<20} L2: {l2.item():>8.4f} ± {l2_std.item():<10.2f}"
                          f" Mean: {mean.item():>8.4f} ± {std.item():<10.4f} "
                          f"Min: {min_val.item():<10.4f} "
                          f"Max: {max_val.item():<10.4f}")
                print("=" * 80)

            optimizer.zero_grad()
            if opt.loss == 'bpr':
                example = example.astype(np.int64)
                users = torch.from_numpy(example[:, 0]).cuda()
                items = torch.from_numpy(example[:, 1]).cuda()
                neg_items = torch.from_numpy(example[:, 2]).cuda()
                pos = model(users, items)
                neg = model(users, neg_items)
                batch_loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean() \
                             + model.get_l2_reg(opt.l2, users, items, neg_items)
            else:
                # Cross Entropy Loss
                example, target = example
                users = torch.from_numpy(example[:, 0]).cuda()
                items = torch.from_numpy(example[:, 1]).cuda()
                target = torch.from_numpy(target).cuda()
                score = model(users, items)
                reg = opt.l2 * model.item_memory(items).norm() \
                      + opt.l2 * model.user_memory(users).norm()
                batch_loss = F.binary_cross_entropy_with_logits(score, target) + reg

            batch_loss.backward()
            optimizer.step()
            loss.append(batch_loss.data.item())
            progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(epoch, batch_loss.item()))
        print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(epoch, np.mean(loss)))
        model.eval()
        EVAL_AT = [10, 25, 50]
        batch_scores = []
        test_batch_size = 32
        index = np.arange(dataset.user_count, dtype=np.int64)
        batch_count = int(np.ceil(dataset.user_count / test_batch_size))
        # batch_count = min(batch_count, test_batch_size*20)
        print(f"\n\nEvaluating on {batch_count * test_batch_size} users!")
        train_prec, train_recall, test_prec, test_recall = [], [], [], []
        user_offset = 0
        with torch.no_grad():
            for i in tqdm(range(batch_count), desc="Evaluation", leave=False):
                lo = i * test_batch_size
                hi = lo + test_batch_size
                idx = torch.from_numpy(index[lo:hi]).cuda()
                s = model.recommend(idx).data.cpu().numpy()
                if len(s.shape) == 1:
                    s = s.reshape(1, -1)
                batch_scores.append(s)
                # Evaluate every 10, else we can run out of memory
                if len(batch_scores) > 20 or i == (batch_count - 1):
                    scores = np.concatenate(batch_scores)
                    batch_scores = []
                    batch_train_prec, batch_train_recall = precision_recall_score(
                            scores, dataset.train_relevance, EVAL_AT, return_all=True, offset=user_offset)
                    train_prec.append(batch_train_prec)
                    train_recall.append(batch_train_recall)

                    batch_test_prec, batch_test_recall = precision_recall_score(
                            scores, dataset.test_relevance, EVAL_AT, return_all=True,
                            offset=user_offset)
                    test_prec.append(batch_test_prec)
                    test_recall.append(batch_test_recall)
                    user_offset += scores.shape[0]

        s = "\n\n" + ("=" * 80) + "\n"
        p = ("-" * 80) + "\n"
        train_prec = np.concatenate(train_prec, 1).mean(1)
        train_recall = np.concatenate(train_recall, 1).mean(1)
        test_prec = np.concatenate(test_prec, 1).mean(1)
        test_recall = np.concatenate(test_recall, 1).mean(1)

        summary = {}
        for _idx in range(len(EVAL_AT)):
            K = EVAL_AT[_idx]
            summary.update({
                f"train_recall{K}": float(train_recall[_idx]),
                f"recall{K}": float(test_recall[_idx]),
                f"train_prec{K}": float(train_prec[_idx]),
                f"prec{K}": float(test_prec[_idx]),
            })
            s += "Train Recall@{k:>3}:     {train:<20} Test Recall@{k:>3}:     {test:<20}\n".format(
                    train="{:.5f}".format(train_recall[_idx]),
                    test="{:.5f}".format(test_recall[_idx]),
                    k=K)
            p += "Train Prec@{k:>3}:       {train:<20} Test Prec@{k:>3}:       {test:<20}\n".format(
                    train="{:.5f}".format(train_prec[_idx]),
                    test="{:.5f}".format(test_prec[_idx]),
                    k=K)

        if opt.model_file:
            summary['epoch'] = epoch
            summary['loss'] = float(np.mean(loss))
            with open(os.path.join(opt.model_file, 'results.json'), 'a') as f:
                f.write(json.dumps(summary, sort_keys=True))
                f.write("\n")
            torch.save({'opt': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'params': opt, 'epoch': epoch,
                        'summary': summary,
                        'loss': opt.loss},
                       open(os.path.join(opt.model_file, 'model.pth'), 'wb'))
            # Save just the model every epoch
            torch.save({'model': model.state_dict(),
                        'params': opt, 'epoch': epoch,
                        'summary': summary,
                        'loss': opt.loss},
                       open(os.path.join(opt.model_file, 'model_%s.pth' % epoch), 'wb'))

        # Weight statistics
        print(s + p + ("-" * 80))
        for name, param in model.named_parameters():
            if ('user' in name or 'item' in name) and len(param.shape) == 2:
                l2 = param.norm(dim=1)
                l2_std = l2.std()
                l2 = l2.mean()
                min_val = param.min(dim=1)[0].mean()
                max_val = param.max(dim=1)[0].mean()
            else:
                l2 = param.norm()
                l2_std = torch.tensor([0.0])
                min_val = param.min()
                max_val = param.max()
            mean = param.mean()
            std = param.std()
            print(f"{name:<20} L2: {l2.item():>8.4f} ± {l2_std.item():<10.2f}"
                  f" Mean: {mean.item():>8.4f} ± {std.item():<10.4f} "
                  f"Min: {min_val.item():<10.4f} "
                  f"Max: {max_val.item():<10.4f}")

        print("=" * 80)


if __name__ == '__main__':
    opt = parser.parse_args()
    # logging.basicConfig(level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    print("Loading dataset....")
    dataset = Dataset(opt.dataset)
    print("dataset = ", opt.dataset)

    # Dataset Specific
    opt.item_count = dataset.item_count
    opt.user_count = dataset.user_count

    if opt.grid_search:
        opt.tie = True
        for l2, embed, n, init in [
            (1e-5, 40, 2, 'he'),
            (1e-4, 40, 2, 'he'),
            (1e-6, 40, 2, 'he'),

            (1e-5, 40, 2, 'xavier'),
            (1e-6, 40, 2, 'xavier'),
            (1e-4, 40, 2, 'xavier'),

            (1e-5, 40, 3, 'he'),
            (1e-5, 40, 3, 'xavier'),

            (1e-4, 40, 3, 'he'),
            (1e-6, 40, 3, 'he'),

            (1e-5, 50, 3, 'he'),
            (1e-5, 50, 4, 'he'),
            # (1e-7, 50, 2, 'he'),
            # (1e-7, 50, 2, 'xavier'),
            # (1e-7, 50, 4, 'he'),
            # (1e-7, 50, 4, 'xavier'),
        ]:
            opt.l2 = l2
            opt.embed_size = embed
            opt.n_layers = n
            opt.init = init

            if opt.l2 in [1e-2, 1e-3, 1e-4]:
                opt.model_file = f"results/{opt.model}/{opt.embed_size}_{opt.n_layers}l/tie/l2{str(opt.l2)}/"
            else:
                opt.model_file = f"results/{opt.model}/{opt.embed_size}_{opt.n_layers}l/tie/l2{str(opt.l2).replace('0','')}/"
            if opt.model == 'mlp' and opt.init != 'default':
                opt.model_file = opt.model_file.replace("/tie/", f"/{opt.init}/")
            if os.path.exists(opt.model_file):  # and os.path.exists(os.path.join(opt.model_file, f"model_{opt.iters}.pth")):
                print("Epoch Completed?", os.path.exists(os.path.join(opt.model_file, f"model_{opt.iters}.pth")))
                print("Already Completed: ", opt)
            else:
                opt.start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                run(opt, dataset)

<<<<<<< HEAD
        """
        # for model in ['mf', 'gmf', 'nmf']:
        #     opt.model = model
        # for L2 in [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        for L2 in [0, 1e-5]:
            opt.l2 = L2
            for EMBED_SIZE in [50]:
                opt.embed_size = EMBED_SIZE
                if opt.model in ['mlp', 'nmf']:
                    for N_LAYERS in [2, 3, 4]: # 2,3
                        opt.n_layers = N_LAYERS

                        if opt.l2 in [1e-2, 1e-3, 1e-4]:
                            opt.model_file = f"results/{opt.model}/{opt.embed_size}_{opt.n_layers}l/tie/l2{str(opt.l2)}/"
                        else:
                            opt.model_file = f"results/{opt.model}/{opt.embed_size}_{opt.n_layers}l/tie/l2{str(opt.l2).replace('0', '')}/"

                        if opt.model == 'mlp' and opt.init != 'default':
                            opt.model_file = opt.model_file.replace("/tie/", f"/{opt.init}/")
                        if os.path.exists(opt.model_file):# and os.path.exists(os.path.join(opt.model_file, f"model_{opt.iters}.pth")):
                            print("Epoch Completed?", os.path.exists(os.path.join(opt.model_file, f"model_{opt.iters}.pth")))
                            print("Already Completed: ", opt)
                            continue
                        opt.start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        run(opt, dataset)
                else:
                    # RUN GMF/MF
                    if opt.l2 in [1e-2, 1e-3, 1e-4]:
                        opt.model_file = f"results/{opt.model}/{opt.embed_size}/l2{str(opt.l2)}"
                    else:
                        opt.model_file = f"results/{opt.model}/{opt.embed_size}/l2{str(opt.l2).replace('0', '')}"
                    if os.path.exists(opt.model_file):# and os.path.exists(os.path.join(opt.model_file, f"model_{opt.iters}.pth")):
                        final_path = os.path.join(opt.model_file, f"model_{opt.iters}.pth")
                        print("Epoch Completed?",
                              os.path.exists(final_path), final_path)
                        print("Already Completed: ", opt)
                        continue
                    opt.start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                    run(opt, dataset)
        """
=======
>>>>>>> be21c49 (adding code and data)
    else:
        run(opt, dataset)
