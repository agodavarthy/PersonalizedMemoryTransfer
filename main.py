"""
TODO: Try the following

- Fix Movie Titles, use movie id in conversations and preprocess rather than
    storing the redundant data.

- Use Cross-Entropy Loss
- Try different initialization
- Try other interaction functions
- Try NetFlix datset? Combine with MovieLens?

- Initialize embeddings with caffe init
- L2 normalize embeddings prior to conversation rec
- Dataset we use eg ML is dense but our final task is sparse, refilter so its sparse?
- Use item/user bias?

Exact match retrieval might be better if the title is a hit

Preprocessing/Encoder
- Handle Smilies
- Add movie genre to the movie plot stuff?
- Remove Year from title?
- Dont perform tokenization just feed in raw text
- Finetune model on some conversational like dataset eg reddit movies


Rank discrepency from Model updates vs Retrieved, I think we need some sort of way
to not deviate from the retrieved ones

Try synthetic datsets?

Notes:
    - The magnitude of the pretrained weights appears to have a large impact on the performance.
    - Applying L2 weight normalization prior to init seems to work for
      pretrained model optimized with Adam

"""
import util.helper
import util.results
import util.simulator
from util.simulator import MovieConversationSimulator
import json
import sys
import util.latent_factor
from util.latent_factor import AvgLatentFactorAgent
import datetime as dt
import argparse
import torch
import select
import os
import pickle
from util.data import Movie
import numpy as np

parser = util.helper.get_parser()
parser.add_argument('-rescale', '--weight_rescale', help='Rescale embeddings weights?',
                    type=str, default='none', choices=['none', 'l2', 'l2_item'])
parser.add_argument('-debug', '--debug', help='Perform Debugging dump',
                    type='bool', default=False)
parser.add_argument('-skip', '--skip', help='Automatically skip if already run',
                    type='bool', default=True)
parser.add_argument('-cv', '--cross_validation', help='Grid search',
                    type='bool', default=False)
# parser.add_argument('-gs', '--grid_search', help='Grid search',
#                     type='bool', default=False)
parser.add_argument('-ul', '--update_length', help='Min length of msg to perform an update',
                    type=int, default=0)
parser.add_argument('-init', '--init', help='Rescale embeddings weights?',
                    type=str, default='var', choices=['random', 'mean', 'var'])
parser.add_argument('-save', '--save', help='Save results in mongodb',
                    type='bool', default=True)
parser.add_argument('-sent', '--sentiment', help='Sentiment integration style',
                    type=str, default='xent', choices=['xent', 'none'])
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# hp_parser = parser.add_argument_group("Hyperparameters", description="HI")
# hp_parser.add_argument('-hi')
# parser.print_help()
# opt = parser.parse_args("-mf gmf -d da".split())


def run(opt, keeper):
    print()
    print(json.dumps(opt.__dict__, indent=2, sort_keys=True))
    print()

    all_results = []

    for fold_index in range(5):
        p = os.path.join(opt.data, "{}.pkl".format(fold_index))
        data = pickle.load(open(p, 'rb'))
        episodes = data['data']
        movies_list = [Movie.from_dict(m) for m in data['movies']
                       if m['matrix_id'] != -1]

        simulator = MovieConversationSimulator(episodes, movies_list)
        if opt.model.startswith("avg"):
            agent = AvgLatentFactorAgent(opt, simulator.movies_list)
        elif opt.model == 'nmf':
            agent = util.latent_factor.NeuMFAgent(opt, simulator.movies_list)
        else:
            agent = util.latent_factor.LatentFactorConvMovieAgent(opt, simulator.movies_list)

        runner = util.simulator.ModelRunner(opt, agent, simulator)
        results = runner.run()

        # r10.append(results['r@10'])
        # r25.append(results['r@25'])
        # ndcg10.append(results['ndcg@10'])
        # ndcg25.append(results['ndcg@25'])
        # mrr10.append(results['mrr@10'])
        # mrr25.append(results['mrr@25'])

        # data/gorecdial/${ENCODER}/gorecdial_flr1e-6_l21e-5/test.pkl
        p = p.split('/')
        m = opt.model_file.split('/')
        out_f = './eval_results/0911/{}_{}_{}_{}_{}.txt'.format(m[2], m[3][:2], p[1], p[2], fold_index)

        with open(out_f, 'w') as f:
            json.dump(results, f)

        print()
        if opt.debug:
            dump_filename = opt.model_file + ".debug"
            print(f"[Saving Debug Dump to {dump_filename}]")
            pickle.dump(runner.debug_info, open(dump_filename, 'wb'))
            print("DONE!")

        all_results.append(results.copy())
        if opt.cross_validation:
            print("Performing CV, exiting first fold")
            break
        # if opt.train:
        #     finetune_filename = opt.model_file.replace(".pth", "") + "_finetune.pth"
        #     print(f"[Saving Fine Tunned to {finetune_filename}]")
        #     with open(finetune_filename, 'wb') as f:
        #         torch.save({'model': agent.model.state_dict(), 'params': agent.model_params}, f)

    summary = {
        'results': all_results,
        'params': agent.model_params.__dict__
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
    if keeper:
        keeper.report(summary)
        print("Saved to file...")


def grid_search_v2(opt, keeper):
    import glob
    model_files = glob.glob(f"results/{opt.model.replace('avg', '')}/**/model_*.pth", recursive=True)
    model_files = [fname for fname in model_files
                   if fname.find("model_5.pth") == -1
                    or fname.find("model_1.pth") == -1
                   or (opt.model.replace('avg', '') == 'nmf' and "tie" not in fname)]

    print(f"Found {len(model_files):,} Runs....")
    for i, mf in enumerate(model_files):
        print(f"[Progress: {i+1} / {len(model_files)}  - {(i+1)/len(model_files)}% ]")
        opt.model_file = mf
        if not keeper.should_run(opt.__dict__):
            print(f"Already Complted: {opt.model} with {opt.model_file}")
            continue
        run(opt, keeper)

# def grid_search(opt, keeper):
#     tie = "tie"
#
#     for L2 in ["0.1", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]:
#         # for ul in [3, 4, 5]:
#             # opt.update_length = ul
#         for INDEX in range(1, 5):
#             for EMBED_SIZE in range(10, 60, 10):
#                 if opt.model == 'nmf':
#                     for N_LAYERS in [2, 3]:
#                         # Already completed
#                         opt.model_file = f"results/{opt.model.replace('avg', '')}/{EMBED_SIZE}_{N_LAYERS}l/{tie}/l2{L2}/model_{INDEX}.pth"
#                         if not os.p   ath.exists(opt.model_file):
#                             print(f"Missing: {opt.model_file}")
#                             continue
#                         if not keeper.should_run(opt.__dict__):
#                             print(f"Already Complted: {opt.model_file}")
#                             continue
#                         run(opt, keeper)
#                 else:
#                     opt.model_file = f"results/{opt.model}/{EMBED_SIZE}/l2{L2}/model_{INDEX}.pth"
#                     if not os.path.exists(opt.model_file):
#                         print(f"Missing: {opt.model_file}")
#                         continue
#                     if not keeper.should_run(opt.__dict__):
#                         print(f"Already Complted: {opt.model_file}")
#                         continue
#                     run(opt, keeper)


if __name__ == '__main__':
    opt = parser.parse_args()
    # ignore_keys = {'gpu', 'overwrite', 'save', 'verbose', 'skip', 'debug', 'grid_search',}
    ignore_keys = {'gpu', 'overwrite', 'save', 'skip', 'debug', 'grid_search',}

    unique_keys = [key for key, _ in opt._get_kwargs() if key not in ignore_keys]

    run_exp = True
    keeper = None
    # if opt.save:
    #     keeper = util.results.MongoDBTracker(unique_keys,
    #                                          collection_name='redial',
    #                                          ignore_keys=ignore_keys)
    #     keeper.update_with_defaults({'cross_validation': False})
    #     if not keeper.should_run(opt.__dict__) and opt.skip:
    #         print("Skipping flag is set and exiting....")
    #         sys.exit()

    run(opt, keeper)

    # Should call cleanup once in a while
    # # keeper.cleanup()
    # if opt.grid_search:
    #     grid_search_v2(opt, keeper)
    # else:
        # if not keeper.should_run(opt.__dict__):
        #     print("\n[Run already completed....]\n")
        #
        #     if opt.skip:
        #         print("Skipping flag is set and exiting....")
        #         sys.exit()
        #
        #     # Ask if we should run anyway?
        #     if not opt.overwrite:
        #         print("", end="\rRerun? (y/n) (10 seconds timeout): ")
        #         i, o, e = select.select([sys.stdin], [], [], 10)
        #         if sys.stdin.readline().strip() != 'y':
        #             print("\nExit...")
        #             sys.exit()
        #

    # s = None
    # # We are debugging, use these arguments not from command line
    # if getattr(sys, 'gettrace', None)():
    #     print("\nDebugging\n")
    #     s = "-mf results/mf/10/model_1.pth -m mf -d data/transformer/1k.pkl -save f -train t".split()
    #
    # opt = parser.parse_args(s)
    # if opt.verbose:
    #     opt.save = False
    #
    # if opt.debug:
    #     print("[ DEBUG MODE........ ]")
    #     opt.save = False
    #
    # if opt.save:
    #     ignore_keys = {'gpu', 'overwrite', 'save', 'verbose', 'skip', 'debug'}
    #     unique_keys = [key for key, _ in opt._get_kwargs() if key not in ignore_keys]
    #     keeper = util.results.ExperimentTracker("results/conv.csv",
    #                                             unique_keys)
    #     if not keeper.should_run(opt.__dict__):
    #         print("Already completed run.....")
    #         if opt.skip:
    #             print("Skipping flag is set and exiting....")
    #             sys.exit()
    #
    #         if not opt.overwrite:
    #             print("", end="\rRerun? (y/n) (10 seconds timeout): ")
    #             i, o, e = select.select([sys.stdin], [], [], 10)
    #             if sys.stdin.readline().strip() != 'y':
    #                 print("\nExit...")
    #                 sys.exit()
    # else:
    #     print("\nNote not saving....\n")
    #
    # run(opt)

