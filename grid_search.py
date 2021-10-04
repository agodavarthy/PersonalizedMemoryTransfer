import subprocess
import os
import copy
from itertools import product
import numpy as np


def run_model(args, run_cmd=True, filename='main.py'):
    cmd_args = ['python', filename]
    items = list(args.items())
    items.sort(key=lambda x: x[0])
    for k, v in items:
        cmd_args.append("--{}".format(k).strip())
        v = str(v).strip()
        if len(v):
            cmd_args.append(v)
    print("=" * 80)
    print("Running Command:")
    print(' '.join(cmd_args))
    print("=" * 80)
    print()
    if run_cmd:
        subprocess.check_call(cmd_args, stderr=subprocess.STDOUT)



if __name__ == '__main__':
    model_file = 'results/gmf/40/l21e-4/model_3.pth'

    data = 'data/redial/transformer/imdb_flr1e-6_l21e-6/splits.pkl'

    default_args = {
        'model_file': model_file,
        #'model': 'gmf',
        'save': 't',
        'data': data,
        'lr': 0.1,
        'weighted_count': 300,
        'momentum': 0.0,
        'neg_count': 10,
        'cross_validation': True,
        'sentiment': 'xent'
    }

    new_args = [
        [{'model_file': x} for x in [
            'results/mf/40/l2.1/model_4.pth',
            # 'results/gmf/40/l21e-4/model_3.pth'
        ]],
        #[{'lr': x} for x in [0.1, 0.25, 0.5]],
        # [{'momentum': x} for x in [0.5, 0.0]],
        [{'data': x} for x in [
            # 'data/redial/transformer/',
            'data/redial/combined_transformer/',
            # 'data/redial/dan/',
        ]],
        [{'neg_count': x} for x in
            [#10, 15, 20, 25, 30, 35, 40, 45, 50
                50, 70, 90
             ]],
        [{'weighted_count': x} for x in [
            #50, 60,
            #  80, 90,
            70,
            100,# 110, 120, 130, 140, 150, 160, 170, 180
        ]],
    ]

    all_trials = list(product(*new_args))
    # Shuffle so we can run it parallel
    np.random.shuffle(all_trials)

    print(f"Total Trials: {len(all_trials)}")
    for tid, new in enumerate(all_trials, 1):
        args = copy.deepcopy(default_args)
        print()
        print("-" * 80)
        print(f"Running {tid} / {len(all_trials)}")
        # If not dictionary then update values
        if not isinstance(new, dict):
            [args.update(n) for n in new]
        else:
            args.update(new)

        if '/gmf/' in args['model_file']:
            args['model'] = 'gmf'
        else:
            args['model'] = 'mf'
        if args['neg_count'] >= args['weighted_count']:
            continue
        run_model(args)
