#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   Travis A. Ebesu
@created:  2017-05-08
@summary:  Helper functions for GraphKeys, Optimizer Arguments,
           creating experiment directories and configurations.
"""
import argparse
import json
import logging
import sys
import shutil
import os
import pickle
from itertools import chain
from logging.config import dictConfig
import time
import datetime as dt
import math

# import tensorflow as tf

import torch
import torch.nn.init


def str2bool(value):
    """Helper Function for argument parser"""
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parser_add_str2bool(parser: argparse.ArgumentParser):
    """
    Adds Str2Bool type for argparser

    :param parser:
    """
    parser.register('type', 'bool', str2bool)


def get_parser():
    """
    Get argparser for main script

    :return:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_add_str2bool(parser)
    parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=int,
                        default=0)
    parser.add_argument('-d', '--data', help='MoviesData Input', type=str,
                        required=True)
    parser.add_argument('-mf', '--model_file', help='path to model.pth', type=str,
                        required=True)
    parser.add_argument('-wc', '--weighted_count', help='Number of top item latent factors to consider', type=int,
                        default=300)
    parser.add_argument('-n', '--neg_count', help='Number of negatives to sample', type=int,
                        default=10)
    parser.add_argument('-lr', '--lr', help='Learning Rate', type=float,
                        default=0.1)
    parser.add_argument('-mom', '--momentum', help='Momentum for optimizer',
                        type=float, default=0.0)
    # L2 is not setup to work I think
    parser.add_argument('-l2', '--l2', help='l2 Regularization', type=float, default=0.0)
    parser.add_argument('-m', '--model', help='Model type', required=True,
                        type=str, choices=['gmf', 'mf', 'nmf', 'mlp',
                                           'avggmf', 'avgmf', 'avgnmf'])
    # parser.add_argument('-s', '--sampling', help='path to model directory',
    #                     type=str, default='random', choices=['hard', 'random'])
    # parser.add_argument('-save', '--save', help='Save results to the conv file?'
    #                                             ' Does not apply to multi_exp.py',
    #                     type='bool', default=True)
    parser.add_argument('-overwrite', '--overwrite', help='overwrite the saved entry',
                        type='bool', default=False)
    parser.add_argument('-seed', '--seed', help='Random seed',
                        type=int, default=42)
    parser.add_argument('-norm', '--norm', help='path to model directory',
                        type=str, default='softmax', choices=['softmax', 'l2'])
    parser.add_argument('-update', '--update', help='When to update the model? after seeker speaks or after every turn',
                        type=str, default='seeker', choices=['seeker', 'all', 'rec'])
    parser.add_argument('-v', '--verbose', help='Print out each conversation results',
                        type='bool', default=False)
    parser.add_argument('-replay', '--replay', help='Use replay buffer?',
                        type='bool', default=False)
    parser.add_argument('-train', '--train', help='Perform fine tunning on this dataset',
                        type='bool', default=False)
    return parser


def caffe_init(tensor: torch.Tensor):
    """
    Initialization is done in place, inits to Uniform(sqrt(1/fan_in))
    Convolutional Architecture for Fast Feature Embedding
        factor=1.0 mode='FAN_IN' uniform=True

    :param tensor:
    """
    if tensor.dim() == 1:
        init = math.sqrt(1.0 / tensor.shape[0])
    else:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        init = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        tensor.uniform_(-init, init)


def setup_exp(opt):
    """
    Creates directories, copies main file to it and saves the configuration
    :param opt: Output from parser.parse_args()
    """
    if not hasattr(opt, 'model_file'):
        print("[Options Passed has no attribute model_file]")
        return

    if opt.model_file is None:
        return

    # creating output folders
    os.makedirs(opt.model_file, exist_ok=True)

    # Copy file to result directory for reproducability
    full_path = os.path.realpath(sys.argv[0])
    # _fname = os.path.basename(sys.argv[0])
    # directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                          _fname)
    backup_path = os.path.join(opt.model_file, os.path.basename(full_path))

    if not os.path.exists(backup_path):
        print(f"Copying {full_path} ==> {backup_path}")
        shutil.copy2(full_path, backup_path)
        json.dump(opt.__dict__, open(os.path.join(opt.model_file, "config.json"), 'w'))


def copy_files(model_directory, copy_dirs=None):
    """
    Copies the current file and all copy_dirs to the model_directory preserving
    the directory structure of the copy_dirs.

    :param model_directory: Path to where to copy the files, it will create it
                            recursively if directory does not exist.
    :param copy_dirs: list of directories to copy
    :return:
    """
    main_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             os.path.basename(sys.argv[0]))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    shutil.copy2(main_file, os.path.join(model_directory, main_file))

    if copy_dirs is not None:
        for d in copy_dirs:
            shutil.copytree(d, model_directory + "/" + d,
                            ignore=shutil.ignore_patterns('*.pyc', '.*', '#*',
                                                          '*.mtx', '*.pkl'))



def create_exp_directory(cwd: str=''):
    """
    Creates a new directory to store experiment to save data. It will try to
    create a new folder name sequentially from 1 to 10000, if all of them exist
    it throws an error.

    :param cwd: Current working directory to create a new experiment in
    :return: The newly created experiment directory
    """
    created = False
    for i in range(1, 10000):
        exp_dir = str(i).zfill(3)
        path = os.path.join(cwd, exp_dir)
        if not os.path.exists(path):
            # Create directory
            os.mkdir(path)
            created = True
            break
    if not created:
        raise Exception('Could not create directory for experiments')
    return path + '/'


def set_logging_config(save_directory: str):
    """
    Get a logging dictionary configuration which will also add a filehandler to
    the save_directory

    :param save_directory:
    :return: dict
    """
    # Setup Logging
    dictConfig(dict(
        version=1,
        formatters={
            # For files
            'detailed': {
                'format': "[%(asctime)s - %(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            },
            # For the console
            'console': {
                'format':"[%(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            }
        },
        handlers={
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging.INFO,
                'formatter': 'console',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.DEBUG,
                'formatter': 'detailed',
                'filename': "{}/log".format(save_directory),
                'mode': 'a',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5
            }
        },
        loggers={
            'tensorflow': {
                'level': logging.INFO,
                'handlers': ['console', 'file'],
            }
        },
        disable_existing_loggers=False,
    ))


class Timer(object):
    """
    Computes elapsed time.
    https://github.com/facebookresearch/ParlAI/blob/52493d792d9f361c2711ca1340a5ed3b4d372456/parlai/core/utils.py#L172
    """
    def __init__(self):
        self.running = True
        self._total = 0
        self._start = time.time()

    def reset(self):
        """Reset the time"""
        self.running = True
        self._total = 0
        self._start = time.time()
        return self

    def resume(self):
        """Resume if we stopped"""
        if not self.running:
            self.running = True
            self._start = time.time()
        return self

    def stop(self):
        """Stop timer"""
        if self.running:
            self.running = False
            self._total += time.time() - self._start
        return self

    def time(self):
        """Return the number of seconds since started"""
        if self.running:
            return self._total + time.time() - self._start
        return self._total

    def eta(self, current, total):
        """
        Return a ETA given the progress
        :param current: Current number of count
        :param total: Number of total size
        :return: string
        """
        if current > 0:
            eta = self.time() / current * (total-current)
            return str(dt.timedelta(seconds=eta)).split(".")[0]
        else:
            return "???"

    def __str__(self):
        return str(dt.timedelta(seconds=self.time())).split(".")[0]

    def __repr__(self):
        return str(dt.timedelta(seconds=self.time())).split(".")[0]


class ProgressBar(object):
    def __init__(self, fill='=', empty=' ', tip='>', begin='[', end=']',
                 precision=1, clear=True):
        """

        Adapted from Eric Wu

        fill        - Optional : bar fill character                         [str] (ex: 'â– ', 'â–ˆ', '#', '=')
        empty       - Optional : not filled bar character                   [str] (ex: '-', ' ', 'â€¢')
        tip         - Optional : character at the end of the fill bar       [str] (ex: '>', '')
        begin       - Optional : starting bar character                     [str] (ex: '|', 'â–•', '[')
        end         - Optional : ending bar character                       [str] (ex: '|', 'â–', ']')
        clear       - Optional : display completion message or leave as is  [str]

        :param fill:
        :param empty:
        :param tip:
        :param begin:
        :param end:
        :param precision: precision for showing percentage
        :param clear:
        """
        self.fill = fill
        self.empty = empty
        self.tip = tip
        self.begin = begin
        self.end = end
        self.clear = clear
        self.precision = precision
        self.timer = Timer()
        self.lifetime = Timer()

    def print(self, step: int, total_steps: int, prefix: str='', suffix: str='',
              done: str="[DONE]", length: int=100):
        """
        Print iterations progress.
        Call in a loop to create terminal progress bar

        [Prefix String][Begin Marker][Progress][End Marker]
        [Percent Done] % [Lifetime Total Time] (eta [Estimated Time for Epoch])
        [Suffix String]


        :param step: Current step for progress
        :param total_steps: Total number of steps
        :param prefix: Prefix string to show
        :param suffix: Suffix string to show
        :param done: (Optional) display message when 100% is reached
        :param length: (Optional) character length of bar
        """
        # percent = ("{0:." + str(self.decimals) + "f}").format(100 * (step / float(total_steps)))
        percent = "{0:.{precision}f}".format(100 * (step / float(total_steps)),
                                             precision=self.precision)
        percent = "{}% {} (eta {})".format(percent, self.timer, self.timer.eta(step, total_steps))

        filled_length = int(length * step // total_steps)
        bar = self.fill * filled_length
        if step != total_steps:
            bar = bar + self.tip
        bar = bar + self.empty * (length - filled_length - len(self.tip))
        display = '\r{prefix}{begin}{bar}{end} {percent}{suffix} ' \
            .format(prefix=prefix, begin=self.begin, bar=bar, end=self.end,
                    percent=percent, suffix=suffix)
        print(display, end=''),  # comma after print() required for python 2
        if step == total_steps:  # print with newline on complete
            # display given complete message with spaces to 'erase' previous progress bar
            finish = '\r{prefix}{done}'.format(prefix=prefix, done=done)
            # Pad the right with spaces
            print(finish.ljust(max(len(display) - len(finish), 0)))

            if self.clear:
                # print("".ljust(256))
                print('')

            # Reset epoch timer
            self.timer.reset()


if __name__ == '__main__':
    CEND = '\33[0m'
    # black letters on white background use ESC[30;47m
    CSI = "\x1B["
    #Should we use: \33 or \x1B ?
    # 31 = Font FontColor
    # 40 = ;40
    print(CSI + "31m" + "Colored Text" + CSI + "0m")
    print(CSI + "92m" + "Colored Text" + CSI + "0m")
    print(CSI + "32m" + "Colored Text" + CSI + "0m")

    pbar = ProgressBar()
    print()
    # pbar.print(5, 10)
    for i in range(10):
        time.sleep(0.25)
        pbar.print(i, 9)
    print()

