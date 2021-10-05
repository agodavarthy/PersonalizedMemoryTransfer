import os, glob
from tqdm import tqdm
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from bs4 import BeautifulSoup


def write_csv(output_filename, filenames, truncate=None):
    """
    Write out in csv format
    label,sentence

    :param output_filename: Filename to write out
    :param filenames: List of filenames to use as input
    :param truncate: optional, length to truncate
    :return:
    """
    num_tokens = defaultdict(list)
    with open(output_filename, 'w') as fout:
        fout.write('label,sentence\n')
        for filename in tqdm(filenames):
            sentiment = int('pos' in filename)
            # sentiment = int(os.path.basename(filename)[:-4].split("_")[1])

            with open(filename) as f:
                doc = f.read()
            # Remove HTML
            soup = BeautifulSoup(doc)
            doc = soup.get_text()

            if truncate:
                doc = " ".join(doc.split()[:truncate])

            num_tokens[sentiment].append(len(doc.split()))
            fout.write("{},{}\n".format(sentiment, doc))

    for k, v in num_tokens.items():
        print("Sentiment {}: Count: {:<10,} Tokens Mean: {:<10,.2f} Min: {:<5} Max: {}".format(
                k, len(v), np.mean(v), np.min(v), np.max(v)))


if __name__ == '__main__':
    # test_negatives = glob.glob("data/sentiment/imdb/test/neg/*")
    # test_positives = glob.glob("data/sentiment/imdb/test/pos/*")
    # train_negatives = glob.glob("data/sentiment/imdb/train/neg/*")
    # train_positives = glob.glob("data/sentiment/imdb/train/pos/*")
    neg = glob.glob("data/sentiment/imdb/test/neg/*") + glob.glob("data/sentiment/imdb/train/neg/*")
    pos = glob.glob("data/sentiment/imdb/test/pos/*") + glob.glob("data/sentiment/imdb/train/pos/*")
    total_size = len(neg)+len(pos)
    MAX_LENGTH = 100

    print("Total Size: {:,}".format(total_size))
    # Split: 70/10/20 train/valid/test; stratified
    test_size = 0.2
    # this is equal to 10% of the initial size
    dev_size = 0.125

    train_negatives, test_negatives = train_test_split(neg, test_size=test_size)
    train_positives, test_positives = train_test_split(pos, test_size=test_size)
    train_negatives, valid_negatives = train_test_split(train_negatives, test_size=dev_size)
    train_positives, valid_positives = train_test_split(train_positives, test_size=dev_size)

    print("Writing out test size: {:,}".format(len(test_negatives) + len(test_positives)))
    write_csv('data/sentiment/imdb/test_binary_sent.csv', test_negatives + test_positives, MAX_LENGTH)

    print("Writing out training size: {:,}".format(len(train_positives) + len(train_negatives)))
    write_csv('data/sentiment/imdb/train_binary_sent.csv', train_positives + train_negatives, MAX_LENGTH)

    print("Writing out dev set: {:,}".format(len(valid_negatives) + len(valid_positives)))
    write_csv('data/sentiment/imdb/dev_binary_sent.csv', valid_positives + valid_negatives, MAX_LENGTH)

    print("DONE!")
