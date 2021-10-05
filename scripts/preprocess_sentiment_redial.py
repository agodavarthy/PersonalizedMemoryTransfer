import os
import json
import pandas as pd
from collections import deque
import regex
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import pickle

CONV_SPLITS_FILENAME = '../data/redial/conv_split_ids.pkl'

def preprocess_split(data):
    """
    5 fold on conversation ids

    :param data:
    :return:
    """
    kf = KFold(5, shuffle=True, random_state=12345)

    conv_ids = data.conversationId.values

    folds = []
    print("KFold Splits")
    for fold, (train_index, test_index) in enumerate(kf.split(conv_ids)):
        folds.append({'test': [conv_ids[idx] for idx in test_index],
                      'train': [conv_ids[idx] for idx in train_index]
                      })
    with open(CONV_SPLITS_FILENAME, 'wb') as f:
        pickle.dump(folds, f)


def write_csv(output_filename, sentiment_data):
    with open(output_filename, 'w') as fout:
        fout.write('label,sentence\n')
        for sentiment, doc in sentiment_data:
            fout.write("{},{}\n".format(sentiment, doc))


def replace_movie(s: str, row):
    """
    Replaces the movie id with the corresponding movie name

    :param s: utterance with movie ids
    :param row: pd.Series from the redial dataset, containing moviementions
    :return: string with replaced movie names, list of movie ids
    """
    # TODO: Fix how we do this...
    mentions = row['movieMentions']
    questions = row['initiatorQuestions']
    match = regex.search(r"(@\d+)", s)
    ids = []

    while match:
        start, end = match.span()
        movie_id = match.group(0)[1:]
        # ConvMovie: ID
        ids.append(movie_id)
        if movie_id not in mentions:
            print("Could not find movie in mentions...", movie_id)
            movie_name = "movie"
        else:
            movie_name = mentions[movie_id].strip()
        # Seeker mentioned
        s = f"{s[:start]} {movie_name} {s[end:]}"
        match = regex.search(r"(@\d+)", s)
    # Remove double whitespace
    s = regex.sub(r"\s{2,}", " ", s).strip()
    return s, ids


def sentiment_splits(data):
    """
    Splits sentiment data according to the conv folds

    :param data:
    :return:
    """
    with open(CONV_SPLITS_FILENAME, 'rb') as f:
        splits = pickle.load(f)

    sentiment_data = {}
    print("Preprocessing data")
    for _, row in data.iterrows():
        # No movies mentioned...
        if len(row.movieMentions) == 0:
            continue

        convId = row.conversationId

        # Questions = {'203371': {'suggested': 1, 'seen': 0, 'liked': 1}}
        questions = row['initiatorQuestions']

        dialogue = deque(maxlen=2)
        ex = set()
        for i, msg in enumerate(row.messages):
            # Replace movie ids with names, return ids
            text, ids = replace_movie(msg['text'], row)
            dialogue.append(text.strip())
            for _id in ids:
                if _id in questions and questions[_id]['liked'] < 2:
                    ex.add((questions[_id]['liked'], '. '.join(dialogue)))
        sentiment_data[convId] = list(ex)

    for n, split in enumerate(splits):
        test = []
        for idx in split['test']:
            if idx in sentiment_data:
                test.extend(sentiment_data[idx])

        train = []
        for idx in split['train']:
            if idx in sentiment_data:
                train.extend(sentiment_data[idx])

        print(f"Split {n}")
        base_dir = '../data/sentiment/redial/{}/'.format(n)
        os.makedirs(base_dir, exist_ok=True)
        print("Writing out test size: {:,}".format(len(test)))
        write_csv('{}/test_binary_sent.csv'.format(base_dir), test)

        print("Writing out training size: {:,}".format(len(train)))
        write_csv('{}/train_binary_sent.csv'.format(base_dir), train)
        print()

    # # train_test_split
    # y, x = zip(*sentiment_data)
    #
    # test_size = 0.2
    # dev_size = 0.125
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    # x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=dev_size)
    #
    # print("Writing out dev set: {:,}".format(len(x_dev)))
    # write_csv('data/sentiment/redial/dev_binary_sent.csv', zip(x_dev, y_dev))



if __name__ == '__main__':
    data = []
    redial_path = "../data/redial_dataset/"

    print("Reading in data.....")
    # load testing data
    with open(os.path.join(redial_path, "test_data.jsonl"), 'r') as f:
        data.extend([json.loads(line) for line in f])

    # Load training data
    with open(os.path.join(redial_path, "train_data.jsonl"), 'r') as f:
        data.extend([json.loads(line) for line in f])
    data = pd.DataFrame.from_dict(data)

    preprocess_split(data)
    sentiment_splits(data)

