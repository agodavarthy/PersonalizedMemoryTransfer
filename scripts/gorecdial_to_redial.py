import json
import pandas as pd
import regex

# test = [{'a': 'b'}, {'a': 'b'}, {'a': 'b'}]
#
# with open("data.json", 'w') as f:
#     for item in test:
#         f.write(json.dumps(item) + "\n")

def construct_message(text, id, seeker=True):
    if seeker:
        if text == 'ACCEPT':
            liked = 1
        elif text == 'REJECT':
            liked = 0
        else:
            liked = -1
        return {'text': text,
            'senderWorkerId': id}, liked
    else:
        movie = ''
        match = regex.search(r"(MID\d+)", text)
        if match:
            movie = text[match.span()[0]:match.span()[1]][3:]
            movie = int(movie)
            text = regex.sub(r"(MID)", "@", text)

        return {'text': text,
            'senderWorkerId': id}, movie


def gorecdial_to_redial(expert_text, conversation_start_id, global_id, mode):
    episodes = []
    all_movies = []
    for line in expert_text:

        text = line.split('\t')
        if text[6] == 'episode_start:True':
            # start of the episode

            # generate seeker id, expert id, conversation id
            conversation_start_id += 1
            seeker_start_id = global_id
            global_id += 1
            expert_start_id = global_id
            global_id += 1
            episode = {'movieMentions': {},
                       'messages': [],
                       'conversationId': conversation_start_id,
                       'respondentWorkerId': expert_start_id,
                       'initiatorWorkerId': seeker_start_id, }

            movies = []
            reactions = []

            seeker_text = text[1].split('\\n')[-1]
            expert_text = text[2][7:]


        else:
            seeker_text = text[1][5:]
            expert_text = text[2][7:]

        m, liked = construct_message(seeker_text, seeker_start_id, True)
        if liked != -1:
            reactions.append(liked)
        episode['messages'].append(m)

        m, movie_id = construct_message(expert_text, expert_start_id, False)
        if movie_id:
            episode['movieMentions'][movie_id] = movie_id2title[movie_id]
            movies.append(movie_id)
            all_movies.append(movie_id)
        episode['messages'].append(m)

        if text[7] == 'episode_done:True':
            if len(movies) == len(reactions):

                questions = {}
                for j in range(len(movies)):
                    questions[movies[j]] = {'liked': reactions[j]}
                episode['initiatorQuestions'] = questions
                episodes.append(episode)

    with open("/home/ywang/convmovie/data/{}_data.jsonl".format(mode), 'w') as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")

    return all_movies

if __name__ == '__main__':

    # path to the folder with train and test gorecdial file
    gorecdial_folder_path = '/home/ywang/convmovie/data/GoRecDial_raw/'

    train_expert_text_path = gorecdial_folder_path + 'dialrec-Expert-all-train.txt'
    test_expert_text_path = gorecdial_folder_path + 'dialrec-Expert-all-test.txt'

    # movie lens dataset
    movielens_df = pd.read_csv('/home/ywang/convmovie/data/ml-latest/movies.csv')

    movie_id2title = dict(zip(movielens_df.movieId, movielens_df.title))

    with open(train_expert_text_path, 'r') as f:
        train_expert_text = f.readlines()

    with open(test_expert_text_path, 'r') as f:
        test_expert_text = f.readlines()

    movie_ids = []
    movie_ids += gorecdial_to_redial(train_expert_text, 0, 0, 'train')

    movie_ids += gorecdial_to_redial(test_expert_text, 1000000, 1000000, 'test')

    df = pd.DataFrame(list(set(movie_ids)), columns=['id'])
    df.to_csv('/home/ywang/convmovie/data/movies_gorecdial.csv', index=False)
