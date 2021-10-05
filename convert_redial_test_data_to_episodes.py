import time
import os
import json
import pickle
import re

#{'seeker': False, 'text': 'If you like horror you can check out Evil Dead', 'movie_id': ['136636'], 'ml_id': [101739], 'convId': '21648', 'query_vec': array([-0.00013433, -0.02305225,  0.02107571, -0.06726646,  0.02599562,
#        0.07697688,  0.01728551], dtype=float32), 'sentiment': array([0.9566811], dtype

ofilename = "data/redial/tfifd_test_episode.pkl"
ofd = open(ofilename, "wb")
movie_name_id_dict = {}
movie_id_name_dict = {}
ifilename = "data/redial_dataset/movies_with_mentions.csv"
ifd = open(ifilename, "r")
for line in ifd:
        tokens = line.split(",")
        if len(tokens) == 3:
                mid = tokens[0]
                mname = tokens[1]
        else:
                mid = tokens[0]
                mname = tokens[0]+", "+tokens[1]
        movie_name_id_dict[mname] = mid
        mname = re.sub('\([^()]*\)', '', mname) 
        movie_id_name_dict[mid] = mname.strip(" ")

new_episodes = []
for i in range(5):
    opt= "data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl/"
    p = os.path.join(opt, "{}.pkl".format(i))
    data = pickle.load(open(p, 'rb'))
    episodes = data['data']
    for eps in episodes:
        new_eps_turn = []
        for eps_turn in eps:
            convId = eps_turn["convId"]
            text = eps_turn["text"].lower()
            movie_ids = eps_turn["movie_id"]
            #print("convId = ", convId)
            #print("movie_ids = ", movie_ids)
        
            for m in movie_ids:
                if m in movie_id_name_dict:
                        mname = movie_id_name_dict[m].lower()
                        if text == mname: 
                                text = '@'+m
                        else:
                               text = text.replace(mname, '@'+m)
                        eps_turn["text"] = text
                        new_eps_turn.append(eps_turn)
        new_episodes.append(new_eps_turn)
data['data'] = new_episodes
pickle.dump(data, ofd)
ofd.close()                        
                        
                        
