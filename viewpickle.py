import json
import pickle
ifilename = "./data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl/0.pkl"
ifilename = "data/redial/redial_test.pkl/4.pkl"
ifilename = "data/redial/transformer/full/test.pkl"
ifilename = "data/redial/sep29/transformer/redial_flr1e-6_l21e-5_transformer/test.pkl/0.pkl"
ifilename = "data/redial/0911/bert/redial_flr1e-6_l21e-5_bert_10/test.pkl/4.pkl"
ifilename = "./data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl/0.pkl"
#ifilename = "/data/shared/test.pkl/3.pkl"
#ifilename = "./data/gorecdial/transformer/gorecdial_flr1e-6_l21e-5/test.pkl/0.pkl"
#ifd = open(ifilename, "rb")
#data = pickle.load(open(os.path.join(opt.data, "{}.pkl".format(fold_index)), 'rb'))
data = pickle.load(open(ifilename, 'rb'))
episode = data['data']

print(len(episode))
conv_msg_gt_lst = [] 
for item in episode:
        #print(item["convId"])
        cnt = len(item)
        for i in item:
                #if str(i["convId"]) == '20001':
                        print(str(i["convId"]), "\t", cnt, "\t", str(i["movie_id"]), "\t",  str(i['text'])) 
                #conv_msg_gt_lst.append({"convId": str(i["convId"]), 'cnt': cnt, "gt": i["movie_id"], 'text': i['text']}) 
        #print("---------------------------")
ofd = open("redial_msg_gt.pkl", "wb")
conv_msg_gt_dict = {}
for item in conv_msg_gt_lst:
        if item["convId"] in conv_msg_gt_dict:
                conv_msg_gt_dict[item["convId"]].append((item["text"], item["gt"]))
        else:
                conv_msg_gt_dict[item["convId"]] = [(item["text"], item["gt"])]
pickle.dump(conv_msg_gt_dict, ofd)
ofd.close()
'''
turns = [0.0]*5
for i in range(5):
    ifilename = "eval_results/content_gorecdial_"+str(i)+".txt"
    ifd = open(ifilename, "r")
    data = json.loads(ifd.read())
    turn = data["turn"]
    print(len(turn))
    for j in range(5):
        turns[j] += turn[j]["r@25"]

for t in turns:
        print(t/5)
'''    
