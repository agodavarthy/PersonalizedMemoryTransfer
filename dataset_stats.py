import json

#ifilename = "/home/agodavarthy/Backup2020/Projects/ConversationalRecommendation/convrec_aug/convmovie_yuan_0917/data/gorecdial_dataset/test_data.jsonl"
ifilename = "/home/agodavarthy/Backup2020/Projects/ConversationalRecommendation/convrec_aug/convmovie_yuan_0917/data/redial_dataset/test_data.jsonl"

ifd = open(ifilename, "r")

num_turns = 0
cnt = 0

for line in ifd:
        jsonObj = json.loads(line)
        num_turns += len(jsonObj["messages"])
        cnt += 1
print("num_turns = ", num_turns)
print("cnt = ", cnt)
print("#turns = ", num_turns/cnt)
