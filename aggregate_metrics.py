import json
import os
import csv

embed = '40'
p1 = []
p3 = []
r1 = []
r3 = []
r5 = []
r10 = []
r25 = []
ndcg1 = []
ndcg3 = []
ndcg5 = []
ndcg10 = []
ndcg25 = []
mrr1 = []
mrr3 = []
mrr5 = []
mrr10 = []
mrr25 = []

<<<<<<< HEAD
bert_num_map = {
            '1' : 'sentence-transformers/all-distilroberta-v1',
            '2' : 'sentence-transformers/distilroberta-base-msmarco-v1',
            '3' : 'sentence-transformers/distilroberta-base-msmarco-v2',
            '4' : 'sentence-transformers/distilroberta-base-paraphrase-v1',
            '5' : 'sentence-transformers/msmarco-distilroberta-base-v2',
            '6' : "sentence-transformers/all-MiniLM-L6-v2",  # R@10 ~ 2.5,
            '7' : 'sentence-transformers/bert-base-nli-mean-tokens', # R@10 ~ 1.5,
            '8' : 'sentence-transformers/all-MiniLM-L12-v2',  # R@10 ~ 2.8
            '9' : 'sentence-transformers/all-mpnet-base-v2',  # R@10 ~ 2.6
            '10': 'sentence-transformers/paraphrase-mpnet-base-v2',  #
            '11' : 'sentence-transformers/paraphrase-MiniLM-L12-v2',  # R@10 ~ 2.6

            '12': 'sentence-transformers/LaBSE',
            '13': 'sentence-transformers/bert-large-nli-mean-tokens',
            '14': 'sentence-transformers/bert-base-nli-max-tokens',
            '15': 'sentence-transformers/bert-base-nli-stsb-mean-tokens',
            '16': 'sentence-transformers/bert-large-nli-cls-token'
}


#ofilename = "metrics/redial.txt"
#ofilename = "redial.txt"
#ofd = open(ofilename, "w+")
#ipath = "/home/agodavarthy/Backup2020/Projects/ConversationalRecommendation/convrec_yuan/eval_results/"
#for encoder in ['pop', 'content', 'tfidf']:
#    print("Encoder = ", encoder)
    #encoder = 'transformer'
    #bert_num = 10
ipath = "/home/agodavarthy/Backup2020/Projects/ConversationalRecommendation/convrec_yuan/eval_results/"
ipath = "eval_results/0925/"
ipath = "eval_results/0911/"
ipath = "/home/agodavarthy/Backup2020/Projects/ConversationalRecommendation/convrec_aug/convrec_yuan/eval_results/"
ipath = "eval_results/"
ipath = "eval_results/oct2/"
for i in range(5):
            #ifilename = ipath+"mf_40_gorecdial_"+encoder+"_"+str(i)+".txt"
            #ifilename = ipath+"40_2l_ti_gorecdial_"+str(encoder)+"_"+str(i)+"_bertnone.txt"
            #ifilename = ipath+str(encoder)+"_redial_"+str(i)+".txt"
            ifilename = ipath+"tfidf_redial_"+str(i)+".txt"  
            #ifilename = ipath+"mf_40_redial_transformer_"+str(i)+"_bertnone.txt"
            ifilename = ipath+"tfidf_redial_"+str(i)+".txt"  
            ifilename = ipath+"dropout_gorecdial"+"_"+str(i)+".txt"
            ifilename = ipath+"content_redial_"+str(i)+".txt"  
            ifilename = ipath+"mf_40_redial_transformer_"+str(i)+"_bertnone.txt"
            ifilename = ipath+"mf_40_redial_0911_"+str(i)+"_bert10.txt"
            ifilename = ipath+"mf_40_redial_transformer_"+str(i)+".txt"  
            ifilename = ipath+"40_2l_ti_gorecdial_transformer_"+str(i)+"_bertnone.txt"
            ifilename = ipath+"40_2l_ti_gorecdial_0911_"+str(i)+"_bertnone.txt"
            ifilename = ipath+"mf_40_redial_transformer_"+str(i)+".txt"
            ifilename = ipath+"40_2l_ti_gorecdial_transformer_"+str(i)+".txt"
            #ifilename = ipath+"tfidf_gorecdial_"+str(i)+".txt"  
            #ifilename = ipath+"pop_gorecdial_"+str(i)+".txt"  
            #ifilename = ipath+"content_gorecdial_"+str(i)+".txt"  
            #ifilename = ipath+"40_2l_ti_gorecdial_transformer_"+str(i)+".txt"
            ifilename = ipath+"dropout_redial_"+str(i)+".txt"
            ifilename = ipath+"_gm_redial_transformer_"+str(i)+"_bertnone.txt"
            ifilename = ipath+"tfidf_redial_"+str(i)+".txt"  
=======
ipath = "eval_results/"
for i in range(5):
            ifilename = ipath+"mf_40_redial_transformer_"+str(i)+".txt"  
>>>>>>> be21c49 (adding code and data)
            ifd = open(ifilename, "r")
            for line in ifd:
                    jsonObj = json.loads(line)
                    p1.append(jsonObj["p@1"])
                    p3.append(jsonObj["p@3"])
                    r1.append(jsonObj["r@1"])
                    r3.append(jsonObj["r@3"])
                    r5.append(jsonObj["r@5"])
                    r10.append(jsonObj["r@10"])
                    r25.append(jsonObj["r@25"])

                    ndcg1.append(jsonObj["ndcg@1"])
                    ndcg3.append(jsonObj["ndcg@3"])
                    ndcg5.append(jsonObj["ndcg@5"])
                    ndcg10.append(jsonObj["ndcg@10"])
                    ndcg25.append(jsonObj["ndcg@25"])

                    mrr1.append(jsonObj["mrr@1"])
                    mrr3.append(jsonObj["mrr@3"])
                    mrr5.append(jsonObj["mrr@5"])
                    mrr10.append(jsonObj["mrr@10"])
                    mrr25.append(jsonObj["mrr@25"])
print("P@1\t"+str(sum(p1)/len(p1)))
print("P@3\t"+str(sum(p3)/len(p3)))
print("R@1\t"+str(sum(r1)/len(r1)))
print("R@3\t"+str(sum(r3)/len(r3)))
#print("R@5\t"+str(sum(r5)/len(r5)))
print("R@10\t"+str(sum(r10)/len(r10)))
print("R@25\t"+str(sum(r25)/len(r25)))
#print("MRR@1\t"+str(sum(mrr1)/len(mrr1)))
print("MRR@3\t"+str(sum(mrr3)/len(mrr3)))
#print("MRR@5\t"+str(sum(mrr5)/len(mrr5)))
print("MRR@10\t"+str(sum(mrr10)/len(mrr10)))
print("MRR@25\t"+str(sum(mrr25)/len(mrr25)))
#print("NDCG@1\t"+str(sum(ndcg1)/len(ndcg1)))
print("NDCG@3\t"+str(sum(ndcg3)/len(ndcg3)))
#print("NDCG@5\t"+str(sum(ndcg5)/len(ndcg5)))
print("NDCG@10\t"+str(sum(ndcg10)/len(ndcg10)))
print("NDCG@25\t"+str(sum(ndcg25)/len(ndcg25)))
