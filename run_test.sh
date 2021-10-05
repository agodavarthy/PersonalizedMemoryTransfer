#!/usr/bin/env bash

DATASET=redial
for MODEL_TYPE in "gmf"
do
  #for EMB in 60, 80, 90, 100, 200, 500
  for EMB in 40 
  do
     #for bert_num in 10
     #do
        MODEL=results/${DATASET}/${MODEL_TYPE}/${EMB}_2l/tie/l21e-4/model.pth
        for ENCODER in "bert"
        do
          DATASET=data/redial/transformer/redial_flr1e-6_l21e-5/test.pkl
          python main.py --overwrite true -d ${DATASET} \
                              -mf ${MODEL} -m ${MODEL_TYPE} \
                              "$@" -g 2 
          echo ---------------------------------------------
          echo
        done
    #done
  done
done







