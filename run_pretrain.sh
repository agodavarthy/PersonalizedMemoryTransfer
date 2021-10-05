#!/usr/bin/env bash

GPU=2
MODEL=mf
#EMBED_SIZE=10
echo GPU ${GPU}

N_LAYERS=2
for EMBED_SIZE in 50 60 
do

  L2=1e-4
  SAVE_DIR=results/redial/sep29/${MODEL}/${EMBED_SIZE}_${N_LAYERS}l/tie/l2${L2}
  echo ${SAVE_DIR}
  mkdir -p ${SAVE_DIR}
  python pretrain.py -g ${GPU} -d data/redial/ -m ${MODEL} -e ${EMBED_SIZE} -mf ${SAVE_DIR} --n_layers ${N_LAYERS} --l2 ${L2} --tie t
done
