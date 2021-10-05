#!/usr/bin/env bash

# change the following parameters to desired one.
FLR=1e-6
L2=1e-5
ENCODER=$1
GPU=$2

# pick the desired sentiment dataset
DATASET=redial

OUTPUT_DIR=data/redial/sep27/${ENCODER}/${DATASET}_flr${FLR}_l2${L2}
BASE_SENTIMENT_PATH=results/sentiment/${DATASET}_${ENCODER}_flr${FLR}_l2${L2}/

##
SAVE_NAME=${OUTPUT_DIR}_kfold/splits.pkl
#if [ ! -f ${SAVE_NAME} ]; then
    echo
    echo Creating Dataset in ${SAVE_NAME}
    echo
    python scripts/preprocess_conv_data.py -redial data/${SENT_DATASET}_dataset/ \
        -movie_map data/${SENT_DATASET}/movie_map.csv \
        -movie_plot data/${SENT_DATASET}/movie_plot.csv \
        -g ${GPU} -enc ${BASE_SENTIMENT_PATH} \
        -o ${SAVE_NAME}
#fi

    python scripts/preprocess_conv_data.py -redial data/${SENT_DATASET}_dataset/ \
        -movie_map data/${SENT_DATASET}/movie_map.csv \
        -movie_plot data/${SENT_DATASET}/movie_plot.csv \
        -g ${GPU} -enc /module/ \
        -o ${SAVE_NAME}

# =============================================================================
# Create test.pkl
# =============================================================================

SAVE_NAME=${OUTPUT_DIR}/test.pkl
if [ ! -f ${SAVE_NAME} ]; then
    echo
    echo Creating Testing Dataset in ${SAVE_NAME}
    echo
    python scripts/preprocess_conv_data.py -redial data/${SENT_DATASET}_dataset/ \
    -movie_map data/${SENT_DATASET}/movie_map.csv \
    -movie_plot data/${SENT_DATASET}/movie_plot.csv \
    -g ${GPU} -enc ${BASE_SENTIMENT_PATH}/ \
    -o ${OUTPUT_DIR}/test.pkl -test t
fi
