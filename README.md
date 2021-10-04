## Pretrain
```bash
    run_pretrain.sh
```
The file did the pretraining of nmf, mf or gmf model. Uncomment and line of codes to do the pretraining.
Look at the parameters input to ```pretrain.py``` to make sure you pretrain on the correct data (redial or GoRecDial).

This file could also be used for parameter search. ```pretrain.py``` has 'for loops' in the file, that these loops are input
different combination of hyperparameters.

## Test
In the paper, we are experimenting transformer, bert, dan and elmo.
### Sentiment model
For each of the language models mention above, we need to train the sentiment model for later test phase.
For example, to train the sentiment model for elmo
```bash
preprocess_sentiment.sh elmo 0
```
The argument 0 means to train the model on GPU 0. You need to rerun the scripts for other language models too.

### Test split
For each for the language models, run ```run_exp.sh``` to split the dataset.
```bash
run_exp.sh elmo 0
```
This example split the dataset with elmo.

### Test model
To get the results on dataset GoRecDial, run:
```bash
run_test_gorecdial.sh
```

To get the results on dataset Redial, run:
```bash
run_test_redial.sh
```

## Baselines
Baselines models scripts can be found at ```run_baseline_gorecdial.sh``` and ```run_baseline_redial.sh```. 
