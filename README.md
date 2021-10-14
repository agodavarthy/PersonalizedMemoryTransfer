**PERSONALIZED MEMORY TRANSFER(PMT)**


## Preprocessing Dataset

### GoRecDial
3. ```movie_map.csv``` file is created by duplicate the movie_id column as in GoRecDial dataset they used
the same id as in the movie-lens dataset. 
4. Run the following preprocessing scripts:  
    Get the redial dataset and movielens dataset then use scripts/match_movies.py to create the matched up ids of movies, movie_match.csv.
    ```bash
    python scripts/reformat_gorecdial.py --movies_merged data/gorecdial/movies_gorecdial.csv --ml_movies data/ml-latest/movies.csv --output movie_match.csv 
    ```
     extract_movie_summary.py -- extracts the movie plots from my database
    ```bash
    python scripts/extract_movie_summary.py --movie_match data/gorecdial/movie_match.csv --output movie_plot.csv
    ```
   
5. Make sure you saved the output files to a specific folder so that you wont mess these files up with the Recdial files.

### Redial
Get the redial dataset and movielens dataset then use scripts/match_movies.py to create the matched up ids of movies, movie_match.csv.

```bash
python scripts/reformat.py --redial data/movies_merged.csv --ml_movies data/ml-latest/movies.csv --output movie_match.csv 
```

extract_movie_summary.py -- extracts the movie plots from my database
```bash
python scripts/extract_movie_summary.py --movie_match data/movie_match.csv --output movie_plot.csv
```


## Pretrain
```bash
    run_pretrain.sh
```
The file did the pretraining of nmf, mf or gmf model. Uncomment and line of codes to do the pretraining.
Look at the parameters input to ```pretrain.py``` to make sure you pretrain on the correct data (redial or GoRecDial).

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

