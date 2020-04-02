import gensim
import gensim.downloader as api

from sklearn.model_selection import ParameterGrid
from torchtext.vocab import GloVe

import os
import time
import torchtext_sentiment
import preprocessing

# INPUTS
############

CREATE_EMBEDDINGS = False
PROCESS_DATASETS = False
TRAINING_MODULE = True

# TODO CREATE OWN EMBEDDINGS
if CREATE_EMBEDDINGS:
    glove_vec = GloVe(name="6B", dim=100)
    glove_vec.save('glove_6B_100.kv')

# TODO TEST EMBEDDINGS AND PLOT RESULTS

if PROCESS_DATASETS:
    dataset_path = os.getcwd()
    dataset_path = os.path.join(dataset_path, "data")
    dataset_path = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    preprocessing.preprocess_text(dataset_path, stem=False)

params = [
  {'MAX_VOCAB_SIZE': [10e3, 25e3],
   'min_freq': [1, 10],
   'freeze_embeddings': [True, False],
   'pretrained': [True],
   'vectors': ['glove_6B_100', 'word2vec_google_news_300']},
  {'MAX_VOCAB_SIZE': [10e3, 25e3],
   'min_freq': [1, 10],
   'freeze_embeddings': [True, False],
   'pretrained': [False],
   'vectors': [None]}]

params = [
          {'MAX_VOCAB_SIZE': [3e3, 10e3, 25e3],
           'min_freq': [1, 5, 10],
           'freeze_embeddings': [False],
           'pretrained': [False],
           'vectors': [None]}]
EMBEDDING_DIM = 100
N_EPOCHS = 20
if TRAINING_MODULE:
    param_grid = list(ParameterGrid(params))
    print(f"Number of items in parameter grid {len(param_grid)}")

    for i, param in enumerate(param_grid):
        print(f"params {param}")
        if param['vectors'] == None:
            model_name = f"own_{param['MAX_VOCAB_SIZE']}_{param['min_freq']}_freeze_{param['freeze_embeddings']}"
        else:
            model_name = f"{param['vectors']}_{param['MAX_VOCAB_SIZE']}_{param['min_freq']}_freeze_{param['freeze_embeddings']}"
        print(f"{i+1}/{len(param_grid)} testing {model_name}")

        start_time = time.time()
        test_loss, test_acc = torchtext_sentiment.analyse_sentiments(params=param,
                                                                     N_EPOCHS=N_EPOCHS,
                                                                     model_name=model_name,
                                                                     embedding_dim=EMBEDDING_DIM)
        end_time = time.time()
        print(f"Training lasted for {round((end_time - start_time) / 60, 1)} min")

# TODO DO TESTS AND PLOT RESULT
