import os
from sklearn.model_selection import ParameterGrid
import time

from model.embeddings import create_embeddings
from model.preprocessing import preprocess_text
from model.torchtext_sentiment_v2 import analyse_sentiments
from model.utils import get_model_name


# INPUTS
############
PROCESS_DATASETS = False
CREATE_EMBEDDINGS = True
TRAINING_MODULE = True

if PROCESS_DATASETS:
    dataset_path = os.getcwd()
    dataset_path = os.path.join(dataset_path, "data")
    dataset_path = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    preprocess_text(dataset_path, stem=False)


if CREATE_EMBEDDINGS:
    # TODO CREATE OWN EMBEDDINGS
    embedding_params = [{
        'min_count': [1],  # valitaan tähän vakioarvo
        'max_vocab_size': [1000e3],  # valitaan tähän vakioarvo, esim. 50k
        'window_size': [7],  # Testataanko: [5, 10] for skip-gram usually around 10, for CBOW around 5
         'vector_size': [100],  # Testataanko [10, 100, 300]
         'noise_words': [20],  # for large datasets between 2-5 valitaan yksi
         'use_skip_gram': [1],  # 1 for skip-gram, 0 for CBOW, testi molemmilla?
         'cbow_mean': [0],  # if using cbow
         'w2v_iters': [10]  # onko tarpeeksi?
         }]

    param_grid_emb = list(ParameterGrid(embedding_params))
    print(f"Number of items in parameter grid {len(param_grid_emb)}")
    for i, param in enumerate(param_grid_emb):
        print(f"{i+1}/{len(param_grid_emb)} Creating word2vec model with params {param}")
        create_embeddings(param, i)

    # TODO TEST EMBEDDINGS AND PLOT RESULTS

if TRAINING_MODULE:
    params = [
        {'MAX_VOCAB_SIZE': [500e3],  # needs to match pretrained word2vec model params
         'min_freq': [1],  # needs to match pretrained word2vec model params
         'embedding_dim': [100],  # only needed if not pretrained
         'pretrained': [True],
         'vectors': ['word2vec_twitter_skipgram_v100.mdl'],  # needs to match pretrained word2vec model params
         'RNN_FREEZE_EMDEDDINGS': [True, False],  # freeze
         'RNN_HIDDEN_DIM': [256],  # 128 tai 256
         'RNN_N_LAYERS': [1],  # 3 layers in  Howard et. al (2018)
         'RNN_DROPOUT': [0.4],  # 0.4
         'RNN_USE_GRU': [True],  # True: use GRU, False: use LSTM
         'RNN_BATCH_SIZE': [64],  # Kagglessa käytettiin 1024
         'RNN_EPOCHS': [10]  # onko riittävä?
         }]

    param_grid = list(ParameterGrid(params))
    print(f"Number of items in parameter grid {len(param_grid)}")

    test_accs = []
    for i, param in enumerate(param_grid):
        print(f"params {param}")
        model_name = get_model_name(param)
        print(f"{i+1}/{len(param_grid)} testing {model_name}")
        start_time = time.time()
        test_loss, test_acc = analyse_sentiments(params=param,
                                                 model_name=model_name)
        end_time = time.time()
        print(f"Training lasted for {round((end_time - start_time) / 60, 1)} min")

        test_accs.append(test_acc)

    for i, param in enumerate(param_grid):
        print(f"param {param}")
        print(f"test accuracy: {test_accs[i]}")


# TODO DO TESTS AND PLOT RESULT
