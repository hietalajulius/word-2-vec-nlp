import os
from sklearn.model_selection import ParameterGrid
import time

from embeddings import create_embeddings
from torchtext_sentiment import analyse_sentiments
from preprocessing import preprocess_text

# INPUTS
############
PROCESS_DATASETS = False
CREATE_EMBEDDINGS = False
TRAINING_MODULE = True

if PROCESS_DATASETS:
    dataset_path = os.getcwd()
    dataset_path = os.path.join(dataset_path, "data")
    dataset_path = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    preprocess_text(dataset_path, stem=False)


if CREATE_EMBEDDINGS:
    # TODO CREATE OWN EMBEDDINGS
    embedding_params = [{
        'min_count': [3],  # valitaan tähän vakioarvo
        'max_vocab_size': [50000],  # valitaan tähän vakioarvo, esim. 50k
        'window_size': [5],  # Testataanko: [5, 10] for skip-gram usually around 10, for CBOW around 5
         'vector_size': [100],  # Testataanko [10, 100, 300]
         'noise_words': [3],  # for large datasets between 2-5 valitaan yksi
         'use_skip_gram': [0],  # 1 for skip-gram, 0 for CBOW
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
        {'MAX_VOCAB_SIZE': [50000],  # needs to match pretrained word2vec model params
         'min_freq': [3],  # needs to match pretrained word2vec model params
         'embedding_dim': [100],  # needs to match pretrained word2vec model params
         'pretrained': [True],
         'vectors': ['word2vec_twitter_v0.mdl'],  # needs to match pretrained word2vec model params
         'RNN_FREEZE_EMDEDDINGS': [False],
         'RNN_HIDDEN_DIM': [256],  # 128 tai 256
         'RNN_N_LAYERS': [1, 3],  # 3 layers in  Howard et. al (2018)
         'RNN_DROPOUT': [0.0, 0.4],  # 0.4
         'RNN_USE_GRU': [True, False],  # False -> use LSTM
         'EPOCHS' : [5]  # onko riittävä?
         }]
    RUN_NAME = '5_epochs'

    param_grid = list(ParameterGrid(params))
    print(f"Number of items in parameter grid {len(param_grid)}")

    for i, param in enumerate(param_grid):
        print(f"params {param}")
        if param['vectors'] == None:
            model_name = f"own_{RUN_NAME}_{param['MAX_VOCAB_SIZE']}_{param['min_freq']}"
        else:
            model_name = f"{RUN_NAME}_{param['vectors']}_{param['MAX_VOCAB_SIZE']}_{param['min_freq']}"
        print(f"{i+1}/{len(param_grid)} testing {model_name}")

        start_time = time.time()
        test_loss, test_acc = analyse_sentiments(params=param,
                                                 model_name=model_name)
        end_time = time.time()
        print(f"Training lasted for {round((end_time - start_time) / 60, 1)} min")

    # TODO DO TESTS AND PLOT RESULT
