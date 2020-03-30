from torchtext.vocab import GloVe

import os
import torchtext_sentiment
import preprocessing

MAX_VOCAB_SIZE = 10000
min_freq = 10
pretrained = False
#vectors = GloVe(name="6B", dim=100)
vectors = None
freeze_embeddings = True
N_EPOCHS = 2
model_name = 'gru_own_vocab'

PROCESS_DATASETS = False
if PROCESS_DATASETS:
    dataset_path = os.getcwd()
    dataset_path = os.path.join(dataset_path, "data")
    dataset_path = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    preprocessing.preprocess_text(dataset_path)


# TODO CREATE OWN EMBDEDDINGS
# TODO TEST EMBEDDINGS AND PLOT RESULTS

# TODO WRITE A LOOP FOR DIFFERENT CASES
test_loss, test_acc = torchtext_sentiment.analyse_sentiments(MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,
                                                             min_freq=min_freq,
                                                             pretrained=pretrained,
                                                             vectors=vectors,
                                                             freeze_embeddings=freeze_embeddings,
                                                             N_EPOCHS=N_EPOCHS,
                                                             model_name=model_name)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
# TODO DO TESTS AND PLOT RESULTS