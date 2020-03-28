

from torchtext.vocab import GloVe

import train_embeddings

MAX_VOCAB_SIZE = 10000
min_freq = 10
pretrained = False
vectors = GloVe(name="6B", dim=100)

model_name = 'gru_own_vocab'
train_embeddings.create_embeddings(MAX_VOCAB_SIZE=10000, min_freq=10,
                                  pretrained=False,
                                  vectors=None,
                                  freeze_embeddings=False,
                                   model_name=model_name)