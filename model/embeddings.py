from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim import utils
import os
import pandas as pd

from torchtext.vocab import Vectors

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__2(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            print(line)
            print(utils.simple_preprocess(line))
            yield utils.simple_preprocess(line)

    def __iter__(self):
        df = pd.read_csv("data/processed_train.csv")
        for i in range(len(df)):
            line = str(df['text'][i])
            preprocessed = utils.simple_preprocess(line)
            # print(line)
            # print(preprocessed)
            yield preprocessed


def create_embeddings(embedding_params, i):
    """

    :param embedding_params:
    :param i:
    :return:
    """
    """
    glove_vec = GloVe(name="6B", dim=100)
    glove_vec.save('glove_6B_100.kv')
    """

    # Params
    min_count = embedding_params['min_count']  # Testataanko: [3, 5, 10]
    max_vocab_size = embedding_params['max_vocab_size']
    window_size = embedding_params['window_size']  # Testataanko: [3, 5, 10]
    vector_size = embedding_params['vector_size']  # Testataanko [10, 100, 300]
    noise_words = embedding_params['noise_words']  # usually between 5-20 (negative sampling)
    use_skip_gram = embedding_params['use_skip_gram']  # 1 for skip-gram, 0 for CBOW
    cbow_mean = embedding_params['cbow_mean']  # if using cbow
    iters = embedding_params['w2v_iters']  # epochs

    w2v_model = Word2Vec(
        min_count=min_count,
        max_vocab_size=max_vocab_size,
        sg=use_skip_gram,
        window=window_size,
        size=vector_size,
        negative=noise_words,
        cbow_mean=cbow_mean)
    sentences = MyCorpus()
    w2v_model.build_vocab(sentences, progress_per=100000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=iters, report_delay=1)

    if use_skip_gram:
        save_name = f"word2vec_twitter_skipgram_v{vector_size}.mdl"
    else:
        save_name = f"word2vec_twitter_cbow_v{vector_size}.mdl"
    print(f"save_name {save_name}")
    path_to_embeddings_file = os.getcwd()
    path_to_embeddings_file = os.path.join(path_to_embeddings_file, "data")
    path_to_embeddings_file = os.path.join(path_to_embeddings_file, save_name)
    w2v_model.wv.save_word2vec_format(path_to_embeddings_file)


def load_vectors(fname):
    """

    :param fname:
    :return:
    """
    path_to_embeddings_file = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    #print(f"path_to_embeddings_file {path_to_embeddings_file}")
    path_to_embeddings_file = os.path.join(path_to_embeddings_file, "data")
    print(f"path_to_embeddings_file {path_to_embeddings_file}, {fname}")
    vectors = Vectors(name=f"{fname}",
                      cache=path_to_embeddings_file)
    return vectors
