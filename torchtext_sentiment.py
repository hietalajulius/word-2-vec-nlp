import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchtext.vocab
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset

from utils import epoch_time
from gru import GRU


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            # print(batch.SentimentText)
            if batch.SentimentText.nelement() > 0:
                predictions = model(batch.SentimentText).squeeze(1)

                loss = criterion(predictions, batch.Sentiment)

                acc = binary_accuracy(predictions, batch.Sentiment)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            # else:
            # print(f"Found a non-empty Tensorlist {batch.SentimentText}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.SentimentText).squeeze(1)

        loss = criterion(predictions, batch.Sentiment)

        acc = binary_accuracy(predictions, batch.Sentiment)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return model, epoch_loss / len(iterator), epoch_acc / len(iterator)


def analyse_sentiments(params=None,
                       N_EPOCHS = 1,
                       model_name='sent_model',
                       embedding_dim=100):
    """
    :param params:
    :param N_EPOCHS:
    :param model_name:
    :return:
    """

    vectors = params['vectors']
    MAX_VOCAB_SIZE = params['MAX_VOCAB_SIZE']
    min_freq = params['min_freq']
    pretrained = params['pretrained']
    freeze_embeddings = params['freeze_embeddings']
    EMBEDDING_DIM = embedding_dim

    TEXT = torchtext.data.Field(tokenize='spacy',
                                tokenizer_language='en_core_web_sm',
                                lower=True)
    LABEL = torchtext.data.LabelField(dtype=torch.float)
    datafields = [('Sentiment', LABEL), ('SentimentText', TEXT)]
    train_set, val_set, test_set = TabularDataset.splits(path='data/',
                                             train='processed_train.csv',
                                            validation='processed_val.csv',
                                            test='processed_test.csv',
                                            format='csv',
                                            skip_header=True,
                                            fields=datafields)

    if pretrained:
        vectors = Vectors(name=f"{vectors}.kv")
        TEXT.build_vocab(train_set,
                         vectors=vectors,
                         max_size=MAX_VOCAB_SIZE,
                         min_freq=min_freq)
    else:
        TEXT.build_vocab(train_set,
                         max_size=MAX_VOCAB_SIZE,
                         min_freq=min_freq)
    LABEL.build_vocab(train_set)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # minimise badding for each sentence
    train_iterator, val_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                                                        (train_set, val_set, test_set),
                                                                        batch_size=64,
                                                                        sort_key=lambda x: len(x.SentimentText),
                                                                        sort_within_batch=False,
                                                                        device=device)
    INPUT_DIM = len(TEXT.vocab)
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT1 = 0.1
    drop_out_dense = 0.4
    model = GRU(vocab_size=INPUT_DIM,
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=OUTPUT_DIM,
                n_layers=N_LAYERS,
                bidirectional=True,
                dropout=DROPOUT1,
                drop_out_dense=drop_out_dense
                )
    print(model)

    if pretrained:
        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        model.embedding.weight.data[unk_idx] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[pad_idx] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # freeze embeddings
    if freeze_embeddings:
        model.embedding.weight.requires_grad = False
    else:
        model.embedding.weight.requires_grad = True

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model, train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{model_name}.pt")

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load(f"{model_name}.pt"))
    # print(model)

    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    return test_loss, test_acc