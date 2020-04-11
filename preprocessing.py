
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
import pandas as pd
import re


def decode_sentiment(label):
    decode_map = {0: 0, 2: "NEUTRAL", 4: 1}
    return decode_map[int(label)]


def preprocess(text, stop_words, stem=False, stemmer=None):
    # Remove link,user and special characters
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def preprocess_text(dataset_path, stem=False):
    """

    :param dataset_path:
    :return:
    """
    print(f"Preprocessing twitter dataset. "
          f"Removing stop words and cleaning hastags etc."
          f"")
    if stem:
        print(f"Applying stemmer")
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    # dataset_path = r'train.csv'
    df = pd.read_csv(dataset_path,
                     encoding=DATASET_ENCODING,
                     names=DATASET_COLUMNS,
                     usecols=[0, 5])
    df.target = df.target.apply(lambda x: decode_sentiment(x))

    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")

    df.text = df.text.apply(lambda x: preprocess(x, stop_words, stem=stem, stemmer=stemmer))

    print(f"Preprocessing results in empty tweets. How do we drop empty sentences from dataset?")
    # How to drop tweets which don't have any words left after processing? dropna does not work
    df.dropna(axis=0, inplace=True)

    # Split in to train val test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=10, shuffle=True)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=10, shuffle=True)

    df_train.to_csv("data/processed_train.csv", index=False)
    df_val.to_csv("data/processed_val.csv", index=False)
    df_test.to_csv("data/processed_test.csv", index=False)