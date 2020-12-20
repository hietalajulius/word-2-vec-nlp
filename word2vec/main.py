from gensim.test.utils import datapath
import pandas as pd
import gensim.models

class TweetCorpusWithStops(object):
    """An interator that yields sentences (lists of str)."""
            
    def __iter__(self, filename="../data/processed_all_stops_included.csv"):
        df = pd.read_csv(filename)
        for i in range(len(df)):
            line = str(df['text'][i])
            split = line.split()
            yield split
            
class TweetCorpusNoStops(object):
    """An interator that yields sentences (lists of str)."""
            
    def __iter__(self, filename="../data/processed_all_stops_removed.csv"):
        df = pd.read_csv(filename)
        for i in range(len(df)):
            line = str(df['text'][i])
            split = line.split()
            yield split

def test_with_stops():
    window_size_list = [8] 
    vector_size_list = [600] 
    noise_words_list = [2] 
    iters_list = [10]
    cbows = [True]
    sentences = TweetCorpusWithStops()
    cols = ['model','capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative', 'gram3-superl', 'participle', 'nationality-adj', 'past-tense', 'plural noun', 'plural verb', 'total-accuracy']
    results_df = pd.DataFrame(columns=cols)

    
    for a in window_size_list:
        for b in vector_size_list:
            for c in noise_words_list:
                for d in iters_list:
                    for cbow in cbows:
                        
                        run_name = "with_stops_cbow_"+str(cbow)+"_window_"+str(a)+"_size_"+str(b)+"_noise_"+str(c)+"_iters_"+str(d)
                        print("Running test for", run_name)
                        if cbow:
                            model = gensim.models.Word2Vec(sentences=sentences, window=a, sample=0.00001, iter=d, min_count=3, size=b, sg=0, negative=c)
                        else:
                            model = gensim.models.Word2Vec(sentences=sentences, window=a, sample=0.00001, iter=d, min_count=3, size=b, sg=1, hs=0, negative=c)
                        print("Calculating accuracy")
                        accuracy = model.wv.accuracy('questions-words.txt', restrict_vocab=80000)
                        total = 0
                        correct = 0
                        accs = [run_name]
                        for i in range(14): 
                            total += len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])
                            correct += len(accuracy[i]['correct'])
                            denom = len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])
                            if denom == 0:
                                cat_acc = 0
                            else:
                                cat_acc = len(accuracy[i]['correct']) / denom
                            accs.append(cat_acc)
                        accuracy = correct / total
                        accs.append(accuracy)
                        res_row = pd.DataFrame([accs],columns=cols)
                        results_df = results_df.append(res_row)

                        fname = "with_stops_cbow_"+str(cbow)+"_window_"+str(a)+"_size_"+str(b)+"_noise_"+str(c)+"_iters_"+str(d)+"_accuracy_"+str(accuracy)+".kv"
                        print("Accuracy:", accuracy, "For:", fname)
                        results_df.to_csv('with_stops_results-full-bad.csv')
                        model.wv.save_word2vec_format('vectors/'+fname)
'''
def test_no_stops():
    window_size_list = [15] 
    vector_size_list = [300,600] 
    noise_words_list = [2,5,20] 
    iters_list = [10, 30, 100]
    cbows = [True,False]
    sentences = TweetCorpusNoStops()
    cols = ['model','capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative', 'total-accuracy']
    results_df = pd.DataFrame(columns=cols)

    for a in window_size_list:
        for b in vector_size_list:
            for c in noise_words_list:
                for d in iters_list:
                    for cbow in cbows:
                        
                        run_name = "no_stops_cbow_"+str(cbow)+"_window_"+str(a)+"_size_"+str(b)+"_noise_"+str(c)+"_iters_"+str(d)
                        print("Running test for", run_name)
                        if cbow:
                            model = gensim.models.Word2Vec(sentences=sentences, window=a, sample=0.00001, iter=d, size=b, min_count=3, sg=0, negative=c)
                        else:
                            model = gensim.models.Word2Vec(sentences=sentences, window=a, sample=0.00001, iter=d, size=b, min_count=3, sg=1, hs=0, negative=c)
                        print("Calculating accuracy")
                        accuracy = model.wv.accuracy('questions-words.txt', restrict_vocab=80000)
                        total = 0
                        correct = 0
                        accs = [run_name]
                        for i in range(8): 
                            total += len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])
                            correct += len(accuracy[i]['correct'])
                            denom = len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])
                            if denom == 0:
                                cat_acc = 0
                            else:
                                cat_acc = len(accuracy[i]['correct']) / denom
                            accs.append(cat_acc)
                        accuracy = correct / total
                        accs.append(accuracy)
                        res_row = pd.DataFrame([accs],columns=cols)
                        results_df = results_df.append(res_row)

                        fname = "no_stops_cbow_"+str(cbow)+"_window_"+str(a)+"_size_"+str(b)+"_noise_"+str(c)+"_iters_"+str(d)+"_accuracy_"+str(accuracy)+".kv"
                        print("Accuracy:", accuracy, "For:", fname)
                        results_df.to_csv('no_stops_results.csv')
                        model.wv.save_word2vec_format('vectors/'+fname)
'''
test_with_stops()