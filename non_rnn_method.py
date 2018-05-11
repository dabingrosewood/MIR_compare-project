import gensim
import Levenshtein
import pandas as pd
from simhash import Simhash
import numpy as np
from scipy import spatial
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import time

model=gensim.models.KeyedVectors.load_word2vec_format('data/numberbatch-en-17.06.txt',binary=False)

def stemming(doc):
    ps = PorterStemmer()
    # sen = 'It is important to by very pythonly while you are pythoning with python'
    words = word_tokenize(doc)

    after = ''
    for w in words:
        after += (ps.stem(w) + ' ')
    # print(after)

    return after

#levenstein
def levenstein_method(doc1,doc2,threshold):
    #distance less than threshod distance
    if Levenshtein.distance(doc1,doc2)<threshold*(len(a)+len(b))/2:
        result=1
    else:
        result=0
    return result



#word2vec based #
def word2vec_wmd_method(doc1,doc2,threshold):
    distance = model.wmdistance(doc1, doc2)
    if distance<threshold:
        result=1
    else:
        result=0
    return result


#word2vec based avg method#
index2word_set = set(model.index2word)
def avg_feature(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def word2vec_avg_method(doc1,doc2,threshold):

    sentence1_vec = avg_feature(doc1, model=model, num_features=300, index2word_set=index2word_set)
    sentence2_vec = avg_feature(doc2, model=model, num_features=300, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(sentence1_vec, sentence2_vec)
    # print(sim)
    if sim>threshold:
        result=1
    else:
        result=0
    return result



#simhash method#
def simhash_method(doc1,doc2,threshold):
    distance=Simhash(doc1).distance(Simhash(doc2))
    # print(distance) #to check the similarity is same as it should to be#
    if distance<threshold:
        result=1
    else:
        result=0
    return result




TRAIN_CSV = 'data/test.csv'
size=50000
df = pd.read_csv(TRAIN_CSV,usecols=[3,4,5])
tp1 = tp2 = tp3 = tp4 = tp5 = 0
tn1 = tn2 = tn3 = tn4 = tn5 = 0
fp1 = fp2 = fp3 = fp4 = fp5 = 0
fn1 = fn2 = fn3 = fn4 = fn5 = 0

th1 = 0.64
th2 = 19
th3 = 0.84
th4 = 0.14
th5 = 0.54




start=time.time()
for i in range(size):
    a = df.iloc[i][0]
    b = df.iloc[i][1]
    c = df.iloc[i][2]

    pre1 = levenstein_method(a, b, th1)
    if (c == pre1 == 1):
        tp1 += 1
    elif (c == pre1 == 0):
        tn1 += 1
    elif (c == 1 and pre1 == 0):
        fn1 += 1
    else:
        fp1 += 1

end=time.time()

print('levenstein acc = ',(tp1+tn1)/size,'\n')
print('timecost =',end-start,'s\n')


start=time.time()
for i in range(size):
    a = df.iloc[i][0]
    b = df.iloc[i][1]
    c = df.iloc[i][2]

    pre2 = simhash_method(a, b, th2)
    if (c == pre2 == 1):
        tp2 += 1
    elif (c == pre2 == 0):
        tn2 += 1
    elif (c == 1 and pre2 == 0):
        fn2 += 1
    else:
        fp2 += 1

end=time.time()

print('simhash acc = ',(tp2+tn2)/size,'\n')
print('timecost =',end-start,'s\n')

start=time.time()
for i in range(size):
    a = df.iloc[i][0]
    b = df.iloc[i][1]
    c = df.iloc[i][2]

    pre3 = word2vec_avg_method(a, b, th3)
    if (c == pre3 == 1):
        tp3 += 1
    elif (c == pre3 == 0):
        tn3 += 1
    elif (c == 1 and pre3 == 0):
        fn3 += 1
    else:
        fp3 += 1

end=time.time()

print('word2vec_avg_method acc = ',(tp3+tn3)/size,'\n')
print('timecost =',end-start,'s\n')



start=time.time()
for i in range(size):
    a = df.iloc[i][0]
    b = df.iloc[i][1]
    c = df.iloc[i][2]

    pre4 = word2vec_wmd_method(a, b, th4)
    if (c == pre4 == 1):
        tp4 += 1
    elif (c == pre4 == 0):
        tn4 += 1
    elif (c == 1 and pre4 == 0):
        fn4 += 1
    else:
        fp4 += 1

end=time.time()

print('word2vec_wmd_method acc = ',(tp4+tn4)/size,'\n')
print(start,end)
print('timecost =',end-start,'s\n')


start=time.time()
for i in range(size):
    a = df.iloc[i][0]
    b = df.iloc[i][1]
    c = df.iloc[i][2]

    pre5 = levenstein_method(a, b, th5)
    if (c == pre5 == 1):
        tp5 += 1
    elif (c == pre5 == 0):
        tn5 += 1
    elif (c == 1 and pre5 == 0):
        fn5 += 1
    else:
        fp5 += 1

end=time.time()

print('stemmed levenstein acc = ',(tp5+tn5)/size,'\n')
print('timecost =',end-start,'s\n')
