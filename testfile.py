import gensim
import metapy
import Levenshtein
import pandas as pd
from simhash import Simhash
import numpy as np
import re
from scipy import spatial


model=gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)
# model=gensim.models.KeyedVectors.load_word2vec_format('data/numberbatch-en-17.06.txt',binary=False)

#levenstein#
def levenstein_method(doc1,doc2,threshold):
    #distance less than threshod distance
    if Levenshtein.distance(doc1,doc2)<threshold*(len(a)+len(b))/2:
        result=1
    else:
        result=0
    return result
#levenstein#


#word2vec based #
def word2vec_wmd_method(doc1,doc2,threshold):
    distance = model.wmdistance(doc1, doc2)

    # print ('distance = %.3f' % distance)
    if distance<threshold:
        result=1
    else:
        result=0
    return result
#end word2vec based#

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
#word2vec based avg method#

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
# end o f word2vec based avg method#



#simhash method#
def simhash_method(doc1,doc2,threshold):
    distance=Simhash(doc1).distance(Simhash(doc2))
    # print(distance) #to check the similarity is same as it should to be#
    if distance<threshold:
        result=1
    else:
        result=0
    return result
#end of simhash method#

# # a=model.similarity('france', 'germany')
# print(model.wv.similarity('man','woman'))
# print(model.wv.similarity('queen','elizabeth'))
# print(model.wv.similarity('sport','basketball'))
# print(model.wv.similarity('sperm','universe'))
# print(model.wv.most_similarity('man','woman'))
# print(model.wv.similarity(doc_1,doc_2))
# print(model.most_similar(positive=doc_1, negative=doc_2))




TRAIN_CSV = 'data/test.csv'

# Load training set
df = pd.read_csv(TRAIN_CSV,usecols=[3,4,5])
# print(len(df))

tp1=tp2=tp3=tp4=0
tn1=tn2=tn3=tn4=0
fp1=fp2=fp3=fp4=0
fn1=fn2=fn3=fn4=0
for i in range(5000):
    a=df.iloc[i][0]
    b=df.iloc[i][1]
    c=df.iloc[i][2]

    pre1=levenstein_method(a, b, 0.75)
    pre2=simhash_method(a, b,27)
    pre3=word2vec_avg_method(a,b,0.74)
    pre4=word2vec_wmd_method(a,b,0.89)

    # print(pre1,pre2,pre3,pre4,c)

    if (c==pre1==1):
        tp1+=1
    elif(c==pre1==0):
        tn1+=1
    elif(c==1 and pre1==0):
        fn1+=1
    else:
        fp1+=1

    if (c==pre2==1):
        tp2+=1
    elif(c==pre2==0):
        tn2+=1
    elif(c==1 and pre2==0):
        fn2+=1
    else:
        fp2+=1

    if (c==pre3==1):
        tp3+=1
    elif(c==pre3==0):
        tn3+=1
    elif(c==1 and pre3==0):
        fn3+=1
    else:
        fp3+=1

    if (c==pre4==1):
        tp4+=1
    elif(c==pre4==0):
        tn4+=1
    elif(c==1 and pre4==0):
        fn4+=1
    else:
        fp4+=1


    # if c==simhash_method(a,b,11):
    #     sum1+=1
    # if c==word2vec_avg_method(a,b,0.83):
    #     sum2+=1
    # if c==word2vec_wmd_method(a,b,0.5):
    #     sum3+=1


# print("acc=",sum1/50000,sum2/50000,sum3/50000)
print("F1_levenstein=",2*tp1/(2*tp1+fp1+fn1),'\n')
print("F1_simhash=",2*tp2/(2*tp2+fp2+fn2),'\n')
print("F1_word2doc_avg=",2*tp3/(2*tp3+fp3+fn3),'\n')
print("F1_word2doc_wmd=",2*tp4/(2*tp4+fp4+fn4),'\n')


# print(Simhash('xxx').value)
# print (Simhash(' How can I succeed in medical school').distance(Simhash('What are some tips for success in medical school?')))
# print (Simhash('Is IMS noida good for BCA').distance(Simhash('How good is IMS Noida for studying BCA?')))
# simhash_method(' How can I succeed in medical school','What are some tips for success in medical school?',10)



# print ('distance = %.3f' % distance)
#
# distance = model.wmdistance('i love to sleep', 'i love to sleep too!')
#
# print ('distance = %.3f' % distance)
#
# distance = model.wmdistance('fdsafjiodshfbiuasdfnoasdfj ifjdsaoifas fsjdifoajsdfoasdf fsia', 'f x')
#
# print ('distance = %.3f' % distance)