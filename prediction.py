import tensorflow as tf
import numpy as np
maxSeqLength=30
def vectorize(text):
    spilt_sentence=text.split()
    with open('wordlist', 'r') as myfile:
        wordlist = myfile.readlines()
        wordlist = [word.lower().strip() for word in wordlist]


    question_one_ids = np.zeros((1, maxSeqLength), dtype='int32')
    wordcounter=0
    for word in spilt_sentence:
        try:
            question_one_ids[0][wordcounter] = wordlist.index(word)
            wordcounter += 1
        except ValueError:
            question_one_ids[0][wordcounter] = 3999999
            wordcounter += 1
    # vec=question_one_ids

    wordVectors = np.load('word_vectors.npy')
    vec = tf.nn.embedding_lookup(wordVectors, question_one_ids)
    # print(vec)
    return vec

vectorize('How can I succeed in medical school')

def lstm_method(input1,input2):
    x1=vectorize(input1)
    x2=vectorize(input2)
    #这两个是打印结果是 Tensor("embedding_lookup:0", shape=(1, 30, 300), dtype=float64)

    x = tf.placeholder(tf.float32, [None, 1600])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "model/siamese.ckpt-2001")
    result = sess.run(y, feed_dict={x: data})


    result=0
    return result


