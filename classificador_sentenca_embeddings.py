# -*- coding: utf-8 -*-

import sys
import time
import json
import numpy as np
# import fasttext
# import gensim


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from gensim.models import KeyedVectors, Word2Vec

pontuacao = ['.', ',', ' ', '"', '!', '(', ')', '-', '=', '+', '/', '*', ';', ':'
                                                                              '[', ']', '{', '}', '$', '#', '@', '%',
             '&', '?']


def div(n):
    return n[0] / n[1]


def to_sentences(abstracts, senteces_max=None):
    sentences = []
    labels = []
    abstracts_sentences = []
    abstracts_labels = []
    ids = []

    for id, abstract in enumerate(abstracts):
        if senteces_max and len(abstract) > senteces_max:
            continue

        tmp_sentences = []
        tmp_labels = []

        for label, text in abstract:
            sentences.append(text)
            labels.append(label)

            tmp_sentences.append(text)
            tmp_labels.append(label)
            ids.append(id)

        abstracts_sentences.append(tmp_sentences)
        abstracts_labels.append(tmp_labels)

    assert (len(sentences) == len(labels))
    assert (len(abstracts_sentences) == len(abstracts_labels))

    return sentences, labels, abstracts_sentences, abstracts_labels, ids


def loadFromJson(file):
    data = []
    with open(file, 'r') as f:
        data = json.load(f, encoding='cp1252')

    return to_sentences(data)


def abstracts_to_sentences(abstracts, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []
    abstracts_idx = []

    for i, (sentences_labels, sentences) in enumerate(zip(labels, abstracts)):
        for j, (label, sentence) in enumerate(zip(sentences_labels, sentences)):
            ret.append(sentence)
            ret_pos.append(j)
            ret_labels.append(label)
            abstracts_idx.append(i)

            if j - 1 >= 0:
                ret_prev.append(sentences[j - 1])
            else:
                ret_prev.append("")

            if j + 1 < len(sentences):
                ret_next.append(sentences[j + 1])
            else:
                ret_next.append("")

    return ret, ret_prev, ret_next, ret_pos, ret_labels, abstracts_idx


def extract_features(X_sentences, model, model_size, vocabulary):
    features = []
    nvocab = 0
    palavras_corpus = 0
    # lista_div = [model_size] * model_size
    for s in X_sentences:
        n = 0
        sentence_feature = [0] * model_size
        sentences = str(s).split()
        for word in sentences:
            if len(word) > 2 and word in vocabulary:
                word_feature = model[word]
                # word_feature = KeyedVectors.word_vec(model, word, use_norm=False)
                sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
                n += 1
            # elif len(word) > 2 and word not in vocabulary:
            #     print(word + " nao estah no vocabulario")
            #     nvocab += 1
        lista_div = [n] * model_size
        sentence_feature = list(map(div, zip(sentence_feature, lista_div)))
        features.append(sentence_feature)
    return np.array(features)


def classificador():
    corpus = 'corpus/output366.json'
    # corpus = 'corpus/output466.json'
    # corpus = 'corpus/output832.json'

    model_name = 'cbow_s50.txt'
    # model_name = 'cbow_s300.txt'
    # model_name = 'skip_s1000.txt'

    model_size = 50

    print(time.asctime(time.localtime(time.time())))

    print("lendo corpus")
    _, _, data, labels, _ = loadFromJson(corpus)
    X_sentences, _, _, X_pos, Y_sentences, _ = abstracts_to_sentences(data, labels)

    print("Abrindo modelo embedding e extraindo features")
    try:
        # model = Word2Vec.load(model_name)
        model = KeyedVectors.load(model_name)
        print("Loading Embedding")
    except:
        # model = KeyedVectors.load_word2vec_format(fname=model_name, binary=False, unicode_errors="ignore") # 1
        model = KeyedVectors.load_word2vec_format(model_name)
        print("Loading word2vec embeddings")
    vocabulary = model.vocab
    X_sentences = extract_features(X_sentences, model, model_size, vocabulary)

    print("Inicializando classificador SVM")
    clf = SVC(kernel='linear')
    clf = clf.fit(X_sentences, Y_sentences)
    print("Predição...")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)
    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("")
    print(confusion_matrix(Y_sentences, pred))

    print("Inicializando classificador KNN")
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
    clf = clf.fit(X_sentences, Y_sentences)
    print("Predição...")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)
    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("")
    print(confusion_matrix(Y_sentences, pred))

    print("Inicializando classificador NB")
    clf = MultinomialNB()
    clf = clf.fit(X_sentences, Y_sentences)
    print("Predição...")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)
    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("")
    print(confusion_matrix(Y_sentences, pred))

    print("Inicializando classificador DT")
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_sentences, Y_sentences)
    print("Predição...")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)
    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("")
    print(confusion_matrix(Y_sentences, pred))

    print(time.asctime(time.localtime(time.time())))


# reload(sys)
# sys.setdefaultencoding('utf8')
classificador()
