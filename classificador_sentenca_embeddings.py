# -*- coding: utf-8 -*-

import time
import json
from collections import defaultdict

import scipy
import numpy as np
import sklearn_crfsuite

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from gensim.models import KeyedVectors


pontuacao = ['.', ',', ' ', '"', '!', '(', ')', '-', '=', '+', '/', '*', ';', ':',
                '[', ']', '{', '}', '$', '#', '@', '%', '&', '?']


# classe retirada de http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# classe retirada de http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


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


def extract_features_we(X_sentences, model, model_size, vocabulary):
    features = []
    for s in X_sentences:
        # n = 0
        sentence_feature = [0] * model_size
        sentences = str(s).split()
        for word in sentences:
            if len(word) > 2 and word in vocabulary:
                word_feature = model[word]
                sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
        features.append(sentence_feature)
    return np.array(features)


def classificador():
    cross_val = 10

    corpora = ['corpus/output366.json', 'corpus/output466.json', 'corpus/output832.json']
    # corpora = ['corpus/output360.json', 'corpus/output465.json', 'corpus/output825.json']

    model_name = 'cbow_s50.txt'
    # model_name = 'cbow_s100.txt'
    # model_name = 'cbow_s300.txt'
    # model_name = 'cbow_s600.txt'
    # model_name = 'cbow_s1000.txt'

    # model_name = 'skip_s50.txt'
    # model_name = 'skip_s100.txt'
    # model_name = 'skip_s300.txt'
    # model_name = 'skip_s600.txt'
    # model_name = 'skip_s1000.txt'

    # model_name = 'glove_s50.txt'
    # model_name = 'glove_s100.txt'
    # model_name = 'glove_s300.txt'
    # model_name = 'glove_s600.txt'
    # model_name = 'glove_s1000.txt'

    model_size = 50
    ngrama = 1
    kchi = 500

    print(time.asctime(time.localtime(time.time())))

    print("Abrindo modelo embedding")
    model = KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
    vocabulary = model.vocab

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        _, _, data, labels, _ = loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = abstracts_to_sentences(data, labels)

        print("Extraindo caracteristicas")
        X_sentences = extract_features_we(X_sentences, model, model_size, vocabulary)

        #################################################################################################
        # - - - - - - - - - - - - - - Combinando embeddings com tfidf") - - - - - - - - - - - - - - - - #
        #################################################################################################
        # print("Aplicando tfidf")
        # vectorizer = TfidfVectorizer(ngram_range=(1, ngrama))
        # X_sentences = vectorizer.fit_transform(X_sentences)
        # X_prev = vectorizer.transform(X_prev)
        # X_next = vectorizer.transform(X_next)
        #
        # print("Aplicando chi-quadrado")
        # selector = SelectKBest(chi2, k=kchi)
        # X_sentences = selector.fit_transform(X_sentences, Y_sentences)
        # X_prev = selector.transform(X_prev)
        # X_next = selector.transform(X_next)
        #
        # # X_sentences = np.sum([X_sentences, X_sentences_we], axis=0)
        # # X_sentences = np.sum([X_sentences, X_prev], axis=0)
        # # X_sentences = np.sum([X_sentences, X_next], axis=0)
        # #
        # # if corpus == 'corpus/output366.json':
        # #     corpus_size = 366
        # # elif corpus == 'corpus/output466.json':
        # #     corpus_size = 466
        # # else:
        # #     corpus_size = 832
        # #
        # # X_pos = np.array(X_pos)
        # # X_pos = np.repeat(X_pos, model_size).reshape(corpus_size, model_size)
        # # X_sentences = np.sum([X_sentences, X_pos], axis=0)
        #
        # print("Adicionando anterior e posterior")
        # X_sentences = hstack([X_sentences_we, X_sentences, X_prev, X_next, np.expand_dims(np.array(X_pos), -1)])
        # X_sentences = X_sentences.todense()
        #################################################################################################

        print("SVM RBF")
        clf = SVC(kernel='rbf')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(classification_report(Y_sentences, pred))
        print(confusion_matrix(Y_sentences, pred))
        print("")

        print("SVM linear")
        clf = SVC(kernel='linear')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(classification_report(Y_sentences, pred))
        print(confusion_matrix(Y_sentences, pred))
        print("")

        print("KNN")
        clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(classification_report(Y_sentences, pred))
        print(confusion_matrix(Y_sentences, pred))
        print("")

        print("NB")
        # clf = MultinomialNB()
        clf = GaussianNB()
        clf = clf.fit(X_sentences, Y_sentences)
        pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(classification_report(Y_sentences, pred))
        print(confusion_matrix(Y_sentences, pred))
        print("")

        print("DT")
        clf = DecisionTreeClassifier(random_state=0)
        clf = clf.fit(X_sentences, Y_sentences)
        pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(classification_report(Y_sentences, pred))
        print(confusion_matrix(Y_sentences, pred))

        # print("CRF")
        # clf = sklearn_crfsuite.CRF()
        # X_sentences = scipy.sparse.csr_matrix(X_sentences)
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(classification_report(Y_sentences, pred))
        # print(confusion_matrix(Y_sentences, pred))

    print(time.asctime(time.localtime(time.time())))


# reload(sys)
# sys.setdefaultencoding('utf8')
classificador()
