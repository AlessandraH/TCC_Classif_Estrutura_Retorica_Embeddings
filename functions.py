# -*- coding: utf-8 -*-

import warnings
import json

import numpy as np
import operator


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
    with open(file, 'r') as f:
        data = json.load(f, encoding='cp1252')

    return to_sentences(data)


def loadJson(file):
    with open(file, 'r') as f:
        data = json.load(f, encoding='cp1252')

    return data


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


def div(n):
    return n[0]/n[1]


def extract_features_we_media_pond(X_sentences, vectorizer, model, model_size, vocabulary):
    features = []

    total_words = 0
    words_in_embeddings = 0

    vectorizer.fit(X_sentences)
    idfs = vectorizer.idf_
    vec_vocab = vectorizer.vocabulary_

    for s in X_sentences:
        total_weight = 0
        sentence_feature = [0] * model_size
        sentences = str(s).split()
        for word in sentences:
            if len(word) > 2 and word in vocabulary:
                total_words += 1
                words_in_embeddings += 1
                if word not in vec_vocab.keys():
                    word_idf = 1
                else:
                    word_idf = idfs[vec_vocab[word]]
                total_weight += word_idf
                word_idf_list = [word_idf] * model_size
                word_feature = list(map(operator.mul, model[word], word_idf_list))
                sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
            elif len(word) > 2:
                total_words += 1
        if total_weight == 0:
            total_weight = 1
        divisor = [total_weight] * model_size
        sentence_feature = list(map(div, zip(sentence_feature, divisor)))
        features.append(sentence_feature)

    print("")
    print("Total words in embedding model: %d" % words_in_embeddings)
    print("Total words in abstract: %d" % total_words)
    print("Words present in embedding model: %.2f" % (words_in_embeddings/total_words))
    print("")

    return np.array(features)


def extract_features_we_media(X_sentences, model, model_size, vocabulary):
    features = []
    for s in X_sentences:
        n = 0
        sentence_feature = [0] * model_size
        sentences = str(s).split()
        for word in sentences:
            if len(word) > 2 and word in vocabulary:
                n += 1
                word_feature = model[word]
                sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
        divisor = [n] * model_size
        sentence_feature = list(map(div, zip(sentence_feature, divisor)))
        features.append(sentence_feature)
    return np.array(features)


def extract_features_we(X_sentences, model, model_size, vocabulary):
    features = []
    for s in X_sentences:
        sentence_feature = [0] * model_size
        sentences = str(s).split()
        for word in sentences:
            if len(word) > 2 and word in vocabulary:
                word_feature = model[word]
                sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
        features.append(sentence_feature)
    return np.array(features)


warnings.filterwarnings("ignore")
