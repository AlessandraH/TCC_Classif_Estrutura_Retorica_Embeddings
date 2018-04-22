# -*- coding: utf-8 -*-
import functions as f
import warnings

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
    return f.np.array(features)


def sent2features(i, we, tfidf, tfidf_prev, tfidf_next, pos):
    features = {
        'we': f.np.sum(we),
        'tfidf': f.np.sum(tfidf),
        'tfidf_prev': f.np.sum(tfidf_prev),
        'tfidf_next': f.np.sum(tfidf_next),
        'pos': pos[i],
    }

    if pos[i] == 0:
        features['boa'] = True
    elif pos[i+1] == 0:
        features['eoa'] = True

    return features


def abstract2features(abstract, we, tfidf, tfidf_prev, tfidf_next, pos, c):
    return [sent2features(c+i, we, tfidf[c+i], tfidf_prev[c+i], tfidf_next[c+i], pos) for i in range(len(abstract))]


def abstract2labels(abstract):
    return [label for label, abstract in abstract]


def classificador():
    corpora = ['corpus/output366.json', 'corpus/output466.json', 'corpus/output832.json']

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
    kchi = 100

    print("Abrindo modelo embedding")
    model = f.KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
    vocabulary = model.vocab

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        abstracts = f.loadJson(corpus)
        _, _, data, labels, _ = f.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

        X_sentences_we = extract_features_we(X_sentences, model, model_size, vocabulary)

        print("Extraindo tfidf e chi2")
        vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
        selector = f.SelectKBest(f.chi2, k=kchi)

        X_sentences = vectorizer.fit_transform(X_sentences)
        X_prev = vectorizer.transform(X_prev)
        X_next = vectorizer.transform(X_next)
        X_sentences = selector.fit_transform(X_sentences, Y_sentences)
        X_prev = selector.transform(X_prev)
        X_next = selector.transform(X_next)

        X_pos.append(0)
        X_crf = []
        c = 0
        for a in abstracts:
            X_crf.append(abstract2features(a, X_sentences_we, X_sentences, X_prev, X_next, X_pos, c))
            c += len(a)
        Y_crf = [abstract2labels(a) for a in abstracts]

        print("CRF")
        clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                     max_iterations=100, all_possible_transitions=True)
        clf = clf.fit(X_crf, Y_crf)
        pred = f.cross_val_predict(clf, X_crf, Y_crf, cv=10)
        print("Classification_report:")
        labels = list(clf.classes_)
        f.metrics.flat_f1_score(Y_crf, pred, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(f.metrics.flat_classification_report(Y_crf, pred, labels=sorted_labels, digits=3))
        print("")


warnings.filterwarnings("ignore")
print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))
