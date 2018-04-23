# -*- coding: utf-8 -*-
import functions as f
import arff
import warnings


def sent2features(i, tfidf, tfidf_prev, tfidf_next, pos):
    features = {
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


def abstract2features(abstract, tfidf, tfidf_prev, tfidf_next, pos, c):
    return [sent2features(c+i, tfidf[c+i], tfidf_prev[c+i], tfidf_next[c+i], pos) for i in range(len(abstract))]


def abstract2labels(abstract):
    return [label for label, abstract in abstract]


def classificador():
    # corpus = 'corpus/output366.json'
    # azfeat = 'azport_features/azfeatures366.arff'
    corpus = 'corpus/output466.json'
    azfeat = 'azport_features/azfeatures466.arff'

    print("lendo corpus ", corpus)
    abstracts = f.loadJson(corpus)
    _, _, data, labels, _ = f.loadFromJson(corpus)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

    ####### 4 6 6 #######
    del X_sentences[206]
    del X_prev[206]
    del X_next[206]
    del X_pos[206]
    del Y_sentences[206]
    #####################

    dataset = arff.load(open(azfeat, 'r'))
    X_data = f.np.array(dataset['data'])
    features = X_data[:, 0:8]
    Y_labels = X_data[:, 8]

    X_pos.append(0)
    X_crf = []
    c = 0
    for a in abstracts:
        X_crf.append(abstract2features(a, X_sentences, X_prev, X_next, X_pos, c))
        c += len(a)
    Y_crf = [abstract2labels(a) for a in abstracts]

    print("CRF")
    clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                 max_iterations=100, all_possible_transitions=True)
    clf = clf.fit(features, Y_crf)
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
