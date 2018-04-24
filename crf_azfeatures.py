# -*- coding: utf-8 -*-
import functions as f
import arff
import warnings


def sent2features(i, x_features, pos):
    features = {
        'length': x_features[i][0],
        'location': x_features[i][1],
        'citacion': x_features[i][2],
        'formulaic': x_features[i][3],
        'tense': x_features[i][4],
        'voice': x_features[i][5],
        'modal': x_features[i][6],
        'history': x_features[i][7],
        'pos': pos[i],
    }

    if pos[i] > 0:
        features.update({
            '-1:length': x_features[i-1][0],
            '-1:location': x_features[i-1][1],
            '-1:citacion': x_features[i-1][2],
            '-1:formulaic': x_features[i-1][3],
            '-1:tense': x_features[i-1][4],
            '-1:voice': x_features[i-1][5],
            '-1:modal': x_features[i-1][6],
            '-1:history': x_features[i-1][7],
            '-1:pos': pos[i-1],
        })
    else:
        features['boa'] = True

    if pos[i+1] > 0:
        features.update({
            '+1:length': x_features[i+1][0],
            '+1:location': x_features[i+1][1],
            '+1:citacion': x_features[i+1][2],
            '+1:formulaic': x_features[i+1][3],
            '+1:tense': x_features[i+1][4],
            '+1:voice': x_features[i+1][5],
            '+1:modal': x_features[i+1][6],
            '+1:history': x_features[i+1][7],
            '+1:pos': pos[i+1],
        })
    else:
        features['eoa'] = True

    return features


def abstract2features(abstract, x_features, pos, c):
    return [sent2features(c+i, x_features, pos) for i in range(len(abstract))]


def abstract2labels(abstract):
    return [label for label, abstract in abstract]


def classificador():
    corpus366 = 'corpus/output366.json'
    azfeat366 = 'azport_features/azfeatures366.arff'
    corpus466 = 'corpus/output466.json'
    azfeat466 = 'azport_features/azfeatures466.arff'
    corpus832 = 'corpus/output832.json'
    azfeat832 = 'azport_features/azfeatures832.arff'

    print("lendo corpus ", corpus366)
    abstracts = f.loadJson(corpus366)
    _, _, data, labels, _ = f.loadFromJson(corpus366)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

    dataset = arff.load(open(azfeat366, 'r'))
    X_data = f.np.array(dataset['data'])
    features = X_data[:, 0:8]
    # Y_labels = X_data[:, 8]

    X_pos.append(0)
    X_crf = []
    c = 0
    for a in abstracts:
        X_crf.append(abstract2features(a, features, X_pos, c))
        c += len(a)
    Y_crf = [abstract2labels(a) for a in abstracts]

    print("CRF corpus366")
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

    print("lendo corpus ", corpus466)
    abstracts = f.loadJson(corpus466)
    _, _, data, labels, _ = f.loadFromJson(corpus466)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

    del abstracts[17][7]
    del X_sentences[206]
    del X_prev[206]
    del X_next[206]
    del X_pos[206]
    del Y_sentences[206]

    dataset = arff.load(open(azfeat466, 'r'))
    X_data = f.np.array(dataset['data'])
    features = X_data[:, 0:8]
    # Y_labels = X_data[:, 8]

    X_pos.append(0)
    X_crf = []
    c = 0
    for a in abstracts:
        X_crf.append(abstract2features(a, features, X_pos, c))
        c += len(a)
    Y_crf = [abstract2labels(a) for a in abstracts]

    print("CRF corpus466")
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

    print("lendo corpus ", corpus832)
    abstracts = f.loadJson(corpus832)
    _, _, data, labels, _ = f.loadFromJson(corpus832)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

    del abstracts[69][7]
    del X_sentences[572]
    del X_prev[572]
    del X_next[572]
    del X_pos[572]
    del Y_sentences[572]

    dataset = arff.load(open(azfeat832, 'r'))
    X_data = f.np.array(dataset['data'])
    features = X_data[:, 0:8]
    # Y_labels = X_data[:, 8]

    X_pos.append(0)
    X_crf = []
    c = 0
    for a in abstracts:
        X_crf.append(abstract2features(a, features, X_pos, c))
        c += len(a)
    Y_crf = [abstract2labels(a) for a in abstracts]

    print("CRF corpus832")
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
