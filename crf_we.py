# -*- coding: utf-8 -*-
import functions as f


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


def sent2features(abstract, i, we, pos):
    label = abstract[i][0]
    sentence = abstract[i][1]

    features = {
        'we': f.np.sum(we),
        'label': label,
    }

    if pos[i] == 0:
        features['boa'] = True
    elif pos[i+1] == 0:
        features['eoa'] = True

    return features


def abstract2features(abstract, we, pos):
    return [sent2features(abstract, i, we, pos) for i in range(len(abstract))]


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

    porcent = 0.2
    model_size = 50

    print("Abrindo modelo embedding")
    model = f.KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
    vocabulary = model.vocab

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        abstracts = f.loadJson(corpus)
        _, _, data, labels, _ = f.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)
        # ind = round(len(data) * porcent)
        # train_data = abstracts[ind:]
        # test_data = abstracts[:ind]
        # x_train, x_test, y_train, y_test = (data[ind:], data[:ind], labels[ind:], labels[:ind])
        # x_train_sentences, x_train_prev_sentences, x_train_next_sentences, x_train_sentences_pos, y_train_sentences, train_ind = f.abstracts_to_sentences(x_train, y_train)
        # x_test_sentences, x_test_prev_sentences, x_test_next_sentences, x_test_sentences_pos, y_test_sentences, test_ind = f.abstracts_to_sentences(x_test, y_test)

        # x_train_we = extract_features_we(x_train_sentences, model, model_size, vocabulary)
        # x_test_we = extract_features_we(x_test_sentences, model, model_size, vocabulary)

        X_sentences_we = extract_features_we(X_sentences, model, model_size, vocabulary)

        # x_train_sentences_pos.append(0)
        # x_test_sentences_pos.append(0)
        # x_train_crf = [abstract2features(a, x_train_we, x_train_sentences_pos) for a in train_data]
        # x_test_crf = [abstract2features(a, x_test_we, x_test_sentences_pos) for a in test_data]
        # y_train_crf = [abstract2labels(a) for a in train_data]
        # y_test_crf = [abstract2labels(a) for a in test_data]

        X_pos.append(0)
        X_crf = [abstract2features(a, X_sentences_we, X_pos) for a in abstracts]
        Y_crf = [abstract2labels(a) for a in abstracts]

        print("CRF")
        clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                     max_iterations=100, all_possible_transitions=True)
        # clf = clf.fit(x_train_crf, y_train_crf)
        # pred = clf.predict(x_test_crf)
        clf = clf.fit(X_crf, Y_crf)
        pred = f.cross_val_predict(clf, X_crf, Y_crf, cv=10)
        print("Classification_report:")
        labels = list(clf.classes_)
        f.metrics.flat_f1_score(Y_crf, pred, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(f.metrics.flat_classification_report(Y_crf, pred, labels=sorted_labels, digits=3))
        print("")


print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))
