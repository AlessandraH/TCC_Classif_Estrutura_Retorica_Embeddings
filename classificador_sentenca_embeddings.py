# -*- coding: utf-8 -*-
from scipy.linalg._expm_frechet import vec

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


# def sent_we(sentence, model, model_size, vocabulary):
#     sentence_feature = [0] * model_size
#     for word in sentence:
#         if len(word) > 2 and word in vocabulary:
#             word_feature = model[word]
#             sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
#     return f.np.array(sentence_feature)


def sent2features(abstract, i, tfidf, tfidf_prev, tfidf_next, pos, we):
    label = abstract[i][0]
    sentence = abstract[i][1]

    features = {
        # 'word_embeddings': f.np.sum(we[i]),
        'tfidf': f.np.sum(tfidf[i]),
        'tfidf_prev': f.np.sum(tfidf_prev[i]),
        'tfidf_next': f.np.sum(tfidf_next[i]),
        'posicao': pos[i],
        'label': label,
    }
    if pos[i] == 0:
        features['boa'] = True
    if pos[i+1] == 0:
        features['eoa'] = True

    return features


def abstract2features(abst, tfidf, tfidf_prev, tfidf_next, pos, we):
    return [sent2features(abst, i, tfidf, tfidf_prev, tfidf_next, pos, we)
            for i in range(len(abst))]


def abstract2labels(abst):
    return [label for label, sentence in abst]


def classificador():
    cross_val = 10

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

        ind = int(round(len(abstracts)*porcent, 0))
        train_abstracts = abstracts[:ind]
        test_abstracts = abstracts[ind:]
        x_train = X_sentences[:ind]
        x_test = X_sentences[ind:]
        x_train_prev = X_prev[:ind]
        x_test_prev = X_prev[ind:]
        x_train_next = X_next[:ind]
        x_test_next = X_next[ind:]
        x_train_pos = X_pos[:ind]
        x_test_pos = X_pos[ind:]
        y_train = Y_sentences[:ind]

        print("Extraindo caracteristicas")
        # X_sentences_we = extract_features_we(X_sentences, model, model_size, vocabulary)
        x_train_we = extract_features_we(x_train, model, model_size, vocabulary)
        x_test_we = extract_features_we(x_test, model, model_size, vocabulary)

        #################################################################################################
        # - - - - - - - - - - - - - - Combinando embeddings com tfidf") - - - - - - - - - - - - - - - - #
        #################################################################################################
        print("Aplicando tfidf")
        vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
        # X_sentences = vectorizer.fit_transform(X_sentences)
        # X_prev = vectorizer.transform(X_prev)
        # X_next = vectorizer.transform(X_next)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        x_train_prev = vectorizer.transform(x_train_prev)
        x_test_prev = vectorizer.transform(x_test_prev)
        x_train_next = vectorizer.transform(x_train_next)
        x_test_next = vectorizer.transform(x_test_next)

        print(x_train.shape)
        print("Aplicando chi-quadrado")
        selector = f.SelectKBest(f.chi2, k=kchi)
        # X_sentences = selector.fit_transform(X_sentences, Y_sentences)
        # X_prev = selector.transform(X_prev)
        # X_next = selector.transform(X_next)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        x_train_prev = selector.transform(x_train_prev)
        x_test_prev = selector.transform(x_test_prev)
        x_train_next = selector.transform(x_train_next)
        x_test_next = selector.transform(x_test_next)

        # print("Adicionando anterior e posterior")
        # X_sentences = f.hstack([X_sentences_we, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
        # X_sentences = X_sentences.todense()
        # # x_train = f.hstack([x_train_we, x_train, x_train_prev, x_train_next, f.np.expand_dims(f.np.array(x_train_pos), -1)])
        # # x_test = f.hstack([x_test_we, x_test, x_test_prev, x_test_next, f.np.expand_dims(f.np.array(x_test_pos), -1)])
        # # x_train = x_train.todense()
        # # x_test = x_test.todense()
        #################################################################################################

        # print("SVM RBF")
        # clf = f.SVC(kernel='rbf')
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))
        # print("")
        #
        # print("SVM linear")
        # clf = f.SVC(kernel='linear')
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))
        # print("")
        #
        # print("KNN")
        # clf = f.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))
        # print("")
        #
        # print("NB")
        # # clf = f.MultinomialNB()
        # clf = f.GaussianNB()
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))
        # print("")
        #
        # print("DT")
        # clf = f.DecisionTreeClassifier(random_state=0)
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))

        print("CRF")
        x_train_pos.append(0)
        x_test_pos.append(0)
        print(len(train_abstracts))
        print(x_train.shape)
        print(x_train_prev.shape)
        print(x_train_next.shape)
        print(x_train_we)
        x_train = [abstract2features(a, x_train, x_train_prev, x_train_next, x_train_pos, x_train_we)
                   for a in train_abstracts]
        x_test = [abstract2features(a, x_test, x_test_prev, x_test_next, x_test_pos, x_test_we)
                  for a in test_abstracts]
        y_train = [abstract2labels(a) for a in train_abstracts]
        y_test = [abstract2labels(a) for a in test_abstracts]
        clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                     max_iterations=100, all_possible_transitions=True)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        print("Classification_report:")
        labels = list(clf.classes_)
        f.metrics.flat_f1_score(y_test, pred, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(f.metrics.flat_classification_report(Y_sentences, pred, labels=sorted_labels, digits=3))


# reload(sys)
# sys.setdefaultencoding('utf8')
print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))