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

    print("Abrindo modelo embedding")
    model = f.KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
    vocabulary = model.vocab

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        _, _, data, labels, _ = f.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

        print("Extraindo caracteristicas")
        X_sentences = extract_features_we(X_sentences, model, model_size, vocabulary)

        #################################################################################################
        # - - - - - - - - - - - - - - Combinando embeddings com tfidf") - - - - - - - - - - - - - - - - #
        #################################################################################################
        # print("Aplicando tfidf")
        # vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
        # X_sentences = vectorizer.fit_transform(X_sentences)
        # X_prev = vectorizer.transform(X_prev)
        # X_next = vectorizer.transform(X_next)
        #
        # print("Aplicando chi-quadrado")
        # selector = f.SelectKBest(chi2, k=kchi)
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
        # X_sentences = f.hstack([X_sentences_we, X_sentences, X_prev, X_next, np.expand_dims(np.array(X_pos), -1)])
        # X_sentences = X_sentences.todense()
        #################################################################################################

        print("SVM RBF")
        clf = f.SVC(kernel='rbf')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("SVM linear")
        clf = f.SVC(kernel='linear')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("KNN")
        clf = f.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
        clf = clf.fit(X_sentences, Y_sentences)
        pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("NB")
        # clf = f.MultinomialNB()
        clf = f.GaussianNB()
        clf = clf.fit(X_sentences, Y_sentences)
        pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("DT")
        clf = f.DecisionTreeClassifier(random_state=0)
        clf = clf.fit(X_sentences, Y_sentences)
        pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))

        # print("CRF")
        # clf = f.sklearn_crfsuite.CRF()
        # X_sentences = f.scipy.sparse.csr_matrix(X_sentences)
        # clf = clf.fit(X_sentences, Y_sentences)
        # pred = f.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        # print("Classification_report:")
        # print(f.classification_report(Y_sentences, pred))
        # print(f.confusion_matrix(Y_sentences, pred))


# reload(sys)
# sys.setdefaultencoding('utf8')
print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))