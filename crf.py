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


def sent2features(features, i, label):
    features = {
        'features': features[i],
        'label': label[i],
    }
    return features


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
    ngrama = 1
    kchi = 100

    print("Abrindo modelo embedding")
    model = f.KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
    vocabulary = model.vocab

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        _, _, data, labels, _ = f.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)
        ind = round(len(data) * porcent)
        x_train, x_test, y_train, y_test = (data[ind:], data[:ind], labels[ind:], labels[:ind])
        x_train_sentences, x_train_prev_sentences, x_train_next_sentences, x_train_sentences_pos, y_train_sentences, train_ind = f.abstracts_to_sentences(x_train, y_train)
        x_test_sentences, x_test_prev_sentences, x_test_next_sentences, x_test_sentences_pos, y_test_sentences, test_ind = f.abstracts_to_sentences(x_test, y_test)

        x_train_we = extract_features_we(x_train_sentences, model, model_size, vocabulary)
        x_test_we = extract_features_we(x_test_sentences, model, model_size, vocabulary)

        print("Extraindo tfidf e chi2")
        vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
        selector = f.SelectKBest(f.chi2, k=kchi)

        x_train_sentences = vectorizer.fit_transform(x_train_sentences)
        x_test_sentences = vectorizer.transform(x_test_sentences)
        x_train_prev_sentences = vectorizer.transform(x_train_prev_sentences)
        x_train_next_sentences = vectorizer.transform(x_train_next_sentences)
        x_test_prev_sentences = vectorizer.transform(x_test_prev_sentences)
        x_test_next_sentences = vectorizer.transform(x_test_next_sentences)

        x_train_sentences = selector.fit_transform(x_train_sentences, y_train_sentences)
        x_test_sentences = selector.transform(x_test_sentences)
        x_train_prev_sentences = selector.transform(x_train_prev_sentences)
        x_train_next_sentences = selector.transform(x_train_next_sentences)
        x_test_prev_sentences = selector.transform(x_test_prev_sentences)
        x_test_next_sentences = selector.transform(x_test_next_sentences)

        x_train_sentences = f.hstack([x_train_we, x_train_sentences, x_train_prev_sentences, x_train_next_sentences, f.np.expand_dims(f.np.array(x_train_sentences_pos), -1)])
        x_test_sentences = f.hstack([x_test_we, x_test_sentences, x_test_prev_sentences, x_test_next_sentences, f.np.expand_dims(f.np.array(x_test_sentences_pos), -1)])
        from scipy.sparse import csr_matrix
        x_train_sentences = csr_matrix(x_train_sentences)
        x_test_sentences = csr_matrix(x_test_sentences)
        print("Gerando features")
        print(x_train_sentences.shape)
        print(len(y_train))
        # x_train_crf = [sent2features(x_train_sentences, i, y_train) for i in range(len())]
        # x_test_crf = [sent2features(x_test_sentences, i, y_test) for i in x_test_sentences.shape[0]]

        print("CRF")
        clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                     max_iterations=100, all_possible_transitions=True)
        clf.fit(x_train_sentences, y_train)
        pred = clf.predict(x_test_sentences)
        print("Classification_report:")
        labels = list(clf.classes_)
        f.metrics.flat_f1_score(y_test, pred, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(f.metrics.flat_classification_report(y_test, pred, labels=sorted_labels, digits=3))


print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))