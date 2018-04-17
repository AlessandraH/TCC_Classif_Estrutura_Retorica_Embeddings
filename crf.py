# -*- coding: utf-8 -*-
import functions as f


def sent_we(sentence, model, model_size, vocabulary):
    sentence_feature = [0] * model_size
    for word in sentence:
        if len(word) > 2 and word in vocabulary:
            word_feature = model[word]
            sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
    return f.np.array(sentence_feature)


def sent_tfidf(abstract, i, ngrama, kchi, fittransf):
    label = abstract[i][0]
    sentence = abstract[i][1]
    vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
    selector = f.SelectKBest(f.chi2, k=kchi)
    if fittransf:
        sentence = vectorizer.fit_transform(sentence)
        sentence = selector.fit_transform(sentence)
    else:
        sentence = vectorizer.transform(sentence)
        sentence = selector.transform(sentence)
    return sentence


def abstract_tfidf(abstract, ngrama, kchi, fittransf):
    return [sent_tfidf(abstract, i, ngrama, kchi, fittransf) for i in range(len(abstract))]


def sent2features(abstract, i, tfidf, tfidf_prev, tfidf_next, pos, model, model_size, vocabulary):
    label = abstract[i][0]
    sentence = abstract[i][1]
    features = {
        'word_embeddings': f.np.sum(sent_we(sentence, model, model_size, vocabulary)),
        'tfidf': f.np.sum(tfidf),
        'tfidf_prev': f.np.sum(tfidf_prev),
        'tfidf_next': f.np.sum(tfidf_next),
        'posicao': pos[i],
        'label': label,
    }

    if i == 0:
        features['boa'] = True
    elif i > 0:
        features['eoa'] = True

    return features


def abstract2features(abst, tfidf, tfidf_prev, tfidf_next, pos, model, model_size, vocabulary):
    return [sent2features(abst, i, tfidf, tfidf_prev, tfidf_next, pos, model, model_size, vocabulary)
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

        ind = int(round(len(abstracts) * porcent, 0))
        train_abstracts = abstracts[:ind]
        test_abstracts = abstracts[ind:]
        # x_train = X_sentences[:ind]
        # x_test = X_sentences[ind:]
        # x_train_prev = X_prev[:ind]
        # x_test_prev = X_prev[ind:]
        # x_train_next = X_next[:ind]
        # x_test_next = X_next[ind:]
        x_train_pos = X_pos[:ind]
        x_test_pos = X_pos[ind:]
        # y_train = Y_sentences[:ind]

        print("CRF")
        x_train = [abstract_tfidf(a, ngrama, kchi, True) for a in train_abstracts]
        x_train_pos.append(0)
        x_test_pos.append(0)
        x_train = [abstract2features(a, x_train, x_train_prev, x_train_next, x_train_pos, model, model_size, vocabulary)
                   for a in train_abstracts]
        x_test = [abstract2features(a, x_test, x_test_prev, x_test_next, x_test_pos, model, model_size, vocabulary)
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


print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))