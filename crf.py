# -*- coding: utf-8 -*-
import functions as f


def abstracts_prev_next_pos(abstracts):
    prev = []
    next = []
    pos = []
    for abstract in abstracts:
        pos_aux = []
        prev_aux = abstract.copy()
        next_aux = abstract.copy()
        for i in range(len(abstract)):
            pos_aux.append(i)
        prev_aux = prev_aux[:-1]
        prev_aux.insert(0, ["", ""])
        next_aux = next_aux[1:]
        next_aux.insert(-1, ["", ""])
        prev.append(prev_aux)
        next.append(next_aux)
        pos.append(pos_aux)
    return prev, next, pos


def label_abstract(abstract, i):
    return (a for label, a in abstract[i])


def sent_we(sentence, model, model_size, vocabulary):
    sentence_feature = [0] * model_size
    for word in sentence:
        if len(word) > 2 and word in vocabulary:
            word_feature = model[word]
            sentence_feature = list(map(sum, zip(sentence_feature, word_feature)))
    return f.np.array(sentence_feature)


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
    else:
        features['eoa'] = True

    return features


def abstract2features(abst, tfidf, tfidf_prev, tfidf_next, pos, model, model_size, vocabulary):
    return [sent2features(abst, i, tfidf, tfidf_prev, tfidf_next, pos, model, model_size, vocabulary)
            for i in range(len(abst))]


def abstract2labels(abst):
    return [label for label, sentence in abst]


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
        abstracts = f.loadJson(corpus)
        prev, next, pos = abstracts_prev_next_pos(abstracts)
        ind = int(round(len(abstracts) * porcent, 0))
        train_abstracts = abstracts[:ind]
        test_abstracts = abstracts[ind:]
        train_prev = prev[:ind]
        test_prev = prev[ind:]
        train_next = next[:ind]
        test_next = next[ind:]
        train_pos = pos[:ind]
        test_pos = pos[ind:]

        print("Extraindo tfidf e chi2")
        x_train = [label_abstract(train_abstracts, i) for i in range(len(train_abstracts))]
        x_train_prev = [label_abstract(train_prev, i) for i in range(len(train_prev))]
        x_train_next = [label_abstract(train_next, i) for i in range(len(train_next))]
        x_test = [label_abstract(test_abstracts, i) for i in range(len(test_abstracts))]
        x_test_prev = [label_abstract(test_prev, i) for i in range(len(test_prev))]
        x_test_next = [label_abstract(test_next, i) for i in range(len(test_next))]
        y_train_crf = [abstract2labels(a) for a in train_abstracts]
        y_test_crf = [abstract2labels(a) for a in test_abstracts]

        vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
        selector = f.SelectKBest(f.chi2, k=kchi)

        x_train = vectorizer.fit_transform(x_train)
        x_train_prev = vectorizer.transform(x_train_prev)
        x_train_next = vectorizer.transform(x_train_next)
        x_test = vectorizer.transform(x_test)
        x_test_prev = vectorizer.transform(x_test_prev)
        x_test_next = vectorizer.transform(x_test_next)

        print(x_train.shape)
        print(len(y_train_crf))
        x_train = selector.fit_transform(x_train, y_train_crf)
        x_train_prev = selector.transform(x_train_prev)
        x_train_next = selector.transform(x_train_next)
        x_test = selector.transform(x_test)
        x_test_prev = selector.transform(x_test_prev)
        x_test_next = selector.transform(x_test_next)

        print("Gerando features")
        x_train_crf = [abstract2features(a, x_train, x_train_prev, x_train_next, train_pos, model, model_size,
                                         vocabulary) for a in train_abstracts]
        x_test_crf = [abstract2features(a, x_test, x_test_prev, x_test_next, test_pos, model, model_size, vocabulary)
                      for a in test_abstracts]

        print("CRF")
        clf = f.sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                                     max_iterations=100, all_possible_transitions=True)
        clf.fit(x_train_crf, y_train_crf)
        pred = clf.predict(x_test_crf)
        print("Classification_report:")
        labels = list(clf.classes_)
        f.metrics.flat_f1_score(y_test_crf, pred, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(f.metrics.flat_classification_report(y_test_crf, pred, labels=sorted_labels, digits=3))


print(f.time.asctime(f.time.localtime(f.time.time())))
classificador()
print(f.time.asctime(f.time.localtime(f.time.time())))