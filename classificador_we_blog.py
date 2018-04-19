# -*- coding: utf-8 -*-
import functions as f


def main():
    cross_val = 10

    corpora = ['corpus/output366.json', 'corpus/output466.json', 'corpus/output832.json']

    model_name = 'cbow_s50.txt'
    # model_name = 'cbow_s100.txt'
    # model_name = 'cbow_s300.txt'
    # model_name = 'cbow_s600.txt'
    # model_name = 'cbow_s1000.txt'

    with open(model_name, "rb") as lines:
        w2v = {line.split()[0]: f.np.array(map(float, line.split()[1:]))
               for line in lines}

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        _, _, data, labels, _ = f.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

        mev = f.MeanEmbeddingVectorizer(w2v)
        mev = mev.transform(X_sentences)

        print("SVM Linear WE")
        # clf = Pipeline([
        #     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        #     ("extra trees", c.SVC(kernel='linear'))])
        # clf = clf.transform(X_sentences)
        clf = f.SVC(kernel='linear')
        clf = clf.fit(mev, Y_sentences)
        pred = f.cross_val_predict(clf, mev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("K-NN WE")
        clf = f.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
        clf = clf.fit(mev, Y_sentences)
        pred = f.cross_val_predict(clf, mev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("NB WE")
        clf = f.GaussianNB()
        print(mev.shape)
        clf = clf.fit(mev, Y_sentences)
        pred = f.cross_val_predict(clf, mev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("DT WE")
        clf = f.DecisionTreeClassifier(random_state=0)
        print(mev.shape)
        clf = clf.fit(mev, Y_sentences)
        pred = f.cross_val_predict(clf, mev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        tev = f.TfidfEmbeddingVectorizer(w2v)
        tev = tev.fit(X_sentences)
        tev = tev.transform(X_sentences)

        print("SVM Linear WE+TFIDF")
        # clf = Pipeline([
        #     ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        #     ("extra trees", c.SVC(kernel='linear')),
        #     ("selector", selector)])
        clf = f.SVC(kernel='linear')
        clf = clf.fit(tev, Y_sentences)
        pred = f.cross_val_predict(clf, tev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("K-NN WE+TFIDF")
        clf = f.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
        clf = clf.fit(tev, Y_sentences)
        pred = f.cross_val_predict(clf, tev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("NB WE+TFIDF")
        clf = f.GaussianNB()
        clf = clf.fit(tev, Y_sentences)
        pred = f.cross_val_predict(clf, tev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")

        print("DT WE+TFIDF")
        clf = f.DecisionTreeClassifier(random_state=0)
        clf = clf.fit(tev, Y_sentences)
        pred = f.cross_val_predict(clf, tev, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(f.classification_report(Y_sentences, pred))
        print(f.confusion_matrix(Y_sentences, pred))
        print("")


print(f.time.asctime(f.time.localtime(f.time.time())))
main()
print(f.time.asctime(f.time.localtime(f.time.time())))
