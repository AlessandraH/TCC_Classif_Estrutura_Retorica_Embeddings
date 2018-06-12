from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import functions as f

corpora = ['corpus/output366.json', 'corpus/output466.json', 'corpus/output832.json']
for corpus in corpora:
    print("Reading ", corpus)
    _, _, data, labels, _ = f.loadFromJson(corpus)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

    if corpora == 'corpus/output466.json':
        del X_sentences[206]
        del X_prev[206]
        del X_next[206]
        del X_pos[206]
        del Y_sentences[206]
    elif corpora == 'corpus/output832.json':
        del X_sentences[572]
        del X_prev[572]
        del X_next[572]
        del X_pos[572]
        del Y_sentences[572]

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    selector = f.SelectKBest(f.chi2, k=100)
    X_sentences = vectorizer.fit_transform(X_sentences)
    X_prev = vectorizer.transform(X_prev)
    X_next = vectorizer.transform(X_next)
    X_sentences = selector.fit_transform(X_sentences, Y_sentences)
    X_prev = selector.transform(X_prev)
    X_next = selector.transform(X_next)

    X_train, X_test, y_train, y_test = train_test_split(X_sentences, Y_sentences, test_size=0.1, random_state=0)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # scores = ['precision', 'recall']
    scores = ['f1']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
