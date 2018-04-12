import classificador_sentenca_embeddings as c
from collections import defaultdict
from sklearn.pipeline import Pipeline
import time


# fonte: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec)))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return c.np.array([
            c.np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [c.np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec)))

    def fit(self, X, y):
        tfidf = c.TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return c.np.array([
                c.np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [c.np.zeros(self.dim)], axis=0)
                for words in X
            ])


def main():
    cross_val = 10

    corpora = ['corpus/output366.json', 'corpus/output466.json', 'corpus/output832.json']

    model_name = 'cbow_s50Copia.txt'

    with open(model_name, "rb") as lines:
        w2v = {line.split()[0]: c.np.array(map(float, line.split()[1:]))
               for line in lines}

    for corpus in corpora:
        print("")
        print("lendo corpus ", corpus)
        _, _, data, labels, _ = c.loadFromJson(corpus)
        X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = c.abstracts_to_sentences(data, labels)

        print("SVM Linear WE")
        clf = Pipeline([
            ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
            ("extra trees", c.SVC(kernel='linear'))])
        pred = c.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(c.classification_report(Y_sentences, pred))
        print(c.confusion_matrix(Y_sentences, pred))
        print("")

        print("SVM Linear WE+TFIDF")
        clf = Pipeline([
            ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
            ("extra trees", c.SVC(kernel='linear'))])
        pred = c.cross_val_predict(clf, X_sentences, Y_sentences, cv=cross_val)
        print("Classification_report:")
        print(c.classification_report(Y_sentences, pred))
        print(c.confusion_matrix(Y_sentences, pred))
        print("")


print(time.asctime(time.localtime(time.time())))
main()
print(time.asctime(time.localtime(time.time())))