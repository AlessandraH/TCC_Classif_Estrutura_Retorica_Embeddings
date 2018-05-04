# -*- coding: utf-8 -*-
import functions as f
import warnings
import arff

warnings.filterwarnings("ignore")

corpus366 = 'corpus/output366.json'
azfeat366 = 'azport_features/azfeatures366.arff'
corpus466 = 'corpus/output466.json'
azfeat466 = 'azport_features/azfeatures466.arff'
corpus832 = 'corpus/output832.json'
azfeat832 = 'azport_features/azfeatures832.arff'

# model_name = 'cbow_s50.txt'
model_name = 'cbow_s100.txt'
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

# model_size = 50
model_size = 100
# model_size = 300
# model_size = 600
# model_size = 1000
ngrama = 1
kchi = 100
cross_val = 10

print(f.time.asctime(f.time.localtime(f.time.time())))

print("Abrindo modelo embedding")
model = f.KeyedVectors.load_word2vec_format(fname=model_name, unicode_errors="ignore")
vocabulary = model.vocab

################################### 366 ####################################################
# print("lendo corpus ", corpus366)
abstracts = f.loadJson(corpus366)
_, _, data, labels, _ = f.loadFromJson(corpus366)
X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

X_sentences_wes = f.extract_features_we(X_sentences, model, model_size, vocabulary)
X_sentences_wem = f.extract_features_we_media(X_sentences, model, model_size, vocabulary)

vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
selector = f.SelectKBest(f.chi2, k=kchi)

X_sentences = vectorizer.fit_transform(X_sentences)
X_prev = vectorizer.transform(X_prev)
X_next = vectorizer.transform(X_next)
X_sentences = selector.fit_transform(X_sentences, Y_sentences)
X_prev = selector.transform(X_prev)
X_next = selector.transform(X_next)

dataset = arff.load(open('azport_features/azfeatures366n.arff', 'r'))
X_data = f.np.array(dataset['data'])
azport = X_data[:, :-1].astype(f.np.float)
Y_labels = X_data[:, -1]

all_soma = f.hstack([azport, X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_soma = all_soma.todense()
all_media = f.hstack([azport, X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_media = all_media.todense()
wes_tfidf = f.hstack([X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wes_tfidf = wes_tfidf.todense()
wem_tfidf = f.hstack([X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wem_tfidf = wem_tfidf.todense()
# tfidf = f.hstack([X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# tfidf = tfidf.todense()
# azport_tfidf = f.hstack([azport, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# azport_tfidf = azport_tfidf.todense()
wes_azport = f.np.concatenate((X_sentences_wes, azport), axis=1)
wem_azport = f.np.concatenate((X_sentences_wem, azport), axis=1)

print("ALL FEATURES WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_soma, Y_sentences)
pred = f.cross_val_predict(clf, all_soma, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("ALL FEATURES WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_media, Y_sentences)
pred = f.cross_val_predict(clf, all_media, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(SOMA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wes_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(MEDIA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wem_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

# print("TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")
#
# print("AZPORT+TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(azport_tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, azport_tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")

print("AZPORT+WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_azport, Y_sentences)
pred = f.cross_val_predict(clf, wes_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("AZPORT+WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_azport, Y_sentences)
pred = f.cross_val_predict(clf, wem_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

################################### 466 ####################################################
# print("lendo corpus ", corpus466)
_, _, data, labels, _ = f.loadFromJson(corpus466)
X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

del X_sentences[206]
del X_prev[206]
del X_next[206]
del X_pos[206]
del Y_sentences[206]

X_sentences_wes = f.extract_features_we(X_sentences, model, model_size, vocabulary)
X_sentences_wem = f.extract_features_we_media(X_sentences, model, model_size, vocabulary)

vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
selector = f.SelectKBest(f.chi2, k=kchi)

X_sentences = vectorizer.fit_transform(X_sentences)
X_prev = vectorizer.transform(X_prev)
X_next = vectorizer.transform(X_next)
X_sentences = selector.fit_transform(X_sentences, Y_sentences)
X_prev = selector.transform(X_prev)
X_next = selector.transform(X_next)

dataset = arff.load(open('azport_features/azfeatures466n.arff', 'r'))
X_data = f.np.array(dataset['data'])
azport = X_data[:, :-1].astype(f.np.float)

all_soma = f.hstack([azport, X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_soma = all_soma.todense()
all_media = f.hstack([azport, X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_media = all_media.todense()
wes_tfidf = f.hstack([X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wes_tfidf = wes_tfidf.todense()
wem_tfidf = f.hstack([X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wem_tfidf = wem_tfidf.todense()
# tfidf = f.hstack([X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# tfidf = tfidf.todense()
# azport_tfidf = f.hstack([azport, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# azport_tfidf = azport_tfidf.todense()
wes_azport = f.np.concatenate((X_sentences_wes, azport), axis=1)
wem_azport = f.np.concatenate((X_sentences_wem, azport), axis=1)

print("ALL FEATURES WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_soma, Y_sentences)
pred = f.cross_val_predict(clf, all_soma, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("ALL FEATURES WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_media, Y_sentences)
pred = f.cross_val_predict(clf, all_media, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(SOMA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wes_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(MEDIA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wem_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

# print("TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")
#
# print("AZPORT+TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(azport_tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, azport_tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")

print("AZPORT+WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_azport, Y_sentences)
pred = f.cross_val_predict(clf, wes_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("AZPORT+WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_azport, Y_sentences)
pred = f.cross_val_predict(clf, wem_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")
################################### 832 ####################################################
# print("lendo corpus ", corpus832)
_, _, data, labels, _ = f.loadFromJson(corpus832)
X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = f.abstracts_to_sentences(data, labels)

del X_sentences[572]
del X_prev[572]
del X_next[572]
del X_pos[572]
del Y_sentences[572]

X_sentences_wes = f.extract_features_we(X_sentences, model, model_size, vocabulary)
X_sentences_wem = f.extract_features_we_media(X_sentences, model, model_size, vocabulary)

vectorizer = f.TfidfVectorizer(ngram_range=(1, ngrama))
selector = f.SelectKBest(f.chi2, k=kchi)

X_sentences = vectorizer.fit_transform(X_sentences)
X_prev = vectorizer.transform(X_prev)
X_next = vectorizer.transform(X_next)
X_sentences = selector.fit_transform(X_sentences, Y_sentences)
X_prev = selector.transform(X_prev)
X_next = selector.transform(X_next)

dataset = arff.load(open('azport_features/azfeatures832n.arff', 'r'))
X_data = f.np.array(dataset['data'])
azport = X_data[:, :-1].astype(f.np.float)

all_soma = f.hstack([azport, X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_soma = all_soma.todense()
all_media = f.hstack([azport, X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
all_media = all_media.todense()
wes_tfidf = f.hstack([X_sentences_wes, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wes_tfidf = wes_tfidf.todense()
wem_tfidf = f.hstack([X_sentences_wem, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
wem_tfidf = wem_tfidf.todense()
# tfidf = f.hstack([X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# tfidf = tfidf.todense()
# azport_tfidf = f.hstack([azport, X_sentences, X_prev, X_next, f.np.expand_dims(f.np.array(X_pos), -1)])
# azport_tfidf = azport_tfidf.todense()
wes_azport = f.np.concatenate((X_sentences_wes, azport), axis=1)
wem_azport = f.np.concatenate((X_sentences_wem, azport), axis=1)

print("ALL FEATURES WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_soma, Y_sentences)
pred = f.cross_val_predict(clf, all_soma, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("ALL FEATURES WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(all_media, Y_sentences)
pred = f.cross_val_predict(clf, all_media, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(SOMA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wes_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("WE(MEDIA)+TFIDF")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_tfidf, Y_sentences)
pred = f.cross_val_predict(clf, wem_tfidf, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

# print("TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")
#
# print("AZPORT+TFIDF")
# print("NB Bernoulli")
# clf = f.BernoulliNB()
# clf = clf.fit(azport_tfidf, Y_sentences)
# pred = f.cross_val_predict(clf, azport_tfidf, Y_sentences, cv=cross_val)
# print("Classification_report:")
# print(f.classification_report(Y_sentences, pred))
# print(f.confusion_matrix(Y_sentences, pred))
# print("")

print("AZPORT+WE(SOMA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wes_azport, Y_sentences)
pred = f.cross_val_predict(clf, wes_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print("AZPORT+WE(MEDIA)")
print("NB Bernoulli")
clf = f.BernoulliNB()
clf = clf.fit(wem_azport, Y_sentences)
pred = f.cross_val_predict(clf, wem_azport, Y_sentences, cv=cross_val)
print("Classification_report:")
print(f.classification_report(Y_sentences, pred))
print(f.confusion_matrix(Y_sentences, pred))
print("")

print(f.time.asctime(f.time.localtime(f.time.time())))
