import sys
import json
from gensim.models import Word2Vec


def to_sentences(abstracts, senteces_max=None):
    sentences = []
    labels = []
    abstracts_sentences = []
    abstracts_labels = []
    ids = []

    for id, abstract in enumerate(abstracts):
        if senteces_max and len(abstract) > senteces_max:
            continue

        tmp_sentences = []
        tmp_labels = []

        for label, text in abstract:
            sentences.append(text)
            labels.append(label)

            tmp_sentences.append(text)
            tmp_labels.append(label)
            ids.append(id)

        abstracts_sentences.append(tmp_sentences)
        abstracts_labels.append(tmp_labels)

    assert (len(sentences) == len(labels))
    assert (len(abstracts_sentences) == len(abstracts_labels))

    return sentences, labels, abstracts_sentences, abstracts_labels, ids


def loadFromJson(file):
    data = []
    with open(file, 'r') as f:
        data = json.load(f, encoding='cp1252')

    return to_sentences(data)


def abstracts_to_sentences(abstracts, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []
    abstracts_idx = []

    for i, (sentences_labels, sentences) in enumerate(zip(labels, abstracts)):
        for j, (label, sentence) in enumerate(zip(sentences_labels, sentences)):
            ret.append(sentence)
            ret_pos.append(j)
            ret_labels.append(label)
            abstracts_idx.append(i)

            if j - 1 >= 0:
                ret_prev.append(sentences[j - 1])
            else:
                ret_prev.append("")

            if j + 1 < len(sentences):
                ret_next.append(sentences[j + 1])
            else:
                ret_next.append("")

    return ret, ret_prev, ret_next, ret_pos, ret_labels, abstracts_idx


reload(sys)
sys.setdefaultencoding('utf8')

model_name = 'word2vec_cbow1000.txt'

_, _, data, labels, _ = loadFromJson('corpus/output832.json')

new_data = []
for resumo in data:
    for sentenca in resumo:
        new_data.append(str(sentenca).split())

model = Word2Vec(new_data, size=1000, min_count=1, workers=4, sg=0) # sg: 0 CBOW or 1 Skip-Gram
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['Atualmente'])
# save model
model.save(model_name)
# load model
# new_model = Word2Vec.load(model_name)
# print(new_model)