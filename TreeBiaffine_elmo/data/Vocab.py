from collections import Counter
from data.Dependency import *

class Vocab(object):
    PAD, ROOT, UNK = 0, 0, 2
    def __init__(self, word_counter, tag_counter, rel_counter, relroot='root', min_occur_count = 2):
        self._root = relroot
        self._root_form = '<' + relroot.lower() + '>'
        self._id2word = ['<pad>', self._root_form, '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2extword = ['<pad>', self._root_form, '<unk>']
        self._id2tag = ['<pad>', relroot]
        self._id2rel = ['<pad>', relroot]
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for tag, count in tag_counter.most_common():
            if tag != relroot: self._id2tag.append(tag)

        for rel, count in rel_counter.most_common():
            if rel != relroot: self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: POS tags dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #tags %d, #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)


def creatVocab(corpusFile, min_occur_count):
    word_counter = Counter()
    tag_counter = Counter()
    rel_counter = Counter()
    root = 'root'
    with open(corpusFile, 'r', encoding="utf8") as infile:
        for sentence in readDepTree(infile):
            for dep in sentence:
                word_counter[dep.form] += 1
                tag_counter[dep.tag] += 1
                rel_counter[dep.rel] += 1

    return Vocab(word_counter, tag_counter, rel_counter, root, min_occur_count)
