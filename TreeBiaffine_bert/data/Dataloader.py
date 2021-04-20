from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable


def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r', encoding="utf8") as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab, eval):
    for sentence in sentences:
        if eval:
            yield sentence2id_eval(sentence, vocab)
        else:
            yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for dep in sentence:
        word = dep.form
        trans_word = dep.org_form
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        relid = vocab.rel2id(dep.rel)
        result.append([word, trans_word, tagid, head, relid])

    return result

def sentence2id_eval(sentence, vocab):
    result = []
    for dep in sentence:
        word = dep.form
        trans_word = dep.org_form
        tagid = vocab.tag2id(dep.tag)
        # head = dep.head
        # relid = vocab.rel2id(dep.rel)
        head = -1
        relid = -1
        result.append([word, trans_word, tagid, head, relid])

    return result



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=False):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab, eval=False):
    length = len(batch[0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b]) > length: length = len(batch[b])

    words = [[] for i in range(len(batch))]
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []

    b = 0
    for sentence in sentences_numberize(batch, vocab, eval):
        index = 0
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        for dep in sentence:
            words[b].append(dep[1])
            if dep[2] == None:
                dep[2] = 0
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            masks[b, index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)
    return words, tags, heads, rels, lengths, masks

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx])))
        yield sentence


def batch_pretrain_variable(batch, tokenizer):
    batch_size = len(batch)
    max_bert_len = -1
    max_sent_len = max([len(sent) for sent in batch])
    #if config.max_sent_len < max_sent_len:max_sent_len = config.max_sent_len
    batch_bert_indices = []
    batch_semgents_ids = []
    batch_piece_ids = []
    for sent in batch:
        bert_indice, segments_id, piece_id = tokenizer.bert_ids(' '.join(sent))
        batch_bert_indices.append(bert_indice)
        batch_semgents_ids.append(segments_id)
        batch_piece_ids.append(piece_id)
        assert len(piece_id) == len(sent)
        assert len(bert_indice) == len(segments_id)
        bert_len = len(bert_indice)
        if bert_len > max_bert_len: max_bert_len = bert_len
    bert_indice_input = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_mask = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_segments_ids = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_piece_ids = np.zeros((batch_size, max_sent_len, max_bert_len), dtype=float)

    for idx in range(batch_size):
        bert_indice = batch_bert_indices[idx]
        segments_id = batch_semgents_ids[idx]
        bert_len = len(bert_indice)
        piece_id = batch_piece_ids[idx]
        sent_len = len(piece_id)
        assert sent_len <= bert_len
        for idz in range(bert_len):
            bert_indice_input[idx, idz] = bert_indice[idz]
            bert_segments_ids[idx, idz] = segments_id[idz]
            bert_mask[idx, idz] = 1
        for idz in range(sent_len):
            for sid, piece in enumerate(piece_id):
                avg_score = 1.0 / (len(piece))
                for tid in piece:
                    bert_piece_ids[idx, sid, tid] = avg_score


    bert_indice_input = torch.from_numpy(bert_indice_input).type(torch.LongTensor)
    bert_segments_ids = torch.from_numpy(bert_segments_ids).type(torch.LongTensor)
    bert_piece_ids = torch.from_numpy(bert_piece_ids).type(torch.FloatTensor)
    bert_mask = torch.from_numpy(bert_mask)

    return bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask


