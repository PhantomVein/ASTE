import pickle
from data.Dependency import *
import hanlp


def construct_dependency(txt, triplet):
    origins, targets, sentiment_terms = txt.split('####')
    words = origins.split()
    word_tag = tagger([words])
    polar_map = ['NEU', 'POS', 'NEG']
    corpus_list = [Dependency(i + 1, word, word_tag[0][i], 0, 'O') for i, word in enumerate(words)]
    for target, term, polar in triplet:
        for i, sequence in enumerate(target):
            if i == 0:
                corpus_list[sequence].rel = 'TARGET'
            else:
                corpus_list[sequence].rel = 'CATT'
                corpus_list[sequence].head = corpus_list[sequence].id - 1
        for i, sequence in enumerate(term):
            if i == 0:
                corpus_list[sequence].rel = polar_map[polar]
                corpus_list[sequence].head = corpus_list[target[0]].id
            else:
                corpus_list[sequence].rel = 'CATS'
                corpus_list[sequence].head = corpus_list[sequence].id - 1
    return corpus_list


if __name__ == '__main__':
    tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    database = ['14lap', '14rest', '15rest', '16rest']
    for dataset in database:
        for partition in ['train', 'dev', 'test']:
            file_path = '../triplet_data/{}/{}_{}.txt'.format(dataset, dataset, partition)
            pkl_file = '../triplet_data/{}/{}_pair/{}_pair.pkl'.format(dataset, dataset, partition)
            write_file = '../dataset/{}/{}.{}'.format(dataset, dataset, partition)

            pair_data = pickle.load(open(pkl_file, 'rb'))
            with open(file_path, mode='r', encoding='utf8') as infile:
                sentences = []
                for info, pairs in zip(infile, pair_data):
                    sentence = construct_dependency(info, pairs)
                    sentences.append(sentence)
                sentences.sort(key=lambda x: len(x), reverse=True)
                writeDepTree(write_file, sentences)


