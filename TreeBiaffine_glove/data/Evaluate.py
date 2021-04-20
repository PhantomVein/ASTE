from data.Triplet import Triplet
from data.Metric import Metric


def parse_dependency(sentence):
    target = []
    target_id2position = {}
    triplets = []
    for word in sentence:
        if word.rel == 'TARGET':
            target.append([word.id])
            target_id2position[str(word.id)] = len(target) - 1
        elif word.rel == 'CATT':
            if len(target) > 0:
                target[-1].append(word.id)
    for word in sentence:
        if word.rel == 'CATS':
            if len(triplets) > 0:
                triplets[-1].term.append(word.id)
        elif word.rel in Triplet.polar_id2tag:
            target_id = str(word.head)
            if target_id in target_id2position:
                position = target_id2position[str(word.head)]
            else:
                position = len(target) - 1
            if 0 <= position < len(target):
                triplets.append(Triplet(target[position], Triplet.polar_tag2id[word.rel], [word.id], sentence))
    return set(triplets)


def eval_triplets(gold_insts, predict_insts):
    metric = Metric()

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        gold_triplets_set = parse_dependency(g_inst)
        predict_triplets_set = parse_dependency(p_inst)

        predict_num += len(predict_triplets_set)
        gold_num += len(gold_triplets_set)
        correct_num += len(predict_triplets_set & gold_triplets_set)

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num
    return metric


def eval_target_opinion(gold_insts, predict_insts):
    target_metric = Metric()
    opinion_metric = Metric()
    target_polar_metric = Metric()
    target_opinion_pair_metric = Metric()

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        gold_triplets_set = parse_dependency(g_inst)
        gold_target = set([str(triplet.target) for triplet in gold_triplets_set])
        gold_opinion = set([str(triplet.term) for triplet in gold_triplets_set])
        gold_target_polar = set([str(triplet.target)+'-'+str(triplet.polar) for triplet in gold_triplets_set])
        gold_target_opinion_pair = set([str(triplet.target) + '-' + str(triplet.term) for triplet in gold_triplets_set])
        predict_triplets_set = parse_dependency(p_inst)
        predict_target = set([str(triplet.target) for triplet in predict_triplets_set])
        predict_opinion = set([str(triplet.term) for triplet in predict_triplets_set])
        predict_target_polar = set([str(triplet.target) + '-' + str(triplet.polar) for triplet in predict_triplets_set])
        predict_target_opinion_pair = set([str(triplet.target) + '-' + str(triplet.term) for triplet in predict_triplets_set])

        target_metric.predicated_label_count += len(predict_target)
        target_metric.overall_label_count += len(gold_target)
        target_metric.correct_label_count += len(predict_target & gold_target)
        opinion_metric.predicated_label_count += len(predict_opinion)
        opinion_metric.overall_label_count += len(gold_opinion)
        opinion_metric.correct_label_count += len(predict_opinion & gold_opinion)
        target_polar_metric.predicated_label_count += len(predict_target_polar)
        target_polar_metric.overall_label_count += len(gold_target_polar)
        target_polar_metric.correct_label_count += len(predict_target_polar & gold_target_polar)
        target_opinion_pair_metric.predicated_label_count += len(predict_target_opinion_pair)
        target_opinion_pair_metric.overall_label_count += len(gold_target_opinion_pair)
        target_opinion_pair_metric.correct_label_count += len(predict_target_opinion_pair & gold_target_opinion_pair)

    return target_metric, opinion_metric, target_polar_metric, target_opinion_pair_metric


if __name__ == '__main__':
    from data.Dataloader import *
    data = read_corpus('test')
    for sentence in data:
        triplets = parse_dependency(sentence)
        print(*triplets)
