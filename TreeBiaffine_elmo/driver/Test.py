import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from pathlib import Path
from driver.Config import *
from driver.Model import *
from driver.Parser import *
import pickle
from data.Dataloader import *
from data.Evaluate import *
from driver.Optimizer import *


def evaluate(gold_insts, parser, vocab, outputFile):
    start = time.time()
    parser.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
    predict_insts = []

    for onebatch in data_iter(gold_insts, config.test_batch_size, False):
        words, tags, heads, rels, lengths, masks = batch_data_variable(onebatch, vocab, eval=True)
        count = 0
        arcs_batch, rels_batch = parser.parse(words, tags, lengths, masks)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(onebatch[count], tree)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

            predict_insts.append(tree)

    target_result, opinion_result, target_polar_result, target_opinion_pair_result = eval_target_opinion(gold_insts, predict_insts)
    print("Target:")
    target_result.print()
    print("Opinion:")
    opinion_result.print()
    print("Target-Polar:")
    target_polar_result.print()
    print("(Target, Opinion):")
    target_opinion_pair_result.print()
    eval_result = eval_triplets(gold_insts, predict_insts)

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        gold_triplets_set = parse_dependency(g_inst)
        predict_triplets_set = parse_dependency(p_inst)
        print('------', file=output)
        print('gold triplets', file=output)
        print(*gold_triplets_set, file=output)
        print('predict triplets', file=output)
        print(*predict_triplets_set, file=output)

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(gold_insts), during_time))
    print("Result: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
          (arc_correct_test, arc_total_test, uas, rel_correct_test, arc_total_test, las))
    return eval_result


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creatVocab(config.train_file, config.min_occur_count)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
    print(model)

    if Path(args.model).is_file():  # 指定的文件存在
        model.load_state_dict(torch.load(args.model))
        print('loaded model:' + args.model)
    parser = BiaffineParser(model, vocab.ROOT)
    print(parser)

    train_data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    print("Dev:")
    dev_seg_eval = evaluate(dev_data, parser, vocab, config.data_dir + '.dev.instance')
    print("Dev Triplet:")
    dev_seg_eval.print()
    print("Test:")
    test_seg_eval = evaluate(test_data, parser, vocab, config.data_dir + '.test.instance')
    print("Test Triplet:")
    test_seg_eval.print()

    del parser
