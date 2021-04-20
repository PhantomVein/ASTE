import sys
sys.path.extend(["../../","../","./"])
import time
from pathlib import Path
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
import pickle
from data.Dataloader import *
from data.Evaluate import *
from driver.Optimizer import *
from driver.BertTokenHelper import BertTokenHelper


def train(data, dev_data, test_data, parser, vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)

    global_step = 0
    best_F = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, False):
            words, tags, heads, rels, lengths, masks = \
                batch_data_variable(onebatch, vocab)
            bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = batch_pretrain_variable(words, vocab.tokenizer)
            parser.model.train()

            parser.forward(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, tags, masks)
            loss = parser.compute_loss(heads, rels, lengths)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            arc_correct, label_correct, total_arcs = parser.compute_accuracy(heads, rels)
            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = overall_arc_correct.item() * 100.0 / overall_total_arcs
            las = overall_label_correct.item() * 100.0 / overall_total_arcs
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, loss:%.2f" \
                %(global_step, uas, las, iter, batch_iter, overall_total_arcs, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parser.model.parameters()), max_norm=config.clip)
                optimizer.step()
                parser.model.zero_grad()       
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                dev_seg_eval = evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step))
                print("Dev Triplet:")
                dev_seg_eval.print()
                print("Test:")
                test_seg_eval = evaluate(test_data, parser, vocab, config.test_file + '.' + str(global_step))
                print("Test Triplet:")
                test_seg_eval.print()

                dev_F = dev_seg_eval.getAccuracy()
                if best_F < dev_F:
                    print("Exceed best Full F-score: history = %.4f, current = %.4f" % (best_F, dev_F))
                    best_F = dev_F

                    if config.save_after > 0 and iter > config.save_after:
                        print("save model to ", config.save_model_path + '.' + str(os.getpid()))
                        torch.save(parser.model.state_dict(), config.save_model_path + '.' + str(os.getpid()))
                        print("save ok.")


def evaluate(gold_insts, parser, vocab, outputFile):
    start = time.time()
    parser.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
    predict_insts = []

    for onebatch in data_iter(gold_insts, config.test_batch_size, False):
        words, tags, heads, rels, lengths, masks = batch_data_variable(onebatch, vocab, eval=True)
        bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = batch_pretrain_variable(words,vocab.tokenizer)

        count = 0
        arcs_batch, rels_batch = parser.parse(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, tags, lengths, masks)
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
        print(*gold_triplets_set, '|\t|', *predict_triplets_set, file=output)

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

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--gpu', default='0')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    vocab = creatVocab([config.train_file, config.dev_file, config.test_file], config.min_occur_count)
    vocab.tokenizer = BertTokenHelper(config.bert_dir)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config)

    if Path(args.model).is_file():  # 指定的文件存在
        model.load_state_dict(torch.load(args.model))
        print('loaded model:' + args.model)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
    print(model)

    parser = BiaffineParser(model, vocab.ROOT)
    print(parser)

    train_data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    train(train_data, dev_data, test_data, parser, vocab, config)
