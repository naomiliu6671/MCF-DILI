import copy
import os
import pickle
import sys
from datetime import datetime
import torch
import pandas as pd
from dataset import mydataset, get_test_data_sample, DynamicSmilesTokenizer
from model import finetune_step
from parsers import get_parser, write_record
from utils import get_results, evaluate_total, mean_std
from pretrain import pretrain_model


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args, encoder, in_con, current_time):
    batch_size = args.batch_size
    test_n_shot = args.test_n_shot
    total_metrics = pd.DataFrame(columns=['acc', 'auc', 'f1', 'pre'])
    add_word = f'few-shot'
    os.makedirs(f'{args.result_path}/{args.test_dataset}/{add_word}', exist_ok=True)
    record_path = f'{args.result_path}/{args.test_dataset}/{add_word}/'
    for times in range(args.mean_times):
        print(f'Running {times + 1} times ...')
        write_record(f'{str(args)}\nRunning {times + 1} times ...', record_path, current_time)
        torch.set_num_threads(args.num_threads)
        result_path = f'{args.result_path}/{args.test_dataset}/{add_word}/total_metrics_final_{times+1}.csv'
        test_tasks = args.test_tasks
        test_dataset = args.test_dataset

        max_length = args.max_length
        preload_test_data_path = f'{args.preload_data_path}/{test_dataset}_{test_n_shot}.pickle'
        write_record(f'preload_test_data_path: {preload_test_data_path}', record_path, current_time)
        sys.stdout.flush()
        if os.path.exists(preload_test_data_path):
            with open(preload_test_data_path, 'rb') as f:
                data_dicts = pickle.load(f)
            print("load data from pickle")
            test_data_batches = data_dicts['test_data']
            test_task_id_list = data_dicts['test_task']
            query_smiles = data_dicts['query_smiles']
            args.vocab_size = data_dicts['vocab_size']
        else:
            preload_test_s_data = mydataset(args, test_tasks, max_length, test_dataset + '_support')
            preload_test_q_data = mydataset(args, test_tasks, max_length, test_dataset + '_query')
            vocab = DynamicSmilesTokenizer(os.path.join(args.data_dir, 'vocab.txt'), max_length)
            args.vocab_size = vocab.get_vocab_size()
            query_smiles = preload_test_q_data[0].smiles_list
            test_task_id_list = list(range(test_tasks))
            test_task = [i for i in range(test_tasks)]
            test_data_batches = {}
            for task_id in test_task_id_list:
                db = get_test_data_sample(task_id, test_task, preload_test_s_data, preload_test_q_data, test_n_shot,
                                          batch_size)
                test_data_batches[task_id] = db
            data_dicts = {'test_data': test_data_batches, 'test_task': test_task_id_list,
                          'query_smiles': query_smiles, 'vocab_size': args.vocab_size}
            with open(preload_test_data_path, 'wb') as f:
                pickle.dump(data_dicts, f)
                print("dump data to pickle")
        sys.stdout.flush()

        print(f'Fine tune starting ...')
        pred_result, columns = finetune_step(encoder, args, test_data_batches, test_task_id_list, query_smiles, current_time, in_con, times)

        word = f'Fine tune finished ...'
        print(word)
        write_record(word, record_path, current_time)

        total_results = get_results(pred_result, columns)
        evalu_matrix = evaluate_total(total_results[1], total_results[2], total_results[3])
        acc, auc, f1, pre, recall = evalu_matrix['acc'], evalu_matrix['auc'], evalu_matrix['f1'], evalu_matrix['precision'], evalu_matrix['recall']
        total_metrics.loc[times, :] = [acc, auc, f1, pre]
        class_report, conf_matrix, unique_preds, unique_labels = evalu_matrix['class_report'], evalu_matrix['confusion_matrix'], evalu_matrix['unique_preds'], evalu_matrix['unique_labels']
        word = f'Final Test Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Pre: {pre:.4f}, Recall: {recall:.4f}, Class Report: {class_report}, Confusion Matrix: {conf_matrix}, Unique Predictions: {unique_preds}, Unique Labels: {unique_labels}'
        print(word)
        write_record(word, record_path, current_time)
        total_results_ = pd.DataFrame(columns=['smiles', 'pred', 'true'])
        total_results_['smiles'] = total_results[0]
        total_results_['pred'] = total_results[1]
        total_results_['true'] = total_results[3]
        total_results_.to_csv(result_path, index=False)
        write_record(f'final_result_path: {result_path}', record_path, current_time)

    for i in range(len(total_metrics)):
        word = f'time: {i + 1} acc:{total_metrics.at[i, "acc"]:.4f} auc:{total_metrics.at[i, "auc"]:.4f} f1:{total_metrics.at[i, "f1"]:.4f} pre.:{total_metrics.at[i, "pre"]:.4f}'
        print(word)
        write_record(word, record_path, current_time)
    for i in range(len(total_metrics.columns)):
        means = mean_std(total_metrics.iloc[:, i])
        word = f'{total_metrics.columns[i]}: {means}'
        write_record(word, record_path, current_time)
        print(word)
        sys.stdout.flush()


if __name__ == '__main__':
    args = get_parser()
    encoder = pretrain_model(args)
    in_con = [True, True, True, True]

    print(f'====================in_col: {in_con}====================')
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    encoder_ = copy.deepcopy(encoder)
    main(args, encoder_, in_con, current_time)
