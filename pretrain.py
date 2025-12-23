import os
import pickle
from datetime import datetime
import torch
from dataset import mydataset, get_train_data_sample
from model import TotalEncoder, pretrain_step
from parsers import write_record
import sys


def pretrain_model(args):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f'{args.result_path}/{args.train_dataset}/{args.train_shot}', exist_ok=True)
    record_path = f'{args.result_path}/{args.train_dataset}/{args.train_shot}/'
    train_tasks = args.train_tasks
    train_dataset = args.train_dataset
    batch_size = args.batch_size
    train_shot = args.train_shot
    max_length = args.max_length
    preload_train_data_path = f'{args.preload_data_path}/{train_dataset}_{train_shot}.pickle'
    write_record(f'preload_train_data_path: {preload_train_data_path}', record_path, current_time)
    sys.stdout.flush()
    if os.path.exists(preload_train_data_path):
        with open(preload_train_data_path, 'rb') as f:
            data_dicts = pickle.load(f)
        print("load data from pickle")
        train_data_batches = data_dicts['train_data']
        train_task_id_list = data_dicts['train_task']
    else:
        preload_train_data = mydataset(args, train_tasks, max_length, train_dataset)
        train_task_id_list = list(range(train_tasks))
        train_task = [i for i in range(train_tasks)]
        train_data_batches = {}
        for task_id in train_task_id_list:
            db = get_train_data_sample(task_id, train_task, preload_train_data, train_dataset, train_shot, batch_size)
            train_data_batches[task_id] = db
        data_dicts = {'train_data': train_data_batches, 'train_task': train_task_id_list}
        with open(preload_train_data_path, 'wb') as f:
            pickle.dump(data_dicts, f)
            print("dump data to pickle")
    sys.stdout.flush()

    pretrain_path = f'{args.pretrain_model_save_path}/{args.train_dataset}/pretrain_model.joblib'
    write_record(f'Pretrain model path: {pretrain_path}', record_path, current_time)
    encoder = TotalEncoder(args.in_channels, args.hidden_channels, args.embedding_size,
                           args.fp_channels, args.sequence_channels, args.dropout, args.device)
    if os.path.exists(pretrain_path):
        print(f'Pretrain model loading ...')
        write_record(f'Pretrain model loading ...', record_path, current_time)
        state_dict = torch.load(pretrain_path)
        encoder.load_state_dict(state_dict)
    else:
        print(f'Pretrain starting ...')
        write_record(f'Pretrain starting ...', record_path, current_time)
        encoder = pretrain_step(encoder, args, train_data_batches, train_task_id_list, current_time)
        torch.save(encoder.state_dict(), pretrain_path)
    return encoder
