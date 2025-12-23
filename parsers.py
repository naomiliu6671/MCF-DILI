import argparse


def get_parser():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--seed', type=int, default=42, help='random seed')
    # parsers.add_argument('--save_dir', type=str, default=f'./outputs/{current_time}', help='save directory')
    parsers.add_argument('--device', type=int, default=1, help='device')
    parsers.add_argument('--num_threads', type=int, default=5, help='num threads')
    parsers.add_argument('--train_dataset', type=str, default='tox21', help='datasets name')     # tox21
    parsers.add_argument('--test_dataset', type=str, default='DILIconcern', help='datasets name')
    parsers.add_argument('--data_dir', type=str, default='./datasets', help='data directory')
    parsers.add_argument('--preload_train_data', type=bool, default=True, help='preload train data')
    parsers.add_argument('--preload_data_path', type=str, default='./datasets/preload_data', help='preload data path')

    parsers.add_argument('--in_channels', type=int, default=7, help='input channels')
    parsers.add_argument('--hidden_channels', type=int, default=512, help='hidden channels')
    parsers.add_argument('--embedding_size', type=int, default=128, help='embedding size')
    parsers.add_argument('--out_channels', type=int, default=32, help='output channels')
    parsers.add_argument('--fp_channels', type=int, default=2215, help='output channels')
    parsers.add_argument('--sequence_channels', type=int, default=128, help='output channels')
    parsers.add_argument('--concern_channels', type=int, default=128, help='output channels')
    parsers.add_argument('--lr_pretrain', type=float, default=1e-3, help='learning rate')
    parsers.add_argument('--lr_finetune', type=float, default=1e-5, help='learning rate')
    parsers.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parsers.add_argument('--contrastive_weight', type=float, default=0.1, help='weight decay')
    parsers.add_argument('--train_tasks', type=int, default=12, help='number of tasks')
    parsers.add_argument('--test_tasks', type=int, default=3, help='number of tasks')
    parsers.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parsers.add_argument('--patience', type=int, default=20, help='patience')
    parsers.add_argument('--train_epochs', type=int, default=300, help='weight decay')
    parsers.add_argument('--fine_tune_epochs', type=int, default=1000, help='weight decay')
    parsers.add_argument('--fine_tune_patience', type=int, default=50, help='fine tune patience')
    parsers.add_argument('--batch_size', type=int, default=8, help='batch size')
    parsers.add_argument('--mean_times', type=int, default=5, help='mean times')
    parsers.add_argument('--train_shot', type=int, default=4000, help='number of shots')
    # parsers.add_argument('--train_n_query', type=int, default=50, help='number of queries')
    parsers.add_argument('--test_n_shot', type=int, default=20, help='number of shots')
    parsers.add_argument('--test_n_query', type=int, default=80, help='number of queries')
    parsers.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parsers.add_argument('--max_length', type=int, default=256, help='max length')
    parsers.add_argument('--batch_task', type=int, default=4, help='batch task')
    parsers.add_argument('--head', type=int, default=4, help='number of heads')
    parsers.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parsers.add_argument('--num_inner_steps', type=int, default=1, help='number of inner steps')

    parsers.add_argument('--pretrain_model_save_path', type=str, default='output', help='save path of model')
    parsers.add_argument('--model_save_path', type=str, default='output', help='save path of model')
    parsers.add_argument('--model_load_path', type=str, default='output', help='load path of model')
    parsers.add_argument('--result_path', type=str, default='output', help='load path of model')

    args = parsers.parse_args()

    return args


def write_record(message, path, current_time):
    fw = open(f'{path}record_{current_time}.txt', 'a')
    fw.write(f'{message}\n')
    fw.close()
