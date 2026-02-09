import os
import torch
import random
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm

from utils.dataloader import build_loaders
from utils.experiment import train, inference, adaptation

def build_parser():
    
    parser = argparse.ArgumentParser(description='EHRTTA')

    # Embedding modle details
    parser.add_argument('--use_time', default=False, action='store_true',
                        help='define using time embedding')
    parser.add_argument('--te_dropout', type=float, default=0.1,
                        help='the dropout rate of time series embdding modules')
    parser.add_argument('--use_norm_ema', default=False, action='store_true',
                        help='use normalization with ema')
    parser.add_argument('--norm_ema_alpha', type=float, default=0.1,
                        help='define the alpha value for ema')
    parser.add_argument('--align_n_heads', type=int, default=8,
                        help='the number of heads for cross attention')
    parser.add_argument('--align_dropout', type=float, default=0.0,
                        help='the dropout rate of cross attention modules')
    parser.add_argument('--align_return_weights', default=False, action='store_true',
                        help='the option of return attn weights')
    parser.add_argument('--use_align_gate', default=False, action='store_true',
                        help='use gate mechanism in crossattention')

    # LLM model details
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B',
                        help='define the llm model type to use')
    parser.add_argument('--use_dora', default=False, action='store_true',
                        help='option of using dora method')
    parser.add_argument('--rank_dim', type=int, default=8,
                        help='the dimension of the low-rank matrices')
    parser.add_argument('--peft_alpha', type=int, default=16,
                        help='the scaling factor for the low-rank matrices')
    parser.add_argument('--peft_dropout', type=int, default=0.1,
                        help='the dropout probability of the LoRA layers')
    parser.add_argument('--target_modules', type=list, default=['q_proj', 'v_proj'],
                        help='which parameters applied by peft method. please check the parameter name of the backbone model')
    parser.add_argument('--h_pool', type=str, default='mean',
                        help='the pooling method of output hidden vectors in LLM')
    parser.add_argument('--e_pool', type=str, default='mean',
                        help='the pooling method of vectors in time series embedding modules')
    parser.add_argument("--text_pad_type", type=str, default='longest',
                        help="padding type of text data")

    # TTA methods details
    parser.add_argument('--statsmode', type=str, default='aggregate',
                        help='define the method for collect mean/std values', choices = ["aggregate", "distribution"])
    parser.add_argument('--selectmode', type=str, default='mean_of_dist',
                        help='define the method for using distribution of statistics')

    # experiment parameters 
    parser.add_argument('--cuda', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--adapt_mode', default=False , action='store_true',
                        help='adaptation_mode')
    parser.add_argument('--print_iter', type=int, default=100,
                        help='frequency of checking the loss, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--ckpt_dir', type=str, default='./results/checkpoint',
                        help='the directory where the best model checkpoint will be saved')
    parser.add_argument('--metrics_dir', type=str, default='./results/metrics',
                        help='the directory where the metric for the test set will be saved')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')
    
    # optimizer parameters influencing all models
    parser.add_argument('--n_epochs', type=int, default=1, 
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate (For baselines)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay of optimizer')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size for dataloader')
    parser.add_argument('--scheduler', default=False , action='store_true',
                        help='use scheduler for training')
    parser.add_argument('--early_stop', default=False , action='store_true',
                        help='using early stopping')
    parser.add_argument('--patience', type=int, default=10, 
                        help='patience for early stopping')
    parser.add_argument('--loss_type', type=str, default='crossentropy', 
                        help='objective function for training')
    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')

    # data parameters
    parser.add_argument('--data_path', default='/Users/korea/EHRTTA/data',
                        help='path where data is located')
    parser.add_argument('--var_info_path', default='/Users/korea/EHRTTA/data/concept-dict.json',
                        help='path where variable information is located')
    parser.add_argument('--data_source', type=str, default='miiv',
                        help='the source of the data')
    parser.add_argument('--data_target', type=str, default='miiv',
                        help='the target of the data')
    parser.add_argument('--task', type=str, default='classification', 
                        help='task for training classification or regression')
    parser.add_argument('--task_label', type=str, default='mortality_inunit',
                        help='the name of downstream task')
    parser.add_argument('--num_labels', type=int, default=2, 
                        help='the number of labels for task. if regression task set to be 1')
    parser.add_argument('--pid_col', type=str, default='stay_id',
                        help='the name of patient id column')
    parser.add_argument('--time_col', type=str, default='charttime',
                        help='the name of time column')
    parser.add_argument('--var_col', type=str, default='var_name',
                        help='the name of variable column for using time series data')
    parser.add_argument('--text_var_col', type=str, default='full_var_name',
                        help='the name of variable column for using text generation')
    parser.add_argument('--val_col', type=str, default='value',
                        help='the name of offset column')
    parser.add_argument('--unit_col', type=str, default='fixed_unit',
                        help='the name of offset column')
    parser.add_argument("--train_ratio", default=0.8, type=float,
                        help="train split ratio (0. <= x <= 1.).")
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help="validation split ratio (0. <= x <= 1.).")
    parser.add_argument("--use_ts_trunc", default=False, action='store_true',
                        help="option for using time series truncation")
    parser.add_argument("--max_length", default=1000, type=float,
                        help="max sequence length of time series")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument('--n_time_cols', type=int, default=7,
                        help='the number of time series columns')

    return parser

def fix_seed(seed: int = 0):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def save_metrics(args, results, save_dir, type='train_val'):
    df = pd.DataFrame(results)
    df['exp_type'] = type
    df['backbone'] = args.model_id

    df.to_csv(f'{save_dir}/metrics_result.csv', index = False)

if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()

    # fix seed
    fix_seed(args.seed)
    
    print('Build Dataloaders..')
    
    # Set save dir 
    ckpt_dir = f'{args.ckpt_dir}/{args.data_source}/{args.task}_{args.task_label}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    metrics_dir = f'{args.metrics_dir}/{args.data_source}/{args.task}_{args.task_label}/{args.data_target}'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Train mode
    if not args.adapt_mode:
        # build dataset
        trn_loader, val_loader, tnt_loader = build_loaders(args)

        # train & test
        train_result, scaler = train(args, trn_loader, val_loader, ckpt_dir) # training function returns scaler
        test_result = inference(args, scaler, tnt_loader, ckpt_dir)
    
        # result save 
        save_metrics(args, train_result, metrics_dir, 'train_val')
        save_metrics(args, test_result, metrics_dir, 'test')

    # Evaluation mode (Test-time adaptation)
    else:
        # build dataset
        _, _, target_loader = build_loaders(args)

        ## ※ 학습 끝난 이후에 source model statistics들을 가져오거나 미리 저장해두어야 함.  

        adaptation_result = adaptation(args, target_loader, ckpt_dir)

        # result save 
        save_metrics(args, adaptation_result, metrics_dir, 'adaptation')
