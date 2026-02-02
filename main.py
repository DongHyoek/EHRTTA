import os
import torch
import random
import numpy as np
import argparse

from utils.dataloader import *
from utils.trainer import *
from utils.inference import *

def build_parser():
    parser = argparse.ArgumentParser(description='EHRTTA')
    parser.add_argument('--expt_name', type=str, default='test_lamaml',
                    help='name of the experiment')
    
    # LLM model details
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B',
                        help='define the llm model type to use')
    parser.add_argument('--peft_type', type=str, default='dora',
                        help='define the peft algorithm')
    parser.add_argument('--task', type=str, default='classification', 
                        help='task for training', choices = ['classification', 'regression'])
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

    # TTA methods details
    parser.add_argument('--statsmode', type=str, default='aggregate',
                        help='define the method for collect mean/std values', choices = ["aggregate", "distribution"])
    parser.add_argument('--selectmode', type=str, default='mean_of_dist',
                        help='define the method for using distribution of statistics')
    parser.add_argument('--use_tta', default=True, action='store_true',
                        help='select using tta or not')
    
    # optimizer parameters influencing all models
    parser.add_argument('--use_loss_weight', default=False , action='store_true',
                        help='use second order MAML updates')
    parser.add_argument("--glances", default=1, type=int,
                        help="Number of times the model is allowed to train over a set of samples in the single pass setting") 
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all ' +
                        'experiments). Variable name is from GEM project.')
    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay.')
    parser.add_argument('--memories', type=int, default=5120, 
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (For baselines)')

    
    # experiment parameters
    parser.add_argument('--cuda', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--eval_mode', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--log_every', type=int, default=1000,
                        help='frequency of checking the validation accuracy, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--tf_dir', type=str, default='',
                        help='(not set by user)')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')

    # data parameters
    parser.add_argument('--data_path', default='/Users/korea/EHRTTA/data',
                        help='path where data is located')
    parser.add_argument('--var_info_path', default='/Users/korea/EHRTTA/data/concept-dict.json',
                        help='path where variable information is located')
    parser.add_argument('--data_source', type=str, default='miiv',
                        help='the source of the data')
    parser.add_argument('--task_label', type=str, default='mortality_inunit',
                        help='the name of downstream task')
    parser.add_argument('--use_pad', default=False, action='store_true',
                        help='present tasks in order')
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
    
    
    parser.add_argument('--iterations', type=int, default=5000,
                        help='number of classes in every batch')
    parser.add_argument("--dataset", default="mnist_rotations", type=str,
                    help="Dataset to train and test on.")
    parser.add_argument("--workers", default=0, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-order", "--class_order", default="old", type=str,
                        help="define classes order of increment ",
                        choices = ["random", "chrono", "old", "super"])
    parser.add_argument("-inc", "--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument('--test_batch_size', type=int, default=100000 ,
                        help='batch size to use during testing.')


    # La-MAML parameters
    parser.add_argument('--opt_lr', type=float, default=1e-1,
                        help='learning rate for LRs')
    parser.add_argument('--opt_wt', type=float, default=1e-1,
                        help='learning rate for weights')
    parser.add_argument('--alpha_init', type=float, default=1e-3,
                        help='initialization for the LRs')
    parser.add_argument('--learn_lr', default=False, action='store_true',
                        help='model should update the LRs during learning')
    parser.add_argument('--sync_update', default=False , action='store_true',
                        help='the LRs and weights should be updated synchronously')

    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')
    parser.add_argument("--cifar_batches", default=3, type=int,
                        help="Number of batches in inner trajectory") 
    parser.add_argument('--use_old_task_memory', default=False, action='store_true', 
                        help='Use only old task samples for replay buffer data')    
    parser.add_argument('--second_order', default=False , action='store_true',
                        help='use second order MAML updates')


   # memory parameters for GEM | AGEM | ICARL 
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--steps_per_sample', default=1, type=int,
                        help='training steps per batch')


    # parameters specific to MER 
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma learning rate parameter')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta learning rate parameter')
    parser.add_argument('--s', type=float, default=1,
                        help='current example learning rate multiplier (s)')
    parser.add_argument('--batches_per_example', type=float, default=1,
                        help='the number of batch per incoming example')


    # parameters specific to Meta-BGD
    parser.add_argument('--bgd_optimizer', type=str, default="bgd", choices=["adam", "adagrad", "bgd", "sgd"],
                    help='Optimizer.')
    parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                        help='Optimizer parameters')

    parser.add_argument('--train_mc_iters', default=5, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--std_init', default=5e-2, type=float,
                        help='STD init value (default 5e-2)')
    parser.add_argument('--mean_eta', default=1, type=float,
                        help='Eta for mean step (default 1)')
    parser.add_argument('--fisher_gamma', default=0.95, type=float,
                        help='')

    return parser

def fix_seed(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 



if __name__ == "__main__":

    args = build_parser()

    # fix seed
    fix_seed(args.seed)

    # build dataset
    if not args.eval_mode:
        trn_loader, val_loader, tnt_loader = build_loaders(args)

