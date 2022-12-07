# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''
import torch 
import numpy as np 
import os 
from Log import MyLog
import gc 
from dataset import SampleDataset
from evaluate import * 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from model import COCML, HarCML
from utils import *
from torch.optim import Adam, SGD, Adagrad, lr_scheduler
import os.path as osp 

from scipy import sparse
from scipy.sparse import coo_matrix, dok_matrix
from collections import defaultdict

torch.cuda.empty_cache()
gc.collect()

SUPPORT_MODEL = {
        'COCML': COCML,
        'HarCML': HarCML
    }

def test(model, logger, val_users, evaluators, top_rec_ks, epoch=0):
    
    if not isinstance(top_rec_ks, list):
        top_rec_ks = list(top_rec_ks)
    
    all_results = defaultdict(dict)

    with torch.no_grad():
        model.eval()

        all_results.setdefault('pre', {})
        all_results.setdefault('rec', {})
        all_results.setdefault('ndcg', {})
        all_results.setdefault('dp', {})
        for k in top_rec_ks:

            p_k, r_k, n_k, dp_k = evaluators.max_sum_dispersion(model, val_users, k)
            logger.info("Epoch: {}, precision@{}: {}, recall@{}: {}, ndcg@{}: {}, dispersion@{}: {}".format(epoch, 
                                                                                         k, 
                                                                                         p_k, 
                                                                                         k, 
                                                                                         r_k, 
                                                                                         k, 
                                                                                         n_k,
                                                                                         k,
                                                                                         dp_k))

            all_results['pre'][str(k)] = p_k
            all_results['rec'][str(k)] = r_k
            all_results['ndcg'][str(k)] = n_k
            all_results['dp'][str(k)] = dp_k
            
        _map, _mrr, _auc, _ndcg = evaluators.map_mrr_auc_ndcg(model, val_users)
        logger.info("Epoch: {}, MAP: {}, MRR: {}, AUC: {}, NDCG: {}".format(epoch, _map, _mrr, 
                                                                            _auc, _ndcg))

    return all_results

if __name__ == '__main__':

    args = parse_args()
    set_seeds(args.random_seed)
    
    save_path = os.path.join(args.data_path, 
                             args.model,
                             args.sampling_strategy, 
                             'best',
                             'per_k_{}'.format(args.per_user_k), 
                             'margin_{}'.format(args.margin))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    log_path = '_'.join([
        'lr_{}'.format(args.lr),
        'max_norm_{}'.format(args.max_norm),
        'num_negs_{}'.format(args.num_negs),
        'optim_{}'.format(args.optimizer),
        'reg_{}'.format(args.reg),
        'dim_{}'.format(args.dim),
        'm1_{}'.format(args.m1),
        'm2_{}'.format(args.m2)
    ]) 

    cur_log = MyLog(os.path.join(save_path, log_path + '.log'), log_path + '.pth')

    cur_log.info(args)

    # with open(os.path.join(args.data_path, args.save_path, 'args.txt'), 'w') as f:
    #     f.write('\n'.join([str(k) + ' = ' + str(v) for k, v in sorted(vars(args).items(), key= lambda x: x[0])]))

    if os.path.exists(os.path.join(args.data_path, "np_data")):
        print("load saved data....")
        user_train_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'))
        user_val_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'))
        user_test_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'))

        user_train_matrix = dok_matrix(user_train_matrix)
        user_val_matrix = dok_matrix(user_val_matrix)
        user_test_matrix = dok_matrix(user_test_matrix)

        num_users, num_items = user_train_matrix.shape
        cur_log.info("number of users: {}".format(num_users))
        cur_log.info("number of items: {}".format(num_items))

    else:
        os.makedirs(os.path.join(args.data_path, "np_data"))
        # load data
        user_item_matrix, num_users, num_items = load_data(args, cur_log, data_name='users.dat', threholds=5)

        # split train/val/test and calculate prob 
        user_train_matrix, user_val_matrix, user_test_matrix = split_train_val_test(user_item_matrix, args, cur_log)

        # save data 
        print('saving splited data')
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'), coo_matrix(user_train_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'), coo_matrix(user_val_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'), coo_matrix(user_test_matrix))

    # # load data
    # user_item_matrix, num_users, num_items = load_data(args, cur_log, data_name='users.dat', threholds=5)

    # # split train/val/test
    # user_train_matrix, user_val_matrix, user_test_matrix = split_train_val_test(user_item_matrix, args, cur_log)

    train_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_train_matrix)
    val_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_val_matrix)
    test_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_test_matrix)

    try:
        model = SUPPORT_MODEL[args.model](args, 
                    num_users,
                    num_items,
                    args.margin).cuda()
    except KeyError as e:
        raise e('Do not support model {}'.format(args.model))
    
    
    cur_log.info("Evaluate Model...")
    cur_log.info("Loading Saved Model From {}".format(cur_log.best_model_path))
        
    best_model_path = cur_log.best_model_path
    if not os.path.exists(best_model_path):
        raise ValueError('saved model are not exist at %s' % best_model_path)

    model.load_state_dict(torch.load(best_model_path))

    model.eval()

    torch.cuda.empty_cache()
    gc.collect()

    val_users = np.asarray([i for i in range(model.num_users)])
    if 'ml-10m' in args.data_path.strip().split('/'):
        cur_log.info("======> load test users from: ml_10m_test_users.npy")
        val_users = np.load('ml_10m_test_users.npy')
    all_results = test(model, cur_log, val_users, test_evaluator, [3, 5, 10, 20, 30, 50, 100, 200])

    import pandas as pd 

    all_results = pd.DataFrame(all_results)

    print(all_results.head())

    pd_path = os.path.join(args.data_path, 
                             args.model,
                             args.sampling_strategy, 
                             'per_k_{}'.format(args.per_user_k))
    
    all_results.to_csv(osp.join(pd_path, 'all_dp_results_{}_reg_{}_m1_{}_m2_{}.csv'.format(args.model, args.reg, args.m1, args.m2)))

    
