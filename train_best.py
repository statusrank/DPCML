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

torch.cuda.empty_cache()
gc.collect()

SUPPORT_MODEL = {
        'COCML': COCML,
        'HarCML': HarCML
    }

def test(model, logger, val_users, evaluators, top_rec_ks, epoch=0):
    
    if not isinstance(top_rec_ks, list):
        top_rec_ks = list(top_rec_ks)
    with torch.no_grad():
        model.eval()

        for k in top_rec_ks:
            p_k, r_k, n_k = evaluators.precision_recall_ndcg_k(model, val_users, k)
            logger.info("Epoch: {}, precision@{}: {}, recall@{}: {}, ndcg@{}: {}".format(epoch, 
                                                                                         k, 
                                                                                         p_k, 
                                                                                         k, 
                                                                                         r_k, 
                                                                                         k, 
                                                                                         n_k))

        _map, _mrr, _auc, _ndcg = evaluators.map_mrr_auc_ndcg(model, val_users)
        logger.info("Epoch: {}, MAP: {}, MRR: {}, AUC: {}, NDCG: {}".format(epoch, _map, _mrr, 
                                                                            _auc, _ndcg))

    return p_k

def train(args, model, logger, metric_evaluator, train_loader):
    if 'ml-10m' in args.data_path.strip().split('/'):
        if not os.path.exists('ml_10m_test_users.npy'):
            val_users = np.random.choice(np.arange(model.num_users), size=5000, replace=False)
            cur_log.info("======> saved test users from: ml_10m_test_users.npy")
            np.save('ml_10m_test_users.npy', val_users)
        else:
            cur_log.info("======> load test users from: ml_10m_test_users.npy")
            val_users = np.load('ml_10m_test_users.npy')
    else:
        val_users = np.asarray([i for i in range(model.num_users)])      
    
    opt_dicts = {
        "Adam": Adam(model.parameters(), lr = args.lr),
        "Adagrad": Adagrad(model.parameters(), lr = args.lr)
    }

    train_evaluator, val_evaluator, test_evaluator = metric_evaluator['train_evaluator'], \
                metric_evaluator['val_evaluator'], \
                metric_evaluator['test_evaluator']

    opt = opt_dicts[args.optimizer]

    best_val_auc = 0
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        gc.collect()

        logger.info('\n========> Epoch %3d: '% epoch) 
        model.train()
        train_loader.dataset.generate_triplets_by_sampling() # conduct negative sampling before training

        for it, (user_ids, pos_ids, neg_ids) in enumerate(train_loader):

            user_ids = user_ids.cuda()
            pos_ids = pos_ids.cuda()
            neg_ids = neg_ids.cuda()

            # print(user_ids)

            loss = model(user_ids, pos_ids, neg_ids)

            opt.zero_grad()
            loss.backward()
            
            # clip grad 
            # clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            if it % 100 == 0 or it == len(train_loader) - 1:
                logger.info('Iter %4d/%4d:  loss: %.4f'%(it, len(train_loader), loss.mean().item()))
            
            model.ClipUserNorm()

        logger.info('\n========> Evaluating validation set...')
        _auc = test(model, logger, val_users, val_evaluator, args.k, epoch)

        if _auc > best_val_auc:
            best_val_auc = _auc
            logger.info('\n========> Evaluating test set...')

            test(model, logger, val_users, test_evaluator, args.k, epoch)
            
            logger.save_model(model)

    logger.info('\nFinal results (val set) ====> best_val_auc:  %.6f' % (best_val_auc))

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

    # load data
    user_item_matrix, num_users, num_items = load_data(args, cur_log, data_name='users.dat', threholds=5)

    # split train/val/test
    user_train_matrix, user_val_matrix, user_test_matrix = split_train_val_test(user_item_matrix, args, cur_log)

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
    
    if not args.test:
        extra_sampler_args = {}

        # if args.sampling_strategy == '2st':
        #     extra_sampler_args['model'] = model
        #     extra_sampler_args['embedding_dim'] = args.dim
        #     extra_sampler_args['per_user_k'] = args.per_user_k
        #     extra_sampler_args['num_neg_candidates'] = args.num_neg_candidates

        train_loader =  DataLoader(SampleDataset(user_train_matrix, 
                                                 args.num_negs, 
                                                 args.sampling_strategy, 
                                                 args.random_seed,
                                                 **extra_sampler_args),
                                   batch_size = args.batch_size, 
                                   shuffle = True,
                                   pin_memory=True)
        metric_evaluator = {
            'train_evaluator': train_evaluator,
            'val_evaluator': val_evaluator,
            'test_evaluator': test_evaluator
        }
        train(args, model, cur_log, metric_evaluator, train_loader)
    else:
        cur_log.info("Evaluate Model...")
        cur_log.info("Loading Saved Model From {}".format(osp.join(args.data_path, args.save_path, model_name)))
        
        best_model_path = osp.join(args.data_path, args.save_path, args.model_name)
        if not os.exists(best_model_path):
            raise ValueError('saved model are not exist at %s' % best_model_path)

        model.load_state_dict(torch.load(best_model_path))

        test_users = np.asarray([i for i in range(model.num_users)])

        model.eval()

        torch.cuda.empty_cache()
        gc.collect()

        cur_log.info("Evaluating model with top-k in " + ','.join([str(k) for k in args.k]))

        test(model, cur_log, test_users, test_evaluator, args.k)