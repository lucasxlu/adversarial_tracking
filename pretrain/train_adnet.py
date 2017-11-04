import pickle
import time
import sys
import os

import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from utils.data_prov import RegionDataset
from utils.config import configs
from tracking.model import *

data_path = 'data/vot-otb.pkl'


def set_optimizer(model, lr_base, lr_mult=configs['lr_mult'], momentum=configs['momentum'], w_decay=configs['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)

    return optimizer


def train_adnet():
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp, encoding='iso-8859-1')

    K = len(data)
    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.items()):
        img_list = seq['images']
        gt = seq['gt']
        img_dir = os.path.join(configs['img_home'], seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, configs)

    # Init model
    model = ADNet(configs['init_model_path'])
    if configs['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(configs['ft_layers'])

    # Init criterion and optimizer
    criterion = ADLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, configs['lr_init'])

    best_prec = 0.
    for i in range(configs['n_cycles']):
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()

            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)

            if configs['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()

            pos_score = model(pos_regions)
            neg_score = model(neg_regions)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), configs['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print("Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                  (i, j, k, loss.data[0], prec[k], toc))

        cur_prec = prec.mean()
        print("Mean Precision: %.3f" % cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if configs['use_gpu']:
                model = model.cuda()
            states = {'shared_layers': model.layers.state_dict()}
            print("Save model to %s" % configs['model_path'])
            torch.save(states, configs['model_path'])
            print('model has been saved...')


if __name__ == "__main__":
    train_adnet()
