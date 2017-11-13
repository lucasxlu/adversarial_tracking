from collections import OrderedDict

configs = OrderedDict()
configs['use_gpu'] = True

# configs['init_model_path'] = '../models/imagenet-vgg-m.mat'
configs['init_model_path'] = '/home/lucasx/ModelZoo/imagenet-vgg-m.mat'
configs['img_home'] = '/media/lucasx/Document/DataSet/CV/Tracking'
configs['model_path'] = '../models/adnet_vot-otb.pth'
configs['test_seq_base'] = '/media/lucasx/Document/DataSet/CV/Tracking/'

configs['batch_frames'] = 8
configs['batch_pos'] = 32
configs['batch_neg'] = 96

configs['overlap_pos'] = [0.7, 1]
configs['overlap_neg'] = [0, 0.5]

configs['img_size'] = 107
configs['padding'] = 16

configs['lr_init'] = 0.0001
configs['lr_update'] = 0.0002
configs['w_decay'] = 0.0005
configs['momentum'] = 0.9
configs['grad_clip'] = 10
configs['ft_layers'] = ['conv', 'fc']
configs['lr_mult'] = {'fc': 10}
configs['n_epoch'] = 50

configs['n_bbreg'] = 1000
configs['overlap_bbreg'] = [0.6, 1]
configs['scale_bbreg'] = [0.5, 2]  # [1, 2]

configs['batch_test'] = 256

configs['n_pos_init'] = 500
configs['n_neg_init'] = 5000

configs['overlap_pos_init'] = [0.7, 1]
configs['overlap_neg_init'] = [0, 0.5]

configs['maxiter_init'] = 30
configs['maxiter_update'] = 15
configs['batch_neg_cand'] = 1024

configs['trans_f'] = 0.6
configs['scale_f'] = 1.05
configs['trans_f_expand'] = 1.5
configs['n_samples'] = 256

configs['n_pos_update'] = 50
configs['n_neg_update'] = 200

configs['success_thr'] = 0
configs['overlap_pos_update'] = [0.7, 1]
configs['overlap_neg_update'] = [0, 0.3]

configs['n_frames_short'] = 20
configs['n_frames_long'] = 100
configs['long_interval'] = 10