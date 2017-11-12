"""
pre-process sequence
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.config import *


def load_seq(option_json_path='./options.json'):
    """
    load sequences and groundtruth
    :return:
    """
    with open(option_json_path, mode='rt') as fp:
        options = json.load(fp)
        sequence = options['sequence']

    seq_type = sequence['type']

    if seq_type.startswith('OTB'):
        img_list = os.listdir(os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name'], 'img'))
        img_list.sort()
        img_list = [os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name'], 'img', _) for _ in
                    img_list]

        gt_txt_path = os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name'],
                                   'groundtruth_rect.txt')
        if ',' in open(gt_txt_path, mode='rt').read():
            gt = np.loadtxt(gt_txt_path, delimiter=',')
        elif '\t' in open(gt_txt_path, mode='rt').read():
            gt = np.loadtxt(gt_txt_path, delimiter='\t')

    elif seq_type == 'VOT':
        img_list = os.listdir(os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name']))
        img_list.sort()
        img_list = [os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name'], _) for _ in img_list]
        gt = np.loadtxt(
            os.path.join(configs['test_seq_base'], sequence['type'], sequence['seq_name'], 'groundtruth.txt'),
            delimiter=',')
    else:
        print('Error, unknown benchmark type!')

    init_bbox = gt[0]

    return seq_type, img_list, gt, init_bbox


def draw_sequence(imgs, gts, seq_name):
    """
    draw sequences
    :return:
    """
    draw_seq_save_dir = '../result' + os.sep + seq_name
    if not os.path.exists(draw_seq_save_dir):
        os.makedirs(draw_seq_save_dir)

    for i in range(len(imgs)):
        image = cv2.imread(imgs[i])
        pt1 = (int(gts[i][0]), int(gts[i][1]))
        pt2 = (int(gts[i][0]) + int(int(gts[i][2])), int(gts[i][1]) + int(int(gts[i][3])))

        image = cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
        image = cv2.putText(image, '#%d' % (i + 1), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                            cv2.LINE_AA)
        cv2.imwrite(os.path.join(draw_seq_save_dir, imgs[i].split('/')[-1]), image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    seq_type, img_list, gt, init_bbox = load_seq()

    with open('./options.json', mode='rt') as fp:
        options = json.load(fp)
        sequence = options['sequence']

    with open('../result/result.json', mode='rt') as fp:
        result = json.load(fp)
    predict_bbox = result['res']
    draw_sequence(img_list, predict_bbox, sequence['seq_name'])
