"""
preprocess sequence
"""
import json
import os

import cv2
import numpy as np
import torchvision.transforms
from skimage import io, transform

import torch
from torch.autograd import Variable
import torchvision


def load_seq():
    """
    load sequences and groundtruth
    :return:
    """
    with open('./options.json', mode='rt') as fp:
        options = json.load(fp)
        sequence = options['sequence']

    seq_type = sequence['type']

    if seq_type == 'OTB':
        img_list = os.listdir(os.path.join('../dataset/', sequence['type'], sequence['seq_name'], 'img'))
        img_list.sort()
        img_list = [os.path.join('../dataset/', sequence['type'], sequence['seq_name'], 'img', _) for _ in img_list]
        gt = np.loadtxt(os.path.join('../dataset/', sequence['type'], sequence['seq_name'], 'groundtruth_rect.txt'),
                        delimiter=',')  # groundtruth
    elif seq_type == 'VOT':
        img_list = os.listdir(os.path.join('../dataset/', sequence['type'], sequence['seq_name']))
        img_list.sort()
        img_list = [os.path.join('../dataset/', sequence['type'], sequence['seq_name'], _) for _ in img_list]
        gt = np.loadtxt(os.path.join('../dataset/', sequence['type'], sequence['seq_name'], 'groundtruth.txt'),
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
        cv2.imwrite(os.path.join(draw_seq_save_dir, imgs[i].split('/')[-1]), image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def extract_feature():
    model = torchvision.models.vgg16(pretrained=True)
    seq_type, img_list, gt, init_bbox = load_seq()
    transposed_image = np.transpose(transform.resize(io.imread(img_list[0]), (224, 224, 3)))
    transposed_image -= np.mean(transposed_image, axis=0)  # mean norm the image data
    image = torch.from_numpy(transposed_image).type(torch.FloatTensor)
    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()

        print(model.forward(Variable(image.unsqueeze(0))))


if __name__ == '__main__':
    seq_type, img_list, gt, init_bbox = load_seq()

    with open('../result/result.json', mode='rt') as fp:
        result = json.load(fp)
    predict_bbox = result['res']
    draw_sequence(img_list, predict_bbox, 'DragonBaby')
    # extract_feature()
