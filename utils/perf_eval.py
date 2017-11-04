import sys
import json
import numpy as np

sys.path.insert(0, '../tracking')
from tracking.prepro_seq import *
from utils.calculator import *


def eval_center_pixel(gt_list, res_list):
    avg_dist = 0.
    for i in range(len(gt_list)):
        gt_center = np.array(
            [int(gt_list[i][0]) + int(int(gt_list[i][2]) / 2), int(gt_list[i][1]) + int(int(gt_list[i][3])) / 2])
        res_center = np.array([int(res_list[i][0]) + int(int(res_list[i][2]) / 2),
                               int(res_list[i][1]) + int(int(res_list[i][3])) / 2])
        dist = np.linalg.norm(gt_center - res_center)
        avg_dist += dist

    return avg_dist / len(gt_list)


def eval_success_rate(gt_list, res_list):
    success_num = 0
    for i in range(len(gt_list)):
        gt = np.array(gt_list[i])
        res = np.array(res_list[i])

        iou = overlap_ratio(gt, res)
        if iou > 0.5:
            success_num += 1

    return success_num / len(gt_list)


def get_bbox_res(result_json_path='../result/result.json'):
    with open(result_json_path, mode='rt') as f:
        res = json.load(f)['res']

    return res


if __name__ == '__main__':
    seq_type, img_list, gt, init_bbox = load_seq('../tracking/options.json')
    res = get_bbox_res()
    print(eval_success_rate(gt, res))
