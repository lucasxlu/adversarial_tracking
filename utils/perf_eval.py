import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
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


def get_bbox_res(seq_name):
    result_json_path = '../result/result_%s.json' % seq_name
    with open(result_json_path, mode='rt') as f:
        res = json.load(f)['res']

    return res


if __name__ == '__main__':
    result = load_seq('../tracking/options.json')

    for _ in result:
        res = get_bbox_res(_[-1])
        print('*' * 100)
        print('success rate is %f' % eval_success_rate(_[2], res))
        print('*' * 100)

        print('\n')

        print('*' * 100)
        print('center pixel error is %d' % eval_center_pixel(_[2], res))
        print('*' * 100)
