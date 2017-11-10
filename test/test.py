from pprint import pprint

import cv2
import numpy as np


def blur_img(image):
    blur = cv2.blur(image, (10, 10))
    cv2.imshow('image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_fm(fm, angle_range=[-30, 30]):
    step = 5
    batch, c, h, w = fm.shape
    result = []

    for _ in range(angle_range[0], angle_range[1], step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), _, 1)
        dst = cv2.warpAffine(np.transpose(fm, [2, 3, 1, 0]), M, (w, h))
        result.append(dst)

    return result


if __name__ == '__main__':
    image = cv2.imread('/home/lucasx/Documents/talor.jpg')
    # rotate_img(image, -15)
    # result = feature_map_utils.mask_fm(image)
    # for _ in result:
    #     cv2.imshow('image', _)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
