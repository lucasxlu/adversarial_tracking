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
        r_fm = np.array([batch, c, h, w])
        for each_bat_fm in r_fm[:, c, h, w]:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), _, 1)
            dst = cv2.warpAffine(np.transpose(each_bat_fm, [2, 3, 1, 0]), M, (w, h))
            result.append(dst)

    return result


if __name__ == '__main__':
    if ',' in open('test.txt').read():
        li = np.loadtxt('test.txt', delimiter=',')
        print(li)
    elif '\t' in open('test.txt').read():
        li = np.loadtxt('test.txt', delimiter='\t')
        print(li)
    # image = cv2.imread('/home/lucasx/Documents/talor.jpg')
    # fm = np.array([4, 3, None, None])
    # fm[0][:][:][:] = image
    # cv2.imshow('image', fm[0])