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


def blur_fm(fm):
    h_parts = 3
    w_parts = 3
    batch, c, h, w = fm.shape

    filter = np.zeros([c, h, w])

    result = []
    for i in range(h_parts):
        for j in range(w_parts):
            copy_of_fm = np.copy(fm)
            copy_of_fm[0: batch, 0: c, i * int(h / h_parts): (i + 1) * int(h / h_parts),
            j * int(w / w_parts): (j + 1) * int(w / w_parts)] *= \
                np.ones([batch, c, int(h / h_parts), int(w / w_parts)])
            result.append(copy_of_fm)

    return result


if __name__ == '__main__':
    img = cv2.imread('/home/lucasx/Documents/talor.jpg')
    fm = np.zeros([1, 3, 1024, 1024])
    fm[0] = np.transpose(img, [2, 0, 1])
    result = blur_fm(fm)
    new_img = np.transpose(result[1][0], [1, 2, 0])
    cv2.imwrite('./new_img.jpg', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    image = cv2.imread('/home/lucasx/Documents/talor.jpg')
    fm = np.ones([4, 3, 1024, 1024])
    fm[0] = np.transpose(image, [2, 0, 1])
    new_img = np.transpose(fm[0], [1, 2, 0])
    cv2.imshow('image', new_img)
    cv2.imwrite('./new_img.jpg', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
