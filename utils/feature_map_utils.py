"""
feature map utils with rotation and masking
"""
import numpy as np
import cv2


def rotate_fm(fm, angle_range=[-30, 30]):
    """
    rotate feature map
    :unfinished!!!!
    :return:
    """
    step = 5
    batch, c, h, w = fm.shape
    result = []

    for _ in range(angle_range[0], angle_range[1], step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), _, 1)
        dst = cv2.warpAffine(np.transpose(fm, [2, 3, 1, 0]), M, (w, h))
        result.append(dst)

    return result


def mask_fm(fm):
    """
    mask feature map
    :param fm:
    :return:
    """
    h_parts = 3
    w_parts = 3
    batch, c, h, w = fm.shape

    result = []
    # mask = np.zeros([int(h / h_parts), int(w / w_parts), c])
    for i in range(h_parts):
        for j in range(w_parts):
            copy_of_fm = np.copy(fm)
            copy_of_fm[0: batch, 0: c, i * int(h / h_parts): (i + 1) * int(h / h_parts),
            j * int(w / w_parts): (j + 1) * int(w / w_parts)] = \
                np.zeros([batch, c, int(h / h_parts), int(w / w_parts)])
            result.append(copy_of_fm)

    return result


if __name__ == '__main__':
    input_fm = np.ones([1, 3, 128, 128])
    # result = rotate_fm(np.transpose(input_fm, [2, 3, 1, 0]))
    result = rotate_fm(input_fm)
    cv2.imshow('img', result[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
