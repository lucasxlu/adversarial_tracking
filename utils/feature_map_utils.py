"""
feature map utils with rotation and masking
"""
import numpy as np
import cv2


def rotate_fm(fm, angle_range=[-30, 30]):
    """
    rotate feature map
    :return:
    """
    step = 5
    cols, rows, chs = fm.shape
    result = []

    for _ in range(angle_range[0], angle_range[1], step):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), _, 1)
        dst = cv2.warpAffine(fm, M, (cols, rows))
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
    h, w, c = fm.shape

    result = []
    # mask = np.zeros([int(h / h_parts), int(w / w_parts), c])
    for i in range(h_parts):
        for j in range(w_parts):
            copy_of_fm = np.copy(fm)
            copy_of_fm[i * int(h / h_parts): (i + 1) * int(h / h_parts),
            j * int(w / w_parts): (j + 1) * int(w / w_parts), :] = \
                np.zeros([int(h / h_parts), int(w / w_parts), c])
            result.append(copy_of_fm)

    return result


if __name__ == '__main__':
    print(len(rotate_fm(np.random.rand(20, 20, 10))))
