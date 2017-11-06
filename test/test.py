import cv2

from utils import feature_map_utils


def blur_img(image):
    blur = cv2.blur(image, (10, 10))
    cv2.imshow('image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_img(image, rotate_angle):
    cols, rows, chs = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imshow('image', dst)
    cv2.imwrite('./ratate.jpg', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('/home/lucasx/Documents/talor.jpg')
    # rotate_img(image, -15)
    result = feature_map_utils.mask_fm(image)
    for _ in result:
        cv2.imshow('image', _)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
