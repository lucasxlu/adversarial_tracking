from utils.calculator import *


def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):
    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 16:
            samples_ = generator(bbox, remain * factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx, :]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor * 2

        return samples


def non_max_suppression(boxes, overlap_thresh):
    """
    Non-max suppression
    :param boxes:
    :param overlap_thresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size)  # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    def __call__(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n, 1) * 2 - 1
            samples[:, 2:] *= self.aspect_f ** np.concatenate([ratio, -ratio], axis=1)

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale_f ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            samples[:, :2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            # samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
            samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f
