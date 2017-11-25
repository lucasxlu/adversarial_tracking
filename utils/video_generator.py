import os
import sys

import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


def gen_video(seq_dir):
    # Define the codec and create VideoWriter object
    print('start to generate video from %s' % seq_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('%s.avi' % seq_dir.split('/')[-1], fourcc, 24, (640, 360))
    filelist = [os.path.join(seq_dir, _) for _ in os.listdir(seq_dir)]
    filelist = np.sort(np.array(filelist)).tolist()

    for file in filelist:
        frame = cv2.imread(file)
        # frame = cv2.flip(frame, 0)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    print('finish...')


if __name__ == '__main__':
    gen_video('/home/lucasx/PycharmProjects/adversarial_tracking/result/DragonBaby')
