import cv2
import numpy as np
from skimage.exposure import cumulative_distribution

import filters


def cdf(im):
    c, b = cumulative_distribution(im)
    # pad the beginning and ending pixels and their CDF values
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c


def hist_matching(c, c_t, im):
    pixels = np.arange(256)
    new_pixels = np.interp(c, c_t, pixels)
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
    return im


class MatchHist:
    def __init__(self, *args, **kwargs):
        bg = cv2.imread('images/background.jpg')
        bg[:, :, 2], bg[:, :, 0] = bg[:, :, 0], bg[:, :, 2]
        c_t = []
        for i in range(3):
            c_t.append(cdf(bg[:, :, i]))
        self.c_t = c_t

    def apply(self, *args, **kwargs):
        frame = kwargs['frame'].astype(np.uint8)
        mask = np.all(frame, axis=2) == np.zeros(frame.shape[:2])
        mask = ~mask
        for i in range(3):
            c = cdf(frame[:, :, i][mask])
            frame[:, :, i] = hist_matching(c, self.c_t[i], frame[:, :, i])
        return frame


filters.register_filter("match_hist", MatchHist)
