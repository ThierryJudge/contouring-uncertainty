import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import binary_fill_holes
from skimage.draw import line
from scipy.stats import multivariate_normal


def contour_spline(mu, n=1001, close=False):
    try:
        tck, u = interpolate.splprep([mu[:, 0], mu[:, 1]], k=3, s=0)
        unew = np.linspace(0, 1.0, n)
        spline = np.array(interpolate.splev(unew, tck)).transpose()
    except:
        # from matplotlib import pyplot as plt
        # print(mu.shape)
        # f, ax = plt.subplots()
        # ax.set_xlim([0, 256])
        # ax.set_ylim([256, 0])
        # ax.scatter(mu[:, 0], mu[:, 1], s=5)
        # plt.show()
        spline = mu
    if close:
        spline = np.concatenate((spline, spline[0][None]))
    return spline


def reconstruction(points: np.ndarray, height: int, width: int):
    seg_map = np.zeros((height, width))
    spline = contour_spline(points, n=1000).round().astype(int)
    try:
        seg_map[spline[:, 1].clip(max=height - 1), spline[:, 0].clip(max=width - 1)] = 1
    except:
        plt.scatter(spline[:, 0], -spline[:, 1])
        plt.show()
    points = points.round().astype(int)
    rr, cc = line(points[-1, 1], points[-1, 0], points[0, 1], points[0, 0])
    seg_map[rr.clip(max=height - 1, min=0), cc.clip(max=width - 1, min=0)] = 1
    seg_map = binary_fill_holes(seg_map).astype(int)
    return seg_map


def linear_reconstruction(points: np.ndarray, height: int, width: int):
    points = points.round().astype(int)
    seg_map = np.zeros((height, width), dtype=int)
    for i in range(len(points) - 1):
        # Convert x,y to row,colunm
        rr, cc = line(points[i, 1], points[i, 0], points[i + 1, 1], points[i + 1, 0])
        seg_map[rr.clip(min=0, max=255), cc.clip(min=0, max=255)] = 1
    rr, cc = line(points[-1, 1], points[-1, 0], points[0, 1], points[0, 0])
    seg_map[rr.clip(min=0, max=255), cc.clip(min=0, max=255)] = 1
    seg_map = binary_fill_holes(seg_map).astype(int)
    return seg_map


def make_prob_map(mu, cov, shape=(256, 256)):
    x, y = np.mgrid[0:shape[0]:1, 0:shape[1]:1]
    pos = np.dstack((x, y))
    map = np.zeros(shape)
    for k in range(len(mu)):
        rv = multivariate_normal(mu[k], cov[k])
        map += rv.pdf(pos)

    return np.transpose(map)
