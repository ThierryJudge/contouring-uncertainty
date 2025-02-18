import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt


def confidence_ellipse(x, y, cov, ax, n_std=2.0, facecolor='none', edgecolor='red', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Adapted from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    cov: covariance matrix (2,2)

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # cov = np.flip(cov)

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor,
                      edgecolor=edgecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    # ci = 0.95
    # s = chi2.ppf(ci, 2)
    # scale_x = np.sqrt(cov[0, 0] * s)
    # scale_y = np.sqrt(cov[1, 1] * s)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array

    https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width

    https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = ax or plt.gca()
    ax.add_collection(lc)

    return lc


string_to_latex = {'cov_det_mean': r'$\bar{|\Sigma|}$',
                   'cov_projection_mean': r'$\bar{\Sigma_\perp}$',
                   'cov_projection': r'$\Sigma_\perp$',
                   'cov_eigenvalue_mean': r'$\bar{\sum \lambda(\Sigma)}$',
                   'cov_eigval_sum': r'$\sum \lambda(\Sigma)$',
                   'sigma_fac': r'$\sigma_{fac}$',
                   'fac_prop': r'$prop(\sigma_{fac})$',
                   'sigma_area': r'$\sigma_{area}$',
                   'sigma_gls': r'$\sigma_{gls}$',
                   'gls_prop': r'$prop(\sigma_{gls})$',
                   'cov_xx': r'$\Sigma_{xx}$',
                   'cov_yy': r'$\Sigma_{yy}$',
                   'cov_det': r'$|\Sigma|$'}


def str2tex(string):
    for key, item in string_to_latex.items():
        string = string.replace(key, item)
    # return f"r'{string}'"
    return string


def crop_axis(contour, margin=20):
    """

    Args:
        contour: (K, 2)

    Returns:

    """
    min_x = contour[:, 0].min()
    max_x = contour[:, 0].max()
    min_y = contour[:, 1].min()
    max_y = contour[:, 1].max()

    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    w = max_x - min_x
    h = max_y - min_y

    # Make a square by taking the longuest lenght
    crop_w = max(h, w) + margin
    crop_h = max(h, w) + margin

    x_plot_min = max(mid_x - crop_w // 2, 0)
    x_plot_max = min(mid_x + crop_w // 2, 255)

    y_plot_min = max(mid_y - crop_h // 2, 0)
    y_plot_max = min(mid_y + crop_h // 2, 255)

    return x_plot_min, x_plot_max, y_plot_min, y_plot_max

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    cov = np.array([[2, 0.5],
                    [0.5, 1]])

    W = np.array([[0.5, 0],
                  [0, 0.5]])

    cov2 = np.dot(np.dot(W, cov), W.T)
    cov3 = cov * 0.5 * 0.5

    fig, ax = plt.subplots(1, 1)
    ax.scatter(0, 0)
    confidence_ellipse(0, 0, cov, ax, n_std=2)
    confidence_ellipse(0, 0, cov2, ax, n_std=2, edgecolor='blue')
    confidence_ellipse(0, 0, cov3, ax, n_std=2, edgecolor='green')

    plt.show()
