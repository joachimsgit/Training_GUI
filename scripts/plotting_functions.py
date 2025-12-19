import itertools

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter


def confidence_ellipse(
    ax,
    mean,
    cov,
    n_std=3.0,
    facecolor="none",
    **kwargs,
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

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
    mean_x = mean[0]
    mean_y = mean[1]
    
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        linewidth=2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_heatmap_plot(
    data,
    axis_names=None,
    sigma=3,
    bins=100,
    processing_function=lambda x: np.log(x + 1),
    upper_bounds=None,
    lower_bounds=None,
    title="Full 3D Contrast Heatmap",
    used_channels="BGR",
):
    """
    Generates 2D heatmaps for all 2-channel combinations from 3D contrast data.

    Args:
        data (np.ndarray): Input data of shape (n_samples, 3), representing BGR contrast values.
        axis_names (list or None): List of axis labels for ['B', 'G', 'R']. Defaults to None.
        sigma (float): Standard deviation for Gaussian smoothing.
        bins (int): Number of bins for the histogram.
        processing_function (callable): Function to process heatmap values (e.g., log-scaling).
        upper_bounds (list or None): Upper limits for each contrast dimension.
        lower_bounds (list or None): Lower limits for each contrast dimension.
        title (str): Title for the entire figure.
        used_channels (str): Which contrast channels to use, e.g., "BGR", "BR", etc.

    Returns:
        None: The function plots the heatmaps directly using matplotlib.
    """
    assert data.shape[1] == 3, "Data must be 3D"
    assert len(used_channels) in [2, 3], "Number of used channels must be 2 or 3"

    if not (axis_names is None):
        assert len(axis_names) == 3, "There must be 3 axis names or None"

    # change the font size of the plot
    plt.rcParams.update({"font.size": 20})

    # extract the used channels and all possible combinations
    # these are either 3 or 1 combinations
    used_channel_indices = ["BGR".index(channel) for channel in used_channels]
    channel_combinations = list(itertools.combinations(used_channel_indices, 2))

    fig, axis = plt.subplots(
        1,
        len(channel_combinations),
        figsize=(7 * len(channel_combinations), 7),
        dpi=300,
    )
    fig.suptitle(title)

    # if only one channel is used, we need to add an axis to the list
    if len(channel_combinations) == 1:
        axis = [axis]

    for idx, (i, j) in enumerate(channel_combinations):
        # Generate some test data
        x = data[:, i]
        y = data[:, j]

        if lower_bounds is not None and upper_bounds is not None:
            x_lower = lower_bounds[i]
            x_upper = upper_bounds[i]
            y_lower = lower_bounds[j]
            y_upper = upper_bounds[j]
            img, extent = create_heatmap(
                x,
                y,
                sigma=sigma,
                bins=bins,
                extent=[x_lower, x_upper, y_lower, y_upper],
            )
        else:
            img, extent = create_heatmap(x, y, sigma=sigma, bins=bins)

        img = processing_function(img)

        axis[idx].imshow(
            img, extent=extent, origin="lower", cmap=cm.plasma, aspect="auto"
        )

        axis[idx].set_xlabel(axis_names[i])
        axis[idx].set_ylabel(axis_names[j])

        axis[idx].grid()

    # set space between subplots
    plt.subplots_adjust(wspace=0.3)

    return fig


def create_heatmap(
    x,
    y,
    sigma,
    bins=1000,
    extent=None,
):
    """
    Creates a 2D heatmap by computing a smoothed 2D histogram using Gaussian filtering.

    Args:
        x (np.ndarray): X-coordinates of the data.
        y (np.ndarray): Y-coordinates of the data.
        sigma (float): Standard deviation used for Gaussian filter.
        bins (int): Number of histogram bins per dimension.
        extent (list or None): Limits [x_min, x_max, y_min, y_max] for the histogram.

    Returns:
        Tuple:
            - heatmap (np.ndarray): Smoothed 2D histogram (transposed for correct orientation).
            - extent (list): Used extent for plotting.
    """
    if extent is None:
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    else:
        hist_range = [[extent[0], extent[1]], [extent[2], extent[3]]]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range)

    heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap.T, extent
