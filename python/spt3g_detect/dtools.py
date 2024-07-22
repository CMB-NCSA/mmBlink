import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from scipy.stats import norm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS
from photutils.background import Background2D


def stack_cols_lists(c1, c2, asscalar=False, append=False):

    """
    Custom function to stack two one-dimensional columns

    inputs:
       - c1: list (or list of lists) or 1D numpy array
       - c2: list (or list of lists) or 1D numpy array

    options:
    """

    # Make sure that c1 and c2 are lists, if not we re-cast them as lists
    if not isinstance(c1, list):
        try:
            c1 = list(c1)
        except Exception as err:
            raise Exception(f"Cannot cast c1 as list {err=}, {type(err)=}")
    if not isinstance(c2, list):
        try:
            c2 = list(c2)
        except Exception as err:
            raise Exception(f"Cannot cast c2 as list {err=}, {type(err)=}")

    # Case 1, we stack c1, c2
    if append is False:
        # Make sure that c1 and c2 have the same dimensions
        if len(c1) != len(c1):
            raise Exception("c1 and c2 have different dimensions")
        else:
            # We will store the new list here as "newlist"
            newlist = []
            for i in range(len(c1)):
                # if elements of c1 are list, we append instead of join
                if isinstance(c1[i], list):
                    c1[i].append(c2[i])
                    x = c1[i]
                # if not we join elements of c1 and c2
                else:
                    x = [c1[i], c2[i]]
                newlist.append(x)

    # Case 2, we want to append c2 to existing c1
    else:
        for c in c2:
            if asscalar:
                c1.append(c)
            else:
                c1.append([c])
        newlist = c1
    return newlist


def mean_list_of_list(list, np_array=True):
    u = [np.array(x).mean() for x in list]
    if np_array:
        u = np.array(u)
    return u


def max_list_of_list(list, np_array=True):
    max_val = [np.array(x).max() for x in list]
    if np_array:
        max_val = np.array(max_val)
    return max_val


def compute_rms2D(data, box=(200, 200), filter_size=(3, 3), sigmaclip=None):
    # Change the name of the function

    # in case we want to clip values
    if sigmaclip:
        sigma_clip = SigmaClip(sigma=sigmaclip)
    else:
        sigma_clip = None

    # Set up the background estimator
    bkg_estimator = StdBackgroundRMS(sigma_clip)
    bkg = Background2D(data, box, filter_size=filter_size, edge_method='pad',
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    bkgrms_value = bkg_estimator.calc_background_rms(data)
    print(bkgrms_value)

    mean, sigma = norm.fit(bkg.background.flatten())
    print(mean, sigma)

    return bkg


def detect_with_photutils(data, wgt=None, nsigma_thresh=3.5, npixels=20,
                          rms2D=False, box=(200, 200), filter_size=(3, 3), sigmaclip=None,
                          wcs=None, plot=False, plot_title=None):

    # Compute 2D rms image
    if rms2D:
        bkg = compute_rms2D(data, box=box, filter_size=filter_size, sigmaclip=sigmaclip)
        sigma = bkg.background
    else:
        # Get the mean and std of the distribution
        mean, sigma = norm.fit(data.flatten())

    # Define the threshold
    threshold = nsigma_thresh * sigma
    # Perform segmentation and deblending
    finder = SourceFinder(npixels=npixels, nlevels=32, contrast=0.001, progress_bar=False)
    segm = finder(data, threshold)
    cat = SourceCatalog(data, segm, error=wgt, wcs=wcs)

    # Nicer formatting
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.5f'
    tbl['kron_fluxerr'].info.format = '.5f'
    tbl['max_value'].info.format = '.5f'
    tbl['sky_centroid_dms'] = tbl['sky_centroid'].to_string('dms', precision=0)
    # print(tbl['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms',
    #          'kron_flux', 'kron_fluxerr', 'max_value', 'area'])
    if plot:
        plot_detection(data, segm, cat, mean, sigma, nsigma=nsigma_thresh, plot_title=plot_title)

    return segm, tbl


def plot_detection(data, segm, cat, mean, sigma, nsigma=3, plot_title=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    im1 = ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=-5*sigma, vmax=5*sigma)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    if plot_title:
        ax1.set_title(plot_title)

    im2 = ax2.imshow(segm, origin='lower', cmap=segm.cmap,
                     interpolation='nearest')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=0.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=0.5)
    plot_distribution(data, mean, sigma, axis=ax3, nsigma=nsigma)
    plt.show()


def plot_distribution(data, mean, sigma, axis=None, nsigma=3):

    """Simple function to display distribution of data"""

    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    legend = "$\\mu$: %.6f\n$\\sigma$: %.6f" % (mean, sigma)
    if axis:
        ax = axis
    else:
        plt.figure(figsize=(6, 6))
        ax = plt

    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    hist = ax.hist(data, bins=nbins, density=True, alpha=0.6)
    ymin, ymax = hist[0].min(), hist[0].max()
    xmin, xmax = hist[1].min(), hist[1].max()
    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, sigma)
    ax.plot(x, y)
    xx = [nsigma*sigma, nsigma*sigma]
    yy = [ymin, ymax]
    ax.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*sigma, -nsigma*sigma]
    ax.plot(xx, yy, 'k--', linewidth=1)
    if axis:
        ax.set_ylim(ymin, ymax)
    else:
        ax.ylim(ymin, ymax)
    ax.legend([legend], frameon=False)
    text = f"${nsigma}\\sigma$"
    ax.text(0.05*(xmax-xmin) + nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')
    ax.text(-0.05*(xmax-xmin) - nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')

    if axis:
        ratio = 0.95
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
