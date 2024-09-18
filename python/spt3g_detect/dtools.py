import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from scipy.stats import norm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.stats import SigmaClip
import photutils.background
import copy


def stack_cols_lists(c1, c2, ix1, ix2, pad=False):

    """
    Custom function to stack two one-dimensional columns

    inputs:
       - c1: list (or list of lists) or 1D numpy array
       - c2: list (or list of lists) or 1D numpy array

    options:
    """
    ixall1 = np.arange(len(c1))
    ixnew1 = np.delete(ixall1, ix1)
    ixall2 = np.arange(len(c2))
    ixnew2 = np.delete(ixall2, ix2)
    nidx = len(ix1)

    if len(ix1) != len(ix2):
        raise Exception("ix1 and ix2 have different dimensions")

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

    # We will store the new list here as "newlist"
    # We want to augment c1 with new matches (ix2) from c2
    # Make sure that the newlist elements are lists:
    # c1copy = copy.deepcopy(c1)
    newlist = [c if isinstance(c, list) else [c] for c in c1]

    # Step 1, we stack c1, c2, by indexing (ix1, ix2)
    for k in range(nidx):
        i = ix1[k]
        j = ix2[k]
        newlist[i].append(c2[j])

    # Step 2, we want to append c2 to existing c1
    print(f"ixnew1: {ixnew1}")
    if pad:
        for k in ixnew1:
            c = newlist[k][0]
            newlist[k] = [c, c]
            print("padding ixnew1:", k, newlist[k])

    print(f"ixnew2: {ixnew2}")
    for k in ixnew2:
        c = c2[k]
        if pad:
            newlist.append([c, c])
            print("padding ixnew2:", len(newlist), newlist[-1])
        else:
            newlist.append([c])
    return newlist


def stack_cols_lists_old(c1, c2, asscalar=False, append=False):

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
        if len(c1) != len(c2):
            raise Exception(f"c1 and c2 have different dimensions: {len(c1)} vs {len(c2)}")
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
            print("c2:", c2)
            print(type(c2))
            if asscalar:
                c1.append(c)
            else:
                c1.append(np.array([c]))
        newlist = c1

    print("#### ---- 1 ---- #####", type(newlist))
    print("#### ---- 1 ---- #####", newlist)

    #n = len(newlist)
    #newarray = np.zeros(n, dtype=object)
    #for k in range(n):
    #newarray[k] = np.asarray(newlist[k])

    #print(newlist)
    #print(newarray)
    #exit()

    newlist = np.asarray(newlist, dtype=object)
    print("#### ---- 2 ---- #####", type(newlist))
    print("#### ---- 2 ---- #####", newlist)
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

    """
    Compute a 2D map of the rms using photutils.Background2D and
    photutils.StdBackgroundRMS
    """

    # in case we want to clip values
    if sigmaclip:
        sigma_clip = SigmaClip(sigma=sigmaclip)
    else:
        sigma_clip = None

    # Set up the background estimator
    bkg_estimator = photutils.background.StdBackgroundRMS(sigma_clip)
    bkg = photutils.background.Background2D(data, box, filter_size=filter_size, edge_method='pad',
                                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg


def detect_with_photutils(data, wgt=None, nsigma_thresh=3.5, npixels=20,
                          rms2D=False, box=(200, 200), filter_size=(3, 3), sigmaclip=None,
                          wcs=None, plot=False, plot_title=None, plot_name=None):

    # Get the mean and std of the distribution
    mean, sigma = norm.fit(data.flatten())

    # Define the threshold, array in the case of rms2D
    if rms2D:
        bkg = compute_rms2D(data, box=box, filter_size=filter_size, sigmaclip=sigmaclip)
        sigma2D = bkg.background
        threshold = nsigma_thresh * sigma2D
        print("2D RMS computed")
    else:
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

        if rms2D:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=plot_title)
        plot_segmentation(ax2, segm, cat)
        plot_distribution(ax3, data, mean, sigma, nsigma=nsigma_thresh)
        if rms2D:
            plot_rms2D(bkg.background, ax4)
        if plot_name:
            plt.savefig(f"{plot_name}.pdf")
            print(f"Saved: {plot_name}.pdf")
        else:
            plt.show()

    return segm, tbl


def plot_rms2D(bkg, ax, nsigma_plot=5):

    # Get the stats for the 2D rms
    bkg_mean, bkg_sigma = norm.fit(bkg.flatten())
    vmin = bkg_mean - nsigma_plot*bkg_sigma
    vmax = bkg_mean + nsigma_plot*bkg_sigma
    im = ax.imshow(bkg, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('2D Noise Map')


def plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=None):
    """
    Function to plot the 2D data array and catalog produced by photutils
    """
    vlim = nsigma_plot*sigma
    im1 = ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=-vlim, vmax=+vlim)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    if plot_title:
        ax1.set_title(plot_title)
    cat.plot_kron_apertures(ax=ax1, color='white', lw=0.5)


def plot_segmentation(ax2, segm, cat):
    """
    Function to plot the segmentation image and catalog produce by photutils
    """
    im = ax2.imshow(segm, origin='lower', cmap=segm.cmap,
                    interpolation='nearest')
    divider = make_axes_locatable(ax2)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax2)
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax2, color='white', lw=0.5)


def plot_distribution(ax, data, mean, sigma, nsigma=3):
    """
    Function to display distribution of data as 1D
    """
    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    legend = "$\\mu$: %.6f\n$\\sigma$: %.6f" % (mean, sigma)
    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    hist = ax.hist(data, bins=nbins, density=True, alpha=0.6)
    ymin, ymax = hist[0].min(), hist[0].max()
    xmin, xmax = hist[1].min(), hist[1].max()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, sigma)
    ax.plot(x, y)
    xx = [nsigma*sigma, nsigma*sigma]
    yy = [ymin, ymax]
    ax.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*sigma, -nsigma*sigma]
    ax.plot(xx, yy, 'k--', linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.legend([legend], frameon=False)
    text = f"${nsigma}\\sigma$"
    ax.text(0.05*(xmax-xmin) + nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')
    ax.text(-0.05*(xmax-xmin) - nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')

    ratio = 0.95
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_xlabel("Flux")
    ax.set_title("1D Noise Distribution and Fit")
