#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from astropy.io import fits
from scipy.stats import norm
from astropy import wcs
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy import units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5


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


def plot_distribution(data, nsigma=3):

    """Simple function to display distribution of data"""

    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    # Fit a normal distribution to the data
    mean, std = norm.fit(data)
    legend = "$\mu$: %.6f\n$\sigma$: %.6f" % (mean, std)

    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    plt.hist(data, bins=nbins, density=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    xx = [nsigma*std, nsigma*std]
    yy = [ymin, ymax]
    plt.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*std, -nsigma*std]
    plt.plot(xx, yy, 'k--', linewidth=1)
    plt.ylim(ymin, ymax)
    plt.legend([legend], frameon=False)
    text = f'${nsigma}\sigma$'
    plt.text(0.05*(xmax-xmin) + nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')
    plt.text(-0.05*(xmax-xmin) - nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')
    # plt.show()
    return mean, std


def plot_distribution_axis(data, ax, nsigma=3):

    """Simple function to display distribution of data"""

    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    # Fit a normal distribution to the data
    mean, std = norm.fit(data)
    legend = "$\mu$: %.6f\n$\sigma$: %.6f" % (mean, std)

    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    hist = ax.hist(data, bins=nbins, density=True, alpha=0.6)
    ymin, ymax = hist[0].min(), hist[0].max()
    xmin, xmax = hist[1].min(), hist[1].max()

    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    ax.plot(x, y)
    xx = [nsigma*std, nsigma*std]
    yy = [ymin, ymax]
    ax.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*std, -nsigma*std]
    ax.plot(xx, yy, 'k--', linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.legend([legend], frameon=False)
    text = f'${nsigma}\sigma$'
    ax.text(0.05*(xmax-xmin) + nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')
    ax.text(-0.05*(xmax-xmin) - nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')

    ratio = 0.95
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    return mean, std


def detect_with_photutils(data, wgt=None, nsigma_thresh=3.5, npixels=20,
                          wcs=None, plot=False, plot_title=None):

    # Get the mean and std of the distribution
    mean, sigma = norm.fit(data.flatten())
    # Define the threshold
    threshold = nsigma_thresh * sigma
    # Perform segmentation and deblending
    finder = SourceFinder(npixels=npixels, nlevels=32, contrast=0.001,
                          progress_bar=False)
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
    # print(tbl['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms', 'kron_flux', 'kron_fluxerr', 'max_value', 'area'])
    if plot:
        plot_detection(data, segm, cat, nsigma=nsigma_thresh, plot_title=plot_title)

    return segm, tbl


def plot_detection(data, segm, cat, nsigma=3, plot_title=None):

    mean, sigma = norm.fit(data.flatten())
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
    plot_distribution_axis(data, ax3, nsigma=nsigma)
    plt.show()


if __name__ == "__main__":

    plot = False
    # plot = True
    nsigma = 3.5
    npixels = 20
    scans = ['40-157-9', '40-160-9', '40-161-9', '40-162-9', '40-163-9']

    segm = {}
    cat = {}

    for scan in scans:
        print(scan)
        filename_090 = f"f090/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.fits"
        filename_150 = f"f150/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.fits"

        # fits.info(filename_090)
        hdul = fits.open(filename_090)
        data_090 = hdul[1].data
        wgt_090 = hdul[2].data

        # fits.info(filename_150)
        hdul = fits.open(filename_150)
        data_150 = hdul[1].data
        wgt_150 = hdul[2].data
        wcs_data = wcs.WCS(hdul[1].header)

        # Make the detection image
        data = (data_090*wgt_090 + data_150*wgt_150)/(wgt_090+wgt_150)
        wgt = (wgt_090+wgt_150)

        # data = data_090
        # wgt = wgt_090
        segm[scan], cat[scan] = detect_with_photutils(data, wgt=1/wgt, nsigma_thresh=nsigma,
                                                      npixels=npixels, wcs=wcs_data,
                                                      plot=plot, plot_title=scan)
        cat[scan].add_column(np.array([scan]*len(cat[scan])), name='scan', index=0)

    # Try to do the matching
    stacked = None
    table_centroids = {}  # Table with centroids
    for k in range(len(scans)-1):

        max_sep = 20.0*u.arcsec

        if k == 0:
            scan1 = scans[k]

        scan1 = scans[k]
        scan2 = scans[k+1]
        cat1 = cat[scan1]['sky_centroid']
        cat2 = cat[scan2]['sky_centroid']
        labelID = f"{scan1}_{scan2}"
        print(f"# Doing {scan1} {scan2}")
        # Match method No 1. using match_to_catalog_sky
        # idx1, sep, _ = cat1.match_to_catalog_sky(cat2)
        # idx1_matched = sep < max_sep
        # print(cat[scan1][idx1_matched]['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms', 'kron_flux', 'kron_fluxerr', 'max_value', 'area'])

        # Match method No 2. using match_to_catalog_sky
        print("==============================")
        idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)

        # Get the mean centroid from the matched catalogs
        xx_sky = np.array([cat[scan1][idxcat1]['sky_centroid'].ra, cat[scan2][idxcat2]['sky_centroid'].ra])
        yy_sky = np.array([cat[scan1][idxcat1]['sky_centroid'].dec, cat[scan2][idxcat2]['sky_centroid'].dec])
        xx_pix = np.array([cat[scan1][idxcat1]['xcentroid'], cat[scan2][idxcat2]['xcentroid']])
        yy_pix = np.array([cat[scan1][idxcat1]['ycentroid'], cat[scan2][idxcat2]['ycentroid']])
        xc_sky = np.mean(xx_sky, axis=0)
        yc_sky = np.mean(yy_sky, axis=0)
        xc_pix = np.mean(xx_pix, axis=0)
        yc_pix = np.mean(yy_pix, axis=0)
        ncoords = [2]*len(xc_sky)
        label_col = [labelID]*len(xc_sky)

        # Get the ids with max value
        max_value = np.array([cat[scan1][idxcat1]['max_value'], cat[scan2][idxcat2]['max_value']])
        scan_value = np.array([cat[scan1][idxcat1]['scan'], cat[scan2][idxcat2]['scan']])
        max_value_max = max_value.max(axis=0)
        idmax = max_value.argmax(axis=0)
        scan_max = scan_value.T[0][idmax]

        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        table_centroids[labelID] = Table([label_col, coords, xc_pix, yc_pix, scan_max, max_value_max, ncoords],
                                         names=('labelID', 'sky_centroid', 'xcentroid', 'ycentroid', 'scan_max', 'value_max', 'ncoords'))
        table_centroids[labelID]['xcentroid'].info.format = '.2f'
        table_centroids[labelID]['ycentroid'].info.format = '.2f'
        table_centroids[labelID]['value_max'].info.format = '.6f'
        print(f"centroids Done for {labelID}")

        if k == 0:
            stacked_cat = table_centroids[labelID]['sky_centroid']
            print(table_centroids[labelID])
        else:
            cat1 = stacked_cat
            cat2 = table_centroids[labelID]['sky_centroid']
            idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
            print(table_centroids[labelID][idxcat1])

        fig = plt.figure(figsize=(8, 8))
        x = table_centroids[labelID]['xcentroid']
        y = table_centroids[labelID]['ycentroid']
        plt.scatter(x, y, marker='o', c='k')
        plt.xlabel('x[pixels]')
        plt.ylabel('y[pixels]')
        plt.title(f"{scan1} - {scan2}")
        # plt.show()
    print("Done Part1")
    print("+++++++++++++++++++++++++++++")

    # Find unique centroids
    stacked_centroids = None
    labelIDs = list(table_centroids.keys())
    for k in range(len(labelIDs)-1):

        # Select current and next table IDs
        label1 = labelIDs[k]
        label2 = labelIDs[k+1]

        # Extract the catalogs (i.e. SkyCoord objects) for search_around_sky
        # and make shorcuts of tables
        # For k > 0 we used the stacked/combined catalog
        if k == 0:
            cat1 = table_centroids[label1]['sky_centroid']
            t1 = table_centroids[label1]
        else:
            cat1 = stacked_centroids['sky_centroid']
            t1 = stacked_centroids
        cat2 = table_centroids[label2]['sky_centroid']
        t2 = table_centroids[label2]

        # Find matching objects to avoid duplicates
        idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
        # Define idxnew, the objects not matched in table2/cat2 that need to be appended
        idxall = np.arange(len(cat2))
        idxnew = np.delete(idxall, idxcat2)

        if k == 0:
            xx_sky = stack_cols_lists(t1[idxcat1]['sky_centroid'].ra.data, t2[idxcat2]['sky_centroid'].ra.data)
            yy_sky = stack_cols_lists(t1[idxcat1]['sky_centroid'].dec.data, t2[idxcat2]['sky_centroid'].dec.data)
            xx_pix = stack_cols_lists(t1[idxcat1]['xcentroid'].data, t2[idxcat2]['xcentroid'].data)
            yy_pix = stack_cols_lists(t1[idxcat1]['ycentroid'].data, t2[idxcat2]['ycentroid'].data)
            value_max = stack_cols_lists(t1[idxcat1]['value_max'].data, t2[idxcat2]['value_max'].data)
            scan_max = stack_cols_lists(t1[idxcat1]['scan_max'].data, t2[idxcat2]['scan_max'].data)
        else:
            xx_sky = stack_cols_lists(xx_sky, t2[idxcat2]['sky_centroid'].ra.data)
            yy_sky = stack_cols_lists(yy_sky, t2[idxcat2]['sky_centroid'].dec.data)
            xx_pix = stack_cols_lists(xx_pix, t2[idxcat2]['xcentroid'].data)
            yy_pix = stack_cols_lists(yy_pix, t2[idxcat2]['ycentroid'].data)
            value_max = stack_cols_lists(value_max, t2[idxcat2]['value_max'].data)
            scan_max = stack_cols_lists(scan_max, t2[idxcat2]['scan_max'].data)

        # Here we update the max_values and scan_max label
        # We make them np.array so we can operate on them
        value_max = np.array(value_max)
        scan_max = np.array(scan_max)
        idmax = value_max.argmax(axis=1)
        # We store them back in the same arrays/lists
        value_max = value_max.max(axis=1)
        scan_max = [scan_max[i][idmax[i]] for i in range(len(idmax))]

        # If we have unmatched objects in cat2 (i.e. idxnew has elements), we append these
        if len(idxnew) > 0:
            xx_sky = stack_cols_lists(xx_sky, t2[idxnew]['sky_centroid'].ra.data, append=True)
            yy_sky = stack_cols_lists(yy_sky, t2[idxnew]['sky_centroid'].dec.data, append=True)
            xx_pix = stack_cols_lists(xx_pix, t2[idxnew]['xcentroid'].data, append=True)
            yy_pix = stack_cols_lists(yy_pix, t2[idxnew]['ycentroid'].data, append=True)
            value_max = stack_cols_lists(value_max, t2[idxnew]['value_max'].data, append=True, asscalar=True)
            scan_max = stack_cols_lists(scan_max, t2[idxnew]['scan_max'].data, append=True, asscalar=True)
            # We stacked
            new_stack = vstack([t1[idxcat1], t2[idxnew]])
            stacked_centroids = new_stack
        else:
            stacked_centroids = t1[idxcat1]
            print(f"{label1}-{label2} No new positions to add")

        # Get the average positions so far
        xc_pix = mean_list_of_list(xx_pix)
        yc_pix = mean_list_of_list(yy_pix)
        xc_sky = mean_list_of_list(xx_sky)
        yc_sky = mean_list_of_list(yy_sky)
        # Update the number of coordinates points we have so far
        ncoords = [len(x)+1 for x in xx_pix]

        # Before Update
        print("Before Update")
        print(stacked_centroids)

        # Update centroids with averages
        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        stacked_centroids['sky_centroid'] = coords
        stacked_centroids['xcentroid'] = xc_pix
        stacked_centroids['ycentroid'] = yc_pix
        stacked_centroids['ncoords'] = ncoords
        stacked_centroids['scan_max'] = scan_max
        stacked_centroids['value_max'] = value_max
        stacked_centroids['value_max'].info.format = '.6f'
        stacked_centroids['xcentroid'].info.format = '.2f'
        stacked_centroids['ycentroid'].info.format = '.2f'
        print(f"centroids Done for {label1}")
        print("After Update")
        print(stacked_centroids)


# Plot the final matches in pixels:
fig1 = plt.figure(figsize=(8, 8))
x = stacked_centroids['xcentroid']
y = stacked_centroids['ycentroid']
plt.scatter(x, y, marker='o', c='k')
plt.xlabel('x[pixels]')
plt.ylabel('y[pixels]')
plt.title("Stacked")

# RA/Dec plot
fig2 = plt.figure(figsize=(8, 8))
x = stacked_centroids['sky_centroid'].ra.data
y = stacked_centroids['sky_centroid'].dec.data
plt.scatter(x, y, marker='o', c='k')
plt.xlabel('ra[deg]')
plt.ylabel('dec[deg]')
plt.title("Stacked")
plt.show()
