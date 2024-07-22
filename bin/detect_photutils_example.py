#!/usr/bin/env python
from astropy.io import fits
from astropy import wcs
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.coordinates import FK5
import matplotlib.pyplot as plt
import spt3g_detect.dtools as du
import os

if __name__ == "__main__":

    plot = False
    # plot = True
    nsigma = 3.5
    npixels = 20
    scans = ['40-157-9', '40-160-9', '40-161-9', '40-162-9', '40-163-9']

    segm = {}
    cat = {}

    path = os.environ['SPT3G_DETECT_DIR']
    for scan in scans:
        print(scan)
        filename_090 = f"{path}/etc/f090/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.fits"
        filename_150 = f"{path}/etc/f150/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.fits"

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
        segm[scan], cat[scan] = du.detect_with_photutils(data, wgt=1/wgt, nsigma_thresh=nsigma,
                                                         npixels=npixels, wcs=wcs_data,
                                                         plot=plot, plot_title=scan)
        cat[scan].add_column(np.array([scan]*len(cat[scan])), name='scan', index=0)

    # Try to do the matching
    stacked = None
    table_centroids = {}  # Table with centroids
    for k in range(len(scans)-1):

        max_sep = 20.0*units.arcsec

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
            xx_sky = du.stack_cols_lists(t1[idxcat1]['sky_centroid'].ra.data, t2[idxcat2]['sky_centroid'].ra.data)
            yy_sky = du.stack_cols_lists(t1[idxcat1]['sky_centroid'].dec.data, t2[idxcat2]['sky_centroid'].dec.data)
            xx_pix = du.stack_cols_lists(t1[idxcat1]['xcentroid'].data, t2[idxcat2]['xcentroid'].data)
            yy_pix = du.stack_cols_lists(t1[idxcat1]['ycentroid'].data, t2[idxcat2]['ycentroid'].data)
            value_max = du.stack_cols_lists(t1[idxcat1]['value_max'].data, t2[idxcat2]['value_max'].data)
            scan_max = du.stack_cols_lists(t1[idxcat1]['scan_max'].data, t2[idxcat2]['scan_max'].data)
        else:
            xx_sky = du.stack_cols_lists(xx_sky, t2[idxcat2]['sky_centroid'].ra.data)
            yy_sky = du.stack_cols_lists(yy_sky, t2[idxcat2]['sky_centroid'].dec.data)
            xx_pix = du.stack_cols_lists(xx_pix, t2[idxcat2]['xcentroid'].data)
            yy_pix = du.stack_cols_lists(yy_pix, t2[idxcat2]['ycentroid'].data)
            value_max = du.stack_cols_lists(value_max, t2[idxcat2]['value_max'].data)
            scan_max = du.stack_cols_lists(scan_max, t2[idxcat2]['scan_max'].data)

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
            xx_sky = du.stack_cols_lists(xx_sky, t2[idxnew]['sky_centroid'].ra.data, append=True)
            yy_sky = du.stack_cols_lists(yy_sky, t2[idxnew]['sky_centroid'].dec.data, append=True)
            xx_pix = du.stack_cols_lists(xx_pix, t2[idxnew]['xcentroid'].data, append=True)
            yy_pix = du.stack_cols_lists(yy_pix, t2[idxnew]['ycentroid'].data, append=True)
            value_max = du.stack_cols_lists(value_max, t2[idxnew]['value_max'].data, append=True, asscalar=True)
            scan_max = du.stack_cols_lists(scan_max, t2[idxnew]['scan_max'].data, append=True, asscalar=True)
            # We stacked
            new_stack = vstack([t1[idxcat1], t2[idxnew]])
            stacked_centroids = new_stack
        else:
            stacked_centroids = t1[idxcat1]
            print(f"{label1}-{label2} No new positions to add")

        # Get the average positions so far
        xc_pix = du.mean_list_of_list(xx_pix)
        yc_pix = du.mean_list_of_list(yy_pix)
        xc_sky = du.mean_list_of_list(xx_sky)
        yc_sky = du.mean_list_of_list(yy_sky)
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
