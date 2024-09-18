#!/usr/bin/env python
from spt3g import core, maps
import spt3g_detect.dtools as du
import numpy as np
import sys
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.table import Table, vstack
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # plot = False
    plot = True
    nsigma_thresh = 3.5
    npixels = 20
    rms2D = True
    g3files = sys.argv[1:]

    flux = {}
    flux_wgt = {}
    flux_mask = {}
    segm = {}
    cat = {}
    outdir = "outdir"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for g3filename in g3files:
        print(f"Opening file: {g3filename}")
        for frame in core.G3File(g3filename):
            # only read in the map
            if frame.type != core.G3FrameType.Map:
                continue
            obsID = frame['ObservationID']
            band = frame["Id"]
            key = f"{obsID}_{band}"
            plot_name = os.path.join(outdir, key)
            print(f"Reading:{frame['Id']}")
            print(f"Reading frame: {frame}")
            print(f"ObservationID: {obsID}")
            print(f"Removing weights: {frame['Id']}")
            maps.RemoveWeights(frame, zero_nans=True)
            flux[key] = np.asarray(frame['T'])/core.G3Units.mJy
            flux_wgt[key] = np.asarray(frame['Wunpol'].TT)*core.G3Units.mJy
            print("Min/Max", flux[key].min(), flux[key].max())
            print("Min/Max", flux_wgt[key].min(), flux_wgt[key].max())
            data = flux[key]
            wgt = 1/flux_wgt[key]
            segm[key], cat[key] = du.detect_with_photutils(data, wgt=wgt, nsigma_thresh=nsigma_thresh,
                                                           npixels=npixels, wcs=frame['T'].wcs,
                                                           rms2D=rms2D, plot=plot, plot_name=plot_name,
                                                           plot_title=band)

            cat[key].add_column(np.array([key]*len(cat[key])), name='scan', index=0)

    # Step 1 -- match catalogs for souces that show up more than once
    # Try to do the matching
    max_sep = 20.0*u.arcsec
    stacked = None
    table_centroids = {}  # Table with centroids
    scans = list(cat.keys())
    print(f"scans: {scans}")
    for k in range(len(scans)-1):
        print(scans[k])

        if k == 0:
            scan1 = scans[k]

        scan1 = scans[k]
        scan2 = scans[k+1]
        cat1 = cat[scan1]['sky_centroid']
        cat2 = cat[scan2]['sky_centroid']
        n1 = len(cat[scan1])
        n2 = len(cat[scan2])
        labelID = f"{scan1}_{scan2}"
        print(f"# Doing {scan1} vs {scan2}")
        print(f"# N in cat1: {n1} cat2: {n2}")

        # Altenative match method using match_to_catalog_sky
        # idx1, sep, _ = cat1.match_to_catalog_sky(cat2)
        # idx1_matched = sep < max_sep
        # print(cat[scan1][idx1_matched]['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms', 'kron_flux', 'kron_fluxerr', 'max_value', 'area'])

        # Match method using search_around_sky
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
                                         names=('labelID', 'sky_centroid', 'xcentroid', 'ycentroid', 'scan_max',
                                                'value_max', 'ncoords'))
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

        print(f"### Doing: {k}/{len(labelIDs)-2} --- ###")

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
        n1 = len(cat1)
        n2 = len(cat2)
        nx1 = len(idxcat1)
        nx2 = len(idxcat2)
        idxall = np.arange(n2)
        idxnew = np.delete(idxall, idxcat2)

        print(f"# label1: {label1}")
        print(f"# label2: {label2}")
        print(f"len(idxcat1): {nx1}")
        print(f"len(idxcat2): {nx2}")
        print(f"idxcat1: {idxcat1}")
        print(f"idxcat2: {idxcat2}")
        print(f"idxall: {idxall}")
        print(f"idxnew: {idxnew}")
        print(f"iter:{k} len(cat1): {len(cat1)}")
        print(f"iter:{k} len(cat2): {len(cat2)}")

        for kk in range(nx1):
            i = idxcat1[kk]
            j = idxcat2[kk]
            if k > 0:
                print(i, j)
                print(xx_pix[i], t2['xcentroid'].data[j])

        if k == 0:
            xx_sky = du.stack_cols_lists(t1['sky_centroid'].ra.data, t2['sky_centroid'].ra.data, idxcat1, idxcat2,)
            yy_sky = du.stack_cols_lists(t1['sky_centroid'].dec.data, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = du.stack_cols_lists(t1['xcentroid'].data, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = du.stack_cols_lists(t1['ycentroid'].data, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = du.stack_cols_lists(t1['value_max'].data, t2['value_max'].data, idxcat1, idxcat2, pad=True)
            scan_max = du.stack_cols_lists(t1['scan_max'].data, t2['scan_max'].data, idxcat1, idxcat2, pad=True)
        else:
            xx_sky = du.stack_cols_lists(xx_sky, t2['sky_centroid'].ra.data, idxcat1, idxcat2)
            yy_sky = du.stack_cols_lists(yy_sky, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = du.stack_cols_lists(xx_pix, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = du.stack_cols_lists(yy_pix, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = du.stack_cols_lists(value_max, t2['value_max'].data, idxcat1, idxcat2, pad=True)
            scan_max = du.stack_cols_lists(scan_max, t2['scan_max'].data, idxcat1, idxcat2, pad=True)

        for k in range(len(xx_pix)):
            print(k, xx_pix[k], yy_pix[k], value_max[k], scan_max[k])

        print(xx_pix)
        print(value_max)
        # Here we update the max_values and scan_max label
        # We make them np.array so we can operate on them
        value_max = np.array(value_max)
        scan_max = np.array(scan_max)
        idmax = value_max.argmax(axis=1)
        # We store them back in the same arrays/lists
        value_max = value_max.max(axis=1)
        scan_max = [scan_max[i][idmax[i]] for i in range(len(idmax))]

        print(value_max)
        print(scan_max)
        print(idmax)

        # If we have unmatched objects in cat2 (i.e. idxnew has elements), we append these
        if len(idxnew) > 0:
            new_stack = vstack([t1, t2[idxnew]])
            stacked_centroids = new_stack
        else:
            stacked_centroids = t1
            print(f"{label1}-{label2} No new positions to add")

        # Get the average positions so far
        xc_pix = du.mean_list_of_list(xx_pix)
        yc_pix = du.mean_list_of_list(yy_pix)
        xc_sky = du.mean_list_of_list(xx_sky)
        yc_sky = du.mean_list_of_list(yy_sky)
        # Update the number of coordinates points we have so far
        ncoords = [len(x)+1 for x in xx_pix]

        print("## ----- ##")
        print(xc_pix)
        print(yc_pix)
        print("## ----- ##")

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
        print("#### stacked_centroids")
        print(stacked_centroids)
        #print("#### t1 -------------")
        #print(t1)
        #print("#### t2 -------------")
        #print(t2)
        print("## ---- ##")

    exit()



    # for key in table_centroids.keys():
    #    print(f"key: {key}")
    #    print(table_centroids[key])

    # Find unique centroids
    stacked_centroids = None
    labelIDs = list(table_centroids.keys())
    for k in range(len(labelIDs)-1):

        # Select current and next table IDs
        label1 = labelIDs[k]
        label2 = labelIDs[k+1]

        print(f"{k}/{len(labelIDs)-1}")

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
        # idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
        idxcat2, idxcat1, d2d, _ = cat1.search_around_sky(cat2, max_sep)

        print(f"# label1: {label1}")
        print(f"# label2: {label2}")
        print(f"len(idxcat1): {len(idxcat1)}")
        print(f"len(idxcat2): {len(idxcat2)}")
        print(f"idxcat1: {idxcat1}")
        print(f"idxcat2: {idxcat2}")

        # Define idxnew, the objects not matched in table1/cat1 that need to be appended
        idxall = np.arange(len(cat1))
        idxnew = np.delete(idxall, idxcat1)
        print(f"idxall: {idxall}")
        print(f"idxnew: {idxnew}")

        n1 = len(cat1)
        n2 = len(cat2)
        print(f"{k} len(cat1): {len(cat1)}")
        print(f"{k} len(cat2): {len(cat2)}")

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
            print(f"Adding idxnew: {idxnew}")
            xx_sky = du.stack_cols_lists(xx_sky, t1[idxnew]['sky_centroid'].ra.data, append=True)
            yy_sky = du.stack_cols_lists(yy_sky, t1[idxnew]['sky_centroid'].dec.data, append=True)
            xx_pix = du.stack_cols_lists(xx_pix, t1[idxnew]['xcentroid'].data, append=True)
            yy_pix = du.stack_cols_lists(yy_pix, t1[idxnew]['ycentroid'].data, append=True)
            value_max = du.stack_cols_lists(value_max, t1[idxnew]['value_max'].data, append=True, asscalar=True)
            scan_max = du.stack_cols_lists(scan_max, t1[idxnew]['scan_max'].data, append=True, asscalar=True)
            # We stacked
            new_stack = vstack([t2[idxcat2], t1[idxnew]])
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
        print("#### stacked_centroids")
        print(stacked_centroids)
        print("#### t1 -------------")
        print(t1)
        print("#### t2 -------------")
        print(t2)
        print("#---")
        exit()
