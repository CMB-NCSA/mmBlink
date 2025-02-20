#!/usr/bin/env python
from astropy.io import fits
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
import spt3g_detect.dtools as du
import os

if __name__ == "__main__":

    # plot = False
    plot = True
    nsigma = 3.5
    npixels = 20
    scans = ['40-160-9']
    scans = ['40-162-9']

    # scans = ['40-157-9', '40-160-9', '40-161-9', '40-162-9', '40-163-9']

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

        # Estimate the 2D rms
        bkg = du.compute_rms2D(data, sigmaclip=None)
        im1 = plt.imshow(bkg.background, origin='lower', cmap='Greys_r')
        plt.colorbar(im1)
        plt.show()

        t = np.mean(bkg.background)
        print(t)

        # data = data_090
        # wgt = wgt_090
        segm[scan], cat[scan] = du.detect_with_photutils(data, wgt=1/wgt, nsigma_thresh=nsigma,
                                                         npixels=npixels, wcs=wcs_data,
                                                         plot=plot, plot_title=scan)
        cat[scan].add_column(np.array([scan]*len(cat[scan])), name='scan', index=0)
