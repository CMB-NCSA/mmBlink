#!/usr/bin/env python
from spt3g import core, maps
import spt3g_detect.dtools as du
import numpy as np
import logging
import sys
import os


if __name__ == "__main__":

    # Create logger
    loglevel = 'DEBUG'
    loglevel = 'INFO'
    log_format = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    log_format_date = '%Y-%m-%d %H:%M:%S'

    du.create_logger(level=loglevel,
                     log_format=log_format,
                     log_format_date=log_format_date)
    logger = logging.getLogger(__name__)
    logger.info("Logger Created")

    # plot = False
    plot = False
    nsigma_thresh = 5
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
            logger.info(f"Reading:{frame['Id']}")
            logger.info(f"Reading frame: {frame}")
            logger.info(f"ObservationID: {obsID}")
            logger.info(f"Removing weights: {frame['Id']}")
            maps.RemoveWeights(frame, zero_nans=True)
            flux[key] = np.asarray(frame['T'])/core.G3Units.mJy
            flux_wgt[key] = np.asarray(frame['Wunpol'].TT)*core.G3Units.mJy
            logger.info(f"Min/Max {flux[key].min()} {flux[key].max()}")
            logger.info(f"Min/Max {flux_wgt[key].min()} {flux_wgt[key].max()}")
            data = flux[key]
            wgt = 1/flux_wgt[key]
            segm[key], cat[key] = du.detect_with_photutils(data, wgt=wgt, nsigma_thresh=nsigma_thresh,
                                                           npixels=npixels, wcs=frame['T'].wcs,
                                                           rms2D=rms2D, plot=plot, plot_name=plot_name,
                                                           plot_title=band)
            if cat[key] is None:
                del cat[key]
                del segm[key]
            else:
                cat[key].add_column(np.array([key]*len(cat[key])), name='scan', index=0)
                cat[key].add_column(np.array([key]*len(cat[key])), name='scan_max', index=0)

    # Try to do the matching
    max_sep = 35.0
    plot = True

    # Example 1, find all positions
    # stacked_centroids = du.find_unique_centroids(cat, separation=max_sep, plot=plot)

    # Example 2, find repeating soueces
    table_centroids = du.find_repeating_sources(cat, separation=max_sep, plot=plot)
    stacked_centroids = du.find_unique_centroids(table_centroids, separation=max_sep, plot=plot)
    print(stacked_centroids)
