#!/usr/bin/env python
from spt3g import core, maps
import spt3g_detect.dtools as du
import numpy as np
import logging
import sys
import os
import argparse
import time


def cmdline():

    parser = argparse.ArgumentParser(description="spt3g transient detection")
    parser.add_argument("files", nargs='+',
                        help="Filename(s) to ingest")
    parser.add_argument("--outdir", type=str, action='store', default=None,
                        required=True, help="Location for output files")
    parser.add_argument("--clobber", action='store_true', default=False,
                        help="Clobber output files")

    # Logging options (loglevel/log_format/log_format_date)
    default_log_format = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    default_log_format_date = '%Y-%m-%d %H:%M:%S'
    parser.add_argument("--loglevel", action="store", default='INFO', type=str.upper,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging Level [DEBUG/INFO/WARNING/ERROR/CRITICAL]")
    parser.add_argument("--log_format", action="store", type=str, default=default_log_format,
                        help="Format for logging")
    parser.add_argument("--log_format_date", action="store", type=str, default=default_log_format_date,
                        help="Format for date section of logging")

    # Detection options
    parser.add_argument("--rms2D", action='store_true', default=False,
                        help="Perform 2D map of the rms using photutils Background2D StdBackgroundRMS")
    parser.add_argument("--npixels", action='store', type=int, default=20,
                        help="Compress output files with astropy.io.fits.CompImageHDU")
    parser.add_argument("--nsigma_thresh", action='store', type=float, default=5.0,
                        help="Number of sigmas use to compute the detection threshold")
    parser.add_argument("--max_sep", action='store', type=float, default=35.0,
                        help="Maximum angular separation to match sources in arcsec")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="Plot detection diagnostic?")

    # Use multiprocessing
    parser.add_argument("--np", action="store", default=1, type=int,
                        help="Run using multi-process, 0=automatic, 1=single-process [default]")
    parser.add_argument("--ntheads", action="store", default=1, type=int,
                        help="The number of threads used by numexpr 0=automatic, 1=single [default]")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Keep time
    t0 = time.time()
    args = cmdline()
    d3w = du.detect_3gworker(**args.__dict__)
    t0 = time.time()
    d3w.run_detection_g3files()
    # d3w.run_detection_serial()
    # d3w.run_detection_mp()
    print(f"Total time: {du.elapsed_time(t0)} for [run_detection_g3files]")
    # Example 2, find repeating soueces
    table_centroids = du.find_repeating_sources(d3w.cat, separation=args.max_sep, plot=args.plot)
    stacked_centroids = du.find_unique_centroids(table_centroids, separation=args.max_sep, plot=args.plot)
    print(stacked_centroids)
    exit()

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
    plot = True
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
