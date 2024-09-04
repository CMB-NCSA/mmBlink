#!/usr/bin/env python
import numpy
import spt3g_detect.dtools as du
import os
from spt3g import core, maps
import sys

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
            flux[key] = numpy.asarray(frame['T'])/core.G3Units.mJy
            flux_wgt[key] = numpy.asarray(frame['Wunpol'].TT)*core.G3Units.mJy
            print("Min/Max", flux[key].min(), flux[key].max())
            print("Min/Max", flux_wgt[key].min(), flux_wgt[key].max())
            data = flux[key]
            wgt = 1/flux_wgt[key]
            segm[key], cat[key] = du.detect_with_photutils(data, wgt=wgt, nsigma_thresh=nsigma_thresh,
                                                           npixels=npixels, wcs=frame['T'].wcs,
                                                           rms2D=rms2D, plot=plot, plot_name=plot_name,
                                                           plot_title=band)
