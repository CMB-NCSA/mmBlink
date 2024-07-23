#!/usr/bin/env python
import numpy
import spt3g_detect.dtools as du
import os
from spt3g import core, maps


if __name__ == "__main__":

    # plot = False
    plot = True
    nsigma_thresh = 3.5
    npixels = 20
    scan = '40-160-9'
    scan = '40-162-9'
    rms2D = True

    flux = {}
    flux_wgt = {}
    flux_mask = {}
    segm = {}
    cat = {}

    path = os.environ['SPT3G_DETECT_DIR']
    g3filename_090 = f"{path}/etc/f090/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.g3.gz"
    g3filename_150 = f"{path}/etc/f150/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.g3.gz"
    print(g3filename_090)
    # Read in the g3 files

    plot_name = os.path.basename(g3filename_090)

    for frame in core.G3File(g3filename_090):
        # only read in the map
        if frame.type != core.G3FrameType.Map:
            continue
        band = frame["Id"]
        print(f"Reading:{frame['Id']}")
        print(f"Reading frame: {frame}")
        print(f"Removing weights: {frame['Id']}")
        maps.RemoveWeights(frame, zero_nans=True)
        flux[band] = numpy.asarray(frame['T'])/core.G3Units.mJy
        flux_wgt[band] = (numpy.asarray(frame['Wunpol'])*core.G3Units.mJy)
        print("Min/Max", flux[band].min(), flux[band].max())
        print("Min/Max", flux_wgt[band].min(), flux_wgt[band].max())
        print(band)
        print(flux[band])
        print(flux_wgt[band])
        data = flux[band]
        wgt = flux_wgt[band]
        segm[scan], cat[scan] = du.detect_with_photutils(data, wgt=None, nsigma_thresh=nsigma_thresh,
                                                         npixels=npixels, wcs=frame['T'].wcs,
                                                         rms2D=rms2D, plot=plot, plot_title=band)
