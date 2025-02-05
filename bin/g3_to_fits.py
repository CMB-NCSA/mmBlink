#!/usr/bin/env python

from spt3g import core, maps, sources
import logging
import sys
import time
import spt3g_detect.dtools as du
import numpy as np
from astropy.nddata import Cutout2D
# from astropy.io import fits
import astropy.io.fits
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
FORMAT_DATE = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(FORMAT, FORMAT_DATE)
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_field_bbox(field, wcs, gridsize=100):
    """
    Get the image extent and central position in pixels for a given WCS
    """
    deg = core.G3Units.deg
    (ra, dec) = sources.get_field_extent(field,
                                         ra_pad=1*deg,
                                         dec_pad=2*deg,
                                         sky_pad=True)
    # we convert back from G3units to degrees
    ra = (ra[0]/deg, ra[1]/deg)
    dec = (dec[0]/deg, dec[1]/deg)
    # Get the new ras corners in and see if we cross RA=0
    crossRA0, ra = crossRAzero(ra)
    # Create a grid of ra, dec to estimate the projected extent for the frame WCS
    ras = np.linspace(ra[0], ra[1], gridsize)
    decs = np.linspace(dec[0], dec[1], gridsize)
    ra_grid, dec_grid = np.meshgrid(ras, decs)
    # Transform ra, dec grid to image positions using astropy
    (x_grid, y_grid) = wcs.wcs_world2pix(ra_grid, dec_grid, 0)
    # Get min, max values for x,y grid
    xmin = math.floor(x_grid.min())
    xmax = math.ceil(x_grid.max())
    ymin = math.floor(y_grid.min())
    ymax = math.ceil(y_grid.max())
    # Get the size in pixels rounded to the nearest hundred
    xsize = round((xmax - xmin), -2)
    ysize = round((ymax - ymin), -2)
    xc = round((xmax+xmin)/2.)
    yc = round((ymax+ymin)/2.)
    logger.info(f"Found center: ({xc}, {yc})")
    logger.info(f"Found size: ({xsize}, {ysize})")
    return xc, yc, xsize, ysize


def crossRAzero(ras):
    # Make sure that they are numpy objetcs
    ras = np.array(ras)
    racmin = ras.min()
    racmax = ras.max()
    if (racmax - racmin) > 180.:
        # Currently we switch order. Perhaps better to +/-360.0?
        # Note we want the total extent which is not necessarily the maximum and minimum in this case
        ras2 = ras
        wsm = np.where(ras > 180.0)
        ras2[wsm] = ras[wsm] - 360.
        CROSSRA0 = True
        ras = ras2
    else:
        CROSSRA0 = False
    return CROSSRA0, ras


def write_trimmed_fits(frame, hdr, fitsfile):

    data_sci = np.asarray(frame['T'])
    data_wgt = np.asarray(frame.get('Wpol', frame.get('Wunpol', None)).TT)
    logger.info("Read data and weight")
    wcs = frame['T'].wcs
    # Get the box size and center position to trim
    xc, yc, xsize, ysize = get_field_bbox(frame['SourceName'], wcs)
    # Trim image using astropy cutour2D
    cutout_sci = Cutout2D(data_sci, (xc, yc), (ysize, xsize), wcs=wcs)
    cutout_wgt = Cutout2D(data_wgt, (xc, yc), (ysize, xsize), wcs=wcs)
    hdr.update(cutout_sci.wcs.to_header())
    hdul = astropy.io.fits.HDUList()
    hdr['EXTNAME'] = ('SCI', "Extension Name")
    hdu_sci = astropy.io.fits.ImageHDU(data=cutout_sci.data, header=hdr)
    hdr['EXTNAME'] = ('WGT', "Extension Name")
    hdu_wgt = astropy.io.fits.ImageHDU(data=cutout_wgt.data, header=hdr)
    hdul.append(hdu_sci)
    hdul.append(hdu_wgt)
    hdul.writeto(fitsfile, overwrite=True)
    del data_sci
    del data_wgt


def remove_units(frame, units):
    "Remove units for g3 frame"
    if frame.type != core.G3FrameType.Map:
        return frame

    if frame['T'].weighted:
        t_scale = units
    else:
        t_scale = 1./units
    w_scale = units * units
    for k in ['T', 'Q', 'U']:
        if k in frame:
            frame[k] = frame.pop(k) * t_scale
    for k in ['Wunpol', 'Wpol']:
        if k in frame:
            frame[k] = frame.pop(k) * w_scale
    return frame


def g3_to_fits(g3file, trim=True):
    """Transform into fits file a g3 map"""

    # Set the output name
    fitsfile = g3file.split(".")[0] + ".fits"
    # Get a g3 handle
    g3 = core.G3File(g3file)

    for frame in g3:
        logger.info(f"Reading frame: {frame.type}")
        if frame.type == core.G3FrameType.Observation:
            obsID = frame['ObservationID']
            SourceName = frame['SourceName']

        if frame.type == core.G3FrameType.Map:
            t0 = time.time()
            if 'ObservationID' not in frame:
                logger.info(f"Setting ObservationID to: {obsID}")
                frame['ObservationID'] = obsID
            if 'SourceName' not in frame:
                logger.info(f"Setting SourceName to: {SourceName}")
                frame['SourceName'] = SourceName

            logger.info(f"Transforming to FITS: {frame.type} -- Id: {frame['Id']}")
            maps.RemoveWeights(frame, zero_nans=True)
            maps.MakeMapsUnpolarized(frame)
            remove_units(frame, units=core.G3Units.mJy)
            hdr = maps.fitsio.create_wcs_header(frame['T'])
            hdr['OBSID'] = (frame['ObservationID'], 'Observation ID')
            hdr['FIELD'] = (frame['SourceName'], 'Name of Observing Field')
            hdr['BAND'] = (frame['Id'], 'Observing Frequency')
            if trim:
                logger.info("Will write trimmed FITS file")
                write_trimmed_fits(frame, hdr, fitsfile)
            else:
                maps.fitsio.save_skymap_fits(fitsfile, frame['T'],
                                             overwrite=True,
                                             compress='GZIP_1',
                                             hdr=hdr,
                                             W=frame.get('Wpol', frame.get('Wunpol', None)))

            logger.info(f"Wrote file: {fitsfile}")
            logger.info(f"Time: {du.elapsed_time(t0)}")
    del g3


if __name__ == "__main__":

    g3files = sys.argv[1:]
    for g3file in g3files:
        logger.info(f"Doing file: {g3file}")
        g3_to_fits(g3file, trim=False)
