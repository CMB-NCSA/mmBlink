import spt3g_detect.dtools as du
from spt3g import core, maps, sources
import numpy as np
import logging
import time
import math
from astropy.nddata import Cutout2D
import astropy.io.fits
import re
import os
import copy

# Mapping of metadata to FITS keywords
_keywords_map = {'ObservationStart': ('DATE-BEG', 'Observation start date'),
                 'ObservationStop': ('DATE-END', 'Observation end date'),
                 'ObservationID': ('OBSID', 'Observation ID'),
                 'Id': ('BAND', 'Band name, Observation Frequency'),
                 'SourceName': ('FIELD', 'Name of object'),
                 }
# Logger
logger = logging.getLogger(__name__)


def g3_to_fits(g3file, trim=True, compress=False, quantize_level=16.0, overwrite=True):
    """Transform g3 file with a FlatSkyMap into a FITS file"""

    # Set the output name
    fitsfile = g3file.split(".")[0] + ".fits"
    # Get a g3 handle
    g3 = core.G3File(g3file)
    # Extract metadata that will augment the header
    metadata = {}
    metadata['PARENT'] = (os.path.basename(g3file), 'Name of parent file')
    for frame in g3:
        logger.debug(f"Reading frame: {frame.type}")
        if frame.type == core.G3FrameType.Observation or frame.type == core.G3FrameType.Map:
            logger.debug(f"Extracting metadata from frame: {frame.type}")
            metadata = extract_metadata_frame(frame, metadata)

        if frame.type == core.G3FrameType.Map:
            t0 = time.time()
            logger.info(f"Transforming to FITS: {frame.type} -- Id: {frame['Id']}")
            logger.debug("Removing weights")
            maps.RemoveWeights(frame, zero_nans=True)
            maps.MakeMapsUnpolarized(frame)
            logger.debug("Removing units --> mJy")
            remove_units(frame, units=core.G3Units.mJy)
            hdr = maps.fitsio.create_wcs_header(frame['T'])
            hdr.update(metadata)
            hdr['UNITS'] = ('mJy', 'Data units')
            if trim:
                field = metadata['FIELD'][0]
                logger.info(f"Will write trimmed FITS file for field: {field}")
                save_skymap_fits_trim(frame, fitsfile, field,
                                      hdr=hdr,
                                      compress=compress,
                                      overwrite=overwrite)
            else:
                # Get the T and weight frames
                T = frame['T']
                W = frame.get('Wpol', frame.get('Wunpol', None))
                maps.fitsio.save_skymap_fits(fitsfile, T,
                                             overwrite=overwrite,
                                             compress=compress,
                                             hdr=hdr,
                                             W=W)
            logger.info(f"Wrote file: {fitsfile}")
            logger.info(f"Time: {du.elapsed_time(t0)}")
    del g3


def extract_metadata_frame(frame, metadata=None):
    """
    Extract selected metadata from a frames
    """
    # Loop over all items and select only the ones in the Mapping
    if not metadata:
        metadata = {}
    for k in iter(frame):
        if k in _keywords_map.keys():
            keyword = _keywords_map[k][0]
            # We need to treat BAND diferently to avoid inconsistensies
            # in how Id is defined (i.e Coadd_90GHz, 90GHz, vs combined_90GHz)
            if keyword == 'BAND':
                try:
                    value = re.findall("90GHz|150GHz|220GHz", frame[k])[0]
                except IndexError:
                    continue
            # Need to re-cast G3Time objects
            elif isinstance(frame[k], core.G3Time):
                value = astropy.time.Time(frame[k].isoformat(), format='isot', scale='utc').isot
            else:
                value = frame[k]
            metadata[keyword] = (value, _keywords_map[k][1])

    return metadata


def get_field_bbox(field, wcs, gridsize=100):
    """
    Get the image extent and central position in pixels for a given WCS
    """
    deg = core.G3Units.deg
    (ra, dec) = sources.get_field_extent(field,
                                         ra_pad=1.5*deg,
                                         dec_pad=3*deg,
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
    logger.debug(f"Found center: ({xc}, {yc})")
    logger.debug(f"Found size: ({xsize}, {ysize})")
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


def save_skymap_fits_trim(frame, fitsfile, field, hdr=None, compress=False,
                          overwrite=True):

    if frame.type != core.G3FrameType.Map:
        raise TypeError(f"Input map: {frame.type} must be a FlatSkyMap or HealpixSkyMap")

    ctype = None
    if compress is True:
        ctype = 'GZIP_2'
    elif isinstance(compress, str):
        ctype = compress

    # Get the T and weight frames
    T = frame['T']
    W = frame.get('Wpol', frame.get('Wunpol', None))

    data_sci = np.asarray(T)
    if W is not None:
        data_wgt = np.asarray(W.TT)
    logger.debug("Read data and weight")

    # Get the box size and center position to trim
    xc, yc, xsize, ysize = get_field_bbox(field, T.wcs)
    # Trim sci and wgt image using astropy cutour2D
    cutout_sci = Cutout2D(data_sci, (xc, yc), (ysize, xsize), wcs=T.wcs)
    cutout_wgt = Cutout2D(data_wgt, (xc, yc), (ysize, xsize), wcs=T.wcs)
    if hdr is None:
        hdr = maps.fitsio.create_wcs_header(T)
    hdr.update(cutout_sci.wcs.to_header())
    hdr_sci = copy.deepcopy(hdr)
    hdr_wgt = copy.deepcopy(hdr)
    hdr_wgt['ISWEIGHT'] = True

    hdul = astropy.io.fits.HDUList()
    if compress:
        logger.debug(f"Will compress using: {ctype} compression")
        hdu_sci = astropy.io.fits.CompImageHDU(
            data=cutout_sci.data,
            name='SCI',
            header=hdr_sci,
            compression_type=ctype)
        if W:
            hdu_wgt = astropy.io.fits.CompImageHDU(
                data=cutout_wgt.data,
                name='WGT',
                header=hdr_wgt,
                compression_type=ctype)
    else:
        hdu_sci = astropy.io.fits.ImageHDU(data=cutout_sci.data, header=hdr)
        if W:
            hdu_wgt = astropy.io.fits.ImageHDU(data=cutout_wgt.data, header=hdr)
    hdul.append(hdu_sci)
    hdul.append(hdu_wgt)
    hdul.writeto(fitsfile, overwrite=overwrite)
    del data_sci
    del data_wgt
    del hdr_sci
    del hdr_wgt
    del hdu_sci
    del hdu_wgt


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
