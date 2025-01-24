#!/usr/bin/env python

from spt3g import core, maps
import logging
import sys
import time
import spt3g_detect.dtools as du

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


def g3_to_fits(g3file):
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
            band = frame["Id"]
            logger.info(f"Transforming to FITS: {frame.type} -- Id: {frame['Id']}")
            maps.RemoveWeights(frame, zero_nans=True)
            maps.MakeMapsUnpolarized(frame)
            remove_units(frame, units=core.G3Units.mJy)
            weight = frame.get('Wpol', frame.get('Wunpol', None))
            band = frame['Id']
            hdr = {}
            hdr['OBSID'] = (obsID, 'Observation ID')
            hdr['FIELD'] = (SourceName, 'Name of Observing Field')
            hdr['BAND'] = (band, 'Observing Frequency')
            maps.fitsio.save_skymap_fits(fitsfile, frame['T'],
                                         overwrite=True,
                                         compress='GZIP_1',
                                         hdr=hdr,
                                         W=weight)
            logger.info(f"Wrote file: {fitsfile}")
            logger.info(f"Time: {du.elapsed_time(t0)}")
    del g3


if __name__ == "__main__":

    g3files = sys.argv[1:]
    for g3file in g3files:
        logger.info(f"Doing file: {g3file}")
        g3_to_fits(g3file)
