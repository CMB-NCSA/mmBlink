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
import matplotlib.pyplot as plt
from astropy.time import Time


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
    """
    Transform a G3 file containing a FlatSkyMap into a FITS file.

    Parameters:
    - g3file (str): Input G3 file path.
    - trim (bool): If True, trims the FITS image to a smaller region.
    - compress (bool): If True, compresses the FITS file.
    - quantize_level (float): Quantization level for the transformation.
    - overwrite (bool): If True, overwrites an existing FITS file.
    """
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
    Extract selected metadata from a frame in a G3 file.

    Parameters:
    - frame: A frame object from the G3 file.
    - metadata (dict): A dictionary to store the extracted metadata (optional).

    Returns:
    - metadata (dict): Updated metadata dictionary with extracted values.
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
    Get the image extent and central position in pixels for a given WCS.

    Parameters:
    - field (str): The name of the field.
    - wcs: WCS (World Coordinate System) object.
    - gridsize (int): Number of grid points along each axis.

    Returns:
    - tuple: (xc, yc, xsize, ysize) where xc, yc are the center coordinates
      and xsize, ysize are the image sizes in pixels.
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
    """
    Check if the RA coordinates cross RA=0 and adjust accordingly.

    Parameters:
    - ras (array): An array of RA coordinates.

    Returns:
    - tuple: A tuple (CROSSRA0, ras) where CROSSRA0 is a boolean indicating if RA
      crosses zero, and ras is the adjusted RA array.
    """    # Make sure that they are numpy objetcs
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
    """
    Save a trimmed version of the sky map to a FITS file.

    Parameters:
    - frame: A frame object containing the map data.
    - fitsfile (str): The path to the output FITS file.
    - field (str): The field name to be used for trimming.
    - hdr (astropy.io.fits.Header): Header to be included in the FITS file (optional).
    - compress (bool): If True, compresses the FITS file.
    - overwrite (bool): If True, overwrites the existing FITS file.
    """
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
    """
    Remove units from the frame, scaling the data accordingly.

    Parameters:
    - frame: A G3 frame object containing the data.
    - units: The unit to scale the data to.

    Returns:
    - frame: The modified G3 frame with units removed.
    """
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


def load_fits_stamp(fits_file):
    """
    Load 2-D images from SCI-n extensions in a multi-extension FITS file.

    Parameters:
        fits_file (str): Path to the FITS file.

    Returns:
        dict: A dictionary with OBSID as keys and 2-D numpy arrays as values.
        str: The ID value from the PRIMARY HDU header.
    """
    images = {}
    with astropy.io.fits.open(fits_file) as hdul:
        # Get NFILES and ID from the primary header
        primary_header = hdul[0].header
        nfiles = primary_header['NFILES']
        id = primary_header['ID']

        # Loop through the extensions to get SCI-n images
        for i in range(1, nfiles + 1):
            sci_ext_name = f"SCI_{i}"
            if sci_ext_name in hdul:
                obsid = hdul[sci_ext_name].header['OBSID']
                images[obsid] = hdul[sci_ext_name].data

    return images, id


def load_fits_table(fits_table_file, target_id):
    """
    Load a FITS table and return the row matching the given 'id'.

    Parameters:
    - fits_table_file (str): Path to the FITS table file.
    - target_id (str): The 'id' to match in the table.

    Returns:
    - dict: A dictionary containing the matching row with columns 'id', 'dates_ave', 'flux_SCI', and 'flux_WGT'.
    """
    with astropy.io.fits.open(fits_table_file) as hdul:
        table_data = hdul[1].data
        ids = table_data['id']

        # Find the index where 'id' matches target_id
        match_index = np.where(ids == target_id)[0]

        if len(match_index) == 0:
            raise ValueError(f"ID {target_id} not found in {fits_table_file}")

        # Extract row data
        row = table_data[match_index[0]]
        return row

def plot_stamps(images_dict):

    n_bands = 1
    n_images = len(images_dict)

    fig = plt.figure(figsize=(n_images*1.8, n_bands*1.8))
    gs = fig.add_gridspec(n_bands, n_images, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex='col', sharey='row')

    # Loop over all of the files
    j = 0
    for band in bands:
        i = 0
        axs[j, 0].set_ylabel(f"{band}GHz", size="x-large")

        for ID in selected_IDs:
            fitsfile = f"spt3gJ215423.8-495636.0_{band}GHz_{ID}_fltd.fits"
            hdu_list = fits.open(fitsfile)
            image_data = hdu_list[0].data
            header = hdu_list[0].header
            date_beg = header["DATE-BEG"]
            t = Time(date_beg, format='isot')
            obstime = f"{t.mjd:.2f}"

            if i == 0:
                t0 = float(obstime)
            days = float(obstime) - t0
            days = f"{days:.2f}"

            #axs[j, i].axis('off')
            axs[j, i].imshow(image_data[x1:x2, y1:y2], origin='lower', cmap='gray_r',
                             vmin=vmin[band], vmax=vmax[band])
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
            axs[j, i].set_xticklabels([])
            axs[j, i].set_yticklabels([])
            # axs[2, i].set_xlabel(obstime, size="large")
            axs[2, i].set_xlabel(str(days), size="large")
            axs[0, i].set_title(obstime, size="large")

            i += 1
        j += 1


    #for ax in fig.get_axes():
    #    ax.label_outer()

    # fig.supxlabel('Time[MJD]')
    fig.supxlabel('Time[days]')
    fig.suptitle('Time[MJD]')

    plt.savefig("flare_example.pdf")
    plt.show()




    def plot_fits_data(images_dict, flux_data_dict):
        """
        Create a multi-panel figure:
        - Top half: Plot all SCI images in order without frames.
        - Bottom half: Scatter plot of flux_SCI vs. dates_ave with flux_WGT as error bars.

        Parameters:
        - images_dict: Dictionary with OBSID as keys and 2D numpy arrays (SCI images) as values.
        - flux_data_dict: Dictionary with keys 'flux_SCI', 'dates_ave', and 'flux_WGT'.
        """
        fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 1]})
        # Top half: Plot all SCI images
        n_images = len(images_dict)
        for idx, (obsid, image) in enumerate(images_dict.items(), 1):
            ax = fig.add_subplot(2, n_images, idx)
            ax.imshow(image, origin='lower', cmap='viridis')
            ax.set_title(f"{obsid}")
            ax.axis('off')  # Remove the frame and ticks around the images

        plt.tight_layout()
        axes[0].axis('off')  # Remove the frame and ticks around the images

        # Bottom half: Scatter plot of flux_SCI vs. dates_ave with flux_WGT as error bars
        dates_ave = flux_data_dict['dates_ave']
        flux_SCI = flux_data_dict['flux_SCI']
        flux_WGT = flux_data_dict['flux_WGT']

        # Convert the first MJD date to a calendar date using Astropy
        start_date = Time(dates_ave[0], format='mjd').iso

        # Shift dates_ave to start from the first date
        dates_ave = [date - dates_ave[0] for date in dates_ave]

        # Figure out the error
        axes[1].errorbar(dates_ave, flux_SCI, yerr=1/np.sqrt(flux_WGT), fmt='o', color='blue', ecolor='red', capsize=3)

        # Add start date to the xlabel
        # Only take the 'YYYY-MM-DD' part
        axes[1].set_xlabel(f'Days since first observation (Start Date: {start_date[:10]})')
        axes[1].set_ylabel('Flux (SCI)')
        axes[1].set_title('Flux SCI vs. Days since first observation')

        plt.tight_layout()
        plt.show()
