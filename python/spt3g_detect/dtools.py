import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from scipy.stats import norm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy import units as u
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.coordinates import search_around_sky
from astropy.table import Table, vstack
import photutils.background
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import multiprocessing as mp
import types
import magic
import errno
import time
import spt3g_detect
from spt3g import core, maps
from spt3g import sources
import fitsio
import copy
from astropy.wcs import WCS
from astropy.io import ascii
from collections import OrderedDict


# Logger
LOGGER = logging.getLogger(__name__)
# Set matplotlib logger at warning level to disengable from default logger
plt.set_loglevel(level='warning')


# Naming template
PREFIX = 'SPT3G'
OBJ_ID = "{prefix}J{ra}{dec}"
FITS_OUTNAME = "{outdir}/{objID}_{obsid}_{filter}.{ext}"
LOG_OUTNAME = "{outdir}/{objID}.{ext}"
BASE_OUTNAME = "{objID}"
BASEDIR_OUTNAME = "{outdir}/{objID}"


class detect_3gworker:

    """ A Class to run and manage Transient detections on SPT files/frames"""

    def __init__(self, **keys):

        # Load the configurarion
        self.config = types.SimpleNamespace(**keys)

        # Start Logging
        self.setup_logging()

        # Prepare things
        self.prepare()

        # Check input files vs file list
        self.check_input_files()

    def prepare(self):
        """Intit some functions"""
        # Get the number of processors to use
        self.NP = get_NP(self.config.np)
        create_dir(self.config.outdir)

        # Dictionaries to store the data, we need manager dictionaries when
        # using multiprocessing
        if self.NP > 1:
            manager = mp.Manager()
            self.cat = manager.dict()
            self.segm = manager.dict()
            self.flux = manager.dict()
            self.flux_wgt = manager.dict()
            self.flux_mask = manager.dict()
            self.header = manager.dict()
        else:
            self.flux = {}
            self.flux_wgt = {}
            self.flux_mask = {}
            self.cat = {}
            self.segm = {}
            self.header = {}

    def setup_logging(self):
        """ Simple logger that uses configure_logger() """

        # Create the logger
        create_logger(level=self.config.loglevel,
                      log_format=self.config.log_format,
                      log_format_date=self.config.log_format_date)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging Started at level:{self.config.loglevel}")
        self.logger.info(f"Running spt3g_ingest version: {spt3g_detect.__version__}")

    def check_input_files(self):
        " Check if the inputs are a list or a file with a list"
        # The number of files to process
        self.nfiles = len(self.config.files)

        t = magic.Magic(mime=True)
        if self.nfiles == 1 and t.from_file(self.config.files[0]) == 'text/plain':
            self.logger.info(f"{self.config.files[0]} is a list of files")
            # Now read them in
            with open(self.config.files[0], 'r') as f:
                lines = f.read().splitlines()
            self.logger.info(f"Read: {len(lines)} input files")
            self.config.files = lines
            self.nfiles = len(lines)
        else:
            self.logger.info(f"Detected list of [{self.nfiles}] files")

    def load_g3frames(self, g3filename, k):
        """Read in the data and metadata in a g3frame"""
        t0 = time.time()
        self.logger.info(f"Opening file: {g3filename}")
        self.logger.info(f"Doing: {k}/{self.nfiles} files")

        frames = []
        metadata_extracted = False
        for frame in core.G3File(g3filename):
            # Extract ObservationID and field (SourceName)
            if frame.type == core.G3FrameType.Observation:
                obsID = frame['ObservationID']
                try:
                    SourceName = frame['SourceName']
                    if self.config.field is not None and SourceName != self.config.field:
                        self.logger.warning(f"Extracted SourceName: {SourceName} doesn't match configuration")
                except KeyError:
                    if self.config.field is not None:
                        SourceName = self.config.field
                    self.logger.warning("Could not extract SourceName from Observation frame")
                metadata_extracted = True

            # check if obsID/SourceName are actualy in the frame
            elif frame.type == core.G3FrameType.Map and metadata_extracted is False:
                try:
                    obsID = frame['ObservationID']
                except KeyError:
                    self.logger.warning("Could not extract obsID from frame")
                try:
                    SourceName = frame['SourceName']
                except KeyError:
                    SourceName = ''
                    self.logger.warning("Could not extract SourceName from frame")

            # only read in the map
            if frame.type != core.G3FrameType.Map:
                continue

            if 'ObservationID' not in frame:
                self.logger.info(f"Setting ObservationID to: {obsID}")
                frame['ObservationID'] = obsID
            if 'SourceName' not in frame:
                self.logger.info(f"Setting SourceName to: {SourceName}")
                frame['SourceName'] = SourceName
            frames.append(frame)

        self.logger.info(f"Total metadata time: {elapsed_time(t0)} for: {g3filename}")
        return frames

    def load_fits_map(self, filename):
        # Get header/extensions/hdu
        t0 = time.time()
        header, hdunum = get_headers_hdus(filename)
        key = f"{header['SCI']['OBSID']}_{header['SCI']['BAND']}"
        self.logger.info(f"Setting key as: {key}")
        self.logger.debug(f"Done Getting header, hdus: {elapsed_time(t0)}")
        extnames = header.keys()  # Gets SCI and WGT
        HDU_SCI = hdunum['SCI']
        HDU_WGT = hdunum['WGT']
        self.logger.debug(f"Found EXTNAMES:{extnames}")

        # Intitialize the FITS object
        ifits = fitsio.FITS(filename, 'r')
        self.logger.debug(f"Done loading fitsio.FITS({filename}): {elapsed_time(t0)}")
        self.logger.debug("Reading SCI HDU")
        self.flux[key] = ifits[HDU_SCI].read()
        self.logger.debug("Reading WGT HDU")
        self.flux_wgt[key] = ifits[HDU_WGT].read()
        self.header[key] = header['SCI']
        ifits.close()
        self.logger.debug("Done Reading")
        self.flux_mask[key] = np.where(self.flux_wgt[key] != 0, int(1), 0)
        self.logger.debug(f"Min/Max Flux: {self.flux[key].min()} {self.flux[key].max()}")
        self.logger.debug(f"Min/Max Wgt: {self.flux_wgt[key].min()} {self.flux_wgt[key].max()}")
        self.logger.info(f"Done loading filename: {filename} in {elapsed_time(t0)}")
        return [key]

    def load_g3frame_map(self, frame):
        # Get the metadata
        t0 = time.time()
        obsID = frame['ObservationID']
        band = frame["Id"]
        key = f"{obsID}_{band}"
        field = frame['SourceName']

        # Create a fits header for the frame map
        hdr = frame['T'].wcs.to_header()
        # Add OBSID, BAND and FIELD to the header (for thumbnails, etc)
        hdr['OBSID'] = (obsID, 'Observation ID')
        hdr['FIELD'] = (field, 'Name of Observing Field')
        hdr['BAND'] = (band, 'Observing Frequency')
        self.header[key] = hdr

        self.logger.info(f"Reading frame[Id]: {frame['Id']}")
        self.logger.debug(f"Reading frame: {frame}")
        self.logger.debug(f"ObservationID: {obsID}")
        self.logger.debug(f"Removing weights: {frame['Id']}")
        t1 = time.time()
        maps.RemoveWeights(frame, zero_nans=True)
        self.logger.info(f"Remove Weights time: {elapsed_time(t1)}[s]")
        self.flux[key] = np.asarray(frame['T'])/core.G3Units.mJy
        self.flux_wgt[key] = np.asarray(frame['Wunpol'].TT)*core.G3Units.mJy*core.G3Units.mJy
        self.logger.debug(f"Min/Max Flux: {self.flux[key].min()} {self.flux[key].max()}")
        self.logger.debug(f"Min/Max Wgt: {self.flux_wgt[key].min()} {self.flux_wgt[key].max()}")
        # Now we exctract the mask
        try:
            # Zero is no data, and Ones is data
            g3_mask = frame["T"].to_mask()
            g3_mask_map = g3_mask.to_map()
            flux_mask = np.asarray(g3_mask_map)
            self.flux_mask[key] = np.where(flux_mask == 1, int(1), 0)
        except Exception as e:
            self.logger.warning(e.message)
            self.flux_mask[key] = None
        self.logger.info(f"Map from frame loaded for {obsID} {band}: {elapsed_time(t0)}[s]")
        return key

    def detect_with_photutils_key(self, key):

        data = self.flux[key]
        wgt = self.flux_wgt[key]
        mask = self.flux_mask[key]
        wcs = WCS(self.header[key])
        plot_name = os.path.join(self.config.outdir, f"{key}_cat")
        plot_title = self.header[key]['BAND']
        field = self.header[key]['FIELD']
        self.segm[key], self.cat[key] = detect_with_photutils(data, wgt=wgt, mask=mask,
                                                              nsigma_thresh=self.config.nsigma_thresh,
                                                              npixels=self.config.npixels, wcs=wcs,
                                                              rms2D=self.config.rms2D, box=self.config.rms2D_box,
                                                              plot=self.config.plot,
                                                              plot_name=plot_name, plot_title=plot_title)

        # Test to dump flux wgt into a fits file
        # fits = fitsio.FITS(f"test_{key}.fits", 'rw', clobber=True)
        # fits.write(self.flux[key])
        # fits.write(self.flux_wgt[key])
        # fits.write(self.flux_mask[key])
        # fits.close()

        # Remove objects that match the sources catalog for that field
        if self.cat[key] is not None:
            self.cat[key] = remove_objects_near_sources(self.cat[key], field)

        # if no detections (i.e. None) or no objecs in catalog (i.e. all objects were removed)
        # we remove it from dictionaries
        if self.cat[key] is None or len(self.cat[key]) == 0:
            self.logger.info(f"Removing key: {key} from catalog dictionary ")
            del self.cat[key]
            del self.segm[key]
        else:
            # Here is a good place to plot detections -- experimental
            self.logger.info(f"Writing thumbnails for {key}")
            self.write_thumbnails_fitsio(key)
            catname = f"{key}.cat"
            ascii.write(self.cat[key]['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms',
                        'kron_flux', 'kron_fluxerr', 'max_value', 'elongation',
                        'ellipticity', 'area'], catname,
                        overwrite=True, format='fixed_width')
            self.logger.info(f"Wrote catalog to: {catname}")
        return key

    def add_scan_column_to_cat(self):
        """
        Add scan and scan_max columns to catalogs
        This needs to be done outside the MP call,
        otherwise the dictionaries are not updated as excepeted
        """
        for key in self.cat.keys():
            self.logger.info(f"Adding scan column for {key}")
            self.cat[key].add_column(np.array([key]*len(self.cat[key])), name='scan', index=0)
            self.cat[key].add_column(np.array([key]*len(self.cat[key])), name='scan_max', index=0)

    def run_detection_file(self, filename, k):
        """
        Run the task(s) for one g3file.
        The outputs are stored in self.cat and self.segm
        """
        t0 = time.time()
        # We need to setup logging again for MP
        if self.NP > 1:
            self.setup_logging()
        self.logger.info(f"Opening file: {filename}")
        self.logger.info(f"Doing: {k}/{self.nfiles} files")
        # Check if g3 or FITS file
        filetype = g3_or_fits(filename)
        self.logger.info(f"This file: {filename} is a {filetype} file")

        # We need to loop as each frame can contain more than one map (mult-band case)
        if filetype == "G3":
            # Load framea and extract keys
            frames = self.load_g3frames(filename, k)
            keys = [self.load_g3frame_map(frame) for frame in frames]
        elif filetype == "FITS":
            keys = self.load_fits_map(filename)

        for key in keys:
            self.logger.info(f"Running detection for {key}")
            self.detect_with_photutils_key(key)
        self.logger.info(f"Completed: {k}/{self.nfiles} files")
        self.logger.info(f"Total time: {elapsed_time(t0)} for: {filename}")

    def run_detection_files(self):
        " Run all g3files"
        if self.NP > 1:
            self.logger.info("Running detection jobs with multiprocessing")
            # self.run_detection_async()
            self.run_detection_mp()
        else:
            self.logger.info("Running detection jobs serialy")
            self.run_detection_serial()

        # Finally we add the scan and scan_max columns
        self.add_scan_column_to_cat()

    def run_detection_mp(self):
        " Run g3files using multiprocessing.Process in chunks of NP"
        k = 1
        jobs = []
        self.logger.info(f"Will use {self.NP} processors")
        # Loop one to defined the jobs
        for g3file in self.config.files:
            self.logger.info(f"Starting mp.Process for {g3file}")
            fargs = (g3file, k)
            p = mp.Process(target=self.run_detection_file, args=fargs)
            jobs.append(p)
            k += 1

        # Loop over the process in chunks of size NP
        for job_chunk in chunker(jobs, self.NP):
            for job in job_chunk:
                self.logger.info(f"Starting job: {job.name}")
                job.start()
            for job in job_chunk:
                self.logger.info(f"Joining job: {job.name}")
                job.join()

        # Update with returned dictionary, we need to make them real
        # dictionaries, instead DictProxy objects returned from multiprocessing
        self.logger.info("Updating returned dictionaries")
        self.cat = self.cat.copy()
        self.segm = self.segm.copy()
        p.terminate()

    def run_detection_async(self):
        # It might have memory issues with spt3g pipe()
        " Run g3files using multiprocessing.apply_async"
        with mp.get_context('spawn').Pool() as p:
            p = mp.Pool(processes=self.NP, maxtasksperchild=1)
            self.logger.info(f"Will use {self.NP} processors")
            k = 1
            for g3file in self.config.files:
                fargs = (g3file, k)
                kw = {}
                self.logger.info(f"Starting apply_async.Process for {g3file}")
                p.apply_async(self.run_detection_file, fargs, kw)
                k += 1
            p.close()
            p.join()

        # Update with returned dictionary, we need to make them real
        # dictionaries, instead DictProxy objects returned from multiprocessing
        self.logger.info("Updating returned dictionaries")
        self.cat = self.cat.copy()
        self.segm = self.segm.copy()
        p.terminate()

    def run_detection_serial(self):
        " Run all g3files serialy "
        k = 1
        for g3file in self.config.files:
            self.run_detection_file(g3file, k)
            k += 1

    def write_thumbnails_fitsio(self, key, size=60, clobber=True):
        """Plot the detections as thumbnails"""

        cat = self.cat[key]
        data = self.flux[key]
        wgt = self.flux_wgt[key]
        hdr = self.header[key]
        # Make a FITSHDR object
        if not isinstance(hdr, fitsio.header.FITSHDR):
            hdr = astropy2fitsio_header(hdr)

        dx = int(size/2.0)
        dy = int(size/2.0)
        wcs = WCS(hdr)

        for k in range(len(cat)):
            t0 = time.time()
            x0 = round(cat['xcentroid'][k])
            y0 = round(cat['ycentroid'][k])
            y1 = y0 - dy
            y2 = y0 + dy
            x1 = x0 - dx
            x2 = x0 + dx
            outname = f"{x0}_{y0}.fits"
            thumb = data[int(y1):int(y2), int(x1):int(x2)]
            thumb_wgt = wgt[int(y1):int(y2), int(x1):int(x2)]
            h_section = update_wcs_matrix(hdr, x1, y1)
            # Construct the name of the Thumbmail using BAND/FILTER/prefix/etc
            ra, dec = wcs.wcs_pix2world(x0, y0, 1)
            objID = get_thumbBaseName(ra, dec, prefix='SPT')
            outname = get_thumbFitsName(ra, dec, hdr['BAND'], hdr['OBSID'],
                                        objID=objID, prefix='SPT', outdir=".")

            ofits = fitsio.FITS(outname, 'rw', clobber=clobber)
            ofits.write(thumb, extname='SCI', header=h_section)
            ofits.write(thumb_wgt, extname='WGT', header=h_section)
            ofits.close()
            LOGGER.info(f"Done writing {outname}: {elapsed_time(t0)}")


def configure_logger(logger, logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Configure an existing logger
    """
    # Define formats
    if log_format:
        FORMAT = log_format
    else:
        FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    if log_format_date:
        FORMAT_DATE = log_format_date
    else:
        FORMAT_DATE = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(FORMAT, FORMAT_DATE)

    # Need to set the root logging level as setting the level for each of the
    # handlers won't be recognized unless the root level is set at the desired
    # appropriate logging level. For example, if we set the root logger to
    # INFO, and all handlers to DEBUG, we won't receive DEBUG messages on
    # handlers.
    logger.setLevel(level)

    handlers = []
    # Set the logfile handle if required
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=2000000, backupCount=10)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
        logger.addHandler(fh)

    # Set the screen handle
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    handlers.append(sh)
    logger.addHandler(sh)

    return


def create_logger(logger=None, logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Simple logger that uses configure_logger()
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    configure_logger(logger, logfile=logfile, level=level,
                     log_format=log_format, log_format_date=log_format_date)
    logging.basicConfig(handlers=logger.handlers, level=level)
    logger.propagate = False
    logger.info(f"Logging Started at level:{level}")
    # logger.info(f"Running spt3g_ingest version: {spt3g_detect.__version__}")
    return logger


def elapsed_time(t1, verb=False):
    """
    Returns the time between t1 and the current time now
    I can can also print the formatted elapsed time.
    ----------
    t1: float
        The initial time (in seconds)
    verb: bool, optional
        Optionally print the formatted elapsed time
    returns
    -------
    stime: float
        The elapsed time in seconds since t1
    """
    t2 = time.time()
    stime = "%dm %2.2fs" % (int((t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print("Elapsed time: {}".format(stime))
    return stime


def get_NP(MP):
    """ Get the number of processors in the machine
    if MP == 0, use all available processor
    """
    # For it to be a integer
    MP = int(MP)
    if MP == 0:
        NP = int(mp.cpu_count())
    elif isinstance(MP, int):
        NP = MP
    else:
        raise ValueError('MP is wrong type: %s, integer type' % MP)
    return NP


def create_dir(dirname):
    "Safely attempt to create a folder"
    if not os.path.isdir(dirname):
        LOGGER.info(f"Creating directory: {dirname}")
        try:
            os.makedirs(dirname, mode=0o755, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                LOGGER.warning(f"Problem creating {dirname} -- proceeding with trepidation")


def chunker(seq, size):
    "Chunk a sequence in chunks of a given size"
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def find_unique_centroids(table_centroids, separation=20, plot=False):
    # Find unique centroids
    logger = LOGGER
    max_sep = separation*u.arcsec
    stacked_centroids = None
    labelIDs = list(table_centroids.keys())
    for k in range(len(labelIDs)-1):

        # Select current and next table IDs
        label1 = labelIDs[k]
        label2 = labelIDs[k+1]

        logger.info(f"Doing: {k}/{len(labelIDs)-2}")

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
        n2 = len(cat2)
        idxall = np.arange(n2)
        idxnew2 = np.delete(idxall, idxcat2)

        # Only for the first iteration we append agaist t1, after that we use the output
        if k == 0:
            xx_sky = stack_cols_lists(t1['sky_centroid'].ra.data, t2['sky_centroid'].ra.data, idxcat1, idxcat2,)
            yy_sky = stack_cols_lists(t1['sky_centroid'].dec.data, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = stack_cols_lists(t1['xcentroid'].data, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = stack_cols_lists(t1['ycentroid'].data, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = stack_cols_lists(t1['max_value'].data, t2['max_value'].data, idxcat1, idxcat2, pad=True)
            scan_max = stack_cols_lists(t1['scan_max'].data, t2['scan_max'].data, idxcat1, idxcat2, pad=True)
        else:
            xx_sky = stack_cols_lists(xx_sky, t2['sky_centroid'].ra.data, idxcat1, idxcat2)
            yy_sky = stack_cols_lists(yy_sky, t2['sky_centroid'].dec.data, idxcat1, idxcat2)
            xx_pix = stack_cols_lists(xx_pix, t2['xcentroid'].data, idxcat1, idxcat2)
            yy_pix = stack_cols_lists(yy_pix, t2['ycentroid'].data, idxcat1, idxcat2)
            value_max = stack_cols_lists(value_max, t2['max_value'].data, idxcat1, idxcat2, pad=True)
            scan_max = stack_cols_lists(scan_max, t2['scan_max'].data, idxcat1, idxcat2, pad=True)

        # Here we update the max_values and scan_max label
        # We make them np.array so we can operate on them
        value_max = np.array(value_max)
        scan_max = np.array(scan_max)
        idmax = value_max.argmax(axis=1)
        # We store them back in the same arrays/lists
        value_max = value_max.max(axis=1)
        scan_max = [scan_max[i][idmax[i]] for i in range(len(idmax))]

        # If we have unmatched objects in cat2 (i.e. idxnew has elements), we append these
        if len(idxnew2) > 0:
            new_stack = vstack([t1, t2[idxnew2]])
            stacked_centroids = new_stack
            logger.info(f"{label1}-{label2} Stacked")

        else:
            stacked_centroids = t1
            logger.info(f"{label1}-{label2} No new positions to add")

        # Get the average positions so far
        xc_pix = mean_list_of_list(xx_pix)
        yc_pix = mean_list_of_list(yy_pix)
        xc_sky = mean_list_of_list(xx_sky)
        yc_sky = mean_list_of_list(yy_sky)
        # Update the number of coordinates points we have so far
        ncoords = [len(x) for x in xx_pix]

        # Before Update
        logger.debug("Before Update")
        logger.debug(f"\n{stacked_centroids}\n")

        # Update centroids with averages
        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        stacked_centroids['index'] = np.arange(len(coords)) + 1
        stacked_centroids['sky_centroid'] = coords
        stacked_centroids['xcentroid'] = xc_pix
        stacked_centroids['ycentroid'] = yc_pix
        stacked_centroids['ncoords'] = ncoords
        stacked_centroids['scan_max'] = scan_max
        stacked_centroids['max_value'] = value_max
        stacked_centroids['max_value'].info.format = '.5f'
        stacked_centroids['xcentroid'].info.format = '.2f'
        stacked_centroids['ycentroid'].info.format = '.2f'
        stacked_centroids['segmented_flux'].info.format = '.1f'
        stacked_centroids['segmented_fluxerr'].info.format = '.1f'
        stacked_centroids['kron_flux'].info.format = '.1f'
        stacked_centroids['kron_fluxerr'].info.format = '.1f'
        logger.debug(f"centroids Done for {label1}")
        logger.debug("After Update")
        logger.debug("#### stacked_centroids\n")
        logger.debug(f"\n{stacked_centroids}\n")

    return stacked_centroids


def find_repeating_sources(cat, separation=20, plot=False, outdir=None):
    """
    Match sources in a list of catalogs that show up in at least two
    consecutive catalogs

    inputs:
       - cat: list of catalogs in astropy Table format
    options:
       - separation: maximum separation in arcsec
       - plot: plot positions

     output:
       - table_centroids: list of astropy Table with match postions
    """
    max_sep = separation*u.arcsec
    table_centroids = {}  # Table with centroids
    scans = list(cat.keys())
    logger = LOGGER
    logger.info("++++++++ Starting Match Loop for repeating sources ++++++++++")
    logger.info(f"scans: {scans}")
    for k in range(len(scans)-1):
        logger.info(scans[k])
        scan1 = scans[k]
        scan2 = scans[k+1]
        cat1 = cat[scan1]['sky_centroid']
        cat2 = cat[scan2]['sky_centroid']
        n1 = len(cat[scan1])
        n2 = len(cat[scan2])
        labelID = f"{scan1}_{scan2}"
        logger.info("==============================")
        logger.info(f"Doing {scan1} vs {scan2}")
        logger.info(f"N in cat1: {n1} cat2: {n2}")

        # Altenative match method using match_to_catalog_sky
        # idx1, sep, _ = cat1.match_to_catalog_sky(cat2)
        # idx1_matched = sep < max_sep
        # print(cat[scan1][idx1_matched]['label', 'xcentroid', 'ycentroid',
        #                                'sky_centroid_dms', 'kron_flux', 'kron_fluxerr', 'max_value', 'area'])
        # Match method using search_around_sky
        idxcat1, idxcat2, d2d, _ = cat2.search_around_sky(cat1, max_sep)
        logger.info(f"N matches: {len(idxcat1)} -- {len(idxcat2)}")

        if len(idxcat1) == 0:
            logger.info(f"*** No matches for: {labelID} ***")
            continue

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
        tblidx = np.arange(len(xc_sky)) + 1

        # Get the ids with max value
        max_value = np.array([cat[scan1][idxcat1]['max_value'], cat[scan2][idxcat2]['max_value']])
        scan_value = np.array([cat[scan1][idxcat1]['scan'], cat[scan2][idxcat2]['scan']])
        max_value_max = max_value.max(axis=0)
        idmax = max_value.argmax(axis=0)
        scan_max = scan_value.T[0][idmax]

        # Create a Skycoord object
        coords = SkyCoord(xc_sky, yc_sky, frame=FK5, unit='deg')
        table_centroids[labelID] = Table([tblidx, label_col, coords, xc_pix, yc_pix, scan_max, max_value_max, ncoords],
                                         names=('index', 'labelID', 'sky_centroid', 'xcentroid', 'ycentroid',
                                                'scan_max', 'max_value', 'ncoords'))
        table_centroids[labelID]['xcentroid'].info.format = '.2f'
        table_centroids[labelID]['ycentroid'].info.format = '.2f'
        table_centroids[labelID]['max_value'].info.format = '.5f'
        logger.info(f"centroids Done for: {labelID}")
        print(table_centroids[labelID])
        if plot:
            plot_name = os.path.join(outdir, f"{labelID}_match")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            # Set limits for centroids
            xmin = min(cat[scan1]['xcentroid'].min(), cat[scan2]['xcentroid'].min())
            ymin = min(cat[scan1]['ycentroid'].min(), cat[scan2]['ycentroid'].min())
            xmax = max(cat[scan1]['xcentroid'].max(), cat[scan2]['xcentroid'].max())
            ymax = max(cat[scan1]['ycentroid'].max(), cat[scan2]['ycentroid'].max())
            dx = 0.2*(xmax-xmin)
            dy = 0.2*(ymax-ymin)
            xmin = xmin-dx
            xmax = xmax+dx
            ymin = ymin-dy
            ymax = ymax+dy

            x = cat[scan1]['xcentroid']
            y = cat[scan1]['ycentroid']
            ax1.scatter(x, y, marker='o', c='red')
            ax1.set_xlabel('x[pixels]')
            ax1.set_ylabel('y[pixels]')
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_title(f"{scan1}")

            x = cat[scan2]['xcentroid']
            y = cat[scan2]['ycentroid']
            ax2.scatter(x, y, marker='o', c='blue')
            ax2.set_xlabel('x[pixels]')
            ax2.set_ylabel('y[pixels]')
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ymin, ymax)
            ax2.set_title(f"{scan2}")

            x = table_centroids[labelID]['xcentroid']
            y = table_centroids[labelID]['ycentroid']
            ax3.scatter(x, y, marker='o', c='k')
            ax3.set_xlabel('x[pixels]')
            ax3.set_ylabel('y[pixels]')
            ax3.set_xlim(xmin, xmax)
            ax3.set_ylim(ymin, ymax)
            ax3.set_title(f"{scan1} - {scan2}")
            plt.savefig(f"{plot_name}.pdf")
            plt.close()

    logger.info("Done Matching Loop for repeating sources")
    logger.info("+++++++++++++++++++++++++++++")
    return table_centroids


def stack_cols_lists(c1, c2, ix1, ix2, pad=False):

    """
    Custom function to stack two one-dimensional columns containing lists

    inputs:
       - c1: list (or list of lists) or 1D numpy array
       - c2: list (or list of lists) or 1D numpy array
       - ix1: list of indicices of c1 that are matched in c2
       - ix2: list of indicices of c2 that are matched in c1

    options:
       - pad: pad lists that have only one element

    """

    # Get elements that are not in c1 but not c2
    ixall1 = np.arange(len(c1))
    ixnew1 = np.delete(ixall1, ix1)
    # Get elements that are not in c2 but not c1
    ixall2 = np.arange(len(c2))
    ixnew2 = np.delete(ixall2, ix2)
    nidx = len(ix1)

    if len(ix1) != len(ix2):
        raise Exception("ix1 and ix2 have different dimensions")

    # Make sure that c1 and c2 are lists, if not we re-cast them as lists
    if not isinstance(c1, list):
        try:
            c1 = list(c1)
        except Exception as err:
            raise Exception(f"Cannot cast c1 as list {err=}, {type(err)=}")
    if not isinstance(c2, list):
        try:
            c2 = list(c2)
        except Exception as err:
            raise Exception(f"Cannot cast c2 as list {err=}, {type(err)=}")

    # We will store the new list here as a list called "newlist"
    # and make sure that the newlist elements are also lists:
    newlist = [c if isinstance(c, list) else [c] for c in c1]

    # Step 1, we stack the elements of c1, c2, by indexing (ix1, ix2) and
    # we want to augment c1 lists with new matches (ix2) from c2
    for k in range(nidx):
        i = ix1[k]
        j = ix2[k]
        newlist[i].append(c2[j])

    # Step 2, we want to append new elememts in c2 to existing c1
    LOGGER.debug(f"ixnew1: {ixnew1}")
    if pad:
        for k in ixnew1:
            c = newlist[k][0]
            newlist[k] = [c, c]

    LOGGER.debug(f"ixnew2: {ixnew2}")
    for k in ixnew2:
        c = c2[k]
        if pad:
            newlist.append([c, c])
            LOGGER.debug(f"padding ixnew2: {len(newlist)} {newlist[-1]}")
        else:
            newlist.append([c])
    return newlist


def mean_list_of_list(list, np_array=True):
    u = [np.array(x).mean() for x in list]
    if np_array:
        u = np.array(u)
    return u


def max_list_of_list(list, np_array=True):
    max_val = [np.array(x).max() for x in list]
    if np_array:
        max_val = np.array(max_val)
    return max_val


def compute_rms2D(data, mask=None, box=200, filter_size=(3, 3), sigmaclip=None):

    """
    Compute a 2D map of the rms using photutils.Background2D and
    photutils.StdBackgroundRMS
    """

    # in case we want to clip values
    if sigmaclip:
        sigma_clip = SigmaClip(sigma=sigmaclip)
    else:
        sigma_clip = None

    # Set up the background estimator
    bkg_estimator = photutils.background.StdBackgroundRMS(sigma_clip)
    # Masking does not work, as images have a large section that it's empty, instead we trick it
    # by masking the input data with Nans
    if mask is not None:
        data = np.where(mask, data, np.nan)
    bkg = photutils.background.Background2D(data, box, mask=None, filter_size=filter_size,
                                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg


def detect_with_photutils(data, wgt=None, mask=None, nsigma_thresh=3.5, npixels=20,
                          rms2D=False, box=(200, 200), filter_size=(3, 3), sigmaclip=None,
                          wcs=None, plot=False, plot_title=None, plot_name=None):

    """
    Use photutils SourceFinder and SourceCatalog to create a catalog of sources

    inputs:
       - npixels: The minimum number of connected pixels,
       - nsigma_thresh: Number of sigmas use to compute the detection threshold

    """

    t0 = time.time()
    if mask is not None:
        # Select only the indices with flux
        idx = np.where(mask == 1)
        # Create a bool mask for the maked array, False is NOT masked
        LOGGER.info("Selecting indices for boolean mask")
        gmask = np.where(mask == 1, False, True)
        # Make the data array a masked array (better plots)
        data = ma.masked_array(data, gmask)
    else:
        idx = np.where(mask)
        gmask = None
    # Get the mean and std of the distribution
    mean, sigma = norm.fit(data[idx].flatten())

    # Define the threshold, array in the case of rms2D
    if rms2D:
        bkg = compute_rms2D(data, mask=mask, box=box, filter_size=filter_size, sigmaclip=sigmaclip)
        sigma2D = np.where(mask, bkg.background, np.nan)
        # sigma2D = bkg.background
        threshold = nsigma_thresh * sigma2D
        LOGGER.debug("2D RMS computed")
        # Test to dump 2D rms image into a fits file
        hdr = wcs.to_header()
        hdr = astropy2fitsio_header(hdr)
        fits = fitsio.FITS(f"{plot_name}_bkg.fits", 'rw', clobber=True)
        fits.write(sigma2D, header=hdr)
        fits.close()
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
            plot_distribution(ax1, data[idx], mean, sigma, nsigma=nsigma_thresh)
            plot_rms2D(bkg.background, ax2, gmask=gmask)
            plt.savefig(f"{plot_name}_bkg.pdf")
            LOGGER.info(f"Created: {plot_name}_bkg.pdf")
    else:
        threshold = nsigma_thresh * sigma

    # Perform segmentation and deblending
    finder = SourceFinder(npixels=npixels, nlevels=32, contrast=0.001, progress_bar=False)
    segm = finder(data, threshold)
    # We stop if we don't find source
    if segm is None:
        LOGGER.info("No sources found, returning Nones")
        return None, None
    cat = SourceCatalog(data, segm, error=wgt, wcs=wcs, progress_bar=True)
    # Make sure these are added.
    cat.default_columns.append('elongation')
    cat.default_columns.append('ellipticity')

    LOGGER.info(f"detect_with_photutils runtime: {elapsed_time(t0)} [s]")
    LOGGER.info(f"Found: {len(cat)} objects")

    # Nicer formatting
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['segment_flux'].info.format = '.2f'
    tbl['segment_fluxerr'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'
    tbl['kron_fluxerr'].info.format = '.2f'
    tbl['max_value'].info.format = '.2f'
    tbl['elongation'].info.format = '.2f'
    tbl['ellipticity'].info.format = '.2f'
    tbl['eccentricity'].info.format = '.2f'
    tbl['sky_centroid_dms'] = tbl['sky_centroid'].to_string('hmsdms', precision=0)
    print(tbl['label', 'xcentroid', 'ycentroid', 'sky_centroid_dms',
              'kron_flux', 'kron_fluxerr', 'max_value',
              'eccentricity', 'elongation', 'ellipticity', 'area'])

    if plot:
        t1 = time.time()
        if rms2D:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=plot_title)
        plot_segmentation(ax2, segm, cat, gmask=gmask)
        plot_distribution(ax3, data[idx], mean, sigma, nsigma=nsigma_thresh)
        if rms2D:
            # Make the background image a masked arrays
            plot_rms2D(bkg.background, ax4, gmask=gmask)
        if plot_name:
            plt.savefig(f"{plot_name}.pdf")
            LOGGER.info(f"Saved: {plot_name}.pdf")
        else:
            plt.show()
        plt.close()
        LOGGER.info(f"detect_with_photutils PLOT runtime: {elapsed_time(t1)} [s]")

    LOGGER.info(f"detect_with_photutils TOTAL runtime: {elapsed_time(t0)} [s]")
    return segm, tbl


def g3_or_fits(filename):
    """Check based on the filename extension whether this is a FITS or G3 file"""

    ext = ".".join(filename.split(".")[1:])
    if ext == "fits" or ext == "fits.gz" or ext == "fits.fz":
        filetype = "FITS"
    elif ext == "g3" or ext == "g3.gz":
        filetype = "G3"
    else:
        msg = f"Could not find filetype for file {filename}"
        LOGGER.warning(msg)
        raise ValueError(msg)
    return filetype


def plot_rms2D(bkg, ax, gmask=None, nsigma_plot=3.5):

    # Plot a masked array if gmask is passed
    if gmask is not None:
        bkg = ma.masked_array(bkg, gmask)
    im = ax.imshow(bkg, origin='lower', cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('2D Noise Map')


def plot_detection(ax1, data, cat, sigma, nsigma_plot=5, plot_title=None):
    """
    Function to plot the 2D data array and catalog produced by photutils
    """
    vlim = nsigma_plot*sigma
    im1 = ax1.imshow(data, origin='lower', cmap='viridis', vmin=-vlim, vmax=+vlim)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    if plot_title:
        ax1.set_title(plot_title)
    cat.plot_kron_apertures(ax=ax1, color='white', lw=0.5)


def plot_segmentation(ax2, segm, cat, gmask=None):
    """
    Function to plot the segmentation image and catalog produce by photutils
    """
    # Plot a masked array if gmask is passed
    if gmask is not None:
        segm_plot = ma.masked_array(segm, gmask)
    else:
        segm_plot = segm
    im = ax2.imshow(segm_plot, origin='lower', cmap=segm.cmap,
                    interpolation='nearest')
    divider = make_axes_locatable(ax2)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax2)
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax2, color='white', lw=0.5)


def plot_distribution(ax, data, mean, sigma, nsigma=3):
    """
    Function to display distribution of data as 1D
    """
    # Flatten the data if needed
    if data.ndim != 1:
        data = data.flatten()

    legend = "$\\mu$: %.6f\n$\\sigma$: %.6f" % (mean, sigma)
    # Plot data and fit
    nbins = int(data.shape[0]/5000.)
    hist = ax.hist(data, bins=nbins, density=True, alpha=0.6)
    ymin, ymax = hist[0].min(), hist[0].max()
    xmin, xmax = hist[1].min(), hist[1].max()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, sigma)
    ax.plot(x, y)
    xx = [nsigma*sigma, nsigma*sigma]
    yy = [ymin, ymax]
    ax.plot(xx, yy, 'k--', linewidth=1)
    xx = [-nsigma*sigma, -nsigma*sigma]
    ax.plot(xx, yy, 'k--', linewidth=1)
    ax.set_ylim(ymin, ymax)
    ax.legend([legend], frameon=False)
    text = f"${nsigma}\\sigma$"
    ax.text(0.05*(xmax-xmin) + nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')
    ax.text(-0.05*(xmax-xmin) - nsigma*sigma, (ymax-ymin)/20., text, horizontalalignment='center')

    ratio = 0.95
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_xlabel("Flux")
    ax.set_title("1D Noise Distribution and Fit")


def get_sources_catalog(field):
    """
    Function to read the sources catalog for a field.
    inputs:
      - field: field name (str)
    output:
      - pscat: SkyCoord astropy object
    """
    # Get the catalog with masked sources
    point_source_file = sources.get_field_source_list(field, analysis="lightcurve")
    _, psra, psdec, __ = sources.read_point_source_mask_file(point_source_file)
    LOGGER.debug(f"Loading source mask positions from file: {point_source_file}")
    # Create a SkyCoord object with the sources catalog to make the matching
    # psra and psdec are in G3 units and need to be converted back to degrees to be use in astropy
    psra = psra/core.G3Units.deg
    psdec = psdec/core.G3Units.deg
    pscat = SkyCoord(ra=psra*u.degree, dec=psdec*u.degree)
    return pscat


def remove_objects_near_sources(cat, field, max_dist=5*u.arcmin):
    """
    Funtion to remove objects from astropy catalog near sources catalog
    inputs:
      - field: field name (str)
      - cat: astropy catalog
      - max_dist: maximum separation distancen (using 5 arcmin)
    output:
      - cat: input astropy catalog without matched sources
    """
    # Get a astropy SkyCoord object catalog to match
    try:
        pscat = get_sources_catalog(field)
    except KeyError:
        LOGGER.warning(f"Cannot get sources catalog for field: {field}")
        return cat

    # Extract the SkyCoord object
    cat1 = cat['sky_centroid']
    inds1, inds2, dist, _ = search_around_sky(cat1, pscat, max_dist)
    if len(inds1) > 0:
        LOGGER.info(f"Found {len(inds1)} matches, will remove them from catalog")
        cat = cat[~np.isin(np.arange(cat1.size), inds1)]
    else:
        LOGGER.info("No matches found in sources catalog")
    return cat


def astropy2fitsio_header(header):
    """
    Translate and astropy header object into a fitsio FITSHDR object
    """
    # Make the header a FITSHDR object
    hlist = []
    for key in header:
        hlist.append({'name': key, 'value': header[key], 'comment': header.comments[key]})
    h = fitsio.FITSHDR(hlist)
    return h


def update_wcs_matrix(header, x0, y0, proj='ZEA'):
    """
    Update the wcs header object with the right CRPIX[1, 2] CRVAL[1, 2] for a
    given subsection

    Parameters:
    header: fits style header
        The header to work with
    x0, y0: float
        The new center of the image
    naxis1, naxis2: int
        The number of pixels on each axis.

    Returns:
        fits style header with the new center.
    """

    # We need to make a deep copy/otherwise if fails
    h = copy.deepcopy(header)
    # Get the astropy.wcs object
    wcs = WCS(h)

    if proj == 'TAN':
        # Recompute CRVAL1/2 on the new center x0,y0
        CRVAL1, CRVAL2 = wcs.wcs_pix2world(x0, y0, 0)
        # Recast numpy objects as floats
        CRVAL1 = float(CRVAL1)
        CRVAL2 = float(CRVAL2)
        # Asign CRPIX1/2 on the new image
        CRPIX1 = 1
        CRPIX2 = 1
        # Update the values
        h['CRVAL1'] = CRVAL1
        h['CRVAL2'] = CRVAL2
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'

    elif proj == 'ZEA':
        CRPIX1 = float(h['CRPIX1']) - x0
        CRPIX2 = float(h['CRPIX2']) - y0
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        LOGGER.debug(f"Updated to CRPIX1:{CRPIX1}, CRPIX2:{CRPIX2}")

    else:
        raise NameError(f"Projection: {proj} not implemented")

    return h


def get_thumbFitsName(ra, dec, filter, obsid,
                      objID=None, prefix=PREFIX, ext='fits', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """

    # Format RA,DEC using astropy coordinates
    coo = FK5(ra*u.degree, dec*u.degree)
    ra = f'{coo.ra.to_string(unit=u.hourangle, sep="", precision=0, pad=True)}'
    dec = f'{coo.dec.to_string(sep="", precision=1, alwayssign=True, pad=True)}'
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    outname = FITS_OUTNAME.format(**kw)
    return outname


def get_thumbBaseDirName(ra, dec, objID=None, prefix=PREFIX, outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    # Format RA,DEC using astropy coordinates
    coo = FK5(ra*u.degree, dec*u.degree)
    ra = f'{coo.ra.to_string(unit=u.hourangle, sep="", precision=0, pad=True)}'
    dec = f'{coo.dec.to_string(sep="", precision=1, alwayssign=True, pad=True)}'
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    basedir = BASEDIR_OUTNAME.format(**kw)
    return basedir


def get_thumbBaseName(ra, dec, objID=None, prefix=PREFIX):
    """ Common function to set the Fits thumbnail name """
    # Format RA,DEC using astropy coordinates
    coo = FK5(ra*u.degree, dec*u.degree)
    ra = f'{coo.ra.to_string(unit=u.hourangle, sep="", precision=0, pad=True)}'
    dec = f'{coo.dec.to_string(sep="", precision=1, alwayssign=True, pad=True)}'
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    outname = BASE_OUTNAME.format(**kw)
    return outname


def get_headers_hdus(filename):

    header = OrderedDict()
    hdu = OrderedDict()

    is_compressed = False
    with fitsio.FITS(filename) as fits:
        # Case 1 -- for well-defined fitsfiles with EXTNAME
        for k in range(len(fits)):
            h = fits[k].read_header()
            # Is compressed
            if h.get('ZIMAGE'):
                is_compressed = True
            # Make sure that we can get the EXTNAME
            if not h.get('EXTNAME'):
                continue
            extname = h['EXTNAME'].strip()
            if extname == 'COMPRESSED_IMAGE':
                is_compressed = True
                continue
            header[extname] = h
            hdu[extname] = k

        # Case 2 -- files without EXTNAME
        if len(header) < 1:
            LOGGER.debug("Getting EXTNAME by compression")
            if is_compressed:
                sci_hdu = 1
                wgt_hdu = 2
            else:
                sci_hdu = 0
                wgt_hdu = 1
            # Assign headers and hdus
            header['SCI'] = fits[sci_hdu].read_header()
            hdu['SCI'] = sci_hdu
            try:
                header['WGT'] = fits[wgt_hdu].read_header()
                hdu['WGT'] = wgt_hdu
            except IOError:
                LOGGER.warning(f"No WGT HDU for: {filename}")
    fits.close()
    return header, hdu
