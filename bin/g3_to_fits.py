#!/usr/bin/env python

import logging
import spt3g_detect.ftools as ft
import spt3g_detect.dtools as du
import time
import argparse


def cmdline():

    parser = argparse.ArgumentParser(description="Transform g3 FlatSkyMap files to FITS")
    parser.add_argument("files", nargs='+',
                        help="Filenames to transform from g3 to FITS")
    parser.add_argument("--clobber", action='store_true', default=False,
                        help="Clobber output files")
    parser.add_argument("--trim", action='store_true', default=False,
                        help="Trim map to field extent definitions")
    parser.add_argument("--compress", action="store", default='RICE_1', type=str.upper,
                        choices=['RICE_1', 'RICE_ONE', 'PLIO_1', 'GZIP_1', 'GZIP_2',
                                 'HCOMPRESS_1', 'NOCOMPRESS', 'NONE'],
                        help="Compression type (Optional)")
    parser.add_argument("--quantize_level", action="store", default=16, type=int,
                        help="Floating point quantization level")
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    t0 = time.time()
    args = cmdline()
    # Create logger
    logger = logging.getLogger(__name__)
    logger = du.create_logger(logger=logger, level=args.loglevel,
                              log_format=args.log_format,
                              log_format_date=args.log_format_date)
    k = 1
    nfiles = len(args.files)
    for g3file in args.files:
        t1 = time.time()
        logger.info(f"Doing: {k}/{nfiles} files")
        logger.info(f"Doing file: {g3file}")
        ft.g3_to_fits(g3file, trim=args.trim,
                      compress=args.compress,
                      quantize_level=args.quantize_level,
                      overwrite=args.clobber)
        logger.info(f"Total time for file {g3file}: {du.elapsed_time(t1)}")
        k += 1
    logger.info(f"Grand total time for {nfiles} files: {du.elapsed_time(t0)}")
