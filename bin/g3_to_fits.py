#!/usr/bin/env python

import logging
import spt3g_detect.ftools as ft
import spt3g_detect.dtools as du
import sys
import time

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

if __name__ == "__main__":

    g3files = sys.argv[1:]
    for g3file in g3files:
        t0 = time.time()
        logger.info(f"Doing file: {g3file}")
        ft.g3_to_fits(g3file, trim=True, compress='RICE_1')
        logger.info(f"Total time for file {g3file}: {du.elapsed_time(t0)}")
