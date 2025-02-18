#!/usr/bin/env python

import spt3g_detect.ftools as ft
import sys

filename_stamp = sys.argv[1]
filename_table = sys.argv[2]

images, id = ft.load_fits_stamp(filename_stamp)
lc = ft.load_fits_table(filename_table, id)

ft.plot_fits_data(images, lc)
