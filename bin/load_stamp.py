#!/usr/bin/env python

import spt3g_detect.ftools as ft
import sys

filename_stamp1 = sys.argv[1]
filename_stamp2 = sys.argv[2]
filename_table1 = sys.argv[3]
filename_table2 = sys.argv[4]

lc = {}
images = {}
headers = {}

images['90GHz'], headers['90GHz'], id = ft.load_fits_stamp(filename_stamp1)
images['150GHz'], headers['150GHz'], id = ft.load_fits_stamp(filename_stamp2)
#images['220GHz'], headers['220GHz'], id = ft.load_fits_stamp(filename_stamp2)

print(images.keys())

lc['90GHz'] = ft.load_fits_table(filename_table1, id)
lc['150GHz'] = ft.load_fits_table(filename_table2, id)

ft.plot_stamps_lc(images, headers, lc)


#ft.plot_fits_data(images['90GHz'], lc)
