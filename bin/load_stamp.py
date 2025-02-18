#!/usr/bin/env python

import spt3g_detect.ftools as ft
import sys

filename_stamp1 = sys.argv[1]
filename_stamp2 = sys.argv[2]
filename_table = sys.argv[3]

images = {}
headers = {}

images['90GHz'], headers['90GHz'], id = ft.load_fits_stamp(filename_stamp1)
images['150GHz'], headers['150GHz'], id = ft.load_fits_stamp(filename_stamp2)
#images['220GHz'], headers['220GHz'], id = ft.load_fits_stamp(filename_stamp2)

print(images.keys())

lc = ft.load_fits_table(filename_table, id)

ft.plot_stamps(images, headers)



#ft.plot_fits_data(images, lc)
