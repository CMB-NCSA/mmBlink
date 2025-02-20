#!/usr/bin/env python

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]
hdu1 = fits.open(file1)
hdu2 = fits.open(file2)

w1 = WCS(hdu1[1].header)
w2 = WCS(hdu2[0].header)

print(w1)


ra0 = 2.1122771
dec0 = -58.5289813
# ra = "0:08:26.9465"
# dec = "-58:31:44.333"
x0 = 8735.8765 - 1
y0 = 5749.4217 - 1

sky0 = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree)
print("--- sky0 ---")
print(sky0)
print(sky0.to_string('hmsdms'))
print("------")
x00, y00 = w1.world_to_pixel(sky0)
ra00, dec00 = w1.wcs_pix2world(x0, y0, 0)
print(ra0, dec0)
print(ra00, dec00)
print(x00, y00)
print(x0, y0)
sky00 = w1.pixel_to_world(x0, y0)
print("--- sky00 ---")
print(sky00)
print(sky00.to_string('hmsdms'))
print("------")


x00, y00 = w1.world_to_pixel(sky0)
sky00 = w1.pixel_to_world(x00, y00)
print(sky00)
print(sky00.to_string('hmsdms'))
print(x00, y00)
print("------")


sky1 = w1.pixel_to_world(x0, y0)
print("--- sky 1 ---")
print(sky1)
print(sky1.to_string('hmsdms'))
print("------")

(x1, y1) = w1.world_to_pixel(sky1)
(x2, y2) = w2.world_to_pixel(sky1)
print(x1, y1)
print(x2, y2)


sky2 = w2.pixel_to_world(x2, y2)
print("--- sky 2 ---")
print(sky2)
print(sky2.to_string('hmsdms'))
print("------")
