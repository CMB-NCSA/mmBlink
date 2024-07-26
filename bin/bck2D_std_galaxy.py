#!/usr/bin/env python

from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS
from photutils.background import Background2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spt3g import core, maps
import numpy
import time

g3file = '196095895_offline_pointing_gain_calibration_transient_filtered_map.g3'

# Keep the 2D arrays in a dictionary keyed to band
flux = {}
flux_mask = {}
# Loop over frames
for frame in core.G3File(g3file):

    if frame.type != core.G3FrameType.Map:
        continue

    band = frame["Id"]
    print("Reading:", frame["Id"])
    print("Removing weights:", frame["Id"])
    maps.RemoveWeights(frame, zero_nans=True)
    flux[band] = numpy.asarray(frame['T'])/core.G3Units.mJy
    print("Min/Max", flux[band].min(), flux[band].max())
    # Now we exctract the mask -- not sure if we will use it
    g3_mask = frame["T"].to_mask()
    g3_mask_map = g3_mask.to_map()
    flux_mask[band] = numpy.asarray(g3_mask_map)
    flux_mask[band] = numpy.where(flux_mask[band] == 1, int(1), 0)

# Set up the background estimator
sigma_clip = SigmaClip(sigma=5)  # in case we want to clip values
sigma_clip = None
bkg_estimator = StdBackgroundRMS()


for band in flux.keys():
    data = flux[band][1650:2670, 360:4050]
    data_mask = flux_mask[band][1650:2670, 360:4050]
    print(f"Computing background for {band}")
    t1 = time.time()
    bkg = Background2D(data, (35, 35), filter_size=(3, 3),
                       # coverage_mask=data_mask,
                       # adding mask makes it much slower and results won't change
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    t2 = time.time()
    stime = "%dm %2.2fs" % (int((t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    print(f"Done {band}, time: {stime}")
    fig = plt.figure(figsize=(18, 6))
    ima = plt.imshow(bkg.background, origin='lower', cmap='Greys_r')
    plt.title(f"Noise map {band}")
    plt.xlabel('x[pixels]')
    plt.ylabel('y[pixels]')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    clb = plt.colorbar(ima, cax=cax)
    clb.ax.set_title('$\\sigma$ [mJy]', fontsize=12)
    plt.tight_layout()
    fig.savefig(f"galaxy2Dnoise_{band}.png")
    print(f"Saved: galaxy2Dnoise_{band}.png")


plt.show()
