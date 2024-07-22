#!/usr/bin/env python
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS
from photutils.background import Background2D
import matplotlib.pyplot as plt

scan = '40-157-9'
filename_090 = f"f090/mapmaker_RISING_SCAN_{scan}_noiseweighted_map_nside4096_flt_small.fits"
hdul = fits.open(filename_090)
data_090 = hdul[1].data

# sigma_clip = SigmaClip(sigma=10)
# bkgrms = StdBackgroundRMS(sigma_clip)
# bkgrms.calc_background_rms(data_090)

sigma_clip = SigmaClip(sigma=10)
bkg_estimator = StdBackgroundRMS()
bkg = Background2D(data_090, (150, 150),
                   filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


im1 = plt.imshow(bkg.background, origin='lower', cmap='Greys_r')
plt.colorbar(im1)
plt.show()
