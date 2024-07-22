#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from astropy.io import fits
from scipy.stats import norm
import math
from astropy import wcs
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def find_distribution(data, wgt=None, nsigma=3):
    data1D = data.flatten()
    # Fit the data
    mean, std = norm.fit(data1D)
    text = """
    $\mu$: %.6f
    $\sigma$: %.6f
    """ % (mean, std)
    print(mean, std)

    # Plot data and fit
    nbins = int(data1D.shape[0]/5000.)
    plt.hist(data1D, bins=nbins, density=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    n = nsigma
    xx = [n*std, n*std]
    yy = [ymin, ymax]
    plt.plot(xx, yy, 'k--', linewidth=1)
    xx = [-n*std, -n*std]
    plt.plot(xx, yy, 'k--', linewidth=1)
    plt.ylim(ymin, ymax)
    plt.legend([text], frameon=False)
    text = f'${nsigma}\sigma$'
    plt.text(0.05*(xmax-xmin) + nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')
    plt.text(-0.05*(xmax-xmin) - nsigma*std, (ymax-ymin)/20., text, horizontalalignment='center')

    # Compute the weighted mean, std
    if wgt is not None:
        wgt1D = wgt.flatten()
        mu = np.average(data1D, weights=wgt1D)
        var = np.average((data1D - mu)**2, weights=wgt1D)
        print(mu, math.sqrt(var))
        title = "$\hat{\mu}$: %.6f, $\hat{\sigma}$: %.6f" % (mu, math.sqrt(var))
        plt.title(title)

    #plt.savefig("dist.png")
    return mean, std


# Read in the data
filename = 'f150/mapmaker_RISING_SCAN_40-161-9_noiseweighted_map_nside4096_flt_small.fits'
fits.info(filename)
hdul = fits.open(filename)
data_150 = hdul[1].data
wgt_150 = hdul[2].data

wcs_data = wcs.WCS(hdul[1].header)

filename = 'f090/mapmaker_RISING_SCAN_40-161-9_noiseweighted_map_nside4096_flt_small.fits'
fits.info(filename)
hdul = fits.open(filename)
data_090 = hdul[1].data
wgt_090 = hdul[2].data

data = (data_090*wgt_090 + data_150*wgt_150)/(wgt_090+wgt_150)
wgt = (wgt_090+wgt_150)

data = data_090
wgt = wgt_090

nsigma = 3.5
npixels = 20

mean, sigma = find_distribution(data, wgt=wgt, nsigma=nsigma)
threshold = nsigma * sigma

# Perform segmentation and deblending
finder = SourceFinder(npixels=npixels, nlevels=32, contrast=0.001,
                      progress_bar=False)
segm_deblend = finder(data, threshold)
print(segm_deblend)

cat = SourceCatalog(data, segm_deblend, error=1/wgt, wcs=wcs_data)
print(cat)

tbl = cat.to_table()
tbl['xcentroid'].info.format = '.2f'  # optional format
tbl['ycentroid'].info.format = '.2f'
tbl['kron_flux'].info.format = '.2f'
tbl['kron_fluxerr'].info.format = '.2f'
tbl['max_value'].info.format = '.2f'
tbl['max_value'] = tbl['max_value']*1000
tbl['kron_flux'] = tbl['kron_flux']*1000

print(tbl['label', 'sky_centroid', 'xcentroid', 'ycentroid', 'max_value', 'area'])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
im1 = ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=-5*sigma, vmax=5*sigma)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1)
ax1.set_title(filename)

im2 = ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
                 interpolation='nearest')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2)
ax2.set_title('Segmentation Image')
cat.plot_kron_apertures(ax=ax1, color='yellow', lw=0.5)
cat.plot_kron_apertures(ax=ax2, color='yellow', lw=0.5)

plt.show()
