# mmBlink
**mmBlink** is a Python pipeline for detecting transient and variable sources in millimeter-wave survey data. Designed for compatibility with CMB surveys, it works G3 and FITS sky maps. It uses astropy and photutils libraries as the engine for source detection.

It provides tools to:

- Detect flux-variable, transient sources
- Extract light curves acrosss observations
- Create image cutouts (postage stamps) for each detection.


# Requirements
- spt3g
- fitsio
- astropy
- photutils
- numpy
- scipy
- magic
- yaml
- pandas
