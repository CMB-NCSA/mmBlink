# mmBlink
**mmBlink** is a Python pipeline for detecting transient and variable sources in millimeter-wave surveys maps. Designed for compatibility with CMB surveys, it works G3 and FITS sky maps. It uses ``astropy`` and ``photutils`` libraries as the engine for source detection.

It provides tools to:

- Detect flux-variable, transient sources
- Extract light curves across observations
- Create image cutouts (postage stamps) for each detection.


# Requirements
- ``spt3g`` (to load g3 FlatSkyMaps and source lists)
- ``fitsio`` (the engine to make the cutouts)
- ``astropy``
- ``photutils``
- ``numpy``
- ``scipy``
- ``magic``
- ``yaml``
- ``pandas``
