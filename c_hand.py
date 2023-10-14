# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import label


def c_hand(dem, gage_el, opix):
    """
    Returns inundation array of the same shape as dem.
    dem and gage_el should have the same vertical datum.
        Parameters:
            dem : ndarray
                2D float array containing terrain elevation data
            gage_el : float
                Constant water surface elevation to apply to DEM
            opix : (int, int)
                Tuple with (row, col) of an ocean cell on dem
    """
    # initialize array with nan values
    inun = np.full(dem.shape, np.nan, dtype=np.float32)
    # initial inun array: 0 if (DEM ≥ gage_el) else (gage_el - DEM)
    inun = np.where(dem >= (gage_el), 0, (gage_el) - dem)
    # masked inun array: 255 if inun > 0 else 0
    inun_mask = np.where(inun == 0, 0, 255)
    # label connected regions of inundation
    regions = label(inun_mask)
    # # find region containing ocean pixel
    # opix_region = regions[opix]  # opix: (row, col)
    # # only keep region containing the ocean pixel
    inun = np.where(regions == regions[opix], inun, 0)
    # uncomment below line to use a max. method to combine tidal inundation with e.g. riverine inundation
    # inun_riv = np.where(inun <= riv_inun, riv_inun, inun)
    # return masked array if fed one
    if isinstance(dem, np.ma.MaskedArray):
        inun = np.ma.masked_array(inun, dem.mask)

    # mask out zero inundation cells
    # do not uncomment unless have a good reason
    # inun = np.ma.masked_where(inun==0,inun,copy=True)

    return inun
