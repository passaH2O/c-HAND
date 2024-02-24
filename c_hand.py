# -*- coding: utf-8 -*-


import contextily as cx
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.plot import plotting_extent
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

    # only keep region containing the ocean pixel
    inun = np.where(regions == regions[opix], inun, 0)

    # return masked array if fed one
    if isinstance(dem, np.ma.MaskedArray):
        inun = np.ma.masked_array(inun, dem.mask)

    # mask out zero inundation cells
    # inun = np.ma.masked_where(inun==0,inun,copy=True)

    return inun


def plot_raster(raster=None, profile=None, label=None, **kwargs):
    fig, ax = plt.subplots(dpi=200)

    # show inundation map
    im = ax.imshow(
        raster,
        extent=plotting_extent(raster, profile["transform"]),
        zorder=2,
        **kwargs,
    )

    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(im, cax=cax, label=label)

    # add basemap
    cx.add_basemap(
        ax,
        crs=profile["crs"],
        source=cx.providers.Esri.WorldImagery,
        zoom=10,
        attribution_size=2,
        zorder=1,
    )

    # add scalebar
    ax.add_artist(ScaleBar(1, box_alpha=0, location="lower right", color="white"))

    # add north arrow
    x, y, arrow_length = 0.9, 0.3, 0.15
    ax.annotate(
        "N",
        color="white",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="white", edgecolor="white", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=10,
        xycoords=ax.transAxes,
    )

    return fig, ax
