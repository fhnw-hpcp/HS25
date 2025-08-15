#
# Copyright (C) 2012-2025 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
:file: python/SHE_PPT/she_io/stamps.py

:date: 15/02/23
:author: Gordon Gibb

"""

from dataclasses import dataclass
from typing import List
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.units import degree
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D

import logging as log

from stampextraction.vis_exposures import VisExposure
from stampextraction.profiling import io_stats


logger = log.getLogger(__name__)


@dataclass
class Stamp:
    """Container for a postage stamp's data"""

    header: fits.header
    wcs: WCS
    sci: np.ndarray
    rms: np.ndarray
    flg: np.ndarray
    wgt: np.ndarray
    bkg: np.ndarray
    seg: np.ndarray
    dpd: "DpdVisCalibratedFrame"  # noqa: F821


def extract_stamps_from_exposures(exposures: List[VisExposure], ra, dec, size, x_buffer=0, y_buffer=0) -> List[Stamp]:
    """
    Extracts a list of stamps from a list of VisExposure objects

    Inputs:
      - exposures: a list of VisExposure objects (or subclasses of)
      - ra: the right ascension of the object
      - dec: the declination of the object
      - size: the size of the stamp in pixels
      - x_buffer: number of pixels around the x edge of the image to exclude objects from. E.g., if
        x_buffer = 3, then objects within 3 pixels of the edge of the image will not be considered within
        that CCD. Negative x_buffer includes objects outside of the CCD
      - y_buffer: number of pixels around the y edge of the image to exclude objects from.

    Returns:
      - stamps: a list of Stamp objects. If no stamp can be extracted (e.g. the input coords are outside the
        FOV of the exposure) then None is returned in the list.

    """
    stamps = []
    for exp in exposures:
        stamps.append(extract_exposure_stamp(exp, ra, dec, size, x_buffer, y_buffer))
    return stamps


@io_stats(prof_queue=True)
def extract_exposure_stamp(exposure: VisExposure, ra, dec, size, x_buffer=0, y_buffer=0):
    """
    Extracts a stamp from a VisExposure object

    Inputs:
      - exposure: a VisExposure object (or subclass of)
      - ra: the right ascension of the object
      - dec: the declination of the object
      - size: the size of the stamp in pixels
      - x_buffer: number of pixels around the x edge of the image to exclude objects from. E.g., if
        x_buffer = 3, then objects within 3 pixels of the edge of the image will not be considered within
        that CCD. Negative x_buffer includes objects outside of the CCD
      - y_buffer: number of pixels around the y edge of the image to exclude objects from.

    Returns:
      - stamp: a Stamp dataclass. If no stamp can be extracted (e.g. the input coords are outside the FOV of
        the exposure) then None is returned.

    """

    wcs = None
    det_id = None

    if "LINEAR" in exposure.get_wcs_list()[0].wcs.ctype:
        # fudge for the static test data which use a linear WCS
        skycoord = (ra, dec)

        for i, w in enumerate(exposure.get_wcs_list()):

            nx, ny = w.pixel_shape
            x, y = w.all_world2pix([ra], [dec], 0)
            if (x_buffer < x <= nx - x_buffer) and (y_buffer < y <= ny - y_buffer):
                wcs = w
                det_id = i
                break

    else:
        # The proper way
        skycoord = SkyCoord(ra, dec, unit=degree)

        # determine which detector this object is in
        wcs = None
        det_id = None
        for i, w in enumerate(exposure.get_wcs_list()):
            if wcs_with_buffer(w, x_buffer, y_buffer).footprint_contains(skycoord):
                wcs = w
                det_id = i
                break

    if wcs is None:
        logger.warning("Object not in observation")
        return None

    det = exposure[det_id]

    header = det.header

    sci_cutout = Cutout2D(det.sci, skycoord, size, wcs=wcs, mode="partial", fill_value=0, copy=True)
    sci = sci_cutout.data
    centred_wcs = sci_cutout.wcs

    rms = (
        Cutout2D(det.rms, skycoord, size, wcs=wcs, mode="partial", fill_value=0, copy=True).data
        if det.rms is not None
        else None
    )
    flg = (
        Cutout2D(det.flg, skycoord, size, wcs=wcs, mode="partial", fill_value=1, copy=True).data
        if det.flg is not None
        else None
    )
    bkg = (
        Cutout2D(det.bkg, skycoord, size, wcs=wcs, mode="partial", fill_value=0, copy=True).data
        if det.bkg is not None
        else None
    )
    wgt = (
        Cutout2D(det.wgt, skycoord, size, wcs=wcs, mode="partial", fill_value=0, copy=True).data
        if det.wgt is not None
        else None
    )
    seg = (
        Cutout2D(det.seg, skycoord, size, wcs=wcs, mode="partial", fill_value=0, copy=True).data
        if det.seg is not None
        else None
    )

    stamp = Stamp(header, centred_wcs, sci, rms, flg, wgt, bkg, seg, det.dpd)

    return stamp


def wcs_with_buffer(wcs, x_buffer=0, y_buffer=0):
    """Creates a WCS with an expanded/contracted detector size to accommodate the pixel buffer"""
    if x_buffer == 0 and y_buffer == 0:
        # no changes needed
        return wcs

    if type(x_buffer) is not int or type(y_buffer) is not int:
        raise ValueError("WCS pixel buffer must be an integer")

    nx, ny = wcs.pixel_shape

    xmin, xmax = x_buffer, nx - x_buffer
    ymin, ymax = y_buffer, ny - y_buffer

    # slice the wcs (this is python/c convention, so [y,x] not [x,y])
    resized_wcs = wcs[ymin:ymax, xmin:xmax]

    # set the new array size
    resized_wcs.pixel_shape = (nx - 2 * x_buffer, ny - 2 * y_buffer)

    return resized_wcs
