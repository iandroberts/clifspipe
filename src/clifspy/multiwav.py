from astropy.coordinates import SkyCoord
from clifspy.galaxy import Galaxy
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import clifspy.utils as utils
import os
import logging
import re
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from photutils.segmentation import (detect_sources,
                                    make_2dgaussian_kernel)
from astropy.convolution import convolve_fft
from photutils.background import Background2D

logger = logging.getLogger("CLIFS_Pipeline")

def subtract_background_galex(img, img_h):
    kernel = make_2dgaussian_kernel(3.0, size = 5)
    convolved_data = convolve_fft(img, kernel)
    mean, med, std = sigma_clipped_stats(convolved_data)
    threshold = med + 5 * std
    segm = detect_sources(convolved_data, threshold, npixels = 5)
    srcmask = segm.data > 0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xcent, ycent = 0.5 * (img.shape[1] - 1), 0.5 * (img.shape[0] - 1)
    rdeg = img_h["CDELT2"] * np.sqrt((xx - xcent) ** 2 + (yy - ycent) ** 2)
    coverage_mask = rdeg > 0.55
    bkg = Background2D(convolved_data, 50, mask = srcmask, coverage_mask = coverage_mask)
    return convolved_data - bkg.background, bkg.background_rms

def find_cfht_field(coord):
    ra_cent = [193.869, 193.869, 193.869, 194.917, 194.926, 194.926, 195.983, 195.983, 195.982]
    dec_cent = [27.058, 27.991, 28.925, 27.997, 27.058, 28.925, 27.058, 28.925, 27.990]
    coord_cent = SkyCoord(ra_cent, dec_cent, unit = "deg")
    ind = np.argmin(coord.separation(coord_cent).arcminute)
    if ra_cent[ind] == 195.982 and dec_cent[ind] == 27.990:
        return "G008.{}+{}".format(ra_cent[ind], dec_cent[ind])
    else:
        return "G007.{}+{}".format(ra_cent[ind], dec_cent[ind])

def find_lofar_field(coord):
    ra_cent = [192.945, 195.856, 195.339]
    dec_cent = [27.2272, 27.2426, 29.7498]
    fields = ["p192+27", "p195+27", "p195+30"]
    coord_cent = SkyCoord(ra_cent, dec_cent, unit = "deg")
    ind = np.argmin(coord.separation(coord_cent).arcminute)
    return fields[ind]

def units_to_MJysr(img, img_h, telescope, filter):
    if telescope == "cfht":
        fv = img * np.power(10, (8.90 - img_h["PHOTZP"]) / 2.5) * u.Jy
        Sv = fv / (img_h["CD2_2"] * u.deg) ** 2
        return Sv.to(u.MJy / u.sr).value
    elif telescope == "herschel" and ("pacs" in filter):
        Sv = (img * u.Jy) / (img_h["CD2_2"] * u.deg) ** 2
        return Sv.to(u.MJy / u.sr).value
    elif telescope == "herschel" and ("spire" in filter):
        if filter == "spire250":
            Abm = 469.4 * (u.arcsec ** 2)
        elif filter == "spire350":
            Abm = 831.2 * (u.arcsec ** 2)
        elif filter == "spire500":
            Abm = 1804.3 * (u.arcsec ** 2)
        else:
            raise ValueError("Invalid SPIRE filter")
        Sv = (img * u.Jy) / Abm
        return Sv.to(u.MJy / u.sr).value
    elif telescope == "galex":
        if filter == "nuv":
            fv = img * 3.37289e-5 * u.Jy
        elif filter == "fuv":
            fv = img * 1.07647e-4 * u.Jy
        else:
            raise ValueError("Invalid GALEX filter")
        Sv = fv / (img_h["CDELT2"] * u.deg) ** 2
        return Sv.to(u.MJy / u.sr).value
    elif telescope == "lofar":
        Abm = 2 * np.pi * (img_h["BMIN"] * u.deg) * (img_h["BMAJ"] * u.deg) * gaussian_fwhm_to_sigma ** 2
        Sv = (img * u.Jy) / Abm
        return Sv.to(u.MJy / u.sr).value
    else:
        raise ValueError("Telescope ({}) is  not supported".format(telescope))

def format_output_header(hdr, shape, telescope, filter):
    hdr["BUNIT"] = ("MJy/sr", "Images units")
    hdr["TELESCOP"] = telescope
    if telescope == "cfht":
        hdr["INSTRUME"] = "Megacam"
    elif telescope == "herschel" and ("pacs" in filter):
        hdr["INSTRUME"] = "PACS"
    elif telescope == "herschel" and ("spire" in filter):
        hdr["INSTRUME"] = "SPIRE"
    else:
        pass
    hdr["FILTER"] = filter
    return hdr

def cutout_from_image(galaxy, telescope, filter):
    coord = SkyCoord(galaxy.ra_pnt, galaxy.dec_pnt, unit = "deg")
    if telescope == "cfht":
        field = find_cfht_field(coord)
        img_path = "/arc/projects/CLIFS/multiwav/{}-{}/{}.{}.fits".format(telescope, filter, field, filter)
    elif telescope == "herschel":
        wav = int(re.search(r'\d+', filter).group())
        img_path = "/arc/projects/CLIFS/multiwav/{}-{}/HATLAS_NGP_DR2_BACKSUB{}.FITS".format(telescope, filter, wav)
    elif telescope == "galex":
        img_path = "/arc/projects/CLIFS/multiwav/{}/mosaic_{}.fits".format(telescope, filter)
    elif telescope == "lofar":
        field = find_lofar_field(coord)
        img_path = "/arc/projects/CLIFS/multiwav/{}-{}/mosaic-blanked-{}.fits".format(telescope, filter, field)
    else:
        raise ValueError("Telescope ({}) is  not supported".format(telescope))
    hdul = fits.open(img_path)
    img, img_h = hdul[0].data, hdul[0].header
    wcs = WCS(img_h)
    img = units_to_MJysr(img, img_h, telescope, filter)
    img_cut, img_cut_h = utils.sky_cutout_from_image(img, coord, 1.5 * u.arcmin, wcs)
    out_hdr = format_output_header(img_cut_h, img_cut.shape, telescope, filter)
    outdir = "/arc/projects/CLIFS/multiwav/cutouts/clifs{}".format(galaxy.clifs_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    hdu = fits.PrimaryHDU(data = img_cut, header = out_hdr)
    hdu.writeto(outdir + "/{}-{}.fits".format(telescope, filter), overwrite = True)
    hdul.close()

def make_multiwav_cutouts(galaxy):
    all_filters = {
                   "galex": ["fuv", "nuv"],
                   "cfht": ["U", "G", "I2"],
                   "herschel": ["pacs100", "pacs160", "spire250", "spire350", "spire500"],
                   "lofar": ["hba"],
                  }
    for telescope in list(all_filters.keys()):
        for filter in all_filters[telescope]:
            logger.info("Making {}: {}".format(telescope, filter))
            cutout_from_image(galaxy, telescope, filter)
