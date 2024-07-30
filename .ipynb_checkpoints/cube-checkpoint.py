## TO-DO:
# - Pass value for alpha to ivar_sum that does not break the spatial downsampling function

import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from tqdm import trange
import numpy as np
from spectral_cube import SpectralCube
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from photutils.aperture import SkyEllipticalAperture
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import logging
from clifspipe.astrometry import find_astrometry_solution
import subprocess

logger = logging.getLogger("CLIFS_Pipeline")

def _do_nothing():
    pass

def _ivar_sum(arr, axis):
    N = 2
    alpha = 1.28
    noise_correction = 1.0 + alpha * np.log10(N)
    var = 1 / arr
    new_var = noise_correction * np.nansum(var, axis = axis)
    return 1 / new_var
    
def _standard_err_mean(arr, axis):
    return np.nanmean(arr, axis = axis) / np.sqrt(2)
    
def preprocess_cube(fname, config, hdul, arm, downsample_wav = False, ext = 1, ext_ivar = 2, ext_fluxcal = 5, bkgsub = False,
                    clobber = False, fill_ccd_gaps = False, verbose = False, fullfield = False):
    cal_data = hdul[ext].data * hdul[ext_fluxcal].data[:, None, None]
    cal_ivar = hdul[ext_ivar].data * (1 / hdul[ext_fluxcal].data[:, None, None] ** 2)

    if config["cube"]["xmin"] == -99:
        fullfield = True

    if not fullfield:
        xmin_py = config["cube"]["xmin"] - 1
        xmax_py = config["cube"]["xmax"] - 1
        ymin_py = config["cube"]["ymin"] - 1
        ymax_py = config["cube"]["ymax"] - 1
        cal_data = cal_data[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]
        cal_ivar = cal_ivar[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]

    wcs = WCS(hdul[ext].header)
    cdelt = hdul[ext].header["CD2_2"]
    nwave = cal_data.shape[0]
    coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave)+1]).T
    wave_orig = wcs.all_pix2world(coo, 1)[:,2]*wcs.wcs.cunit[2].to('angstrom')
    Afibre = np.pi * (2.6 / 2.) ** 2
    Apx = (3600 * cdelt) ** 2
    cal_data /= ((Afibre / Apx) * 1e-17)
    cal_ivar /= ((Apx / Afibre) ** 2 * 1e34)

    if downsample_wav:
        wave, cal_data = downsample_wav_axis(wave_orig, cal_data, "flux", return_wave = True, verbose = verbose)
        dwave = np.median(np.diff(wave))
        cal_ivar = downsample_wav_axis(wave_orig, cal_ivar, "ivar", verbose = verbose)
    else:
        wave = wave_orig.copy()
        dwave = np.median(np.diff(wave))

    if fill_ccd_gaps:
        cal_data = fill_ccd_gaps(wave, cal_data, arm, verbose = verbose)
        cal_ivar = fill_ccd_gaps(wave, cal_ivar, arm, verbose = verbose)
    
    head_new = wcs.to_header()
    head_new["NAXIS"] = (3, "Number of array dimensions")
    head_new["NAXIS1"] = cal_data.shape[2]
    head_new["NAXIS2"] = cal_data.shape[1]
    head_new["NAXIS3"] = cal_data.shape[0]
    head_new["PC3_3"] = (dwave * 1e-10, "Coordinate transformation matrix element")
    head_new["CDELT1"] = (1., "[deg] Coordinate increment at reference point")
    head_new["CDELT2"] = (1., "[deg] Coordinate increment at reference point")
    head_new["CDELT3"] = (1., "[m] Coordinate increment at reference point")
    head_new["CRVAL3"] = (wave[0] * 1e-10, "[m] Coordinate value at reference point")
    head_new["CTYPE3"] = ("AWAV", "Air wavelength")
    if not fullfield:
        head_new["CRPIX1"] = (hdul[ext].header["CRPIX1"] - xmin_py, "Pixel coordinate of reference point")
        head_new["CRPIX2"] = (hdul[ext].header["CRPIX2"] - ymin_py, "Pixel coordinate of reference point")
    else:
        head_new["CRPIX1"] = (hdul[ext].header["CRPIX1"], "Pixel coordinate of reference point")
        head_new["CRPIX2"] = (hdul[ext].header["CRPIX2"], "Pixel coordinate of reference point")
    head_new["CRPIX3"] = (1.0, "Pixel coordinate of reference point")
    
    if bkgsub:
        cal_data = bkg_sub(config, cal_data, WCS(head_new), verbose = verbose)
    
    prim_hdu = fits.PrimaryHDU(header = hdul[0].header)
    #head_new = hdul[ext].header
    head_new["BUNIT"] = ("1E-17 erg/(s cm2 Ang)", "units of image")
    data_hdu = fits.ImageHDU(data = cal_data, header = head_new, name = "FLUX")
    #head_ivar = hdul[ext_ivar].header
    head_new["BUNIT"] = ("1E34 (s2 cm4 Ang2)/erg2", "units of image")
    ivar_hdu = fits.ImageHDU(data = cal_ivar, header = head_new, name = "IVAR")
    hdul_out = fits.HDUList([prim_hdu, data_hdu, ivar_hdu])
    
    name_split = fname.split(".fit")[0]
    hdul_out.writeto(name_split + "_cal.fit", overwrite = clobber)
    hdul_out.close()
    return name_split + "_cal.fit"
        
def bkg_sub(config, data, wcs, verbose = False):
    if config["pipeline"]["bkgsub_galmask"]:
        reff = config["galaxy"]["reff"] * u.arcsec
        ba = 1 - config["galaxy"]["ell"]
        pa = config["galaxy"]["pa"] * u.deg
        coord = SkyCoord(ra = config["galaxy"]["ra"], dec = config["galaxy"]["dec"], unit = "deg")
        aper = SkyEllipticalAperture(coord, 2 * reff, 2 * ba * reff, theta = pa)
        aper_px = aper.to_pixel(wcs.dropaxis(2))
        mask = aper_px.to_mask().to_image(data.shape[1:])
    else:
        mask = np.zeros(data.shape[1:])

    if verbose:
        for ch in trange(data.shape[0], desc = "Subtracting background"):
            sigma_clip = SigmaClip(sigma = 3.0)
            bkg_estimator = MedianBackground()
            bkg = Background2D(data[ch, :, :], (20, 20), filter_size=(3, 3), mask = mask.astype(bool),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            data[ch, :, :] = data[ch, :, :] - bkg.background
    else:
        for ch in range(data.shape[0]):
            sigma_clip = SigmaClip(sigma = 3.0)
            bkg_estimator = MedianBackground()
            bkg = Background2D(data[ch, :, :], (20, 20), filter_size=(3, 3), mask = mask.astype(bool),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            data[ch, :, :] = data[ch, :, :] - bkg.background
        
    return data

def downsample_wav_axis(wave, data, method, return_wave = False, verbose = False):
    new_data = np.zeros((data.shape[0] // 2, data.shape[1], data.shape[2]))
    new_wave = np.zeros(new_data.shape[0])
    k = 0
    if verbose:
        for i in trange(new_data.shape[0]):
            if method == "flux":
                new_data[i, :, :] = 0.5 * (data[k, :, :] + data[k + 1, :, :])
            elif method == "ivar":
                var = 0.5 * (1 / data[k, :, :] + 1 / data[k + 1, :, :])
                new_data[i, :, :] = 1 / (var / 2)
            else:
                raise ValueError
            new_wave[i] = 0.5 * (wave[k] + wave[k + 1])
            k += 2
    else:
        for i in range(new_data.shape[0]):
            if method == "flux":
                new_data[i, :, :] = 0.5 * (data[k, :, :] + data[k + 1, :, :])
            elif method == "ivar":
                var = 0.5 * (1 / data[k, :, :] + 1 / data[k + 1, :, :])
                new_data[i, :, :] = 1 / (var / 2)
            else:
                raise ValueError
            new_wave[i] = 0.5 * (wave[k] + wave[k + 1])
            k += 2
    
    if return_wave:
        return new_wave, new_data
    else:
        return new_data
     
def fill_ccd_gaps(wave, data, arm, verbose = False):
    if arm == "red":
        lgap = [7590, 7695]
        llow = [7485, 7585]
        lhigh = [7700, 7800]
        mask_gap = (wave >= lgap[0]) & (wave <= lgap[1])
        Ngap = mask_gap.sum()
        mask_low = (wave > llow[0]) & (wave < llow[1])
        mask_high = (wave > lhigh[0]) & (wave < lhigh[1])

        if verbose:
            for i in trange(data.shape[1], desc = "Filling ccd gaps..."):
                for j in range(data.shape[2]):
                    data_low = data[mask_low, i, j]
                    data_high = data[mask_high, i, j]
                    mean = np.nanmean(np.concatenate((data_low, data_high)))
                    std = np.nanstd(np.concatenate((data_low, data_high)))
                    noise = np.random.normal(mean, std, size = Ngap)
                    data[mask_gap, i, j] = mean
        else:
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    data_low = data[mask_low, i, j]
                    data_high = data[mask_high, i, j]
                    mean = np.nanmean(np.concatenate((data_low, data_high)))
                    std = np.nanstd(np.concatenate((data_low, data_high)))
                    noise = np.random.normal(mean, std, size = Ngap)
                    data[mask_gap, i, j] = mean
                
        return data
                
    elif arm == "blue":
        lgap = [5491, 5581]
        llow = [5386, 5486]
        lhigh = [5578, 5678]
        mask_gap = (wave >= lgap[0]) & (wave <= lgap[1])
        Ngap = mask_gap.sum()
        mask_low = (wave > llow[0]) & (wave < llow[1])
        mask_high = (wave > lhigh[0]) & (wave < lhigh[1])
        
        for i in trange(data.shape[1], desc = "Filling ccd gaps..."):
            for j in range(data.shape[2]):
                data_low = data[mask_low, i, j]
                data_high = data[mask_high, i, j]
                mean = np.nanmean(np.concatenate((data_low, data_high)))
                std = np.nanstd(np.concatenate((data_low, data_high)))
                noise = np.random.normal(mean, std, size = Ngap)
                data[mask_gap, i, j] = mean
                
        return data
                
    else:
        raise ValueError("Arm must be 'red' or 'blue'")
        
def downsample_cube_spatial(cube, axes, cube_type, factor = 2, verbose = False):
    cube.allow_huge_operations = True
    for a in axes:
        if cube_type == "flux":
            cube = cube.downsample_axis(factor, a, estimator = np.nansum, progressbar = verbose)
        elif cube_type == "ivar":
            cube = cube.downsample_axis(factor, a, estimator = _ivar_sum, progressbar = verbose)
        else:
            raise ValueError
    return cube
        
def reproject_spectral_axis(cube, l_low, l_high, dl, fill_value = 0, verbose = False):
    dl_ang = dl.to(u.AA).value
    new_spectral_axis = np.arange(l_low.to(u.AA).value, l_high.to(u.AA).value + dl_ang / 2, dl_ang) * u.AA
    if verbose:
        cube_newspec = cube.spectral_interpolate(new_spectral_axis, fill_value = fill_value)
    else:
        cube_newspec = cube.spectral_interpolate(new_spectral_axis, fill_value = fill_value, update_function = _do_nothing)
    
    return cube_newspec
    
def stitch_cubes(cube_blue, cube_red, ivar_blue, ivar_red):
    cube_blue.allow_huge_operations = True
    cube_red.allow_huge_operations = True
    ivar_blue.allow_huge_operations = True
    ivar_red.allow_huge_operations = True
    cube_full = (cube_blue * ivar_blue + cube_red * ivar_red) / (ivar_blue + ivar_red)
    ivar_full = (ivar_blue * ivar_blue + ivar_red * ivar_red) / (ivar_blue + ivar_red)
    
    return cube_full, ivar_full
    
def write_fullcube(galaxy, fname_out, fname_blue, fname_red, config, cube_full, ivar_full):
    head = fits.Header()
    head["TELESCOP"] = ("WHT", "4.2m William Herschel Telescope")
    head["DETECTOR"] = ("WEAVELIFU", "WEAVE Large IFU")
    head["INFILE_B"] = fname_blue
    head["INFILE_R"] = fname_red
    head["OBJRA"] = config["galaxy"]["ra"]
    head["OBJDEC"] = config["galaxy"]["dec"]
    
    data = np.nan_to_num(cube_full.unmasked_data[:, :, :].value)
    ivar = np.nan_to_num(ivar_full.unmasked_data[:, :, :].value)
    ivar[ivar < 0] = 0
    mask = np.isnan(data) | np.isnan(ivar)

    if galaxy.config["pipeline"]["fix_astrometry"]:
        new_hdr = find_astrometry_solution(data, WCS(cube_full.header))
        logger.info("Fixed WCS solution")
        prim_hdu = fits.PrimaryHDU(header = head)
        img_hdu = fits.ImageHDU(data = data, header = new_hdr, name = "FLUX")
        ivar_hdu = fits.ImageHDU(data = ivar, header = new_hdr, name = "IVAR")
        mask_hdu = fits.ImageHDU(data = mask.astype(int), header = new_hdr, name = "MASK")
    else:
        prim_hdu = fits.PrimaryHDU(header = head)
        img_hdu = fits.ImageHDU(data = data, header = cube_full.header, name = "FLUX")
        ivar_hdu = fits.ImageHDU(data = ivar, header = cube_full.header, name = "IVAR")
        mask_hdu = fits.ImageHDU(data = mask.astype(int), header = cube_full.header, name = "MASK")
    
    hdul = fits.HDUList([prim_hdu, img_hdu, ivar_hdu, mask_hdu])
    hdul.writeto(fname_out, overwrite = True)

    fname_flux = fname_out.split(".fits")[0] + "_only-flux.fits"
    hdu_flux = fits.PrimaryHDU(data = data.astype("float32"), header = cube_full.header)
    hdu_flux.writeto(fname_flux, overwrite = True)

    if galaxy.config["pipeline"]["hdf5"]:
        subprocess.run(["fits2idia", fname_flux])
        logger.info("Converted cube to HDF5")

def generate_cube(galaxy, fullfield = False):
    if galaxy.config["pipeline"]["downsample_spatial"] and ((galaxy.config["pipeline"]["alpha"] is None) | (galaxy.config["pipeline"]["factor_spatial"] is None)):
        raise ValueError("If 'downsample_spatial = True', 'alpha', 'factor' cannot be None")

    fname_blue = galaxy.config["files"]["cube_blue"]
    fname_red = galaxy.config["files"]["cube_red"]

    hdul_blue = fits.open(fname_blue)
    hdul_red = fits.open(fname_red)
    
    cal_fname_red = preprocess_cube(fname_red, galaxy.config, hdul_red, "red", downsample_wav = galaxy.config["pipeline"]["downsample_wav"],
                                    bkgsub = galaxy.config["pipeline"]["bkgsub"], clobber = galaxy.config["pipeline"]["clobber"],
                                    fill_ccd_gaps = galaxy.config["pipeline"]["fill_ccd_gaps"], verbose = galaxy.config["pipeline"]["verbose"],
                                   fullfield = fullfield)    
    cal_fname_blue = preprocess_cube(fname_blue, galaxy.config, hdul_blue, "blue", downsample_wav = galaxy.config["pipeline"]["downsample_wav"],
                                     bkgsub = galaxy.config["pipeline"]["bkgsub"], clobber = galaxy.config["pipeline"]["clobber"],
                                     fill_ccd_gaps = galaxy.config["pipeline"]["fill_ccd_gaps"], verbose = galaxy.config["pipeline"]["verbose"],
                                    fullfield = fullfield)
    logger.info("Done preprocessing")
    hdul_blue.close()
    hdul_red.close()
    
    cube_blue = SpectralCube.read(cal_fname_blue, hdu = 1)
    cube_red = SpectralCube.read(cal_fname_red, hdu = 1)
    ivar_blue = SpectralCube.read(cal_fname_blue, hdu = 2)
    ivar_red = SpectralCube.read(cal_fname_red, hdu = 2)
    logger.info("Read flux-calibrated cubes")
    
    if galaxy.config["pipeline"]["downsample_spatial"]:
        cube_blue = downsample_cube_spatial(cube_blue, [1, 2], "flux", factor = galaxy.config["pipeline"]["factor_spatial"],
                                            verbose = galaxy.config["pipeline"]["verbose"])
        cube_red = downsample_cube_spatial(cube_red, [1, 2], "flux", factor = galaxy.config["pipeline"]["factor_spatial"],
                                           verbose = galaxy.config["pipeline"]["verbose"])
        ivar_blue = downsample_cube_spatial(ivar_blue, [1, 2], "ivar", factor = galaxy.config["pipeline"]["factor_spatial"],
                                            verbose = galaxy.config["pipeline"]["verbose"])
        ivar_red = downsample_cube_spatial(ivar_red, [1, 2], "ivar", factor = galaxy.config["pipeline"]["factor_spatial"],
                                           verbose = galaxy.config["pipeline"]["verbose"])
        logger.info("Done spatial binning")
    
    cube_blue = reproject_spectral_axis(cube_blue, 3700 * u.AA, 9000 * u.AA, 1. * u.AA, verbose = galaxy.config["pipeline"]["verbose"])
    cube_red = reproject_spectral_axis(cube_red, 3700 * u.AA, 9000 * u.AA, 1. * u.AA, verbose = galaxy.config["pipeline"]["verbose"])
    ivar_blue = reproject_spectral_axis(ivar_blue, 3700 * u.AA, 9000 * u.AA, 1. * u.AA, verbose = galaxy.config["pipeline"]["verbose"])
    ivar_red = reproject_spectral_axis(ivar_red, 3700 * u.AA, 9000 * u.AA, 1. * u.AA, verbose = galaxy.config["pipeline"]["verbose"])
    logger.info("Reprojected red and blue cubes onto common spectral axis")
    cube_full, ivar_full = stitch_cubes(cube_blue, cube_red, ivar_blue, ivar_red)
    logger.info("Combined red and blue cubes")
    
    outdir = galaxy.config["files"]["outdir"]
    if galaxy.config["pipeline"]["downsample_spatial"]:
        if fullfield:
            outfile = outdir + "calibrated_cube_full.fits"
        else:
            outfile = outdir + "calibrated_cube.fits"
        write_fullcube(galaxy, outfile, fname_blue, fname_red, galaxy.config, cube_full, ivar_full)
        logger.info(f"Wrote combined, flux-calibrated cube: {outfile}")
    else:
        outfile = outdir + "calibrated_cube_p5.fits"
        write_fullcube(galaxy, outfile, fname_blue, fname_red, galaxy.config, cube_full, ivar_full)
        logger.info(f"Wrote combined, flux-calibrated cube: {outfile}")

##
