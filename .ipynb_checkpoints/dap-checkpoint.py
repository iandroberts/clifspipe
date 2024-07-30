from pathlib import Path
import numpy
from astropy.io import fits
from astropy.wcs import WCS
from mangadap.datacube.datacube import DataCube
from mangadap.util.sampling import Resample, angstroms_per_pixel
import sys
import matplotlib.pyplot as plt
import astropy.units as u
import toml
import re
import subprocess
import os
import shutil
import logging

logger = logging.getLogger("CLIFS_Pipeline")

class WEAVEDataCube(DataCube):
    r"""
    Container class for WEAVE datacubes.

    Args:
        ifile (:obj:`str`):
            The name of the file to read.
    """

    instrument = 'weave'
    """
    Set the name of the instrument.  This is used to construct the
    output file names.
    """

    def __init__(self, ifile):

        _ifile = Path(ifile).resolve()
        # Extract CLIFS/NGC ID from cube file path. This is a bit hacky and relies on
        # the CLIFS/NGC ID being the only int in the path.  Probably a better way...
        where_ngc = ifile.find("ngc")
        find_ints = re.findall(r'\d+', ifile)
        assert len(find_ints) == 1 # Assert that only one int is found
        if where_ngc == -1:
            config_path = "/arc/projects/CLIFS/config_files/clifs_{}.toml".format(find_ints[0])
        else:
            config_path = "/arc/projects/CLIFS/config_files/ngc_{}.toml".format(find_ints[0])
        
        if not _ifile.exists():
            raise FileNotFoundError(f'File does not exist: {_ifile}')

        # Set the paths
        self.directory_path = _ifile.parent
        self.file_name = _ifile.name

        # Collect the metadata into a dictionary
        config = toml.load(config_path)
        meta = config["galaxy"]
        sres = 2500             

        # Open the file and initialize the DataCube base class
        with fits.open(str(_ifile)) as hdu:
            print('Reading WEAVE datacube data ...', end='\r')
            hdr = hdu["FLUX"].header
            prihdr = hdu[0].header
            wcs = WCS(header=hdr, fix=True)
            flux = hdu["FLUX"].data
            ivar = hdu["IVAR"].data
            err = 1 / numpy.sqrt(ivar)
            mask = hdu["MASK"].data.astype(bool) | numpy.logical_not(err > 0.) | numpy.equal(ivar, 0.)
        print('Reading WEAVE datacube data ... DONE')
        
        # SUB-CUBE
        #xmin_py = config["cube"]["xmin"] - 1
        #xmax_py = config["cube"]["xmax"] - 1
        #ymin_py = config["cube"]["ymin"] - 1
        #ymax_py = config["cube"]["ymax"] - 1
        #flux = flux[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]
        #err = err[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]
        #mask = mask[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]

        # Resample to a geometric sampling
        # - Get the wavelength vector
        spatial_shape = flux.shape[1:][::-1]
        nwave = flux.shape[0]
        coo = numpy.array([numpy.ones(nwave), numpy.ones(nwave), numpy.arange(nwave)+1]).T
        wave = (wcs.all_pix2world(coo, 1)[:,2]*wcs.wcs.cunit[2].to('angstrom')) * u.AA
        # - Convert wavelengths to vacuum
        wlum = wave.to(u.um).value
        wave = ((1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wave).to(u.AA).value
        # - Convert the fluxes to flux density
        #dw = angstroms_per_pixel(wave, regular=False)
        #flux /= dw[:,None,None]
        # - Set the geometric step to the mean value.  This means some
        # pixels will be oversampled and others will be averaged.
        dlogl = numpy.mean(numpy.diff(numpy.log10(wave)))  
        # - Resample all the spectra.  Note that the Resample arguments
        # expect the input spectra to be provided in 2D arrays with the
        # last axis as the dispersion axis.
        r = Resample(flux.T.reshape(-1,nwave), e=err.T.reshape(-1,nwave),
                     mask=mask.T.reshape(-1,nwave), x=wave, inLog=False, newRange=wave[[0,-1]],
                     newLog=True, newdx=dlogl)
        # - Reshape and reformat the resampled data in prep for
        # instantiation

        head_new = wcs.to_header()
        head_new["NAXIS"] = (3, "Number of array dimensions")
        head_new["NAXIS1"] = flux.shape[2]
        head_new["NAXIS2"] = flux.shape[1]
        head_new["NAXIS3"] = flux.shape[0]
        head_new["PC1_1"] = (head_new["CDELT1"], "Coordinate transformation matrix element")
        head_new["PC2_2"] = (head_new["CDELT2"], "Coordinate transformation matrix element")
        head_new["PC3_3"] = (r.outx[0] * dlogl * numpy.log(10) * 1e-10, "Coordinate transformation matrix element")
        head_new["CDELT1"] = (1., "[deg] Coordinate increment at reference point")
        head_new["CDELT2"] = (1., "[deg] Coordinate increment at reference point")
        head_new["CDELT3"] = (1., "[m] Coordinate increment at reference point")
        head_new["CRVAL3"] = (r.outx[0] * 1e-10, "[m] Coordinate value at reference point")
        head_new["CTYPE3"] = ("WAVE-LOG", "Vacuum wavelength")
        #head_new["CRPIX1"] = (hdr["CRPIX1"] - xmin_py, "Pixel coordinate of reference point")
        #head_new["CRPIX2"] = (hdr["CRPIX2"] - ymin_py, "Pixel coordinate of reference point")
        head_new["CRPIX1"] = (hdr["CRPIX1"], "Pixel coordinate of reference point")
        head_new["CRPIX2"] = (hdr["CRPIX2"], "Pixel coordinate of reference point")
        head_new["CRPIX3"] = (1.0, "Pixel coordinate of reference point")
        #head_new["OBJRA"] = prihdr["OBJRA"]
        #head_new["OBJDEC"] = prihdr["OBJDEC"]
        wcs_new = WCS(head_new)

        ivar = r.oute.reshape(*spatial_shape,-1)
        mask = r.outf.reshape(*spatial_shape,-1) < 0.8
        ivar[mask] = 0.0
        gpm = numpy.logical_not(mask)
        ivar[gpm] = 1/ivar[gpm]**2
        ivar[~numpy.isfinite(ivar)] = 0.0
        _sres = numpy.full(ivar.shape, sres, dtype=float)
        flux = r.outy.reshape(*spatial_shape,-1)
        flux[mask] = 0.0
        flux[~numpy.isfinite(flux)] = 0.0
        
        # Default name assumes file names like, e.g., '*_icubew.fits'
        super().__init__(flux, ivar=ivar, mask=mask, sres=_sres,
                         wave=r.outx, meta=meta, prihdr=head_new, wcs=wcs_new,
                         name=_ifile.name.split('_')[0])

def _move_manga_dap_output_files(config, dap_dir_name = "HYB10-MILESHC-MASTARSSP", decompress = False):
    os.rename(config["files"]["outdir_dap"] + "{}/weave-calibrated-LOGCUBE-{}.fits.gz".format(dap_dir_name, dap_dir_name),
              config["files"]["outdir_dap"] + "weave-calibrated-LOGCUBE-{}.fits.gz".format(dap_dir_name))
    os.rename(config["files"]["outdir_dap"] + "{}/weave-calibrated-MAPS-{}.fits.gz".format(dap_dir_name, dap_dir_name),
              config["files"]["outdir_dap"] + "weave-calibrated-MAPS-{}.fits.gz".format(dap_dir_name))
    shutil.rmtree(config["files"]["outdir_dap"] + dap_dir_name)

    if decompress:
        subprocess.run(["gunzip", config["files"]["outdir_dap"] + "weave-calibrated-LOGCUBE-{}.fits.gz".format(dap_dir_name)])
        subprocess.run(["gunzip", config["files"]["outdir_dap"] + "weave-calibrated-MAPS-{}.fits.gz".format(dap_dir_name)])
    
def run_manga_dap(galaxy, decompress = False):
    cube_path = galaxy.config["files"]["cube_sci"]
    dap_config_path = "/arc/projects/CLIFS/config_files/weave.toml"
    out_path = galaxy.config["files"]["outdir_dap"]
    subprocess.run(["manga_dap",
                    "-f",
                    cube_path,
                    "--cube_module",
                    "/arc/projects/CLIFS/clifspipe/dap",
                    "WEAVEDataCube",
                    "--plan_module",
                    "mangadap.config.analysisplan.AnalysisPlan",
                    "-p",
                    dap_config_path,
                    "-o",
                    out_path])
    # Move output files back one step in file tree, probably a way to do this via the DAP call..
    _move_manga_dap_output_files(galaxy.config, decompress = decompress)

    