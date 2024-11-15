import glob
import sys
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import toml
from clifspy.utils import eline_lookup
from astropy.table import Table

class galaxy:
    def __init__(self, clifs_id):
        clifs_cat = Table.read("/arc/projects/CLIFS/catalogs/clifs_master_catalog.fits")
        config_path = f"/arc/projects/CLIFS/config_files/clifs_{clifs_id}.toml"
        self.config = toml.load(config_path)
        self.name = self.config["galaxy"]["name"]
        self.clifs_id = self.config["galaxy"]["clifs_id"]
        self.ra = self.config["galaxy"]["ra"]
        self.dec = self.config["galaxy"]["dec"]
        self.z = self.config["galaxy"]["z"]
        self.c = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg")
        self.reff = self.config["galaxy"]["reff"] * u.arcsec
        self.ell = self.config["galaxy"]["ell"]
        self.pa = self.config["galaxy"]["pa"] * u.deg
        self.r90 = self.config["galaxy"]["r90"] * u.arcsec
        self.manga = self.config["data_coverage"]["manga"]
        self.ra_pnt = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["ra_lifu"])
        self.dec_pnt = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["dec_lifu"])
        self.tail = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["tail_pa"]) >= 0.0

    def get_cutout_image(self, telescope, filter, header = False):
        img_path = "/arc/projects/CLIFS/multiwav/cutouts/clifs{}/{}-{}.fits".format(self.clifs_id, telescope, filter)
        return fits.getdata(img_path, header = header)

    def get_maps(self, ifu = "weave"):
        dap_dir = self.config["files"]["outdir_dap"]
        if ifu == "weave":
            mapsfile = fits.open(dap_dir + "/weave-calibrated-MAPS-HYB10-MILESHC-MASTARSSP.fits")
        elif ifu == "manga":
            find_maps = glob.glob(dap_dir + "/manga-*-MAPS-HYB10-MILESHC-MASTARSSP.fits*")
            if len(find_maps) == 0:
                print("No MaNGA maps file found")
                sys.exit()
            elif len(find_maps) == 1:
                mapsfile = fits.open(find_maps[0])
            else:
                print("Found more than one MaNGA maps file")
                sys.exit()
        else:
            raise ValueError("Invalid ifu")
        return mapsfile

    def get_eline_map(self, line, map = "GFLUX", return_wcs = False, ifu = "weave"):
        mapsfile = self.get_maps(ifu = ifu)
        if return_wcs:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)], WCS(mapsfile["EMLINE_GFLUX"].header).celestial
        else:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)]

class get_cube:
    def __init__(self, galaxy):
        cube = fits.open(galaxy.config["files"]["cube_sci"])
        self.flux = cube["FLUX"].data
        self.ivar = cube["IVAR"].data
        #self.wave = cube["WAVE"].data
        self.sres = 2500
        #self.mask = np.equal(cube["MASK"].data, 0)
        self.wcs = WCS(cube["FLUX"].header)

    def get_center(self):
        x0, y0 = np.round(self.wcs.celestial.world_to_pixel(self.c)).astype(int)
        return x0, y0
