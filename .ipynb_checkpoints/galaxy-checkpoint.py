from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import toml

class galaxy:
    def __init__(self, clifs_id):
        config_path = f"/arc/projects/CLIFS/config_files/clifs_{clifs_id}.toml"
        self.config = toml.load(config_path)
        self.name = self.config["galaxy"]["name"]
        self.clifs_id = self.config["galaxy"]["clifs_id"]
        self.ra = self.config["galaxy"]["ra"]
        self.dec = self.config["galaxy"]["dec"]
        self.c = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg")
        self.reff = self.config["galaxy"]["reff"] * u.arcsec
        self.ell = self.config["galaxy"]["ell"]
        self.pa = self.config["galaxy"]["pa"] * u.deg

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