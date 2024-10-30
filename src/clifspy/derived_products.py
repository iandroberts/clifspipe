from astropy.io import fits
from clifspy.utils import eline_lookup
import numpy as np

def dered_flux(galaxy):
    ha_flux = galaxy.get_eline_map("Ha-6564")
    ha_snr = ha_flux * np.sqrt(galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR"))
    hb_flux = galaxy.get_eline_map("Hb-4862")
    hb_snr = ha_flux * np.sqrt(galaxy.get_eline_map("Hb-4862", map = "GFLUX_IVAR"))
    mask_good = np.greater_equal(ha_snr, 3) & np.greater_equal(hb_snr, 3)
    hb_flux[~mask_good] = np.nan
    ha_flux[~mask_good] = np.nan
    decr = ha_flux / hb_flux
    return 1e-17 * ha_flux * np.power(decr / 2.85, 0.76 * (colour_excess(656.46 * u.nm) + 4.5)) * (u.erg / u.s / u.cm ** 2)

def make_sfr_map(galaxy):
