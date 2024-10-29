import numpy as np
from photutils.aperture import SkyEllipticalAperture, SkyEllipticalAnnulus
from clifspipe.galaxy import get_cube

def _spec_err(ivar, calibrate = 1.0):
    tot_var = np.nansum(1 / ivar, axis = (1, 2))
    N = np.sum(np.logical_not(np.isnan(ivar)))
    return (1 + calibrate * np.log10(N)) * np.sqrt(tot_var)

def radial_mask(galaxy, rout, rin = None):
    cube = get_cube(galaxy)
    if rin is None:
        aper = SkyEllipticalAperture(galaxy.c, rout * galaxy.reff, rout * galaxy.reff * galaxy.ba, theta = galaxy.pa)
        aper_mask = aper.to_pixel(cube.wcs.celestial).to_mask(method = "center")
        return aper_mask.to_image(cube.flux.shape[1:]).astype(bool)

    else:
        aper = SkyEllipticalAnnulus(galaxy.c, rin * galaxy.reff, rout * galaxy.reff, rout * galaxy.reff * galaxy.ba, theta = galaxy.pa)
        aper_mask = aper.to_pixel(cube.wcs.celestial).to_mask(method = "center")
        return aper_mask.to_image(cube.flux.shape[1:]).astype(bool)

def stack_spectrum_radial(galaxy, r1 = 0.5, r2 = 1.5):
    cube = get_cube(galaxy)
    flux_in = cube.flux.copy()
    flux_in[np.logical_not(radial_mask(galaxy, rout = r1))] = np.nan
    ivar_in = galaxy.ivar.copy()
    ivar_in[np.logical_not(radial_mask(galaxy, rout = r1))] = np.nan
    
    flux_out = galaxy.flux.copy()
    flux_out[np.logical_not(radial_mask(galaxy, rout = r2, rin = r1))] = np.nan
    ivar_out = galaxy.ivar.copy()
    ivar_out[np.logical_not(radial_mask(galaxy, rout = r2, rin = r1))] = np.nan
    
    spec_in = np.nansum(flux_in, axis = (1, 2))
    spec_in_err = _spec_err(ivar_in, calibrate = galaxy.config["pipeline"]["alpha"])
    
    spec_out = np.nansum(flux_out, axis = (1, 2))
    spec_out_err = _spec_err(ivar_out, calibrate = galaxy.config["pipeline"]["alpha"])
    return galaxy.wave, spec_in, spec_out