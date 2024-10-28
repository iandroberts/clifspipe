import numpy as np
import matplotlib.pyplot as plt
from clifspipe.galaxy import galaxy
from astropy.wcs import WCS
import matplotlib.gridspec as gs
from astropy.visualization import (PercentileInterval, SqrtStretch,
                                   ImageNormalize, LinearStretch, AsinhStretch)

class panel_image:
    def __init__(self, clifs_id, panels = ["v_star", "vdisp_star", "dn4000", "flux_ha", "v_ha", "vdisp_ha"], figsize = (9.0, 4.0)):
        self.galaxy = galaxy(clifs_id)
        self.fig = plt.Figure(figsize = figsize)
        self.axis_grid = gs.GridSpec(2, 5)
        self.panels = panels
        self.maps = self.galaxy.get_maps()

    def optical(self, gax, rgb = False):
        if rgb:
            print("Not implemented yet...")
        else:
            img, img_h = self.galaxy.get_cutout_image("cfht", "G", header = True)
            ax = self.fig.add_subplot(gax, projection = WCS(img_h).celestial)
            ax.imshow(img)

    def v_star(self, gax, mask = None, xlim = None, ylim = None, vel_min = -75, vel_max = 75, xticks = False, yticks = False):
        vel = self.maps["STELLAR_VEL"].data
        vel[self.maps["BINID"].data[0] == -1] = np.nan
        vel[self.maps["BIN_SNR"].data < 8] = np.nan
        if mask is not None:
            vel[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "RdBu_r",
                       vmin = vel_min,
                       vmax = vel_max,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)

    def vdisp_star(self, gax, mask = None, xlim = None, ylim = None, vel_min = 10, vel_max = 100, xticks = False, yticks = False):
        vel = self.maps["STELLAR_SIGMA"].data
        vel[self.maps["BINID"].data[0] == -1] = np.nan
        vel[self.maps["BIN_SNR"].data < 8] = np.nan
        if mask is not None:
            vel[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "inferno_r",
                       vmin = vel_min,
                       vmax = vel_max,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)

    def dn4000(self, gax, xlim = None, ylim = None, mask = None, yticks = False, xticks = False):
        d4 = self.maps["SPECINDEX"].data[44]
        snr_map = self.maps["SPX_SNR"].data
        d4[snr_map < 3] = np.nan
        if mask is not None:
            d4[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(d4, vmin = 1, vmax = 1.5, cmap = "RdBu_r")
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$\mathrm{D_n4000}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)

    def flux_ha(self, gax, xlim = None, ylim = None, return_mask = False, yticks = False, xticks = False):
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 3
        flux[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        norm = ImageNormalize(flux, interval = PercentileInterval(99.5), stretch = SqrtStretch())
        im = ax.imshow(
                       flux,
                       cmap = "viridis",
                       norm = norm,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.97, "Flux", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)
        if return_mask:
            return mask

    def v_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False, vel_min = -75, vel_max = 75):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GVEL")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 3
        vel[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "RdBu_r",
                       vmin = vel_min,
                       vmax = vel_max,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)  
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        #ax.set_axisbelow(True) ?
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)

    def vdisp_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False, vel_min = 10, vel_max = 100):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GSIGMA")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 3
        vel[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "inferno_r",
                       vmin = vel_min,
                       vmax = vel_max,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticklabel_visible(xticks)
        lat.set_ticklabel_visible(yticks)
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_axislabel(" ", minpad = -5)
        lat.set_axislabel(" ", minpad = -5)  
        ax.tick_params(direction = "in", length = 3.0, width = 0.5)
        #ax.set_axisbelow(True) ?
        ax.set_facecolor("#eeeeee")
        ax.grid(color = "k", alpha = 0.1, lw = 0.5)


    def make(self, filepath):
        if len(self.panels) != 6:
            raise IndexError("List of panels should have precisely six elements")
        self.optical(self.axis_grid[0:2, 0:2])
        for i in range(len(self.panels)):
            r = i // 3
            c = (i % 3) + 2
            getattr(self, self.panels[i])(self.axis_grid[r, c])
        self.fig.savefig(filepath, bbox_inches = "tight", pad_inches = 0.03)

if __name__ == "__main__":
    panel_image(151).make("/arc/projects/CLIFS/panel_img.pdf")
