import numpy as np
import matplotlib.pyplot as plt
from clifspipe.galaxy import galaxy
from astropy.wcs import WCS
import matplotlib.gridspec as gs
from astropy.visualization import (AsymmetricPercentileInterval, PercentileInterval, SqrtStretch,
                                   ImageNormalize, LinearStretch, AsinhStretch)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

class panel_image:
    def __init__(self, clifs_id, panels = ["v_star", "vdisp_star", "dn4000", "flux_ha", "v_ha", "vdisp_ha"], figsize = (9.0, 2.9)):
        self.galaxy = galaxy(clifs_id)
        self.fig = plt.Figure(figsize = figsize)
        self.axis_grid = gs.GridSpec(2, 5)
        self.axis_grid.update(wspace = 0.05, hspace = 0.1)
        self.panels = panels
        self.maps = self.galaxy.get_maps()

    def _offset_axis(self, ax, labels = False, grid = False):
         # Remove the absolute coordinates
        ra = ax.coords["ra"]
        dec = ax.coords["dec"]
        ra.set_ticks_visible(False)
        ra.set_ticklabel_visible(False)
        dec.set_ticks_visible(False)
        dec.set_ticklabel_visible(False)
        ra.set_axislabel("")
        dec.set_axislabel("")
        # Create an overlay with relative coordinates
        aframe = self.galaxy.c.skyoffset_frame()
        overlay = ax.get_coords_overlay(aframe)
        ra_offset = overlay["lon"]
        dec_offset = overlay["lat"]
        if labels:
            ra_offset.set_axislabel(r"$\Delta\,\mathrm{RA}$")
            dec_offset.set_axislabel(r"$\Delta\,\mathrm{Dec}$")
            ra_offset.set_major_formatter("s")
            dec_offset.set_major_formatter("s")
        else:
            ra_offset.set_axislabel(" ", minpad = -5)
            dec_offset.set_axislabel(" ", minpad = -5)
        ra_offset.set_ticks_visible(labels)
        dec_offset.set_ticks_visible(labels)
        ra_offset.set_ticklabel_visible(labels)
        dec_offset.set_ticklabel_visible(labels)
        ra_offset.set_ticks_position("b")
        dec_offset.set_ticks_position("l")
        ra_offset.set_axislabel_position("b")
        dec_offset.set_axislabel_position("l")
        ra_offset.set_ticklabel_position("b")
        dec_offset.set_ticklabel_position("l")
        if grid:
            overlay.grid(color = "k", alpha = 0.1, lw = 0.5)

    def optical(self, gax, rgb = False, xlim = None, ylim = None, Nr = 2):
        img, img_h = self.galaxy.get_cutout_image("cfht", "G", header = True)
        x0, y0 = WCS(img_h).celestial.world_to_pixel(self.galaxy.c)
        cd = img_h["PC2_2"]
        r90 = self.galaxy.config["galaxy"]["r90"]
        xlim = [int(x0 - Nr * (r90 / 3600 / cd)), int(x0 + Nr * (r90 / 3600 / cd))]
        ylim = [int(y0 - Nr * (r90 / 3600 / cd)), int(y0 + Nr * (r90 / 3600 / cd))]
        if rgb:
            imgU, imgU_h = self.galaxy.get_cutout_image("cfht", "U", header = True)
            imgI, imgI_h = self.galaxy.get_cutout_image("cfht", "I2", header = True)
            normU = ImageNormalize(imgU, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
            normG = ImageNormalize(img, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
            normI = ImageNormalize(imgI, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
            rgb_array = np.array([normI(imgI), normG(img), normU(imgU)])
            ax = self.fig.add_subplot(gax, projection = WCS(img_h).celestial)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.imshow(np.moveaxis(rgb_array, 0, -1))
            self._offset_axis(ax, labels = True)
        else:
            norm = ImageNormalize(img, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
            ax = self.fig.add_subplot(gax, projection = WCS(img_h).celestial)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.imshow(img, norm = norm, cmap = "binary")
            self._offset_axis(ax, labels = True)
            #ax.tick_params(direction = "in", length = 3.0, width = 0.5)

    def v_star(self, gax, mask = None, xlim = None, ylim = None, vel_min = -100, vel_max = 100, xticks = False, yticks = False):
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
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [-75, 0, 75])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def vdisp_star(self, gax, mask = None, xlim = None, ylim = None, vel_min = 0, vel_max = 100, xticks = False, yticks = False):
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
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [10, 90])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$\sigma_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

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
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [1.1, 1.4])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$\mathrm{D_n4000}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def flux_ha(self, gax, xlim = None, ylim = None, return_mask = False, yticks = False, xticks = False):
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 4
        flux[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[0].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        norm = ImageNormalize(flux, interval = AsymmetricPercentileInterval(1.0, 95.0), stretch = LinearStretch())
        im = ax.imshow(
                       flux,
                       cmap = "viridis",
                       norm = norm,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.97, r"$F_\mathrm{H\alpha}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")
        if return_mask:
            return mask

    def v_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False, vel_min = -100, vel_max = 100):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GVEL")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 4
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
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [-75, 0, 75])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$V_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def vdisp_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False, vel_min = 0, vel_max = 140):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GSIGMA")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 4
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
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [20, 120])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.07, 0.98, r"$\sigma_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def make(self, filepath, rgb = False, Nr = 2):
        if len(self.panels) != 6:
            raise IndexError("List of panels should have precisely six elements")
        self.optical(self.axis_grid[0:2, 0:2], rgb = rgb, Nr = Nr)
        x0, y0 = WCS(self.maps[0].header).celestial.world_to_pixel(self.galaxy.c)
        cd = self.maps[0].header["PC2_2"]
        r90 = self.galaxy.config["galaxy"]["r90"]
        xlim = [int(x0 - Nr * (r90 / 3600 / cd)), int(x0 + Nr * (r90 / 3600 / cd))]
        ylim = [int(y0 - Nr * (r90 / 3600 / cd)), int(y0 + Nr * (r90 / 3600 / cd))]
        for i in range(len(self.panels)):
            r = i // 3
            c = (i % 3) + 2
            getattr(self, self.panels[i])(self.axis_grid[r, c], xlim = xlim, ylim = ylim)
        self.fig.savefig(filepath, bbox_inches = "tight", pad_inches = 0.03)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("clifs_id", type = int)
    parser.add_argument("--rgb", action = "store_true")
    parser.add_argument("--Nr", default = 1.5, type = float)
    args = parser.parse_args()
    panel_image(args.clifs_id).make("/arc/projects/CLIFS/plots/panel_images/panel_img_clifs{}.pdf".format(args.clifs_id), rgb = args.rgb, Nr = args.Nr)
