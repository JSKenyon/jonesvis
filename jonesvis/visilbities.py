import numpy as np
import xarray
import dask.array as da
import pandas as pd
import panel as pn

from daskms import xds_from_storage_ms, xds_from_storage_table
from ducc0 import wgridder

from jonesvis.utils.gridding import (
    vis_to_stokes_vis,
    wgt_to_stokes_wgt,
    grid_weights,
    imaging_weights,
)
from jonesvis.utils.gains import nb_apply_gains


class Visibilities(object):

    def __init__(self, ms_path):

        self.datasets = xds_from_storage_ms(
            ms_path,
            columns=["TIME", "ANTENNA1", "ANTENNA2", "UVW", "DATA"]
        )

        self.dataset = xarray.combine_by_coords(
            self.datasets,
            combine_attrs="drop_conflicts"
        ).compute()

        spw_dataset = xds_from_storage_table(
            str(ms_path) + "::SPECTRAL_WINDOW"
        )[0]

        feed_datasets = xds_from_storage_table(
            str(ms_path) + "::FEED",
            group_cols="__row__"
        )

        unique_feeds = {
            pt
            for xds in feed_datasets
            for pt in xds.POLARIZATION_TYPE.values.ravel()
        }

        if np.all([feed in "XxYy" for feed in unique_feeds]):
            self.feed_type = "linear"
            correlations = ["XX", "XY", "YX", "YY"]
        elif np.all([feed in "LlRr" for feed in unique_feeds]):
            self.feed_type = "circular"
            correlations = ["RR", "RL", "LR", "LL"]
        else:
            raise ValueError("Unsupported feed type/configuration.")

        self.dataset = self.dataset.assign_coords(
            {
                "chan": (("chan",), spw_dataset.CHAN_FREQ.values[0]),
                "corr": (("corr",), correlations)
            }
        )

        self.dims = {
            "time": np.unique(self.dataset.TIME.values).size,
            "chan": self.dataset.sizes["chan"],
            "ant": self.dataset.ANTENNA2.values.max() + 1,
            "corr": self.dataset.sizes["corr"]
        }

        antenna_dataset = xds_from_storage_table(str(ms_path) + "::ANTENNA")[0]
        self.antenna_positions = antenna_dataset.POSITION.values

        field_dataset = xds_from_storage_table(str(ms_path) + "::FIELD")[0]
        self.phase_dir = tuple(field_dataset.PHASE_DIR.values[0, 0])

        self.set_stokes()  # Set starting stokes params.

        self.pix_x, self.pix_y = 512, 512
        self.cell_x, self.cell_y = 2.5e-6, 2.5e-6

        weights = np.ones(self.dataset.DATA.values.shape, dtype=np.float64)
        flags = np.zeros(self.dataset.DATA.values.shape[:-1], dtype=bool)

        stokes_weight = wgt_to_stokes_wgt(weights, feed_type=self.feed_type)

        gridded_weights = grid_weights(
            self.dataset.UVW.values,
            self.dataset.chan.values,
            (~flags).astype(np.uint8),
            stokes_weight["I"],  # Same for all stokes in this case.
            self.pix_x,
            self.pix_y,
            self.cell_x,
            self.cell_y,
            self.dataset.UVW.values.dtype,
            ngrid=1
        )

        self.img_weight = imaging_weights(
            gridded_weights,
            self.dataset.UVW.values,
            self.dataset.chan.values,
            stokes_weight["I"],  # Same for all stokes in this case.
            self.pix_x,
            self.pix_y,
            self.cell_x,
            self.cell_y,
            0,  # Briggs factor
        )

        self.wsum = self.img_weight.sum()

    def apply_gains(self, gains):

        self.dataset.DATA.values[...] = self.visibility_element

        # Assume full resolution gains.
        nb_apply_gains(
            self.dataset.DATA.values,
            gains,
            self.dataset.ANTENNA1.values,
            self.dataset.ANTENNA2.values,
            np.unique(self.dataset.TIME.values, return_inverse=True)[1]
        )

        # Update the visibilities.
        self.stokes_vis = vis_to_stokes_vis(
            self.dataset.DATA.values,
            feed_type=self.feed_type
        )

    def grid(self, pol="I"):

        return wgridder.vis2dirty(
            uvw=self.dataset.UVW.values,
            freq=self.dataset.chan.values,
            vis=self.stokes_vis[pol],
            wgt=self.img_weight.astype(np.float32),
            npix_x=self.pix_x,
            npix_y=self.pix_y,
            pixsize_x=self.cell_x,
            pixsize_y=self.cell_y,
            epsilon=float(1e-4),
            divide_by_n=True,
            do_wgridding=False,
            nthreads=1
        ) / self.wsum

    def set_stokes(self, stokes=(1, 0, 0, 0),):

        self.stokes = stokes

        if self.feed_type == "linear":
            self.visibility_element = (
                self.stokes[0] + self.stokes[1],
                self.stokes[2] + 1j*self.stokes[3],
                self.stokes[2] - 1j*self.stokes[3],
                self.stokes[0] - self.stokes[1],
            )
        elif self.feed_type == "circular":
            self.visibility_element = (
                self.stokes[0] + self.stokes[3],
                self.stokes[1] + 1j*self.stokes[2],
                self.stokes[1] - 1j*self.stokes[2],
                self.stokes[0] - self.stokes[3],
            )
        else:
            raise ValueError(f"Feed type = {self.feed_type} not understood.")

        self.dataset.DATA.values[...] = self.visibility_element

        self.stokes_vis = vis_to_stokes_vis(
            self.dataset.DATA.values,
            feed_type=self.feed_type
        )