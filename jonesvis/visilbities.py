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

from timedec import timedec


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

        # TODO: Allows this configurable.
        self.dataset.DATA.values[..., (1, 2)] = 0
        self.dataset.DATA.values[..., (0, 3)] = 1

        spw_dataset = xds_from_storage_table(
            str(ms_path) + "::SPECTRAL_WINDOW"
        )[0]

        self.dataset = self.dataset.assign_coords(
            {
                "chan": (("chan",), spw_dataset.CHAN_FREQ.values[0])
            }
        )

        self.dims = {
            "time": np.unique(self.dataset.TIME.values).size,
            "chan": self.dataset.sizes["chan"],
            "ant": self.dataset.ANTENNA2.values.max() + 1,
            "corr": self.dataset.sizes["corr"]
        }

        self.jones_shape = (
            self.dims["time"],
            self.dims["ant"],
            self.dims["chan"],
            1,  # Direction.
            self.dims["corr"]
        )

        self.pix_x, self.pix_y = 512, 512
        self.cell_x, self.cell_y = 2.5e-6, 2.5e-6

        weights = np.ones(self.dataset.DATA.values.shape, dtype=np.float64)
        flags = np.zeros(self.dataset.DATA.values.shape[:-1], dtype=bool)

        self.stokes_vis = vis_to_stokes_vis(self.dataset.DATA.values)
        stokes_weight = wgt_to_stokes_wgt(weights)

        gridded_weights = timedec(grid_weights)(
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

        self.img_weight = timedec(imaging_weights)(
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

        # TODO: Overwrite visibility values.
        self.dataset.DATA.values[:] = 1

        # Assume full resolution gains.
        nb_apply_gains(
            self.dataset.DATA.values,
            gains,
            self.dataset.ANTENNA1.values,
            self.dataset.ANTENNA2.values,
            np.unique(self.dataset.TIME.values, return_inverse=True)[1]
        )

        # Update the visibilities.
        self.stokes_vis = vis_to_stokes_vis(self.dataset.DATA.values)

    def grid(self, pol="I"):

        return timedec(wgridder.vis2dirty)(
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
