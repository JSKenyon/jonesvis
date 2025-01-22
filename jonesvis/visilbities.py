import numpy as np
import xarray
import dask.array as da
import pandas as pd
import panel as pn

from daskms import xds_from_storage_ms, xds_from_storage_table
from ducc0 import wgridder

from jonesvis.utils.gridding import (
    vis_to_stokes,
    grid_weights,
    imaging_weights,
)

from pfb.utils.weighting import (
    _compute_counts,
    counts_to_weights,
    weight_data
)

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

        # TODO: Overwrite visibility values.
        self.dataset.DATA.values[:] = 1

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


    def grid(self, pol="I"):

        pix_x, pix_y = 1024, 1024
        cell_x, cell_y = 2.5e-6, 2.5e-6

        weights = np.ones(self.dataset.DATA.values.shape, dtype=np.float64)
        flags = np.zeros(self.dataset.DATA.values.shape[:-1], dtype=bool)

        stokes_vis, stokes_weight = vis_to_stokes(
            self.dataset.DATA.values,
            weights
        )

        gridded_weights = timedec(grid_weights)(
            self.dataset.UVW.values,
            self.dataset.chan.values,
            (~flags).astype(np.uint8),
            stokes_weight["I"],  # Ask LB.
            pix_x,
            pix_y,
            cell_x,
            cell_y,
            self.dataset.UVW.values.dtype,
            ngrid=1
        )

        img_weight = timedec(imaging_weights)(
            gridded_weights,
            self.dataset.UVW.values,
            self.dataset.chan.values,
            stokes_weight["I"],
            pix_x,
            pix_y,
            cell_x,
            cell_y,
            0,  # Briggs factor
        )

        wsum = img_weight.sum()
        
        return timedec(wgridder.vis2dirty)(
            uvw=self.dataset.UVW.values,
            freq=self.dataset.chan.values,
            vis=stokes_vis[pol],
            wgt=img_weight.astype(np.float32), #np.ones(vis.shape, dtype=np.float32),
            npix_x=pix_x,
            npix_y=pix_y,
            pixsize_x=cell_x,
            pixsize_y=cell_y,
            epsilon=float(1e-5),
            divide_by_n=True,
            do_wgridding=False
        ) / wsum

