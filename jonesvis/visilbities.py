import numpy as np
import xarray
import dask.array as da
import pandas as pd

from daskms import xds_from_storage_ms, xds_from_storage_table
from ducc0 import wgridder

from pfb.utils.weighting import (
    _compute_counts,
    counts_to_weights,
    weight_data
)

from timedec import timedec

from numba import literally


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


    def grid(self):

        pix_x, pix_y = 1024, 1024
        cell_x, cell_y = 2.5e-6, 2.5e-6

        jones = np.zeros(self.jones_shape, dtype=np.complex128)
        jones[..., (0, -1)] = 1

        jones = jones.reshape(self.jones_shape[:-1] + (2, 2))

        _, tbin_idx, tbin_counts = np.unique(
            self.dataset.TIME.values,
            return_counts=True,
            return_index=True
        )

        weights = np.ones(self.dataset.DATA.values.shape, dtype=np.float64)
        flags = np.zeros(self.dataset.DATA.values.shape[:-1], dtype=bool)

        # This can be done much more simply by manually applying the T matrix.
        # Check the PFB code.

        stokes_vis, stokes_weight = timedec(weight_data)(
            self.dataset.DATA.values,
            weights,
            flags,
            jones,
            tbin_idx,
            tbin_counts,
            self.dataset.ANTENNA1.values,
            self.dataset.ANTENNA2.values,
            "linear",
            "I",  # FS for full stokes on joint-corr branch.
            str(self.dims["corr"])
        )

        counts = timedec(_compute_counts)(
            self.dataset.UVW.values,
            self.dataset.chan.values,
            (~flags).astype(np.uint8),
            stokes_weight.astype(np.float64),  # Ask LB.
            pix_x,
            pix_y,
            cell_x,
            cell_y,
            self.dataset.UVW.values.dtype,
            ngrid=1,
            usign=1.0,
            vsign=1.0
        )

        img_weight = timedec(counts_to_weights)(
            counts,
            self.dataset.UVW.values,
            self.dataset.chan.values,
            stokes_weight,
            pix_x,
            pix_y,
            cell_x,
            cell_y,
            0,  # Briggs factor
            usign=1.0,
            vsign=1.0
        )

        wsum = img_weight.sum()

        return timedec(wgridder.vis2dirty)(
            uvw=self.dataset.UVW.values,
            freq=self.dataset.chan.values,
            vis=stokes_vis,
            wgt=img_weight, #np.ones(vis.shape, dtype=np.float32),
            npix_x=pix_x,
            npix_y=pix_y,
            pixsize_x=cell_x,
            pixsize_y=cell_y,
            epsilon=float(1e-5),
            divide_by_n=True,
            do_wgridding=False
        ) / wsum

