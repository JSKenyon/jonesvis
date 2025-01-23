import numpy as np
from math import prod

import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade

import param
import panel as pn

from jonesvis.jones.base import Gain
from jonesvis.utils.math import kron_matvec

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class Delay(Gain):

    std_dev = param.Number(
        label="Standard deviation (in ns)",
        bounds=(0, 1),
        step=0.01,
        default=0
    )

    length_scale_time = param.Number(
        label="Length Scale (Time)",
        bounds=(0, 1),
        step=0.05,
        default=0.25
    )

    _gain_parameters = [
        "std_dev",
        "length_scale_time"
    ]

    def __init__(self, vis, **params):
        super().__init__(vis, **params)

    @pn.depends(*_gain_parameters, watch=True)
    def update_gains(self):

        freqs = self.freqs
        times = self.times
        ntime = times.size
        nchan = freqs.size
        nant = self.n_ant

        rng = np.random.default_rng(12345)  # Set seed.

        # This is not really required. Could leave this in physical units.
        t = (times - times.min()) / (times.max() - times.min())

        # We cannot set independent std as in the kronecker product they end
        # up getting combined.

        # Delay
        tt = np.abs(t[:, None] - t[None, :])
        lt = self.length_scale_time
        Kt = self.std_dev ** 2 * np.exp(-tt**2/(2*lt**2))  # Squared exponential.

        if self.std_dev:
            Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
        else:
            Lt = np.zeros((ntime, ntime))

        delays = np.zeros((ntime, 1, nant, 1, 4), dtype=np.float64)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_delay = rng.standard_normal(size=(ntime,))
                delays[:, 0, p, 0, c] =  Lt @ xi_delay

        jones = np.exp(1j * delays / 1e9 * freqs[None, :, None, None, None])
        jones[..., (1, 2)] = 0  # Diagonal term.

        self.gains = jones

    @pn.depends(*_gain_parameters, watch=True)
    def update_stokes_images(self):

        pn.state.log(f'Plot update triggered.')

        self.vis.apply_gains(self.gains)

        plots = []

        for pol in "IQUV":
            image_data = self.vis.grid(pol)
            plots.append(
                hv.Image(
                    image_data
                ).opts(
                    responsive=True,
                    clim=(self.vmin, self.vmax),
                    title=pol,
                    colorbar=True,
                    cmap="inferno",
                )
            )

        self.stokes_images = plots

    @pn.depends(*_gain_parameters, watch=True)
    def update_jones_images(self):

        plots = [
            hv.Image(np.abs(self.gains[:,:,0,0,0])).opts(responsive=True, colorbar=True),
            hv.Image(np.angle(self.gains[:,:,0,0,0])).opts(responsive=True, colorbar=True)
        ]

        self.jones_images = plots

