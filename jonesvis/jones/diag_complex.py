import numpy as np
from math import prod

import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade

import param
import panel as pn

from jonesvis.utils.math import kron_matvec
from jonesvis.jones.base import Gain

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class DiagComplex(Gain):

    std_dev = param.Number(
        label="Standard deviation",
        bounds=(0, 2),
        step=0.05,
        default=0.1
    )

    length_scale_time = param.Number(
        label="Length Scale (Time)",
        bounds=(0, 1),
        step=0.05,
        default=0.25
    )
    length_scale_freq = param.Number(
        label="Length Scale (Frequency)",
        bounds=(0, 1),
        step=0.05,
        default=0.1
    )

    _gain_parameters = [
        "std_dev",
        "length_scale_time",
        "length_scale_freq",
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
        # nu = 2.5 * (freqs / freqs.mean() - 1.0)
        nu = (freqs - freqs.min()) / (freqs.max() - freqs.min())

        # We cannot set independent std as in the kronecker product they end
        # up getting combined.

        tt = np.abs(t[:, None] - t[None, :])
        lt = self.length_scale_time
        Kt = self.std_dev * np.exp(-tt**2/(2*lt**2))  # Squared exponential.
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
        vv = np.abs(nu[:, None] - nu[None, :])
        lv = self.length_scale_freq
        Kv = self.std_dev * np.exp(-vv**2/(2*lv**2))
        Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
        L = (Lt, Lv)  # np.kron(Lt, Lv) vec(chi)

        jones = np.zeros((ntime, nchan, nant, 1, 4), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = rng.standard_normal(size=(ntime, nchan))
                amp = 1 + kron_matvec(L, xi_amp)  # No guarantee of positivity. #np.exp(-nu[None, :]**2 + kron_matvec(L, xi_amp))
                xi_phase = rng.standard_normal(size=(ntime, nchan))
                phase = kron_matvec(L, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

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
