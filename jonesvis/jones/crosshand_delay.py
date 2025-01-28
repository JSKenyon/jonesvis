import numpy as np

import holoviews as hv

import param
import panel as pn

from jonesvis.jones.base import Gain

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class CrosshandDelay(Gain):

    std_dev = param.Number(
        label="Standard Deviation (in ns)",
        bounds=(0, 2),
        step=0.01,
        default=0
    )

    time_invariant = param.Boolean(
        label="Time Invariant",
        default=True
    )

    length_scale_time = param.Number(
        label="Length Scale (Time)",
        bounds=(0, 5),
        step=0.01,
        default=0.25
    )

    _gain_parameters = Gain._gain_parameters + [
        "std_dev",
        "time_invariant",
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

        rng = np.random.default_rng(self.random_seed)  # Set seed.

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

        xi_delay = rng.standard_normal(size=(ntime,))
        delays[:, 0, :, 0, 0] =  (Lt @ xi_delay)[:, None]

        if self.time_invariant:
            delays[:, 0, :, 0, 0] = delays[0, 0, 0, 0, 0]

        jones = np.exp(1j * delays * 1e-9 * freqs[None, :, None, None, None])
        jones[..., (1, 2)] = 0  # Diagonal term.

        self.delays = delays
        self.gains = jones

    @pn.depends(*_gain_parameters, *Gain._data_parameters, watch=True)
    def update_stokes_images(self):
        super().update_stokes_images()

    @pn.depends(*_gain_parameters, *Gain._selection_parameters, watch=True)
    def update_jones_images(self):

        corr_idx = self.param.correlation.objects.index(self.correlation)

        selected_gains = self.gains[:, :, self.antenna, 0, corr_idx]
        phase = np.rad2deg(np.angle(selected_gains))
        delay = self.delays[:, 0, self.antenna, 0, corr_idx]

        plots = [
            hv.Scatter(
                list(zip(self.times, delay))
            ).opts(
                responsive=True,
                title="Parameters",
                xlabel="Time",
                ylabel="Delay (ns)"
            ).redim(
                x="gain0",
                y="gain1"
            ),
            hv.Image(
                (
                    self.freqs,
                    self.times,
                    phase
                )
            ).opts(
                responsive=True,
                colorbar=True,
                title="Phase Surface",
                xlabel="Frequency",
                ylabel="Time",
                clim=(phase.min(), phase.max()),
                clabel="Phase (deg)",
                xticks=[
                    self.freqs.min(),
                    self.freqs.mean(),
                    self.freqs.max(),
                ]
            ).redim(
                x="surf0",
                y="surf1"
            )
        ]

        self.jones_images = plots

