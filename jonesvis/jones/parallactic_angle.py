import numpy as np

import holoviews as hv

import param
import panel as pn

from jonesvis.jones.base import Gain
from jonesvis.utils.angles import skyfield_parangles

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class ParallacticAngle(Gain):

    time_shift = param.Number(
        label="Time shift (min)",
        bounds=(0, 24 * 60),
        step=10,
        default=0
    )

    time_dilation = param.Number(
        label="Time dilation",
        bounds=(1, 48),
        step=1,
        default=1
    )

    _gain_parameters = Gain._gain_parameters + [
        "time_shift",
        "time_dilation"
    ]

    def __init__(self, vis, **params):
        super().__init__(vis, **params)

    @pn.depends(*_gain_parameters, watch=True)
    def update_gains(self):

        freqs = self.freqs
        times = self.times.copy()
        ntime = times.size
        nchan = freqs.size
        nant = self.n_ant

        antenna_positions = self.vis.antenna_positions
        phase_dir = self.vis.phase_dir

        # Dilate times i.e. increase length of integration.
        times[1:] = times[0] + np.cumsum(self.time_dilation * np.diff(times))

        # Shift times i.e. change start of observation.
        times += self.time_shift * 60

        parangles = skyfield_parangles(times, antenna_positions, phase_dir)

        jones = np.zeros((ntime, nchan, nant, 1, 4), dtype=np.complex128)

        for p in range(nant):
            if self.vis.feed_type == "circular":
                jones[:, :, p, 0, 0] = np.exp(1j * parangles[:, p])[:, None]
                jones[:, :, p, 0, 3] = np.exp(-1j * parangles[:, p])[:, None]
            else:
                jones[:, :, p, 0, 0] = np.cos(parangles[:, p])[:, None]
                jones[:, :, p, 0, 1] = np.sin(parangles[:, p])[:, None]
                jones[:, :, p, 0, 2] = -np.sin(parangles[:, p])[:, None]
                jones[:, :, p, 0, 3] = np.cos(parangles[:, p])[:, None]

        self.paranagles = parangles
        self.gains = jones
        self.scaled_times = times

    @pn.depends(*_gain_parameters, *Gain._data_parameters, watch=True)
    def update_stokes_images(self):
        super().update_stokes_images()

    @pn.depends(*_gain_parameters, *Gain._selection_parameters, watch=True)
    def update_jones_images(self):

        corr_idx = self.param.correlation.objects.index(self.correlation)

        selected_gains = self.gains[:, :, self.antenna, 0, corr_idx]
        phase = np.rad2deg(np.angle(selected_gains))
        parangles = np.rad2deg(self.paranagles[:, self.antenna])

        plots = [
            hv.Scatter(
                list(zip(self.scaled_times, parangles))
            ).opts(
                responsive=True,
                title="Parameters",
                xlabel="Time",
                ylabel="Parallactic Angle (deg)"
            ).redim(
                x="gain0",
                y="gain1"
            ),
            hv.Image(
                (
                    self.freqs,
                    self.scaled_times,
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

