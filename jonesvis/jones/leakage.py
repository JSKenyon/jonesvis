import numpy as np

import holoviews as hv

import param
import panel as pn

from jonesvis.utils.math import kron_matvec
from jonesvis.jones.base import Gain

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class Leakage(Gain):

    amp_std_dev = param.Number(
        label="Amplitude Standard deviation",
        bounds=(0, 0.25),
        step=0.01,
        default=0
    )
    amp_length_scale_freq = param.Number(
        label="Amplitude Length Scale (Frequency)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.2
    )

    phase_std_dev = param.Number(
        label="Phase Standard deviation",
        bounds=(0, np.round(2 * np.pi / 3, 2)),
        step=0.01,
        default=0
    )
    phase_length_scale_freq = param.Number(
        label="Phase Length Scale (Frequency)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.2
    )

    _gain_parameters = Gain._gain_parameters + [
        "amp_std_dev",
        "phase_std_dev",
        "amp_length_scale_freq",
        "phase_length_scale_freq",
    ]

    def __init__(self, vis, **params):
        super().__init__(vis, **params)

        # Select first off diagonal.
        self.correlation = self.param.correlation.objects[1]

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
        # nu = 2.5 * (freqs / freqs.mean() - 1.0)
        nu = (freqs - freqs.min()) / (freqs.max() - freqs.min())

        # We cannot set independent std as in the kronecker product they end
        # up getting combined.
        vv = np.abs(nu[:, None] - nu[None, :])

        # Amplitude
        if self.amp_std_dev:
            lv = self.amp_length_scale_freq
            Kv = self.amp_std_dev ** 2 * np.exp(-vv**2/(2*lv**2))
            Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
            L_amp = Lv
        else:
            L_amp = np.zeros((nchan, nchan))

        # Phase
        if self.phase_std_dev:
            lv = self.phase_length_scale_freq
            Kv = self.phase_std_dev ** 2 * np.exp(-vv**2/(2*lv**2))
            Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
            L_phase = Lv
        else:
            L_phase = (np.zeros((nchan, nchan)))

        jones = np.ones((ntime, nchan, nant, 1, 4), dtype=np.complex128)
        for p in range(nant):
            for c in [1, 2]:
                xi_amp = rng.standard_normal(size=(nchan,))
                amp = np.abs(L_amp @ xi_amp)
                xi_phase = rng.standard_normal(size=(nchan,))
                phase = L_phase @ xi_phase
                jones[:, :, p, 0, c] = (amp * np.exp(1.0j * phase))[None, :]

        self.gains = jones

    @pn.depends(*_gain_parameters, *Gain._data_parameters, watch=True)
    def update_stokes_images(self):
        super().update_stokes_images()

    @pn.depends(*_gain_parameters, *Gain._selection_parameters, watch=True)
    def update_jones_images(self):

        corr_idx = self.param.correlation.objects.index(self.correlation)

        selected_gains = self.gains[0, :, self.antenna, 0, corr_idx]
        amp = np.abs(selected_gains)
        phase = np.rad2deg(np.angle(selected_gains))

        plots = [
            hv.Scatter(
                list(zip(self.freqs, amp))
            ).opts(
                responsive=True,
                title="Amplitude",
                xlabel="Frequency",
                ylabel="Amplitude"
            ).redim(
                x="gain0",
                y="gainamp"
            ),
            hv.Scatter(
                list(zip(self.freqs, phase))
            ).opts(
                responsive=True,
                title="Phase",
                xlabel="Frequency",
                ylabel="Phase (deg)"
            ).redim(
                x="gain0",
                y="gainphase"
            ),
        ]

        self.jones_images = plots
