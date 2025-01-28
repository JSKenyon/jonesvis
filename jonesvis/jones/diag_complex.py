import numpy as np

import holoviews as hv

import param
import panel as pn

from jonesvis.utils.math import kron_matvec
from jonesvis.jones.base import Gain

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class DiagComplex(Gain):

    amp_std_dev = param.Number(
        label="Amplitude Standard deviation",
        bounds=(0, 0.25),
        step=0.01,
        default=0
    )
    amp_length_scale_time = param.Number(
        label="Amplitude Length Scale (Time)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.2
    )
    amp_length_scale_freq = param.Number(
        label="Amplitude Length Scale (Frequency)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.1
    )

    phase_std_dev = param.Number(
        label="Phase Standard deviation",
        bounds=(0, np.round(2 * np.pi / 3, 2)),
        step=0.01,
        default=0
    )
    phase_length_scale_time = param.Number(
        label="Phase Length Scale (Time)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.2
    )
    phase_length_scale_freq = param.Number(
        label="Phase Length Scale (Frequency)",
        bounds=(0.01, 1),
        step=0.01,
        default=0.1
    )

    _gain_parameters = Gain._gain_parameters + [
        "amp_std_dev",
        "phase_std_dev",
        "amp_length_scale_time",
        "amp_length_scale_freq",
        "phase_length_scale_time",
        "phase_length_scale_freq",
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
        # nu = 2.5 * (freqs / freqs.mean() - 1.0)
        nu = (freqs - freqs.min()) / (freqs.max() - freqs.min())

        # We cannot set independent std as in the kronecker product they end
        # up getting combined.

        tt = np.abs(t[:, None] - t[None, :])
        vv = np.abs(nu[:, None] - nu[None, :])

        # Amplitude
        if self.amp_std_dev:
            lt = self.amp_length_scale_time
            Kt = self.amp_std_dev * np.exp(-tt**2/(2*lt**2))
            Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))

            lv = self.amp_length_scale_freq
            Kv = self.amp_std_dev * np.exp(-vv**2/(2*lv**2))
            Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
            L_amp = (Lt, Lv)  # np.kron(Lt, Lv) vec(chi)
        else:
            L_amp = (np.zeros((ntime, ntime)), np.zeros((nchan, nchan)))

        # Phase
        if self.phase_std_dev:
            lt = self.phase_length_scale_time
            Kt = self.phase_std_dev * np.exp(-tt**2/(2*lt**2))
            Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))

            lv = self.phase_length_scale_freq
            Kv = self.phase_std_dev * np.exp(-vv**2/(2*lv**2))
            Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
            L_phase = (Lt, Lv)  # np.kron(Lt, Lv) vec(chi)
        else:
            L_phase = (np.zeros((ntime, ntime)), np.zeros((nchan, nchan)))

        jones = np.zeros((ntime, nchan, nant, 1, 4), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = rng.standard_normal(size=(ntime, nchan))
                amp = 1 + kron_matvec(L_amp, xi_amp)
                xi_phase = rng.standard_normal(size=(ntime, nchan))
                phase = kron_matvec(L_phase, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

        self.gains = jones

    @pn.depends(*_gain_parameters, *Gain._data_parameters, watch=True)
    def update_stokes_images(self):
        super().update_stokes_images()

    @pn.depends(*_gain_parameters, *Gain._selection_parameters, watch=True)
    def update_jones_images(self):

        corr_idx = self.param.correlation.objects.index(self.correlation)

        selected_gains = self.gains[:, :, self.antenna, 0, corr_idx]
        amp = np.abs(selected_gains)
        phase =  np.rad2deg(np.angle(selected_gains))

        plots = [
            hv.Image(
                (
                    self.freqs,
                    self.times,
                    amp
                )
            ).opts(
                responsive=True,
                colorbar=True,
                title="Amplitude Surface",
                xlabel="Frequency",
                ylabel="Time",
                clim=(amp.min(), amp.max()),
                clabel="Amplitude",
                xticks=[
                    self.freqs.min(),
                    self.freqs.mean(),
                    self.freqs.max(),
                ]
            ).redim(
                x="surf0",
                y="surf1"
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
