import numpy as np
from math import prod

import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade

import param
import panel as pn

from jonesvis.utils.math import kron_matvec

pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")

class Gain(param.Parameterized):

    vmin = param.Number(
        label="vmin",
        default=0
    )
    vmax = param.Number(
        label="vmax",
        default=1
    )

    antenna = param.Integer(
        label="Antenna Number",
        default=0
    )

    correlation = param.Selector(
        label="Correlation",
        default="XX",
        objects=["XX", "XY", "YX", "YY"]
    )

    _gain_parameters = []

    _display_parameters = [
        "vmin",
        "vmax"
    ]

    _selection_parameters = [
        "antenna",
        "correlation"
    ]

    def __init__(self, vis, **params):

        self.vis = vis

        self.times = np.unique(self.vis.dataset.TIME.values)
        self.freqs = self.vis.dataset.chan.values

        self.n_ant = self.vis.dataset.ANTENNA2.values.max() + 1

        self.param.antenna.bounds = (0, self.n_ant)

        super().__init__(**params)

        self.update_gains()
        self.update_stokes_images()
        self.update_jones_images()

    @pn.depends(
        *_gain_parameters,
        watch=True
    )
    def update_gains(self):
        freqs = self.freqs
        times = self.times
        ntime = times.size
        nchan = freqs.size
        nant = self.n_ant

        self.gains = np.zeros((ntime, nchan, nant, 1, 4), dtype=np.complex128)
        self.gains[..., (0, 3)] = 1  # Identity gains.

    def update_image(self):

        return hv.Layout(
            [
                self.jones_images[0],
                *self.stokes_images[:2],
                self.jones_images[1],
                *self.stokes_images[2:]
            ]
        ).cols(3)

    @pn.depends("vmin", "vmax", watch=True)
    def set_vlim(self):
        self.stokes_images = [
            img.opts(clim=(self.vmin, self.vmax)) for img in self.stokes_images
        ]

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

    def update_jones_images(self):

        corr_idx = self.param.correlation.objects.index(self.correlation)

        selected_gains = self.gains[:, :, self.antenna, 0, corr_idx]
        amp = np.abs(selected_gains)
        phase =  np.angle(selected_gains)

        plots = [
            hv.Image(
                amp
            ).opts(
                responsive=True,
                colorbar=True
            ),
            hv.Image(
                phase
            ).opts(
                responsive=True,
                colorbar=True
            )
        ]

        self.jones_images = plots

    @property
    def widgets(self):

        widget_opts = {}

        for k in self.param.objects().keys():
            widget_opts[k] = {"sizing_mode": "stretch_width"}

        display_widgets = pn.Param(
            self.param,
            parameters=self._display_parameters,
            name="DISPLAY",
            widgets=widget_opts
        )

        gain_widgets = pn.Param(
            self.param,
            parameters=self._gain_parameters,
            name="JONES",
            widgets=widget_opts
        )

        selection_widgets = pn.Param(
            self.param,
            parameters=self._selection_parameters,
            name="SELECTION",
            widgets=widget_opts
        )

        return pn.Column(
            pn.WidgetBox(display_widgets),
            pn.WidgetBox(gain_widgets),
            pn.WidgetBox(selection_widgets)
        )
