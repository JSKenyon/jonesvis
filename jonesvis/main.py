import panel as pn
import param

from pathlib import Path

import typer
from typing_extensions import Annotated

from jonesvis.visilbities import Visibilities
from jonesvis.jones import JONES_TYPES


def main():
    typer.run(app)


def app(
    ms_path: Annotated[
        Path,
        typer.Argument(
            help="Path to Measurement Set.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True
        )
    ],
    port: Annotated[
        int,
        typer.Option(
            help="Port on which to serve the visualiser."
        )
    ] = 5006
):
    
    jv = JonesVisualiser(ms_path)

    customised_widgets = pn.Param(
        jv.param,
        show_name=False,
        widgets={"gain_type": {"sizing_mode": "stretch_width"}}
    )

    layout = pn.template.MaterialTemplate(
        # site="Panel",
        title="Jones-Visualiser",
        sidebar=[customised_widgets, jv.gain_widgets],
        main=[jv.gain_plots],
    ).servable()

    pn.serve(
        layout,
        port=port,
        show=False
    )

class JonesVisualiser(param.Parameterized):

    gain_type = param.Selector(
        label="Gain Type",
        # default="parallactic_angle",
        objects=list(JONES_TYPES.keys())
    )

    def __init__(self, ms_path, **params):

        super().__init__(**params)

        self.ms_path = ms_path
        self.set_gain_type()

    @pn.depends("gain_type", watch=True)
    def set_gain_type(self):
        vis = Visibilities(self.ms_path)
        self.gain = JONES_TYPES[self.gain_type](vis)

    @pn.depends("gain_type")
    def gain_widgets(self):
        return self.gain.widgets

    @pn.depends("gain_type")
    def gain_plots(self):
        return self.gain.update_image

