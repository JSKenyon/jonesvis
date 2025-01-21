import panel as pn

from pathlib import Path

import typer
from typing_extensions import Annotated

from jonesvis.visilbities import Visibilities
from jonesvis.jones.base import Gain


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
    
    vis = Visibilities(ms_path)

    gains = {}

    gains["complex"] = Gain(vis)

    def get_widgets(value):
        return gains[value].widgets

    def get_plot(value):
        return gains[value].update_image

    gain_type = pn.widgets.Select(
        name="Jones Type",
        options=list(gains.keys()),
        value=list(gains.keys())[0],
        sizing_mode="stretch_width"
    )

    bound_get_widgets = pn.bind(get_widgets, gain_type)
    bound_get_plot = pn.bind(get_plot, gain_type)

    layout = pn.template.MaterialTemplate(
        # site="Panel",
        title="Jones-Visualiser",
        sidebar=[gain_type, bound_get_widgets],
        main=[bound_get_plot],
    ).servable()

    pn.serve(
        layout,
        port=port,
        show=False
    )