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

    axis_map = {}  # Specific inspectors should provide valid mappings.

    vmin = param.Number(
        label="vmin",
        default=0
    )
    vmax = param.Number(
        label="vmax",
        default=1
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

    # Set the bounds during the init step.
    rasterize_when = param.Integer(
        label="Rasterize Limit",
        bounds=(1, None),
        step=10000,
        default=50000,
    )
    pixel_ratio = param.Number(
        label="Pixel ratio",
        bounds=(0.1, 2),
        step=0.05,
        default=0.25
    )
    flag_mode = param.Selector(
        label='FLAGGING MODE',
        objects=["SELECTED ANTENNA", "ALL ANTENNAS"],
        default="SELECTED ANTENNA"
    )
    flag_axis = param.Selector(
        label='FLAGGING AXIS',
        objects=["SELECTION", "SELECTION (X-AXIS)", "SELECTION (Y-AXIS)"],
        default="SELECTION"
    )
    flag = param.Action(
        lambda x: x.param.trigger('flag'),
        label='APPLY FLAGS'
    )
    reset = param.Action(
        lambda x: x.param.trigger('reset'),
        label='RESET FLAGS'
    )
    save = param.Action(
        lambda x: x.param.trigger('save'),
        label='SAVE FLAGS'
    )

    _gain_parameters = [
        "length_scale_time",
        "length_scale_freq",
    ]

    _display_parameters = [
        "vmin",
        "vmax",
        "pixel_ratio",
    ]

    _flag_parameters = [
        "flag",
        "flag_mode",
        "flag_axis",
        "reset",
        "save"
    ]

    def __init__(self, vis, **params):

        super().__init__(**params)

        self.vis = vis

        self.times = np.unique(self.vis.dataset.TIME.values)
        self.freqs = self.vis.dataset.chan.values

        self.n_ant = self.vis.dataset.ANTENNA2.values.max() + 1


        # self.dm = DataManager(data_path, fields=[data_field, flag_field])
        # self.data_field = data_field
        # self.flag_field = flag_field

        # dims = list(self.dm.dataset[self.data_field].dims)

        # for dim in dims:
        #     self.param.add_parameter(
        #         dim,
        #         param.Selector(
        #             label=dim.capitalize(),
        #             objects=self.dm.get_coord_values(dim).tolist()
        #         )
        #     )

        # for i, ax in enumerate(["x_axis", "y_axis"]):
        #     self.param.add_parameter(
        #         ax,
        #         param.Selector(
        #             label=ax.replace("_", " ").capitalize(),
        #             objects=list(self.axis_map.keys()),
        #             default=list(self.axis_map.keys())[i]
        #         )
        #     )

        # # Configure initial selection.
        # self.update_selection()

        # # # Ensure that amplitude is added to data on init. TODO: The plottable
        # # # axes are term dependent i.e. this shouldn't be here.
        # # self.dm.set_otf_columns(amplitude="gains")

        # self.param.watch(self.update_flags, ['flag'], queued=True)
        # self.param.watch(self.write_flags, ['save'], queued=True)
        # self.param.watch(self.reset_flags, ['reset'], queued=True)

        # # Automatically update data selection when these fields change.
        # self.param.watch(
        #     self.update_selection,
        #     dims,
        #     queued=True
        # )

        # # Automatically update on-the-fly columns when these fields change.
        # self.param.watch(
        #     self.update_otf_columns,
        #     ['x_axis', 'y_axis'],
        #     queued=True
        # )

        # # Empty Rectangles for overlay
        # self.rectangles = hv.Rectangles([]).opts(alpha=0.2, color="red")
        # # Attach a BoxEdit stream to the Rectangles
        # self.box_edit = streams.BoxEdit(source=self.rectangles)

        # self.zoom = streams.RangeXY()

        # # Get initial selection so we can reason about it.
        # selection = self.dm.get_selection()
        # # Start in the appropriate state based on size of selection.
        # self.rasterized = prod(selection.sizes.values()) > self.rasterize_when

    # def update_flags(self, event):

    #     if not self.box_edit.data:  # Nothing has been flagged.
    #         return

    #     corners = self.box_edit.data
    #     axes = ["antenna"] if self.flag_mode == "ALL ANTENNAS" else []

    #     for x_min, y_min, x_max, y_max in zip(*corners.values()):

    #         criteria = {}

    #         if self.flag_axis in ["SELECTION", "SELECTION (X-AXIS)"]:
    #             criteria[self.axis_map[self.x_axis]] = (x_min, x_max)
    #         if self.flag_axis in ["SELECTION", "SELECTION (Y-AXIS)"]:
    #             criteria[self.axis_map[self.y_axis]] = (y_min, y_max)

    #         self.dm.flag_selection(self.flag_field, criteria, axes=axes)

    # def reset_flags(self, event=None):
    #     self.dm.reset()

    # def write_flags(self, event=None):
    #     self.dm.write_flags(self.flag_field)

    # def update_selection(self, event=None):
    #     return NotImplementedError(f"update_selection not yet implemented.")

    # def update_otf_columns(self, event=None):
    #     self.dm.set_otf_columns(
    #         **{
    #             self.axis_map[ax]: self.data_field for ax in self.current_axes
    #             if self.axis_map[ax] in self.dm.otf_column_map
    #         }
    #     )

    # @property
    # def current_axes(self):
    #     return [self.x_axis, self.y_axis]

    @property
    def gains(self):

        freqs = self.freqs
        times = self.times
        ntime = times.size
        nchan = freqs.size
        nant = self.n_ant

        rng = np.random.default_rng(12345)  # Set seed.

        t = (times - times.min()) / (times.max() - times.min())
        nu = 2.5 * (freqs / freqs.mean() - 1.0)

        tt = np.abs(t[:, None] - t[None, :])
        lt = self.length_scale_time
        Kt = 0.1 * np.exp(-tt**2/(2*lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
        vv = np.abs(nu[:, None] - nu[None, :])
        lv = self.length_scale_freq
        Kv = 0.1 * np.exp(-vv**2/(2*lv**2))
        Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
        L = (Lt, Lv)

        jones = np.zeros((ntime, nchan, nant, 1, 4), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = rng.standard_normal(size=(ntime, nchan))
                amp = np.exp(-nu[None, :]**2 + kron_matvec(L, xi_amp))
                xi_phase = rng.standard_normal(size=(ntime, nchan))
                phase = kron_matvec(L, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

        return jones


    def update_image(self):

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
                    aspect="square"
                )
            )

        return hv.Layout(plots).cols(2)


        # x_axis = self.axis_map[self.x_axis]
        # y_axis = self.axis_map[self.y_axis]

        # plot_data = self.dm.get_plot_data(
        #     x_axis,
        #     y_axis,
        #     self.data_field,
        #     self.flag_field
        # )

        # n_points = len(plot_data)

        # x_limits = (plot_data[x_axis].min(), plot_data[x_axis].max())
        # y_limits = (plot_data[y_axis].min(), plot_data[y_axis].max())

        # scatter = hv.Scatter(
        #     plot_data,
        #     kdims=[x_axis],
        #     vdims=[y_axis]
        # )
        # self.zoom.source = scatter

        # # Get the points which fall in the current window.
        # visible_points = scatter.apply(filter_points, streams=[self.zoom])

        # # Get the points which we want to datashade - this may be an empty
        # # selection if we are below the threshold.
        # datashade_points = visible_points.apply(
        #     threshold_points,
        #     threshold=self.rasterize_when if self.rasterized else n_points
        # )
        # raw_points = visible_points.apply(
        #     threshold_points,
        #     threshold=self.rasterize_when if self.rasterized else n_points,
        #     inverse=True
        # )

        # # Set inital zoom to plot limits.
        # self.zoom.update(x_range=x_limits, y_range=y_limits)

        # shaded_plot = datashade(
        #     datashade_points,
        #     streams=[self.zoom],
        #     pixel_ratio=self.pixel_ratio
        # ).opts(
        #     responsive=True,
        #     xlabel=self.x_axis,
        #     ylabel=self.y_axis,
        #     xlim=x_limits,
        #     ylim=y_limits
        # )

        # pn.state.log(f'Plot update completed.')

        return shaded_plot * raw_points * self.rectangles

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

        selection_widgets = pn.Param(
            self.param,
            parameters=self._gain_parameters,
            name="JONES",
            widgets=widget_opts
        )

        widget_opts["flag_mode"].update(
            {
                "type": pn.widgets.RadioButtonGroup,
                "orientation": "vertical",
                "name": "FLAGGING MODE"
            }
        )

        widget_opts["flag_axis"].update(
            {
                "type": pn.widgets.RadioButtonGroup,
                "orientation": "vertical",
                "name": "FLAGGING AXIS"
            }
        )

        flagging_widgets = pn.Param(
            self.param,
            parameters=self._flag_parameters,
            name="FLAGGING",
            widgets=widget_opts
        )

        return pn.Column(
            pn.WidgetBox(display_widgets),
            pn.WidgetBox(selection_widgets),
            pn.WidgetBox(flagging_widgets)
        )
