import numpy as np
from numba import njit
from scipy.constants import c as lightspeed


def vis_to_stokes_vis(visibilities, feed_type="linear"):

    stokes_vis = {}

    # NOTE: Presumes unity weights i.e. definitely not correct in general.
    if feed_type == "linear":

        stokes_vis["I"] = 0.5 * (visibilities[..., 0] + visibilities[..., 3])
        stokes_vis["Q"] = 0.5 * (visibilities[..., 0] - visibilities[..., 3])
        stokes_vis["U"] = 0.5 * (visibilities[..., 1] + visibilities[..., 2])
        stokes_vis["V"] = 0.5 * (-1j * visibilities[..., 1] + 1j * visibilities[..., 2])

    elif feed_type == "circular":

        raise NotImplementedError("Circular feeds are not yet supported.")

    return stokes_vis


def wgt_to_stokes_wgt(weights, feed_type="linear"):

    stokes_weights = {}

    # NOTE: Presumes unity weights i.e. definitely not correct in general.
    if feed_type == "linear":

        stokes_weights["I"] = 2 * weights[..., 0]
        stokes_weights["Q"] = 2 * weights[..., 1]
        stokes_weights["U"] = 2 * weights[..., 2]
        stokes_weights["V"] = 2 * weights[..., 3]

    elif feed_type == "circular":

        raise NotImplementedError("Circular feeds are not yet supported.")

    return stokes_weights


@njit(nogil=True, cache=True)
def grid_weights(
    uvw,
    freq,
    mask,
    wgt,
    nx,
    ny,
    cell_size_x,
    cell_size_y,
    dtype,
    k=6,
    ngrid=1,
    usign=1.0,
    vsign=-1.0
):
    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    # shifts fftfreq such that they start at zero
    # convenient to look up the pixel value
    umax = np.abs(-1 / cell_size_x / 2 - u_cell / 2)

    # vfreq
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(-1 / cell_size_y / 2 - v_cell / 2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    counts = np.zeros((nx, ny), dtype=dtype)

    # accumulate counts
    nrow, nchan = wgt.shape # No correlation axis.

    normfreq = freq / lightspeed
    ko2 = k // 2
    beta = 2.3

    for r in range(nrow):
        uvw_row = uvw[r]
        wgt_row = wgt[r]
        mask_row = mask[r]
        for c in range(nchan):
            if not mask_row[c]:
                continue
            # current uv coords
            chan_normfreq = normfreq[c]
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            # pixel coordinates
            ug = (u_tmp + umax) / u_cell
            vg = (v_tmp + vmax) / v_cell
            wrc = wgt_row[c]

            # indices
            u_idx = int(np.round(ug))
            v_idx = int(np.round(vg))
            for i in range(-ko2, ko2):
                x_idx = i + u_idx
                x = x_idx - ug + 0.5
                # np.exp(beta*k*(np.sqrt((1-x)*(1+x)) - 1))
                val = wrc * np.exp(beta * k * (np.sqrt(1 - (x / ko2)**2) - 1))
                for j in range(-ko2, ko2):
                    y_idx = j + v_idx
                    y = y_idx - vg + 0.5
                    counts[x_idx, y_idx] += val * np.exp(beta * k * (np.sqrt(1 - (y / ko2)**2) - 1))

    return counts


@njit(nogil=True, cache=True, parallel=True)
def imaging_weights(
    gridded_weights,
    uvw,
    freq,
    stokes_weight,
    nx,
    ny,
    cell_size_x,
    cell_size_y,
    robust,
    usign=1.0,
    vsign=-1.0
):
    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    umax = np.abs(-1 / cell_size_x / 2 - u_cell / 2)

    # vfreq
    v_cell = 1/(ny * cell_size_y)
    vmax = np.abs(-1 / cell_size_y/ 2 - v_cell / 2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    nrow, nchan = stokes_weight.shape  # No correlaion axis.

    if not gridded_weights.any():
        return stokes_weight

    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5*10**(-robust)
        avgW = (gridded_weights ** 2).sum() / gridded_weights.sum()
        ssq = numsqrt * numsqrt / avgW
        gridded_weights = 1 + gridded_weights * ssq

    normfreq = freq / lightspeed
    for r in range(nrow):
        uvw_row = uvw[r]
        weight_row = stokes_weight[r]
        for c in range(nchan):
            # get current uv
            chan_normfreq = normfreq[c]
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            # get u index
            u_idx = int(np.floor((u_tmp + umax)/u_cell))
            # get v index
            v_idx = int(np.floor((v_tmp + vmax)/v_cell))
            if gridded_weights[u_idx, v_idx]:
                weight_row[c] = weight_row[c]/gridded_weights[u_idx, v_idx]
    return stokes_weight