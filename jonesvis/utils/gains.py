import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def nb_apply_gains(vis, gains, ant1, ant2, row_ind):

    n_row, n_chan, n_corr = vis.shape

    for row in range(n_row):
        r = row_ind[row]
        a1 = ant1[row]
        a2 = ant2[row]

        for f in range(n_chan):

            gp00, gp01, gp10, gp11 = gains[r, f, a1, 0]  # No direction.
            # Note the subtle transpose.
            gqc00, gqc10, gqc01, gqc11 = gains[r, f, a2, 0].conjugate()
            v00, v01, v10, v11 = vis[r, f]

            tmp00 = (gp00*v00 + gp01*v10)
            tmp01 = (gp00*v01 + gp01*v11)
            tmp10 = (gp10*v00 + gp11*v10)
            tmp11 = (gp10*v01 + gp11*v11)

            vis[row, f, 0] = (tmp00*gqc00 + tmp01*gqc10)
            vis[row, f, 1] = (tmp00*gqc01 + tmp01*gqc11)
            vis[row, f, 2] = (tmp10*gqc00 + tmp11*gqc10)
            vis[row, f, 3] = (tmp10*gqc01 + tmp11*gqc11)
