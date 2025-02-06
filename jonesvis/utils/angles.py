import numpy as np

from astropy.time import Time
from astropy import units as u

import skyfield.api as sky
import skyfield.toposlib as skytop
import skyfield.trigonometry as skytrig
from skyfield.positionlib import position_of_radec

import numpy as np
import casacore.measures
import casacore.quanta as pq


def parallactic_angle(ha, dec, lat):
    numer = np.sin(ha)
    denom = np.cos(dec) * np.tan(lat) - np.sin(dec) * np.cos(ha)

    return np.arctan2(numer, denom)


def skyfield_parangles(
    times,
    ant_positions_ecef,
    field_centre
):

    n_ant = ant_positions_ecef.shape[0]
    n_time = np.unique(times).size

    # Create a "Star" object to be the field centre.
    star = sky.Star(
        ra=sky.Angle(radians=field_centre[0]),
        dec=sky.Angle(radians=field_centre[1])
    )

    # Our time vlaues are stored time values as MJD in seconds (which is
    # weird). This example avoids the problem by working with datetimes.
    ts = sky.load.timescale()
    apy_times = Time(times*u.s, format='mjd', scale='utc')
    times = ts.from_astropy(apy_times)

    planets = sky.load('de421.bsp')
    earth = planets['earth']

    # Specify input antenna positions as ITRS positions.
    ant_positions_itrf_sf = [
        skytop.ITRSPosition(sky.Distance(m=pos)) for pos in ant_positions_ecef
    ]

    # Convert ITRS positions into geographic positions.
    t = times[0]  # Time is irrelevant here, but is a required input.
    ant_positions_geo = [
        sky.wgs84.geographic_position_of(pos.at(t))
        for pos in ant_positions_itrf_sf
    ]

    # This is an alternative, more hands on approach that gives much lower
    # errors. The origin of the discrepancy may be hidden in these details.

    sf_angles = np.zeros((n_time, n_ant), dtype=np.float64)

    # Local apparent sidereal time, per antenna, per time.
    last = [sky.Angle(hours=pos.lst_hours_at(times)) for pos in ant_positions_geo]

    for ai in range(n_ant):

        # Apparent ra and dec of source relative to earth at each time.
        field_centre = \
            (earth + ant_positions_geo[ai]).at(times).observe(star)

        app_ra, app_dec, _ = field_centre.apparent().radec(epoch=times)

        app_ha = sky.Angle(radians=(last[ai].radians - app_ra.radians))

        sf_angles[:, ai] = parallactic_angle(
            app_ha.radians,
            app_dec.radians,
            ant_positions_geo[ai].latitude.radians
        )

    return sf_angles


def casa_parangles(time_col, ant_names, ant_positions_ecef,
                    field_centre, epoch):
    """Handles the construction of the parallactic angles using measures.

    Args:
        time_col: Array containing time values for each row.
        ant_names: Array of antenna names.
        ant_positions_ecef: Array of antenna positions in ECEF frame.
        field_centre: Array containing field centre coordinates.
        epoch: Reference epoch for measures calculations.

    Returns:
        angles: Array of parallactic angles per antenna per unique time.
    """

    cms = casacore.measures.measures()

    n_time = time_col.size
    n_ant = ant_names.size

    # Init angles from receptor angles. TODO: This only works for orthogonal
    # receptors. The more general case needs them to be kept separate.
    angles = np.zeros((n_time, n_ant, 2), dtype=np.float64)

    # Assume all antenna are pointed in the same direction.
    field_centre = \
        cms.direction(epoch, *(pq.quantity(fi, 'rad') for fi in field_centre))

    unique_times = np.unique(time_col)
    n_utime = unique_times.size
    angles = np.zeros((n_utime, n_ant, 2), dtype=np.float64)

    zenith_azel = cms.direction(
        "AZEL", *(pq.quantity(fi, 'deg') for fi in (0, 90))
    )

    ant_positions_itrf = [
        cms.position(
            'WGS84', *(pq.quantity(p, 'm') for p in pos)
        ) for pos in ant_positions_ecef
    ]

    for ti, t in enumerate(unique_times):
        cms.do_frame(cms.epoch("UTC", pq.quantity(t, 's')))
        for rpi, rp in enumerate(ant_positions_itrf):
            cms.do_frame(rp)
            angles[ti, rpi, :] += \
                cms.posangle(field_centre, zenith_azel).get_value("rad")

    return angles