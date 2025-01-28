import numpy as np

from astropy.time import Time
from astropy import units as u

import skyfield.api as sky
import skyfield.toposlib as skytop
import skyfield.trigonometry as skytrig

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
    field_centre = sky.Star(
        ra=sky.Angle(degrees=field_centre[0]),
        dec=sky.Angle(degrees=field_centre[1])
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

    # -------------------------------------------------------------------------
    
    # # This is the recommended approach, but it seems to yield large errors.
    # # Unfortunately I am unsure how to debug it.
    
    # apparent_positions = [
    #     (earth + pos).at(times).observe(field_centre).apparent()
    #     for pos in ant_positions_geo
    # ]

    # # Get the zenith for each antenna at each time. This is the part that
    # # doesn't currently work and which I may be misunderstanding.
    # zeniths = [
    #     (earth + pos).at(times).from_altaz(alt_degrees=90, az_degrees=0)
    #     for pos in ant_positions_geo
    # ]

    # # The parallactic angle can then be computed as follows (ordering of
    # # arguments may be incorrect).
    # parallactic_angles = [
    #     skytrig.position_angle_of(a.radec(), z.radec())
    #     for a, z in zip(apparent_positions, zeniths)
    # ]

    # sf_angles = np.zeros((n_time, n_ant), dtype=np.float64)

    # for a, pa in enumerate(parallactic_angles):
    #     sf_angles[:, a] = pa.radians

    # -------------------------------------------------------------------------

    # This is an alternative, more hands on approach that gives much lower
    # errors. The origin of the discrepancy may be hidden in these details.

    sf_angles = np.zeros((n_time, n_ant), dtype=np.float64)

    # Apparent ra and dec of source relative to earth at each time.
    app_ra, app_dec, _ = \
        earth.at(times).observe(field_centre).apparent().radec(epoch='date')

    # Local apparent sidereal time, per antenna, per time.
    last = [sky.Angle(hours=pos.lst_hours_at(times)) for pos in ant_positions_geo]

    for ai in range(n_ant):
        app_ha = sky.Angle(radians=(last[ai].radians - app_ra.radians))

        sf_angles[:, ai] = parallactic_angle(
            app_ha.radians,
            app_dec.radians,
            ant_positions_geo[ai].latitude.radians
        )

    return sf_angles