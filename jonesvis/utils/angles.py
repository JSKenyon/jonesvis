import numpy as np

from astropy.time import Time
from astropy import units as u

import skyfield.api as sky
import skyfield.toposlib as skytop
import skyfield.trigonometry as skytrig
from skyfield.positionlib import position_of_radec

from astropy.coordinates import ICRS, SkyCoord, EarthLocation, TETE
from astropy import units as u
from astropy.time import Time
from astroplan import Observer

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
    # from skyfield.data import iers
    # url = sky.load.build_url('finals2000A.all')
    # with sky.load.open(url) as f:
    #     finals_data = iers.parse_x_y_dut1_from_finals_all(f)
    # iers.install_polar_motion_table(ts, finals_data)    
    
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
        sky.iers2010.geographic_position_of(pos.at(t))
        for pos in ant_positions_itrf_sf
    ]

    sf_angles = np.zeros((n_time, n_ant), dtype=np.float64)

    for ai in range(n_ant):

        # field_centre = \
        #     (earth + ant_positions_geo[ai]).at(times).observe(star).apparent()

        # # Apparent ra and dec of source relative to earth at each time.
        # app_ra, app_dec, _ = field_centre.radec(epoch=times)

        # # Local apparent sidereal time, per antenna, per time.
        # last = sky.Angle(hours=ant_positions_geo[ai].lst_hours_at(times))

        # app_ha = sky.Angle(radians=(last.radians - app_ra.radians))

        # sf_angles[:, ai] = parallactic_angle(
        #     app_ha.radians,
        #     app_dec.radians,
        #     ant_positions_geo[ai].latitude.radians
        # )

        pos_at_time = ant_positions_geo[ai].at(times)

        zenith = pos_at_time.from_altaz(alt_degrees=90.0, az_degrees=0.0)

        # Apparent ra and dec of source relative to earth at each time.
        field_centre = \
            (earth + ant_positions_geo[ai]).at(times).observe(star).apparent()

        pa = skytrig.position_angle_of(
            field_centre.radec(epoch=times),
            zenith.radec(epoch=times),
        )

        # Ensure we are in (-np.pi, np.pi).
        sf_angles[:, ai] = np.atan2(np.sin(pa.radians), np.cos(pa.radians))

    return sf_angles


def casa_parangles(
    times,
    ant_positions_ecef,
    field_centre,
    epoch="J2000"
):
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

    n_time = times.size
    n_ant = ant_positions_ecef.shape[0]

    # Assume all antenna are pointed in the same direction.
    field_centre = \
        cms.direction(epoch, *(pq.quantity(fi, 'rad') for fi in field_centre))

    angles = np.zeros((n_time, n_ant), dtype=np.float64)

    zenith_azel = cms.direction(
        "AZELGEO", *(pq.quantity(fi, 'deg') for fi in (0, 90))
    )

    ant_positions_itrf = [
        cms.position(
            'itrf', *(pq.quantity(p, 'm') for p in pos)
        ) for pos in ant_positions_ecef
    ]

    for ti, t in enumerate(times):
        cms.do_frame(cms.epoch("UTC", pq.quantity(t, 's')))
        for rpi, rp in enumerate(ant_positions_itrf):
            cms.do_frame(rp)
            app_field_centre = cms.measure(field_centre, "APP")
            angles[ti, rpi] = \
                cms.posangle(app_field_centre, zenith_azel).get_value("rad")

    return angles


def astropy_parangles(
    times,
    ant_positions_ecef,
    field_centre
):

    n_ant = ant_positions_ecef.shape[0]

    target = SkyCoord(
        ra=field_centre[0]*u.rad,
        dec=field_centre[1]*u.rad,
        frame="icrs"
    )

    # MS stores time values as MJD in seconds (which is weird). This lets us
    # get the correct times.
    times = Time(times*u.s, format='mjd', scale='utc')  # TODO: Validate!

    _ant_positions_itrf = [
        EarthLocation.from_geocentric(
            pos[0]*u.m,
            pos[1]*u.m,
            pos[2]*u.m,
        )
        for pos in ant_positions_ecef
    ]

    target = target.transform_to(TETE(obstime=times))

    observers = [Observer(location=loc) for loc in _ant_positions_itrf]

    # Kind can be mean or apparent - mean agrees more closely with skyfield.
    # Apparent agrees more closely with casacore.
    parangs = [
        obs.parallactic_angle(times, target, kind="apparent")
        for obs in observers
    ]

    ap_angles = np.zeros((len(times), n_ant), dtype=np.float64)

    for ai in range(n_ant):
        ap_angles[:, ai] = parangs[ai].value

    return ap_angles
