# Thermodynamic calculations for parcel theory
# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units
from metpy.units import concatenate

from scipy.optimize import root_scalar

def moist_lapse(pressure, initial_temperature, reference_pressure=None):
    """
    Equivalent to metpy.calc.moist_lapse, circumventing bugs
    """

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    if pressure.size == 1:
        temperature = mpcalc.moist_lapse(
            pressure, initial_temperature,
            reference_pressure=reference_pressure)
        return np.atleast_1d(temperature)[0]
    else:
        pressure = pressure[1:]
        temperature = mpcalc.moist_lapse(
            pressure, initial_temperature,
            reference_pressure=reference_pressure)
        return concatenate([initial_temperature, temperature])


# objective function for the root-finding algorithm
def remaining_liquid_ratio(
        pressure, initial_pressure, initial_temperature,
        initial_liquid_ratio, min_zero=False):
    """
    Calculates the amount of liquid water left in the parcel.

    It is assumed that the parcel is initially saturated.
    The arguments must be given as plain numbers, without units.

    Args:
        pressure: The pressure level of interest in millibars.
        initial_pressure: The initial pressure of the parcel in
            millibars.
        initial_temperature: The initial temperature of the parcel in
            degrees celsius.
        initial_liquid_ratio: The ratio of the initial mass of liquid
            water to the total mass of the parcel.
        min_zero: Whether or not to return a non-negative result
            (defaults to False).

    Returns:
        The ratio of the remaining mass of liquid water to the total
            mass of the parcel.
    """

    if not hasattr(pressure, 'units'):
        pressure = pressure*units.mbar
    if not hasattr(initial_pressure, 'units'):
        initial_pressure = initial_pressure*units.mbar
    if not hasattr(initial_temperature, 'units'):
        initial_temperature = initial_temperature*units.celsius

    initial_specific_humidity = mpcalc.specific_humidity_from_dewpoint(
        initial_pressure, initial_temperature)
    final_temperature = moist_lapse(
        pressure, initial_temperature, reference_pressure=initial_pressure)
    final_specific_humidity = mpcalc.specific_humidity_from_dewpoint(
        pressure, final_temperature)
    remaining_ratio = (initial_specific_humidity + initial_liquid_ratio
                       - final_specific_humidity)
    if min_zero:
        remaining_ratio = np.max([remaining_ratio, 0])

    return remaining_ratio


def evaporation_level(
        initial_pressure, initial_temperature, initial_liquid_ratio):
    """
    Finds the pressure at which all liquid water evaporates.

    Args:
        initial_pressure: The initial pressure of the parcel.
        initial_temperature: The initial temperature of the parcel.
        initial_liquid_ratio: The ratio of the initial mass of liquid
            water to the total mass of the parcel.

    Returns:
        A tuple containing the pressure at which all liquid water
            evaporates, and the temperature of the parcel at this point.
    """

    solution = root_scalar(
        remaining_liquid_ratio,
        args=(initial_pressure, initial_temperature, initial_liquid_ratio),
        bracket=[initial_pressure.to(units.mbar).m, 2000])
    level = solution.root*units.mbar

    level_temperature = moist_lapse(
            level, initial_temperature, reference_pressure=initial_pressure)

    return level, level_temperature


def extra_liquid_descent_profile(
        pressure, initial_temperature, evaporation_level, level_temperature,
        reference_pressure=None):
    """
    Calculates the temperature of a descending air parcel.

    The parcel has some initial liquid water content and is assumed
    to be initially saturated.

    Args:
        pressure: Pressure levels of interest (must be monotonically
            increasing).
        initial_temperature: Initial temperature of the parcel,
            corresponding to pressure[0].
        evaporation_level: The pressure at which all liquid water in
            the parcel will evaporate.
        level_temperature: Parcel temperature at the evaporation level.
        reference_pressure: (optional) The pressure corresponding to
            initial_temperature. Defaults to pressure[0].

    Returns:
        The temperature of the parcel at the given pressure levels
            as an array.
    """

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]

    # find the position of the evaporation level in the pressure array
    level_index = np.searchsorted(pressure.m, evaporation_level.m, side='left')

    # moist descent to the evaporation level
    if level_index > 0:  # some levels are above EL
        moist_temperature = moist_lapse(
            pressure[:level_index], initial_temperature,
            reference_pressure=reference_pressure)
    else:  # no levels are above EL
        moist_temperature = np.array([])*initial_temperature.units

    # dry descent from the evaporation level
    if level_index != len(pressure):  # some levels are at or below EL
        dry_temperature = mpcalc.dry_lapse(
            pressure[level_index:], level_temperature,
            reference_pressure=evaporation_level)
    else:  # no levels are at or below EL
        dry_temperature = np.array([])*initial_temperature.units

    # join moist and dry descent
    temperature = concatenate([moist_temperature, dry_temperature])
    if temperature.size == 1:
        temperature = temperature.item()

    return temperature


def specific_humidity_from_descent_profile(
        pressure, temperature, evaporation_level, level_temperature):
    """
    Calculates the specific humidity of a descending parcel.

    Args:
        pressure: Pressure levels of interest.
        temperature: Temperature array corresponding to given pressure
            levels.
        evaporation_level: Evaporation level of the parcel.
        level_temperature: Parcel temperature at the evaporation level.

    Returns:
        The specific humidity array corresponding to the specified
        pressure levels.
    """

    pressure = np.atleast_1d(pressure)
    temperature = np.atleast_1d(temperature)
    above_level_mask = pressure <= evaporation_level
    if np.sum(above_level_mask) > 0:
        # dew point above the evaporation level is the temperature
        q_above_level = mpcalc.specific_humidity_from_dewpoint(
                pressure[above_level_mask], temperature[above_level_mask])
    else:
        q_above_level = np.array([])

    # specific humidity below the evaporation level is the specific
    # humidity at the evaporation level
    q_below_level = (
        np.ones(np.sum(~above_level_mask))
        * mpcalc.specific_humidity_from_dewpoint(
             evaporation_level, level_temperature)
    )

    specific_humidity = concatenate([q_above_level, q_below_level])
    if specific_humidity.size == 1:
        specific_humidity = specific_humidity.item()

    return specific_humidity
