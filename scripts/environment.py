# Class for parcel calculations on real soundings
# Thomas Schanzer, UNSW Sydney
# September 2021

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units
from metpy.units import concatenate

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

import sys


class MotionResult():
    """Class for results of parcel motion calculations."""

    pass



def moist_lapse(pressure, initial_temperature, reference_pressure=None):
    """
    Equivalent to metpy.calc.moist_lapse, circumventing bugs
    """

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    if pressure.size == 1:
        if np.abs(pressure - reference_pressure).to(units.mbar).m >= 0.01:
            return mpcalc.moist_lapse(
                pressure, initial_temperature,
                reference_pressure=reference_pressure)
        else:
            return initial_temperature
    else:
        pressure = pressure[1:]
        temperature = mpcalc.moist_lapse(
            pressure, initial_temperature,
            reference_pressure=reference_pressure)
        return concatenate([initial_temperature, temperature])


class Environment():
    """
    Class for parcel calculations on a real atmospheric sounding.
    """

    def __init__(self, pressure, temperature, dewpoint, info='', name=''):
        """
        Instantiates an Environment.

        Allows all variables to be calculated as functions of height
        rather than pressure by assuming hydrostatic balance.

        Args:
            pressure: Pressure array in the sounding.
            temperature: Temperature array in the sounding.
            dewpoint: Dewpoint array in the sounding.
        """

        self.pressure_raw = pressure.to(units.mbar)
        self.temperature_raw = temperature.to(units.celsius)
        self.dewpoint_raw = dewpoint.to(units.celsius)
        self.info = info
        self.name = name

        self.temperature_interp = interp1d(
            self.pressure_raw.m, self.temperature_raw.m,
            fill_value='extrapolate')
        self.dewpoint_interp = interp1d(
            self.pressure_raw.m, self.dewpoint_raw.m,
            fill_value='extrapolate')

        def dzdp(pressure, height):
            """
            Calculates the rate of height change w.r.t. pressure, dz/dp.

            Args:
                pressure: The pressure at the point of interest, in Pa.
                height: The height of the point of interest, in m.

            Returns:
                The derivative dz/dp in m/Pa.
            """

            pressure = pressure*units.pascal
            temperature = self.temperature_from_pressure(pressure)
            dewpoint = self.dewpoint_from_pressure(pressure)

            specific_humidity = mpcalc.specific_humidity_from_dewpoint(
                pressure, dewpoint)
            mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
                specific_humidity)
            density = mpcalc.density(pressure, temperature, mixing_ratio)
            dzdp = - 1 / (density.to(units.kg/units.meter**3).m * const.g)

            return dzdp

        def dpdz(height, pressure):
            return 1 / dzdp(pressure, height)

        # integrate the hydrostatic equation
        p_min = np.min(self.pressure_raw).m * 1e2
        p_max = np.max(self.pressure_raw).m * 1e2
        self.dzdp_sol = solve_ivp(
            dzdp, (p_max, p_min), [0], dense_output=True).sol

        z_max = self.dzdp_sol(p_min).item()
        self.dpdz_sol = solve_ivp(
            dpdz, (0, z_max), [p_max], dense_output=True).sol

    def temperature_from_pressure(self, pressure):
        """
        Finds the environmental temperature at a given pressure.
        """

        temperature = self.temperature_interp(pressure.to(units.mbar).m)
        if temperature.size == 1:
            temperature = temperature.item()
        return temperature*units.celsius

    def dewpoint_from_pressure(self, pressure):
        """
        Finds the environmental dew point at a given pressure.
        """

        dewpoint = self.dewpoint_interp(pressure.to(units.mbar).m)
        if dewpoint.size == 1:
            dewpoint = dewpoint.item()
        return dewpoint*units.celsius

    def pressure(self, height):
        """
        Finds the environmental pressure at a given height.
        """

        height = np.atleast_1d(height).to(units.meter).m
        pressure = self.dpdz_sol(height)[0,:]
        if pressure.size == 1:
            pressure = pressure.item()
        return (pressure*units.pascal).to(units.mbar)

    def height(self, pressure):
        """
        Finds the height at a given environmental pressure.
        """

        pressure = np.atleast_1d(pressure).to(units.pascal).m
        height = self.dzdp_sol(pressure)[0,:]
        if height.size == 1:
            height = height.item()
        return height*units.meter

    def temperature(self, height):
        """
        Finds the environmental temperature at a given height.
        """

        pressure = self.pressure(height)
        temperature = self.temperature_from_pressure(pressure)
        return temperature

    def dewpoint(self, height):
        """
        Finds the environmental dew point at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint_from_pressure(pressure)
        return dewpoint

    def wetbulb_temperature(self, height):
        """
        Finds the environmental wet-bulb temperature at a given height.
        """

        pressure = self.pressure(height)
        temperature = self.temperature_from_pressure(pressure)
        dewpoint = self.dewpoint_from_pressure(pressure)
        wetbulb_temperature = mpcalc.wet_bulb_temperature(
            pressure, temperature, dewpoint)
        return wetbulb_temperature

    def specific_humidity(self, height):
        """
        Finds the environmental specific humidity at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint_from_pressure(pressure)
        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        return specific_humidity

    def temperature_change(self, dq):
        """
        Calculates the temperature change due to evaporation of water.
        """

        dT = (- const.water_heat_vaporization
              * dq / const.dry_air_spec_heat_press)
        return dT.to(units.delta_degC)

    def dq_root(self, dq, height):
        """
        Calculates the saturated vs. actual specific humidity difference.

        Args:
            dq: Specific humidity change resulting from evaporation.
            height: Initial height of the parcel.

        Returns:
            The difference between saturated and actual specific
                humidity at the temperature that results from the
                evaporation.
        """

        pressure = self.pressure(height)
        initial_temperature = self.temperature(height)
        initial_specific_humidity = self.specific_humidity(height)

        final_temperature = initial_temperature + self.temperature_change(dq)
        saturation_mixing_ratio = mpcalc.saturation_mixing_ratio(
            pressure, final_temperature)
        saturation_specific_humidity = \
            mpcalc.specific_humidity_from_mixing_ratio(saturation_mixing_ratio)
        return saturation_specific_humidity - initial_specific_humidity - dq

    def maximum_specific_humidity_change(self, height):
        """
        Calculates the maximum specific humidity increase for the parcel.

        Finds the root of dq_root at the specified height.
        """

        height = np.atleast_1d(height)
        sol = [
            root_scalar(self.dq_root, args=(z,), bracket=[0, 20e-3]).root
            for z in height]
        return np.squeeze(concatenate(sol)) * 1

    def density(self, height):
        """
        Finds the environmental density at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint(height)
        temperature = self.temperature(height)

        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def dry_parcel_density(
            self, height, initial_height, specific_humidity_change):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the specific humidity is initially increased
        by a known amount via evaporation, with the parcel remaining
        subsaturated at all times.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            specific_humidity_change: The change in specific humidity
                that resulted from the precipitation.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self.pressure(initial_height)
        pressure = self.pressure(height)
        initial_temperature = (self.temperature(initial_height)
                               + self.temperature_change(
                                   specific_humidity_change))
        specific_humidity = (self.specific_humidity(initial_height)
                             + specific_humidity_change)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        temperature = mpcalc.dry_lapse(
            pressure, initial_temperature, reference_pressure=initial_pressure)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def saturated_parcel_density(
            self, height, initial_height, initial_temperature):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the parcel is initially saturated via evaporation
        and remains saturated at all times.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            initial_temperature: The initial temperature of the parcel.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self.pressure(initial_height)
        pressure = self.pressure(height)

        temperature = moist_lapse(
            pressure, initial_temperature, reference_pressure=initial_pressure)

        mixing_ratio = mpcalc.saturation_mixing_ratio(pressure, temperature)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def limited_water_parcel_density(
            self, height, initial_height, initial_temperature,
            evaporation_level, level_temperature, initial_liquid_ratio):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the parcel is initially saturated via evaporation,
        but retains only a limited amount of liquid water so that it
        may not remain saturated during descent.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            initial_temperature: The initial temperature of the parcel.
            evaporation_level: The pressure at which the remaining
                liquid water will completely evaporate.
            level_temperature: The parcel's temperature at the
                evaporation level.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self.pressure(initial_height)
        pressure = self.pressure(height)

        temperature = extra_liquid_descent_profile(
            pressure, initial_temperature, evaporation_level,
            level_temperature, reference_pressure=initial_pressure)

        specific_humidity = specific_humidity_from_descent_profile(
            pressure, temperature, evaporation_level, level_temperature)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)

        gas_density = mpcalc.density(pressure, temperature, mixing_ratio)
        liquid_ratio = remaining_liquid_ratio(
                pressure, initial_pressure, initial_temperature,
                initial_liquid_ratio, min_zero=True)

        density = gas_density / (1 - liquid_ratio)  # liquid correction
        return density

    def parcel_buoyancy(self, height, *args, regime='dry'):
        """
        Calculates the buoyancy of a parcel after precipitation.

        Args:
            height: The height of the parcel.
            regime: The buoyancy regime for the calculation: 'dry',
                'saturated' or 'limited'. Default is 'dry'.
            The remaining positional arguments are the same as those for
                the density function corresponding to the regime.

        Returns:
            The buoyant force per unit mass on the parcel.
        """

        if regime not in ['dry', 'saturated', 'limited']:
            raise ValueError(
                "regime must be 'dry', 'saturated' or 'limited'.")

        environment_density = self.density(height)

        density_function = {
            'dry': self.dry_parcel_density,
            'saturated': self.saturated_parcel_density,
            'limited': self.limited_water_parcel_density}[regime]
        density = density_function(height, *args)

        buoyancy = (environment_density - density) / density * const.g
        return buoyancy

    def dry_neutral_buoyancy_level(
            self, initial_height, specific_humidity_change):
        """
        Calculates the neutral buoyancy heights of parcels.

        Assumes the dry buoyancy regime.

        Args:
            initial_height: An array of initial heights.
            specific_humidity_change: An array of initial specific
                humidity changes due to evaporation.

        Returns:
            An array of neutral buoyancy heights, with each row
                corresponding to one initial height and each column
                corresponding to one initial specific humidity change.
        """

        initial_height = np.atleast_1d(initial_height)
        specific_humidity_change = np.atleast_1d(specific_humidity_change)
        sol = np.zeros((len(initial_height), len(specific_humidity_change)))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='dry').m

        for i, z0 in enumerate(initial_height):
            for ii, dq in enumerate(specific_humidity_change):
                sys.stdout.write(
                        '\rCalculating buoyancy level '
                        '{} of {}.       '.format(
                            i*len(specific_humidity_change) + ii + 1,
                            len(specific_humidity_change)*len(initial_height)))
                if dq <= self.maximum_specific_humidity_change(z0):
                    try:
                        sol[i,ii] = root_scalar(
                            root_function,
                            args=(z0, dq),
                            x0=z0.to(units.meter).m, x1=0,
                            bracket=[0, z0.to(units.meter).m]
                        ).root
                    except ValueError:
                        sol[i,ii] = 0
                else:
                    sol[i,ii] = np.nan

        sol = np.squeeze(sol)
        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def saturated_neutral_buoyancy_level(self, initial_height):
        """
        Calculates the neutral buoyancy height of a parcel.

        Assumes the saturated buoyancy regime.

        Args:
            initial_height: The initial height of the parcel.

        Returns:
            The height at which the buoyancy of the parcel is zero.
        """

        initial_height = np.atleast_1d(initial_height)
        initial_temperature = self.wetbulb_temperature(initial_height)
        sol = np.zeros(len(initial_height))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='saturated').m

        for i, z0 in enumerate(initial_height):
            sys.stdout.write(
                '\rCalculating buoyancy level '
                '{} of {}.       '.format(i+1, len(initial_height)))
            try:
                sol[i] = root_scalar(
                    root_function,
                    args=(z0, initial_temperature[i]),
                    x0=z0.to(units.meter).m, x1=0,
                    bracket=[0, z0.to(units.meter).m]
                ).root
            except ValueError:
                sol[i] = 0

        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def limited_neutral_buoyancy_level(self, initial_height, liquid_ratio):
        """
        Calculates the neutral buoyancy height of a parcel.

        Assumes the limited buoyancy regime.

        Args:
            initial_height: An array of initial heights.
            liquid_ratio: An array of initial liquid water amounts,
                as fractions of total parcel mass.

        Returns:
            An array of neutral buoyancy heights, with each row
                corresponding to one initial height and each column
                corresponding to one initial liquid water ratio.
        """

        initial_height = np.atleast_1d(initial_height)
        initial_pressure = self.pressure(initial_height)
        initial_temperature = self.wetbulb_temperature(initial_height)
        liquid_ratio = np.atleast_1d(liquid_ratio)
        sol = np.zeros((len(initial_height), len(liquid_ratio)))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='limited').m

        for i, z0 in enumerate(initial_height):
            for ii, lr in enumerate(liquid_ratio):
                sys.stdout.write(
                        '\rCalculating buoyancy level '
                        '{} of {}.       '.format(
                            i*len(liquid_ratio) + ii + 1,
                            len(liquid_ratio)*len(initial_height)))
                level, level_temperature = evaporation_level(
                    initial_pressure[i], initial_temperature[i], lr)
                try:
                    sol[i,ii] = root_scalar(
                        root_function,
                        args=(z0, initial_temperature[i], level,
                              level_temperature, lr),
                        x0=z0.to(units.meter).m, x1=0,
                        bracket=[0, z0.to(units.meter).m]
                    ).root
                except ValueError:
                    sol[i,ii] = 0

        sol = np.squeeze(sol)
        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def modified_motion(
        self, time, initial_height, initial_velocity, *args, regime='dry'):
        """
        Calculates parcel motion.

        Args:
            time: Array of time points for the solution.
            initial_height: Array of initial heights.
            *args:
                Initial specific humidity change array if regime is
                    'dry',
                Nothing if regime is 'saturated',
                Initial liquid water ratio if regime is 'limited'.
            regime: The buoyancy regime to use: 'dry', 'saturated' or
                'limited'.

        Returns:
            A MotionResult object.
        """

        # independent variables
        initial_height = np.atleast_1d(initial_height).to(units.meter).m
        length = len(initial_height)
        initial_velocity = np.atleast_1d(
            initial_velocity).to(units.meter/units.second).m
        if length != len(initial_velocity):
            raise ValueError(
                    'Initial height and velocity arrays must have the same '
                    'length.')
        initial_state = [
            list(x) for x in zip(initial_height, initial_velocity)]
        time = time.to(units.second).m

        # functions to calculate the extra arguments to pass to motion_ode
        if regime == 'dry':
            dq = np.atleast_1d(args[0])
            if len(dq) != length:
                raise ValueError(
                    'Initial height and specific humidity change arrays must '
                    'have the same length.')
            ode_args = lambda i: (initial_height[i]*units.meter, dq[i])
        elif regime == 'saturated':
            ode_args = lambda i: (
                initial_height[i]*units.meter,
                self.wetbulb_temperature(initial_height[i]*units.meter),
            )
        elif regime == 'limited':
            liquid_ratio = np.atleast_1d(args[0])
            if len(liquid_ratio) != length:
                raise ValueError(
                    'Initial height and liquid ratio arrays must '
                    'have the same length.')
            def ode_args(i):
                height = initial_height[i]*units.meter
                initial_pressure = self.pressure(height)
                initial_temperature = self.wetbulb_temperature(height)
                level, level_temperature = evaporation_level(
                    initial_pressure, initial_temperature, liquid_ratio[i])
                return (
                    height, initial_temperature,
                    level, level_temperature, liquid_ratio[i],
                )
        else:
            raise ValueError(
                "regime must be 'dry', 'saturated' or 'limited'.")

        def motion_ode(time, state, *args):
            """
            Defines the equation of motion for a parcel.
            """

            # for some reason solve_ivp likes to test large negative
            # heights which can cause overflows
            height = np.max([state[0], 0])*units.meter
            buoyancy = self.parcel_buoyancy(
                height, *args, regime=regime)
            return [state[1], buoyancy.magnitude]

        # event function for solve_ivp, zero when parcel reaches min height
        min_height = lambda time, state, *args: state[1]
        min_height.direction = 1  # find zero that goes from - to +
        min_height.terminal = True  # stop integration at minimum height

        # event function for solve_ivp, zero when parcel hits ground
        hit_ground = lambda time, state, *args: state[0]
        hit_ground.terminal = True  # stop integration at ground

        # prepare empty arrays for data
        height = np.zeros((length, len(time)))
        height[:] = np.nan
        velocity = np.zeros((length, len(time)))
        velocity[:] = np.nan

        neutral_buoyancy_time = np.zeros(length)
        hit_ground_time = np.zeros(length)
        min_height_time = np.zeros(length)

        neutral_buoyancy_height = np.zeros(length)
        neutral_buoyancy_velocity = np.zeros(length)
        hit_ground_velocity = np.zeros(length)
        min_height_height = np.zeros(length)

        for i in range(length):
            sys.stdout.write(
                '\rCalculating profile {0} of {1}.'
                '   '.format(i+1, length))

            # event function for solve_ivp, zero when parcel is neutrally
            # buoyant
            neutral_buoyancy = lambda time, state, *args: motion_ode(
                time, state, *args)[1]

            sol = solve_ivp(
                motion_ode,
                [np.min(time), np.max(time)],
                initial_state[i],
                t_eval=time,
                args=ode_args(i),
                events=[neutral_buoyancy, hit_ground, min_height])

            height[i,:len(sol.y[0,:])] = sol.y[0,:]
            velocity[i,:len(sol.y[1,:])] = sol.y[1,:]

            # record times of events
            # sol.t_events[i].size == 0 means the event did not occur
            neutral_buoyancy_time[i] = (  # record only the first instance
                sol.t_events[0][0] if sol.t_events[0].size > 0 else np.nan)
            hit_ground_time[i] = (
                sol.t_events[1][0] if sol.t_events[1].size > 0 else np.nan)
            min_height_time[i] = (
                sol.t_events[2][0] if sol.t_events[2].size > 0 else np.nan)

            # record states at event times
            neutral_buoyancy_height[i] = (  # record only the first instance
                sol.y_events[0][0,0] if sol.y_events[0].size > 0 else np.nan)
            neutral_buoyancy_velocity[i] = (  # record only the first instance
                sol.y_events[0][0,1] if sol.y_events[0].size > 0 else np.nan)
            hit_ground_velocity[i] = (
                sol.y_events[1][0,1] if sol.y_events[1].size > 0 else np.nan)
            min_height_height[i] = (
                sol.y_events[2][0,0] if sol.y_events[2].size > 0 else np.nan)

        result = MotionResult()
        result.height = np.squeeze(height)*units.meter
        result.velocity = np.squeeze(velocity)*units.meter/units.second
        result.neutral_buoyancy_time = (
            np.squeeze(neutral_buoyancy_time)*units.second)
        result.hit_ground_time = np.squeeze(hit_ground_time)*units.second
        result.min_height_time = np.squeeze(min_height_time)*units.second
        result.neutral_buoyancy_height = (
            np.squeeze(neutral_buoyancy_height)*units.meter)
        result.neutral_buoyancy_velocity = (
            np.squeeze(neutral_buoyancy_velocity)*units.meter/units.second)
        result.hit_ground_velocity = (
            np.squeeze(hit_ground_velocity)*units.meter/units.second)
        result.min_height = np.squeeze(min_height_height)*units.meter

        return result


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
